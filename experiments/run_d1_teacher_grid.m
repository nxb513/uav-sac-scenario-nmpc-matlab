function run_d1_teacher_grid(varargin)
%RUN_D1_TEACHER_GRID Stage S3: scenario-NMPC teacher config grid (sharded).
%
% For each (config, teacher-dev case) in this shard, runs a full closed-loop
% scenario-NMPC teacher episode on the case's deterministic reference under its
% sampled plant uncertainty, and records whether the teacher stays within the
% 20-step tracking spec over the whole episode. Writes one checkpoint file per
% (config,case) so a timed-out job resumes by skipping completed tasks. The
% aggregator select_d1_teacher.m then picks g* = argmin episode-violation rate.
%
% Sharding / config via env (workflow) or name-value args (local):
%   D1_CONFIG_INDEX : 0..7 or -1 = all configs      (arg 'configIndex')
%   D1_CASE_BLOCK   : 0,1 or -1 = all cases          (arg 'caseBlock')
%   D1_SELECTION_STEPS : rollout length (default 200; use small for smoke test)
%   D1_RESUME_ROOT  : dir with prior task files to resume from
%   NMPC_MAX_WALL_SECONDS : per-solve wall guard (default 60)
%
% Frozen scope (prereg v1 + amendment v1.2, FULL scope): M=5, Nc=5, 8 configs
% (posAtt penalty {0.5,2} x input penalty {0.5,2} x horizon {20,30}), 24 dev.

opts = parse_args(varargin{:});
add_project_paths();

projectRoot = project_root();
outDir = fullfile(projectRoot, 'results', 'd1_teacher_grid', 'tasks');
if ~exist(outDir, 'dir'); mkdir(outDir); end

weakCfg = targeted_lqr_weak_config();
refCfg = weakCfg.reference;
theta = weakCfg.plant.nominal;
dt = weakCfg.sampleTime;              % 0.05
selSteps = opts.selectionSteps;       % 200 (full episode)

devCases = load_teacher_dev_cases(projectRoot);
configs = build_configs();            % 1x8 struct array of NMPC cfgs + labels

configIdx = resolve_config_indices(opts.configIndex, numel(configs));
if opts.caseIndex > 0
    caseIdx = opts.caseIndex;            % single-case override (smoke test)
else
    caseIdx = resolve_case_indices(opts.caseBlock, opts.numBlocks, ...
        numel(devCases));
end

fprintf('== S3 run_d1_teacher_grid ==\n');
fprintf('configs=%s  cases=%s  selSteps=%d  wall=%ds  devN=%d\n', ...
    mat2str(configIdx - 1), mat2str(caseIdx), selSteps, ...
    opts.maxWallSeconds, numel(devCases));

% Contraction labeling on the common Lyapunov metric V = e'Pe (P = discrete LQR
% Riccati selectedLqr.S; NO fixed error threshold). We run the FULL episode and
% store the tracking-error trajectory; the teacher's finite-horizon contraction of
% V over H steps is a raw diagnostic (NOT the final S7 confidence target). Tracking
% accuracy (RMS/max position/attitude/velocity/rate) is recorded SEPARATELY and is
% not the contraction/divergence definition.
% H comes from the single source-of-truth cfg.contraction.horizonSteps (never
% hard-coded here or in the analysis module).
H = weakCfg.contraction.horizonSteps;

% FAIL-FAST: the D1 main run REQUIRES P = selectedLqr.S. No NaN / identity / Qf /
% recompute fallback is permitted; a bad or missing P stops the run before any
% expensive solve. d1_load_lyapunov_P is the ONLY loader and validates P.
try
    [P_lyap, Pinfo] = d1_load_lyapunov_P();
catch loadErr
    error('D1:LyapunovPUnavailable', ...
        ['D1 main run requires the Lyapunov matrix P = selectedLqr.S but it '...
         'could not be loaded (no fallback permitted).\n  underlying: %s\n'...
         '  loader: d1_load_lyapunov_P'], loadErr.message);
end
fprintf('Lyapunov metric:\n');
fprintf('  P = selectedLqr.S\n');
fprintf('  source = %s\n', Pinfo.source);
fprintf('  minEig(P) = %.6g\n', Pinfo.minEig);
fprintf('  H = %d steps  (cfg.contraction.horizonSteps)\n', H);
fprintf('  Ts = %.4g s\n', dt);
fprintf('  physical contraction horizon = %.4g s\n', H * dt);

for ci = configIdx
    cfg = configs(ci).cfg;
    cfg.solver.maxWallSeconds = opts.maxWallSeconds;
    for kj = caseIdx
        dc = devCases(kj);
        taskName = sprintf('cfg%02d_%s.mat', ci, sanitize(dc.groupId));
        taskPath = fullfile(outDir, taskName);
        if task_done(taskPath, opts.resumeRoot, taskName)
            fprintf('  skip (done): %s\n', taskName);
            continue;
        end
        result = run_one_task(configs(ci), dc, refCfg, theta, dt, ...
            selSteps, H, opts.caseBudgetSeconds, P_lyap);
        save(taskPath, 'result', '-v7');
        fprintf(['  [cfg%02d %s] div=%d reason=%s steps=%d/%d rmsPos=%.3f ' ...
            'maxPos=%.3f contract%%=%.0f meanSolve=%.2fs conv%%=%.0f\n'], ci, ...
            dc.groupId, result.diverged, result.stopReason, ...
            result.stepsCompleted, result.stepsRequested, result.rmsPositionM, ...
            result.maxPositionM, 100 * result.fractionContracting, ...
            result.meanSolveTime, 100 * result.convergedFraction);
    end
end
fprintf('S3 shard complete. Task files in %s\n', outDir);
end

% ------------------------------------------------------------------------
function result = run_one_task(configEntry, dc, refCfg, theta, dt, ...
        selSteps, H, caseBudget, P_lyap)
cfg = configEntry.cfg;
% Deterministic reference for this case (Xref state + Uref flatness feedforward).
[Xref, ~, ~, Uref] = d1_regenerate_reference(dc.family, dc.speed, dc.accel, ...
    dc.rep, refCfg, theta, dt, selSteps);
% Hidden plant theta (targeted envelope) + scenario thetas known to teacher.
plantCfg = cfg.plant;
plantCfg.uncertainty.targeted = targeted_uncertainty(plantCfg);
thetaPlant = quad_sample_uncertainty(plantCfg, 1, 'targeted', ...
    double(d1_case_seed([dc.groupId '|plant'])), 'lhs');
thetaScenarios = quad_sample_uncertainty(plantCfg, cfg.scenario.count, ...
    'train', double(d1_case_seed([dc.groupId '|scen'])), 'lhs');

% Closed-loop teacher rollout over the FULL horizon. NO first-crossing early
% stop and NO fixed error threshold: teacher quality is measured by how the
% tracking error EVOLVES (contracts vs diverges), so we need the whole
% trajectory. We stop only on genuine numerical divergence (non-finite state)
% or a gross-divergence anchor (position error beyond 5x the reference's own
% extent, i.e. the vehicle has clearly left the arena; class-R compute saver,
% not a success threshold), plus a wall-budget safety net.
refCols = size(Xref, 2);
refPosExtent = max(vecnorm(Xref(1:3, :), 2, 1));
grossBound = max(5 * refPosExtent, 5.0);
X = nan(12, selSteps + 1);
X(:, 1) = Xref(:, 1);
solveTimes = zeros(1, selSteps);
exitflags = zeros(1, selSteps);
warmStart = nmpc_default_warm_start(cfg.plant.nominal, cfg.predictionHorizon);
reason = 'complete';
stepsDone = 0;
caseClock = tic;
for k = 1:selSteps
    if ~all(isfinite(X(:, k)))
        reason = 'nonfinite'; break;
    end
    tNow = (k - 1) * dt;
    Xwin = nmpc_reference_window(Xref, k, tNow, cfg);
    Uwin = input_ref_window(Uref, k, cfg.predictionHorizon);
    sol = scenario_nmpc_solve(X(:, k), struct('X', Xwin, 'U', Uwin), ...
        thetaScenarios, cfg, warmStart);
    solveTimes(k) = sol.solveTime;
    exitflags(k) = sol.exitflag;
    X(:, k + 1) = quad_step_rk4(tNow, X(:, k), sol.u0, dt, thetaPlant, []);
    warmStart = nmpc_shift_sequence(sol.U, sol.U(:, end));
    stepsDone = k;
    if ~all(isfinite(X(:, k + 1)))
        reason = 'nonfinite'; break;
    end
    if norm(X(1:3, k + 1) - Xref(1:3, min(k + 1, refCols))) > grossBound
        reason = 'grossdiverge'; break;
    end
    if toc(caseClock) > caseBudget
        reason = 'budget'; break;
    end
end

m = min(stepsDone + 1, refCols);        % clip both to the common length
E = X(:, 1:m) - Xref(:, 1:m);
result = build_result(configEntry, dc, E, H, reason, stepsDone, selSteps, ...
    solveTimes, exitflags, grossBound, P_lyap);
end

% ------------------------------------------------------------------------
function result = build_result(configEntry, dc, E, H, reason, stepsDone, ...
        selSteps, solveTimes, exitflags, grossBound, P_lyap)
% Two SEPARATE outcome families (kept distinct on purpose):
%  (1) TRACKING ACCURACY  - per-channel error magnitudes (position/attitude/
%      velocity/rate). These are accuracy, NOT the contraction/divergence label.
%  (2) LYAPUNOV CONTRACTION - the PRIMARY finite-horizon metric on V = e'Pe
%      (P = discrete LQR Riccati selectedLqr.S), computed by
%      d1_finite_horizon_contraction. NO fixed error threshold is used.
% The FULL error trajectory E is stored so V, the confidence targets, and any
% norm can be recomputed OFFLINE without re-running the teacher.

% --- (1) tracking accuracy (descriptive, threshold-free magnitudes) ---
p = vecnorm(E(1:3, :), 2, 1);            % position error (m)
att = rad2deg(vecnorm(E(4:6, :), 2, 1)); % attitude error (deg, reporting only)
vel = vecnorm(E(7:9, :), 2, 1);
rate = vecnorm(E(10:12, :), 2, 1);

% --- (2) Lyapunov finite-horizon contraction V=e'Pe ---
con = struct('fractionContracting', NaN, 'meanG_H', NaN, 'medianG_H', NaN, ...
    'maxRmax', NaN, 'nInsufficientHorizon', NaN, 'nNonfiniteWindow', NaN, ...
    'Vstart', NaN, 'Vend', NaN);
if ~isempty(P_lyap) && size(E, 2) > H
    c = d1_finite_horizon_contraction(E, P_lyap, H);
    con.fractionContracting = c.fractionContracting;
    con.meanG_H = c.meanG_H;
    con.medianG_H = c.medianG_H;
    con.maxRmax = c.maxRmax;
    con.nInsufficientHorizon = c.nInsufficientHorizon;
    con.nNonfiniteWindow = c.nNonfiniteWindow;
    con.Vstart = c.Vstart;
    con.Vend = c.Vend;
end

result = struct();
result.groupId = dc.groupId;
result.configLabel = configEntry.label;
result.configIndex = configEntry.index;
result.stopReason = reason;
% TRUE divergence = the error blew up (non-finite state or gross-divergence
% anchor). A wall-budget stop is a COMPUTE limit, not a divergence: the tracking
% so far was fine, so it is recorded separately and must NOT be counted as a
% teacher failure (this keeps a long episode from being corrupted by a slow
% shard timing out).
result.diverged = strcmp(reason, 'nonfinite') || strcmp(reason, 'grossdiverge');
result.budgetStopped = strcmp(reason, 'budget');
result.complete = strcmp(reason, 'complete');
result.stepsCompleted = stepsDone;
result.stepsRequested = selSteps;
% (1) tracking accuracy
result.maxPositionM = finite_max(p);
result.rmsPositionM = sqrt(mean(p(isfinite(p)) .^ 2));
result.maxAttitudeDeg = finite_max(att);
result.maxVelocityMps = finite_max(vel);
result.maxBodyRateRadps = finite_max(rate);
% (2) Lyapunov contraction (PRIMARY loss-of-tracking / growth metric)
result.fractionContracting = con.fractionContracting;   % share of states with dV_H<0
result.meanG_H = con.meanG_H;                           % mean (1/H)log(V_{k+H}/V_k)
result.medianG_H = con.medianG_H;
result.maxRmax = con.maxRmax;                           % worst transient V_max/V_k
result.nInsufficientHorizon = con.nInsufficientHorizon; % states excluded (k+H>N)
result.nNonfiniteWindow = con.nNonfiniteWindow;
result.Vstart = con.Vstart;
result.Vend = con.Vend;
% solver + bookkeeping
result.meanSolveTime = mean(solveTimes(1:max(stepsDone, 1)));
result.totalSolveTime = sum(solveTimes(1:stepsDone));
result.convergedFraction = mean(exitflags(1:max(stepsDone, 1)) > 0);
result.finite = all(isfinite(E(:)));
result.grossBound = grossBound;
result.windowH = H;
result.errorTrajectory = E;      % 12 x (stepsDone+1) full tracking error
end

% ------------------------------------------------------------------------
function configs = build_configs()
% FINAL NMPC teacher: a SINGLE fully-specified Bryson configuration -- NO tuning
% knob. Q,R are the physically-normalized Bryson base (step2_nmpc_config), R is
% used as-is (lambda_R = 1), Qf = 0, dU = 0, prediction horizon Np = 20. There is
% nothing to grid-search: S3 becomes VERIFICATION of this one teacher on the 24
% teacher-dev cases (does it track / not diverge), not a selection.
base = step2_nmpc_config();
Np = 20;
cfg = base;
cfg.scenario.count = 5;               % M=5
cfg.predictionHorizon = Np;           % Np = 20
cfg.controlHorizon = 5;               % Nc = 5
cfg.solver.algorithm = 'sqp';
cfg.solver.maxIterations = 80;
cfg.solver.maxFunctionEvaluations = 1500;
cfg.weights.inputReference = 'hover'; % per-scenario hover
cfg.weights.Q = base.weights.Q;       % Bryson
cfg.weights.R = base.weights.R;       % Bryson (lambda_R = 1, no scaling)
cfg.weights.Qf = zeros(12);           % terminal not amplified (stage already weights e_N by Q)
cfg.weights.dU = zeros(4);            % no input-rate spec -> dU = 0
configs = struct('cfg', cfg, 'index', 1, ...
    'label', sprintf('bryson_N%d', Np));
end

% ------------------------------------------------------------------------
function devCases = load_teacher_dev_cases(projectRoot)
bankDir = fullfile(projectRoot, 'results', 'd1_bank');
files = dir(fullfile(bankDir, 'reference_bank_*.mat'));
if isempty(files)
    error('run_d1_teacher_grid:NoBank', ...
        'No reference bank in %s. Run build_d1_reference_bank first.', bankDir);
end
[~, order] = sort([files.datenum], 'descend');
data = load(fullfile(bankDir, files(order(1)).name), 'manifest');
cases = data.manifest.cases;
devCases = cases([cases.isTeacherDev]);
if isempty(devCases)
    error('run_d1_teacher_grid:NoDevCases', 'Bank has no teacher-dev cases.');
end
end

function u = targeted_uncertainty(plantCfg)
weakCfg = targeted_lqr_weak_config();
u.rho = weakCfg.uncertainty.targeted.rho;
end

% ------------------------------------------------------------------------
function done = task_done(taskPath, resumeRoot, taskName)
done = exist(taskPath, 'file') == 2;
if ~done && ~isempty(resumeRoot)
    priorPath = fullfile(resumeRoot, 'tasks', taskName);
    if exist(priorPath, 'file') == 2
        copyfile(priorPath, taskPath);
        done = true;
    end
end
end

function idx = resolve_config_indices(configIndex, nConfig)
if configIndex < 0
    idx = 1:nConfig;
else
    idx = configIndex + 1;   % env is 0-based
end
end

function idx = resolve_case_indices(caseBlock, numBlocks, nCases)
if caseBlock < 0
    idx = 1:nCases;
    return;
end
numBlocks = max(1, round(numBlocks));
edges = round(linspace(0, nCases, numBlocks + 1));
lo = edges(caseBlock + 1) + 1;
hi = edges(caseBlock + 2);
idx = lo:hi;
end

function opts = parse_args(varargin)
opts = struct('configIndex', env_num('D1_CONFIG_INDEX', -1), ...
    'caseBlock', env_num('D1_CASE_BLOCK', -1), ...
    'numBlocks', env_num('D1_NUM_BLOCKS', 3), ...
    'caseIndex', env_num('D1_CASE_INDEX', -1), ...
    'selectionSteps', env_num('D1_SELECTION_STEPS', 200), ...
    'maxWallSeconds', env_num('NMPC_MAX_WALL_SECONDS', 60), ...
    'caseBudgetSeconds', env_num('D1_CASE_BUDGET_SECONDS', 18000), ...
    'resumeRoot', getenv_default('D1_RESUME_ROOT', ''));
for k = 1:2:numel(varargin)
    opts.(varargin{k}) = varargin{k + 1};
end
end

function v = env_num(name, default)
s = getenv(name);
if isempty(s); v = default; else; v = str2double(s); end
if ~isfinite(v); v = default; end
end

function s = getenv_default(name, default)
s = getenv(name); if isempty(s); s = default; end
end

function add_project_paths()
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(genpath(fullfile(root, 'src')));
addpath(fullfile(root, 'configs'));
end

function root = project_root()
root = fileparts(fileparts(mfilename('fullpath')));
end

function s = sanitize(str)
s = regexprep(str, '[^A-Za-z0-9]', '_');
end

function v = finite_max(x)
x = x(isfinite(x));
if isempty(x); v = Inf; else; v = max(x); end
end

function Uwin = input_ref_window(Uref, stepIndex, predictionHorizon)
% Feedforward input window aligned with the state reference window: input at
% prediction step j (j=1..Np) uses the flatness feedforward at time t_{k+j-1},
% i.e. Uref(:, k:k+Np-1), padded with the last column past the trajectory end.
n = size(Uref, 2);
lo = stepIndex;
hi = min(stepIndex + predictionHorizon - 1, n);
Uwin = Uref(:, lo:hi);
if size(Uwin, 2) < predictionHorizon
    Uwin = [Uwin, repmat(Uwin(:, end), 1, predictionHorizon - size(Uwin, 2))];
end
end
