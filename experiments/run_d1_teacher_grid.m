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

% Growth-based labeling: NO fixed error threshold. We run the FULL episode and
% store the tracking-error trajectory; teacher quality = how well it CONTRACTS the
% error (the same forward-growth quantity g that later defines the confidence c and
% the blend weight alpha). H is the forward window = the intervention lead time.
H = 20;

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
            selSteps, H, opts.caseBudgetSeconds);
        save(taskPath, 'result', '-v7');
        fprintf(['  [cfg%02d %s] div=%d reason=%s steps=%d/%d rmsPos=%.3f ' ...
            'maxPos=%.3f expFrac=%.2f meanSolve=%.2fs conv%%=%.0f\n'], ci, ...
            dc.groupId, result.diverged, result.stopReason, ...
            result.stepsCompleted, result.stepsRequested, result.rmsPositionM, ...
            result.maxPositionM, result.expansionFraction, ...
            result.meanSolveTime, 100 * result.convergedFraction);
    end
end
fprintf('S3 shard complete. Task files in %s\n', outDir);
end

% ------------------------------------------------------------------------
function result = run_one_task(configEntry, dc, refCfg, theta, dt, ...
        selSteps, H, caseBudget)
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
    solveTimes, exitflags, grossBound);
end

% ------------------------------------------------------------------------
function result = build_result(configEntry, dc, E, H, reason, stepsDone, ...
        selSteps, solveTimes, exitflags, grossBound)
% Descriptive error signals (per channel) + the forward-growth metric on the
% position-error signal. The FULL error trajectory E is stored so the exact
% growth quantity g, the confidence targets, and any error norm can be recomputed
% OFFLINE without re-running the teacher (which is the expensive part).
p = vecnorm(E(1:3, :), 2, 1);            % position-error signal (m)
att = rad2deg(vecnorm(E(4:6, :), 2, 1)); % attitude error (deg, descriptive)
vel = vecnorm(E(7:9, :), 2, 1);
rate = vecnorm(E(10:12, :), 2, 1);
nCols = numel(p);

% Forward H-step growth of the position error: g_k = p(k+H)/p(k). A near-zero
% floor (1% of the episode's peak error) keeps the ratio finite when the teacher
% is momentarily perfect; it only affects the log-ratio scale, not the sign of
% growth. These aggregates are provisional; the stored E allows the final
% definition to be pinned at S7.
if nCols > H
    p0 = p(1:nCols - H); pH = p(1 + H:nCols);
    floorv = max(1e-6, 0.01 * max(p));
    expansionFraction = mean(pH > p0);                 % share of steps expanding
    meanLogGrowth = mean(log(max(pH, floorv) ./ max(p0, floorv)));
    maxGrowthFactor = max(pH ./ max(p0, floorv));
else
    expansionFraction = NaN; meanLogGrowth = NaN; maxGrowthFactor = NaN;
end

result = struct();
result.groupId = dc.groupId;
result.configLabel = configEntry.label;
result.configIndex = configEntry.index;
result.stopReason = reason;
result.diverged = ~strcmp(reason, 'complete');   % nonfinite/grossdiverge/budget
result.stepsCompleted = stepsDone;
result.stepsRequested = selSteps;
result.maxPositionM = finite_max(p);
result.rmsPositionM = sqrt(mean(p(isfinite(p)) .^ 2));
result.maxAttitudeDeg = finite_max(att);
result.maxVelocityMps = finite_max(vel);
result.maxBodyRateRadps = finite_max(rate);
result.expansionFraction = expansionFraction;
result.meanLogGrowth = meanLogGrowth;
result.maxGrowthFactor = maxGrowthFactor;
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
base = step2_nmpc_config();
posAttScales = [0.5, 2];
inputScales = [0.5, 2];
horizons = [20, 30];
configs = struct('cfg', {}, 'label', {}, 'index', {});
idx = 0;
for pa = posAttScales
    for is = inputScales
        for N = horizons
            idx = idx + 1;
            cfg = base;
            cfg.scenario.count = 5;               % M=5 full scope
            cfg.predictionHorizon = N;
            cfg.controlHorizon = 5;               % Nc=5
            cfg.solver.algorithm = 'sqp';
            % Converged teacher (does not compromise quality): warm-started solves
            % converge in ~40-67 iters, so maxIter=80 reaches the optimum, and
            % MaxFunctionEvaluations=1500 hard-bounds each solve (~60 s on the
            % 2-core CI runner) so none can run away. With 1-case-per-shard each
            % case has the full 5.5 h slot; the divergence check stops lost cases.
            cfg.solver.maxIterations = 80;
            cfg.solver.maxFunctionEvaluations = 1500;
            % Penalize control deviation from the time-varying flatness
            % feedforward u_ref(t), not from constant hover: on aggressive
            % (high-speed/high-accel) trajectories u_ref departs hover by up to
            % ~40% of hover thrust, so a hover-referenced penalty would fight the
            % necessary actuation and bias the input-penalty ranking axis.
            cfg.weights.inputReference = 'feedforward';
            q = diag(base.weights.Q);
            q(1:6) = q(1:6) * pa;                 % position+attitude penalty
            cfg.weights.Q = diag(q);
            cfg.weights.Qf = 4.0 * cfg.weights.Q;
            cfg.weights.R = base.weights.R * is;  % input penalty
            configs(idx) = struct('cfg', cfg, 'index', idx, ...
                'label', sprintf('posAtt%.1f_input%.1f_N%d', pa, is, N)); %#ok<AGROW>
        end
    end
end
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
