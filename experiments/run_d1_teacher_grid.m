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

thresh = struct('positionM', 0.10, 'attitudeDeg', 5.0, ...
    'velocityMps', 0.30, 'bodyRateRadps', 2.0);

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
            selSteps, thresh, opts.caseBudgetSeconds);
        save(taskPath, 'result', '-v7');
        fprintf(['  [cfg%02d %s] viol=%d reason=%s steps=%d/%d maxPos=%.3f ' ...
            'meanSolve=%.2fs conv%%=%.0f\n'], ci, dc.groupId, ...
            result.episodeViolation, result.stopReason, result.stepsCompleted, ...
            result.stepsRequested, result.maxPositionM, result.meanSolveTime, ...
            100 * result.convergedFraction);
    end
end
fprintf('S3 shard complete. Task files in %s\n', outDir);
end

% ------------------------------------------------------------------------
function result = run_one_task(configEntry, dc, refCfg, theta, dt, ...
        selSteps, thresh, caseBudget)
cfg = configEntry.cfg;
% Deterministic reference for this case.
[Xref, ~] = d1_regenerate_reference(dc.family, dc.speed, dc.accel, dc.rep, ...
    refCfg, theta, dt, selSteps);
% Hidden plant theta (targeted envelope) + scenario thetas known to teacher.
plantCfg = cfg.plant;
plantCfg.uncertainty.targeted = targeted_uncertainty(plantCfg);
thetaPlant = quad_sample_uncertainty(plantCfg, 1, 'targeted', ...
    double(d1_case_seed([dc.groupId '|plant'])), 'lhs');
thetaScenarios = quad_sample_uncertainty(plantCfg, cfg.scenario.count, ...
    'train', double(d1_case_seed([dc.groupId '|scen'])), 'lhs');

% Bounded closed-loop teacher rollout with a per-case wall budget so a single
% hard case cannot drag a shard past the job timeout: if the budget is exceeded
% the case is stopped and recorded as a teacher failure (reason 'budget').
X = nan(12, selSteps + 1);
X(:, 1) = Xref(:, 1);
solveTimes = zeros(1, selSteps);
exitflags = zeros(1, selSteps);
warmStart = nmpc_default_warm_start(cfg.plant.nominal, cfg.predictionHorizon);
reason = 'complete';
stepsDone = 0;
caseClock = tic;
refCols = size(Xref, 2);
for k = 1:selSteps
    % Stop BEFORE solving if the teacher has already lost tracking: a diverged
    % state is both an episode violation already and the input that makes
    % fmincon stall for a very long time (which the between-iteration wall guard
    % cannot cap). Catching it here keeps every case bounded.
    posErr = norm(X(1:3, k) - Xref(1:3, min(k, refCols)));
    if ~all(isfinite(X(:, k))) || posErr > 2.0 || max(abs(X(:, k))) > 500
        reason = 'diverged'; break;
    end
    tNow = (k - 1) * dt;
    Xwin = nmpc_reference_window(Xref, k, tNow, cfg);
    sol = scenario_nmpc_solve(X(:, k), Xwin, thetaScenarios, cfg, warmStart);
    solveTimes(k) = sol.solveTime;
    exitflags(k) = sol.exitflag;
    X(:, k + 1) = quad_step_rk4(tNow, X(:, k), sol.u0, dt, thetaPlant, []);
    warmStart = nmpc_shift_sequence(sol.U, sol.U(:, end));
    stepsDone = k;
    if ~all(isfinite(X(:, k + 1)))
        reason = 'nonfinite'; break;
    end
    if toc(caseClock) > caseBudget
        reason = 'budget'; break;
    end
end

usedX = X(:, 1:stepsDone + 1);
[viol, maxErr] = episode_violation(usedX, Xref, thresh);  % clips to min cols
% A truncated case (budget/nonfinite before the full horizon) is a failure.
if ~strcmp(reason, 'complete')
    viol = true;
end
result = struct();
result.groupId = dc.groupId;
result.configLabel = configEntry.label;
result.configIndex = configEntry.index;
result.episodeViolation = viol;
result.stopReason = reason;
result.stepsCompleted = stepsDone;
result.stepsRequested = selSteps;
result.maxPositionM = maxErr.pos;
result.maxAttitudeDeg = maxErr.att;
result.maxVelocityMps = maxErr.vel;
result.maxBodyRateRadps = maxErr.rate;
result.cumulativeNormError = maxErr.cumNorm;
result.meanSolveTime = mean(solveTimes(1:max(stepsDone, 1)));
result.totalSolveTime = sum(solveTimes(1:stepsDone));
result.convergedFraction = mean(exitflags(1:max(stepsDone, 1)) > 0);
result.finite = all(isfinite(usedX(:)));
end

% ------------------------------------------------------------------------
function [viol, maxErr] = episode_violation(X, Xref, thresh)
n = min(size(X, 2), size(Xref, 2));
E = X(:, 1:n) - Xref(:, 1:n);
pos = vecnorm(E(1:3, :), 2, 1);
att = rad2deg(vecnorm(E(4:6, :), 2, 1));
vel = vecnorm(E(7:9, :), 2, 1);
rate = vecnorm(E(10:12, :), 2, 1);
cross = pos >= thresh.positionM | att >= thresh.attitudeDeg | ...
    vel >= thresh.velocityMps | rate >= thresh.bodyRateRadps;
viol = any(cross) || ~all(isfinite(X(:)));
maxErr.pos = finite_max(pos); maxErr.att = finite_max(att);
maxErr.vel = finite_max(vel); maxErr.rate = finite_max(rate);
norm2 = (pos / thresh.positionM) .^ 2 + (att / thresh.attitudeDeg) .^ 2 + ...
    (vel / thresh.velocityMps) .^ 2 + (rate / thresh.bodyRateRadps) .^ 2;
maxErr.cumNorm = sum(norm2(isfinite(norm2)));
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
            % Hard time bound per solve via MaxFunctionEvaluations (fmincon checks
            % it after EVERY evaluation, unlike the between-iteration wall guard
            % which can miss a stalled solve). ~400 evals ~= 12 s local / ~30 s on
            % the 2-core CI runner; the local test showed tracking stays tight
            % (posErr < 0.01) even under-converged. Frozen for all configs.
            cfg.solver.maxIterations = 40;
            cfg.solver.maxFunctionEvaluations = 400;
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
    'caseBudgetSeconds', env_num('D1_CASE_BUDGET_SECONDS', 2700), ...
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
