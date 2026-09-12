function episode = scenario_nmpc_teacher_rollout(x0, reference, thetaPlant, cfg, thetaScenarios, stepCount, plantDisturbanceSpec)
%SCENARIO_NMPC_TEACHER_ROLLOUT Receding-horizon scenario NMPC episode.
%
% thetaPlant is the hidden plant parameter sample used by the simulator.
% thetaScenarios are the parameter samples known to the teacher optimizer.

if nargin < 4 || isempty(cfg)
    cfg = step2_nmpc_config();
end
if nargin < 3 || isempty(thetaPlant)
    thetaPlant = cfg.plant.nominal;
end
if nargin < 5 || isempty(thetaScenarios)
    thetaScenarios = [];
end
if nargin < 6 || isempty(stepCount)
    stepCount = infer_step_count(reference, cfg);
end
if nargin < 7
    plantDisturbanceSpec = cfg.rollout.disturbance;
end

if stepCount <= 0 || stepCount ~= floor(stepCount)
    error('scenario_nmpc_teacher_rollout:BadStepCount', ...
          'stepCount must be a positive integer.');
end

x0 = x0(:);
if numel(x0) ~= 12
    error('scenario_nmpc_teacher_rollout:BadStateSize', ...
          'x0 must have 12 elements.');
end

horizon = cfg.predictionHorizon;
X = zeros(12, stepCount + 1);
U = zeros(4, stepCount);
cost = zeros(1, stepCount);
solveTime = zeros(1, stepCount);
exitflag = zeros(1, stepCount);
X(:, 1) = x0;

warmStart = nmpc_default_warm_start(cfg.plant.nominal, horizon);
solutions = cell(1, stepCount);

for k = 1:stepCount
    tNow = cfg.rollout.startTime + (k - 1) * cfg.sampleTime;
    Xref = nmpc_reference_window(reference, k, tNow, cfg);
    sol = scenario_nmpc_solve(X(:, k), Xref, thetaScenarios, cfg, warmStart);

    U(:, k) = sol.u0;
    cost(k) = sol.cost;
    solveTime(k) = sol.solveTime;
    exitflag(k) = sol.exitflag;
    solutions{k} = sol;

    X(:, k + 1) = quad_step_rk4(tNow, X(:, k), U(:, k), cfg.sampleTime, ...
                                thetaPlant, plantDisturbanceSpec);
    warmStart = nmpc_shift_sequence(sol.U, sol.U(:, end));
end

episode.X = X;
episode.U = U;
episode.cost = cost;
episode.solveTime = solveTime;
episode.exitflag = exitflag;
episode.solutions = solutions;
episode.thetaPlant = thetaPlant;
episode.thetaScenarios = thetaScenarios;
episode.sampleTime = cfg.sampleTime;
episode.stepCount = stepCount;
end

function stepCount = infer_step_count(reference, cfg)
if isnumeric(reference) && ~isempty(reference) && size(reference, 1) == 12 && size(reference, 2) > 1
    stepCount = max(1, size(reference, 2) - cfg.predictionHorizon);
else
    error('scenario_nmpc_teacher_rollout:MissingStepCount', ...
          'stepCount is required for constant or function-handle references.');
end
end
