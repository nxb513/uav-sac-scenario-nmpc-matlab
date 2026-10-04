function [rollout, agent] = step7_rollout_frozen_teacher(scenario, agent, context)
%STEP7_ROLLOUT_FROZEN_TEACHER Run frozen SAC-scenario-NMPC on one OOD plant.

cfg = context.cfg;
steps = cfg.stepCount;
warmup = cfg.warmupSteps;
X = nan(12, steps + 1);
U = nan(4, steps);
action = nan(cfg.sac.action.dimension, steps);
solveTime = zeros(1, steps);
controlTime = zeros(1, steps);
exitflag = ones(1, steps);
horizon = nan(1, steps);
X(:, 1) = scenario.x0;

for k = 1:warmup
    raw = lqr_command(X(:, k), scenario.reference(:, k), ...
        context.selection.selectedLqr);
    U(:, k) = quad_saturate_input(raw, cfg.plant.nominal);
    X(:, k + 1) = plant_step(scenario, X(:, k), U(:, k), k, cfg);
end

predictionResidual = recompute_residual(X, U, warmup + 1, cfg);
previousInput = U(:, warmup);
previousSolveTime = 0;
previousHorizon = cfg.teacher.nmpc.predictionHorizon;
observation = rl_nmpc_make_observation(X(:, warmup + 1), ...
    scenario.reference(:, warmup + 1), previousInput, ...
    predictionResidual, previousSolveTime, previousHorizon, cfg.sac);
thetaScenarios = quad_sample_uncertainty(cfg.plant, ...
    cfg.teacher.nmpc.scenario.count, 'train', scenario.scenarioSeed, ...
    cfg.teacher.nmpc.scenario.method);
warmStart = [];

for k = warmup + 1:steps
    clock = tic;
    [actionCell, agent] = getAction(agent, {observation});
    currentAction = double(actionCell{1}(:));
    [nmpcCfg, mapping] = rl_nmpc_action_to_config(currentAction, cfg.sac);
    warmStart = rl_nmpc_resize_warm_start(warmStart, ...
        nmpcCfg.predictionHorizon, cfg.plant.nominal);
    time = (k - 1) * cfg.sampleTime;
    Xref = nmpc_reference_window(scenario.reference, k, time, nmpcCfg);
    sol = scenario_nmpc_solve(X(:, k), Xref, thetaScenarios, ...
        nmpcCfg, warmStart);
    controlTime(k) = toc(clock);
    U(:, k) = sol.u0;
    action(:, k) = currentAction;
    solveTime(k) = sol.solveTime;
    exitflag(k) = sol.exitflag;
    horizon(k) = mapping.horizon;

    if ~all(isfinite(U(:, k)))
        exitflag(k:end) = 0;
        break;
    end

    X(:, k + 1) = plant_step(scenario, X(:, k), U(:, k), k, cfg);
    if ~all(isfinite(X(:, k + 1)))
        exitflag(k:end) = 0;
        break;
    end
    predictionResidual = X(:, k + 1) - sol.Xpred(:, 2);
    if ~all(isfinite(predictionResidual))
        exitflag(k:end) = 0;
        break;
    end
    warmStart = nmpc_shift_sequence(sol.U, sol.U(:, end));
    observation = rl_nmpc_make_observation(X(:, k + 1), ...
        scenario.reference(:, k + 1), U(:, k), predictionResidual, ...
        sol.solveTime, mapping.horizon, cfg.sac);
    if ~all(isfinite(observation))
        exitflag(k:end) = 0;
        break;
    end
end

rollout.X = X;
rollout.U = U;
rollout.rawU = U;
rollout.uSurrogate = nan(4, steps);
rollout.uFallback = nan(4, steps);
rollout.confidence = nan(1, steps);
rollout.alpha = nan(1, steps);
rollout.controlTime = controlTime;
rollout.solveTime = solveTime;
rollout.saturated = false(1, steps);
rollout.fallbackSaturated = false(1, steps);
rollout.exitflag = exitflag;
rollout.horizon = horizon;
rollout.action = action;
rollout.sampleTime = cfg.sampleTime;
rollout.evaluationStartStep = warmup + 1;
end

function raw = lqr_command(x, reference, design)
raw = design.uEquilibrium - design.K * nmpc_state_error(x, reference);
end

function nextState = plant_step(scenario, x, u, stepIndex, cfg)
time = (stepIndex - 1) * cfg.sampleTime;
theta = quad_apply_model_shift(scenario.thetaPlant, scenario.shift, time);
theta.inputLimits = cfg.plant.nominal.inputLimits;
nextState = quad_step_rk4(time, x, u, cfg.sampleTime, theta, ...
    scenario.disturbance);
end

function residual = recompute_residual(X, U, stepIndex, cfg)
time = (stepIndex - 2) * cfg.sampleTime;
predicted = quad_step_rk4(time, X(:, stepIndex - 1), ...
    U(:, stepIndex - 1), cfg.sampleTime, cfg.plant.nominal, []);
residual = X(:, stepIndex) - predicted;
end
