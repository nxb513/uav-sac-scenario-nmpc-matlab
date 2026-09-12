function [episode, agent] = generate_sac_teacher_dataset_episode( ...
        agent, cfg, specification)
%GENERATE_SAC_TEACHER_DATASET_EPISODE Generate one closed-loop teacher episode.

stepCount = cfg.dataset.stepsPerEpisode;
lookahead = cfg.dataset.referenceLookahead;
referenceCount = stepCount + max(cfg.sac.environment.horizonBank) + ...
    lookahead + 1;
reference = quad_reference_trajectory(specification.referenceType, ...
    cfg.teacher.nmpc.sampleTime, referenceCount, specification.referenceOptions);
[thetaPlant, thetaScenarios] = episode_uncertainty(cfg, specification);
disturbance = quad_generate_disturbance_episode(cfg.teacher.nmpc.plant, ...
    cfg.sac.disturbance, specification.disturbanceType, 'train', ...
    specification.disturbanceLevel, cfg.teacher.nmpc.sampleTime, ...
    stepCount, specification.disturbanceSeed);

previousRng = rng;
cleanup = onCleanup(@() rng(previousRng));
rng(specification.initialStateSeed, 'twister');
x0 = reference(:, 1) + specification.initialStateScale .* ...
    cfg.sac.environment.initialStateStd .* randn(12, 1);

X = nan(12, stepCount + 1);
U = nan(4, stepCount);
action = nan(cfg.sac.action.dimension, stepCount);
solveTime = nan(1, stepCount);
solverCallWallTime = nan(1, stepCount);
policyTime = nan(1, stepCount);
stepWallTime = nan(1, stepCount);
featureTime = zeros(1, stepCount);
horizon = nan(1, stepCount);
exitflag = nan(1, stepCount);
predictionResidual = nan(12, stepCount + 1);
X(:, 1) = x0;
predictionResidual(:, 1) = zeros(12, 1);

sampleCount = stepCount - cfg.dataset.discardInitialSteps;
features = nan(cfg.dataset.featureDimension, sampleCount, cfg.dataset.storageClass);
targets = nan(cfg.dataset.targetDimension, sampleCount, cfg.dataset.storageClass);
sampleAction = nan(cfg.sac.action.dimension, sampleCount, cfg.dataset.storageClass);
sampleSolveTime = nan(1, sampleCount, cfg.dataset.storageClass);
sampleHorizon = zeros(1, sampleCount, 'uint8');
sampleExitflag = zeros(1, sampleCount, 'int16');
sampleStepIndex = zeros(1, sampleCount, 'uint16');

previousInput = quad_hover_input(cfg.teacher.nmpc.plant.nominal);
previousSolveTime = 0.0;
previousHorizon = cfg.teacher.nmpc.predictionHorizon;
warmStart = [];
observation = rl_nmpc_make_observation(X(:, 1), reference(:, 1), ...
    previousInput, predictionResidual(:, 1), previousSolveTime, ...
    previousHorizon, cfg.sac);

episodeClock = tic;
for k = 1:stepCount
    stepClock = tic;
    policyClock = tic;
    [actionCell, agent] = getAction(agent, {observation});
    policyTime(k) = toc(policyClock);
    currentAction = double(actionCell{1}(:));
    [nmpcCfg, mapping] = rl_nmpc_action_to_config(currentAction, cfg.sac);
    warmStart = rl_nmpc_resize_warm_start(warmStart, ...
        nmpcCfg.predictionHorizon, cfg.teacher.nmpc.plant.nominal);
    tNow = (k - 1) * cfg.teacher.nmpc.sampleTime;
    Xref = nmpc_reference_window(reference, k, tNow, nmpcCfg);

    solverClock = tic;
    sol = scenario_nmpc_solve(X(:, k), Xref, thetaScenarios, ...
        nmpcCfg, warmStart);
    solverCallWallTime(k) = toc(solverClock);
    U(:, k) = sol.u0;
    action(:, k) = currentAction;
    solveTime(k) = sol.solveTime;
    horizon(k) = mapping.horizon;
    exitflag(k) = sol.exitflag;

    X(:, k + 1) = quad_step_rk4(tNow, X(:, k), U(:, k), ...
        cfg.teacher.nmpc.sampleTime, thetaPlant(1), disturbance);
    predictionResidual(:, k + 1) = X(:, k + 1) - sol.Xpred(:, 2);

    if k > cfg.dataset.discardInitialSteps
        sampleIndex = k - cfg.dataset.discardInitialSteps;
        featureClock = tic;
        feature = surrogate_build_feature(X(:, k - 3:k), ...
            U(:, k - 4:k - 1), reference(:, k:k + lookahead), ...
            predictionResidual(:, k));
        featureTime(k) = toc(featureClock);
        features(:, sampleIndex) = cast(feature, cfg.dataset.storageClass);
        targets(:, sampleIndex) = cast(U(:, k), cfg.dataset.storageClass);
        sampleAction(:, sampleIndex) = cast(currentAction, cfg.dataset.storageClass);
        sampleSolveTime(sampleIndex) = cast(sol.solveTime, cfg.dataset.storageClass);
        sampleHorizon(sampleIndex) = uint8(mapping.horizon);
        sampleExitflag(sampleIndex) = int16(sol.exitflag);
        sampleStepIndex(sampleIndex) = uint16(k);
    end

    warmStart = nmpc_shift_sequence(sol.U, sol.U(:, end));
    nextReference = reference(:, k + 1);
    observation = rl_nmpc_make_observation(X(:, k + 1), nextReference, ...
        U(:, k), predictionResidual(:, k + 1), sol.solveTime, ...
        mapping.horizon, cfg.sac);
    stepWallTime(k) = toc(stepClock);
end

episode.wallTimeSeconds = toc(episodeClock);
episode.specification = specification;
episode.reference = reference;
episode.thetaPlant = thetaPlant(1);
episode.thetaScenarios = thetaScenarios;
episode.disturbance = disturbance;
episode.X = X;
episode.U = U;
episode.action = action;
episode.predictionResidual = predictionResidual;
episode.solveTime = solveTime;
episode.solverCallWallTime = solverCallWallTime;
episode.policyTime = policyTime;
episode.featureTime = featureTime;
episode.stepWallTime = stepWallTime;
episode.horizon = horizon;
episode.exitflag = exitflag;
episode.features = features;
episode.targets = targets;
episode.sampleAction = sampleAction;
episode.sampleSolveTime = sampleSolveTime;
episode.sampleHorizon = sampleHorizon;
episode.sampleExitflag = sampleExitflag;
episode.sampleStepIndex = sampleStepIndex;
end

function [thetaPlant, thetaScenarios] = episode_uncertainty(cfg, specification)
plantCfg = cfg.teacher.nmpc.plant;
rho = plantCfg.uncertainty.train.rho(:);
if isfield(specification, 'plantXi') && ~isempty(specification.plantXi)
    thetaPlant = quad_apply_uncertainty(plantCfg.nominal, ...
        specification.plantXi(:), rho);
else
    sampled = quad_sample_uncertainty(plantCfg, 1, 'train', ...
        specification.plantSeed, cfg.teacher.nmpc.scenario.method);
    thetaPlant = sampled(1);
end

if isfield(specification, 'scenarioXi') && ~isempty(specification.scenarioXi)
    Xi = specification.scenarioXi;
    if size(Xi, 1) ~= 14 || ...
            size(Xi, 2) ~= cfg.teacher.nmpc.scenario.count
        error('generate_sac_teacher_dataset_episode:BadScenarioXi', ...
            'scenarioXi must be 14-by-scenarioCount.');
    end
    thetaScenarios(1, size(Xi, 2)) = plantCfg.nominal;
    for index = 1:size(Xi, 2)
        thetaScenarios(index) = quad_apply_uncertainty( ...
            plantCfg.nominal, Xi(:, index), rho);
    end
else
    thetaScenarios = quad_sample_uncertainty(plantCfg, ...
        cfg.teacher.nmpc.scenario.count, 'train', ...
        specification.scenarioSeed, cfg.teacher.nmpc.scenario.method);
end
end
