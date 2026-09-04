function [env, observationInfo, actionInfo] = ...
        create_targeted_sac_nmpc_environment(cfg, runDir, contextBank)
%CREATE_TARGETED_SAC_NMPC_ENVIRONMENT SAC-NMPC on frozen LQR-weak contexts.

observationInfo = rlNumericSpec([cfg.observation.dimension, 1]);
observationInfo.Name = 'targeted_online_uav_features';
observationInfo.LowerLimit = -cfg.observation.clip .* ...
    ones(cfg.observation.dimension, 1);
observationInfo.UpperLimit = cfg.observation.clip .* ...
    ones(cfg.observation.dimension, 1);
actionInfo = rlNumericSpec([cfg.action.dimension, 1]);
actionInfo.Name = 'targeted_weight_and_nmpc_structure_action';
actionInfo.LowerLimit = -ones(cfg.action.dimension, 1);
actionInfo.UpperLimit = ones(cfg.action.dimension, 1);

env = rlFunctionEnv(observationInfo, actionInfo, ...
    @step_environment, @reset_environment);

    function [initialObservation, info] = reset_environment()
        workerId = current_worker_id();
        episodeIndex = next_worker_episode_index(runDir, workerId, cfg);
        episodeSeed = cfg.environment.baseEpisodeSeed + ...
            workerId * cfg.environment.workerSeedStride + episodeIndex;
        previousRng = rng;
        cleanup = onCleanup(@() rng(previousRng));
        rng(episodeSeed, 'twister');

        stageIndex = curriculum_stage(episodeIndex, cfg);
        eligible = eligible_contexts(contextBank, stageIndex);
        contextIndex = eligible(randi(numel(eligible)));
        scenario = contextBank(contextIndex);
        startStep = scenario.StartStep;
        lastReferenceStep = startStep + cfg.environment.stepsPerEpisode + ...
            max(cfg.environment.horizonBank) - 1;
        assert(lastReferenceStep <= size(scenario.Reference, 2), ...
            'Local context does not contain the required reference horizon.');
        reference = scenario.Reference(:, startStep:lastReferenceStep);

        scenarioSeed = episodeSeed + cfg.environment.scenarioSeedOffset;
        thetaScenarios = quad_sample_uncertainty(cfg.nmpc.plant, ...
            cfg.nmpc.scenario.count, 'targeted', ...
            scenarioSeed, ...
            cfg.nmpc.scenario.method);

        info.WorkerId = workerId;
        info.EpisodeIndex = episodeIndex;
        info.EpisodeSeed = episodeSeed;
        info.ContextIndex = contextIndex;
        info.ContextClass = string(scenario_field( ...
            scenario, 'ContextClass', 'hard_event'));
        info.ContextEpisodeIndex = scenario.SourceEpisodeIndex;
        info.AbsoluteStartStep = startStep;
        info.StageIndex = stageIndex;
        info.StepIndex = 0;
        info.State = scenario.X0;
        info.PreviousInput = scenario.PreviousInput;
        info.PreviousSolveTime = 0.0;
        info.PreviousHorizon = cfg.nmpc.predictionHorizon;
        info.PreviousControlHorizon = cfg.nmpc.controlHorizon;
        info.PreviousScenarioCount = cfg.nmpc.scenario.count;
        info.WarmStart = [];
        info.Reference = reference;
        info.ReferenceType = scenario.Family;
        info.ThetaPlant = scenario.ThetaPlant;
        info.ThetaScenarios = thetaScenarios;
        info.Disturbance = scenario.Disturbance;
        info.DisturbanceType = scenario.Disturbance.type;
        info.DisturbanceLevel = scenario.Disturbance.levelIndex;
        info.CounterfactualDivergence = scenario_field( ...
            scenario, 'CounterfactualDivergence', false);
        info.AlreadyOutsideEnvelope = scenario_field( ...
            scenario, 'AlreadyOutsideEnvelope', false);
        info.PreDivergenceEligible = scenario_field( ...
            scenario, 'PreDivergenceEligible', false);
        info.DivergenceEventStep = scenario_field( ...
            scenario, 'DivergenceEventStep', Inf);
        info.InterventionLeadSteps = scenario_field( ...
            scenario, 'InterventionLeadSteps', Inf);
        info.LqrRiskScore = scenario_field( ...
            scenario, 'LqrRiskScore', NaN);
        info.LqrLogEnergyGrowthPerStep = scenario_field( ...
            scenario, 'LqrLogEnergyGrowthPerStep', NaN);
        info.Log = initialize_log(cfg.environment.stepsPerEpisode, ...
            cfg.reward.componentNames, cfg.action.dimension);
        if cfg.logging.enabled
            append_manifest(runDir, info, scenarioSeed);
        end
        initialObservation = rl_nmpc_make_observation(info.State, ...
            reference(:, 1), info.PreviousInput, zeros(12, 1), 0.0, ...
            info.PreviousHorizon, cfg, info.PreviousControlHorizon);
        clear cleanup;
    end

    function [observation, reward, isDone, info] = ...
            step_environment(action, info)
        stepIndex = info.StepIndex + 1;
        action = min(max(double(action(:)), -1), 1);
        exceptionIdentifier = '';
        try
            [nmpcCfg, mapping] = rl_nmpc_action_to_config(action, cfg);
            warmStart = rl_nmpc_resize_warm_start(info.WarmStart, ...
                nmpcCfg.predictionHorizon, cfg.nmpc.plant.nominal);
            absoluteStep = info.AbsoluteStartStep + stepIndex - 1;
            time = (absoluteStep - 1) * cfg.environment.sampleTime;
            referenceWindow = nmpc_reference_window(info.Reference, ...
                stepIndex, time, nmpcCfg);
            solution = scenario_nmpc_solve(info.State, referenceWindow, ...
                info.ThetaScenarios(1:mapping.scenarioCount), ...
                nmpcCfg, warmStart);
            input = solution.u0;
            nextState = quad_step_rk4(time, info.State, input, ...
                cfg.environment.sampleTime, info.ThetaPlant, ...
                info.Disturbance);
            predictionResidual = nextState - solution.Xpred(:, 2);
            nextReference = info.Reference(:, min(stepIndex + 1, ...
                size(info.Reference, 2)));
            [reward, components, constraintViolation] = compute_reward( ...
                nextState, nextReference, input, info.PreviousInput, ...
                solution, nmpcCfg, cfg);
            finiteTransition = all(isfinite(nextState)) && ...
                all(isfinite(input)) && isfinite(reward) && ...
                isfinite(solution.solveTime);
            invalidEuler = abs(cos(nextState(5))) < ...
                cfg.environment.terminateCosPitchMargin;
            outsideRegion = any(abs(nextState(1:3)) > ...
                cfg.environment.maxAbsolutePosition);
            isDone = stepIndex >= cfg.environment.stepsPerEpisode || ...
                ~finiteTransition || solution.exitflag <= 0 || ...
                invalidEuler || outsideRegion || constraintViolation;
            info.WarmStart = nmpc_shift_sequence(solution.U, ...
                solution.U(:, end));
            solveTime = solution.solveTime;
            horizon = mapping.horizon;
            controlHorizon = mapping.controlHorizon;
            scenarioCount = mapping.scenarioCount;
            exitflag = solution.exitflag;
            solverTimedOut = solution.timedOut;
        catch exception
            exceptionIdentifier = exception.identifier;
            nextState = info.State;
            input = info.PreviousInput;
            predictionResidual = zeros(12, 1);
            reward = -cfg.reward.weights(end);
            components = zeros(numel(cfg.reward.componentNames), 1);
            components(end) = 1;
            isDone = true;
            solveTime = 0;
            horizon = info.PreviousHorizon;
            controlHorizon = info.PreviousControlHorizon;
            scenarioCount = info.PreviousScenarioCount;
            exitflag = -999;
            solverTimedOut = false;
        end

        nextReference = info.Reference(:, min(stepIndex + 1, ...
            size(info.Reference, 2)));
        observation = rl_nmpc_make_observation(nextState, nextReference, ...
            input, predictionResidual, solveTime, horizon, cfg, ...
            controlHorizon);
        info.Log.State(:, stepIndex) = nextState;
        info.Log.Input(:, stepIndex) = input;
        info.Log.Action(:, stepIndex) = action;
        info.Log.Reward(stepIndex) = reward;
        info.Log.RewardComponents(:, stepIndex) = components;
        info.Log.SolveTime(stepIndex) = solveTime;
        info.Log.Horizon(stepIndex) = horizon;
        info.Log.ControlHorizon(stepIndex) = controlHorizon;
        info.Log.ScenarioCount(stepIndex) = scenarioCount;
        info.Log.Exitflag(stepIndex) = exitflag;
        info.Log.SolverTimedOut(stepIndex) = solverTimedOut;
        info.Log.ExceptionIdentifier{stepIndex} = exceptionIdentifier;
        info.StepIndex = stepIndex;
        info.State = nextState;
        info.PreviousInput = input;
        info.PreviousSolveTime = solveTime;
        info.PreviousHorizon = horizon;
        info.PreviousControlHorizon = controlHorizon;
        info.PreviousScenarioCount = scenarioCount;
        if isDone && cfg.logging.enabled
            save_episode_log(runDir, info, cfg);
        end
    end
end

function stage = curriculum_stage(localEpisode, cfg)
if cfg.environment.forceCurriculumStage > 0
    stage = cfg.environment.forceCurriculumStage;
else
    workerCount = 1;
    if isfield(cfg.training, 'activeWorkerCount')
        workerCount = max(1, cfg.training.activeWorkerCount);
    end
    episodeBudget = cfg.training.candidateEpisodeCeiling;
    if isfield(cfg.training, 'executionEpisodeCeiling')
        episodeBudget = cfg.training.executionEpisodeCeiling;
    end
    localEpisodeBudget = ceil(episodeBudget / workerCount);
    stageBoundaryEpisodes = ceil( ...
        cfg.environment.curriculumTransitionFractions * localEpisodeBudget);
    stage = find(localEpisode <= [stageBoundaryEpisodes, inf], 1, 'first');
end
end

function indices = eligible_contexts(bank, stage)
disturbanceType = string(arrayfun(@(item) item.Disturbance.type, bank, ...
    'UniformOutput', false));
disturbanceLevel = arrayfun(@(item) item.Disturbance.levelIndex, bank);
contextClass = string({bank.ContextClass});
disturbanceType = disturbanceType(:);
disturbanceLevel = disturbanceLevel(:);
contextClass = contextClass(:);
switch stage
    case 1
        mask = disturbanceType == "zero";
    case 2
        mask = disturbanceType == "zero" | disturbanceLevel == 1;
    case 3
        mask = ismember(disturbanceType, ...
            ["gust", "sinusoidal", "stochastic"]);
    otherwise
        mask = contextClass == "hard_event";
end
indices = find(mask);
if isempty(indices)
    indices = 1:numel(bank);
end
end

function workerId = current_worker_id()
task = getCurrentTask();
if isempty(task)
    workerId = 0;
else
    workerId = task.ID;
end
end

function episodeIndex = next_worker_episode_index(runDir, workerId, cfg)
persistent counterByWorker
if isempty(counterByWorker)
    counterByWorker = containers.Map('KeyType', 'char', ...
        'ValueType', 'double');
end
key = sprintf('%s|worker_%02d', char(runDir), workerId);
if ~isKey(counterByWorker, key)
    manifestPath = fullfile(runDir, sprintf('worker_%02d', workerId), ...
        'episode_manifest.csv');
    counterByWorker(key) = manifest_row_count(manifestPath);
end
counterByWorker(key) = counterByWorker(key) + 1;
episodeIndexOffset = 0;
if isfield(cfg.environment, 'episodeIndexOffset')
    episodeIndexOffset = cfg.environment.episodeIndexOffset;
end
episodeIndex = episodeIndexOffset + counterByWorker(key);
end

function rowCount = manifest_row_count(path)
rowCount = 0;
if ~isfile(path)
    return;
end
fileId = fopen(path, 'r');
assert(fileId >= 0, 'Cannot open %s.', path);
cleanup = onCleanup(@() fclose(fileId));
lineCount = 0;
while ischar(fgetl(fileId))
    lineCount = lineCount + 1;
end
rowCount = max(0, lineCount - 1);
clear cleanup;
end

function [reward, components, constraintViolation] = compute_reward( ...
        state, reference, input, previousInput, solution, nmpcCfg, cfg)
error = nmpc_state_error(state, reference);
scaledError = error ./ cfg.observation.errorScale;
inputScale = actuator_ranges(cfg.nmpc.plant.nominal);
hoverInput = quad_hover_input(cfg.nmpc.plant.nominal);
violations = nmpc_state_bound_violations([state, state], nmpcCfg);
positiveViolation = max(violations, 0);
constraintViolation = any(positiveViolation > 1e-8);
components = [sum(scaledError(1:3) .^ 2); ...
    sum(scaledError(4:6) .^ 2); sum(scaledError(7:9) .^ 2); ...
    sum(scaledError(10:12) .^ 2); ...
    sum(((input - hoverInput) ./ inputScale) .^ 2); ...
    sum(((input - previousInput) ./ inputScale) .^ 2); ...
    min(solution.solveTime / cfg.observation.solveTimeScale, 10); ...
    double(constraintViolation) + sum(positiveViolation .^ 2); ...
    double(solution.exitflag <= 0)];
reward = -dot(cfg.reward.weights, components);
end

function ranges = actuator_ranges(theta)
ranges = [theta.inputLimits.T(2) - theta.inputLimits.T(1); ...
    theta.inputLimits.tau(:, 2) - theta.inputLimits.tau(:, 1)];
ranges = max(ranges, eps);
end

function log = initialize_log(stepCount, componentNames, actionDimension)
log.State = nan(12, stepCount);
log.Input = nan(4, stepCount);
log.Action = nan(actionDimension, stepCount);
log.Reward = nan(1, stepCount);
log.RewardComponents = nan(numel(componentNames), stepCount);
log.RewardComponentNames = componentNames;
log.SolveTime = nan(1, stepCount);
log.Horizon = nan(1, stepCount);
log.ControlHorizon = nan(1, stepCount);
log.ScenarioCount = nan(1, stepCount);
log.Exitflag = nan(1, stepCount);
log.SolverTimedOut = false(1, stepCount);
log.ExceptionIdentifier = repmat({''}, 1, stepCount);
end

function append_manifest(runDir, info, scenarioSeed)
workerRoot = ensure_worker_root(runDir, info.WorkerId);
path = fullfile(workerRoot, 'episode_manifest.csv');
newFile = ~isfile(path);
[fileId, message] = fopen(path, 'a');
if fileId < 0
    error('create_targeted_sac_nmpc_environment:ManifestOpenFailed', ...
        '%s', message);
end
cleanup = onCleanup(@() fclose(fileId));
if newFile
    fprintf(fileId, ['worker,episode,episode_seed,context_index,' ...
        'context_episode,context_class,start_step,stage,reference,disturbance,level,' ...
        'scenario_seed,lqr_divergence,already_outside,' ...
        'predivergence_eligible,event_step,lead_steps,lqr_risk,' ...
        'lqr_log_energy_growth\n']);
end
fprintf(fileId, ...
    ['%d,%d,%d,%d,%d,%s,%d,%d,%s,%s,%d,%d,%d,%d,%d,' ...
    '%.0f,%.0f,%.16g,%.16g\n'], ...
    info.WorkerId, info.EpisodeIndex, info.EpisodeSeed, ...
    info.ContextIndex, info.ContextEpisodeIndex, info.ContextClass, ...
    info.AbsoluteStartStep, ...
    info.StageIndex, info.ReferenceType, info.DisturbanceType, ...
    info.DisturbanceLevel, scenarioSeed, info.CounterfactualDivergence, ...
    info.AlreadyOutsideEnvelope, info.PreDivergenceEligible, ...
    info.DivergenceEventStep, info.InterventionLeadSteps, ...
    info.LqrRiskScore, info.LqrLogEnergyGrowthPerStep);
clear cleanup;
end

function save_episode_log(runDir, info, cfg)
workerRoot = ensure_worker_root(runDir, info.WorkerId);
directory = fullfile(workerRoot, 'episode_logs');
if ~isfolder(directory)
    mkdir(directory);
end
episodeLog = info.Log;
episodeLog.workerId = info.WorkerId;
episodeLog.episodeIndex = info.EpisodeIndex;
episodeLog.episodeSeed = info.EpisodeSeed;
episodeLog.contextIndex = info.ContextIndex;
episodeLog.contextEpisodeIndex = info.ContextEpisodeIndex;
episodeLog.contextClass = info.ContextClass;
episodeLog.absoluteStartStep = info.AbsoluteStartStep;
episodeLog.stageIndex = info.StageIndex;
episodeLog.referenceType = info.ReferenceType;
episodeLog.disturbanceType = info.DisturbanceType;
episodeLog.disturbanceLevel = info.DisturbanceLevel;
episodeLog.counterfactualLqrDivergence = info.CounterfactualDivergence;
episodeLog.alreadyOutsideLqrEnvelope = info.AlreadyOutsideEnvelope;
episodeLog.preDivergenceEligible = info.PreDivergenceEligible;
episodeLog.divergenceEventStep = info.DivergenceEventStep;
episodeLog.interventionLeadSteps = info.InterventionLeadSteps;
episodeLog.lqrRiskScore = info.LqrRiskScore;
episodeLog.lqrLogEnergyGrowthPerStep = ...
    info.LqrLogEnergyGrowthPerStep;
episodeLog.stepCount = info.StepIndex;
episodeLog.configName = cfg.name;
filePath = fullfile(directory, sprintf('episode_%06d.mat', ...
    info.EpisodeIndex));
save(filePath, 'episodeLog', '-v7.3');
end

function value = scenario_field(scenario, name, defaultValue)
if isfield(scenario, name)
    value = scenario.(name);
else
    value = defaultValue;
end
end

function root = ensure_worker_root(runDir, workerId)
root = fullfile(runDir, sprintf('worker_%02d', workerId));
if ~isfolder(root)
    mkdir(root);
end
end
