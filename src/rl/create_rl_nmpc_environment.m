function [env, observationInfo, actionInfo] = create_rl_nmpc_environment(cfg, runDir)
%CREATE_RL_NMPC_ENVIRONMENT Build the seeded uncertain-plant SAC environment.

observationInfo = rlNumericSpec([cfg.observation.dimension, 1]);
observationInfo.Name = 'online_uav_features';
observationInfo.LowerLimit = -cfg.observation.clip * ...
    ones(cfg.observation.dimension, 1);
observationInfo.UpperLimit = cfg.observation.clip * ...
    ones(cfg.observation.dimension, 1);
actionInfo = rlNumericSpec([cfg.action.dimension, 1]);
actionInfo.Name = 'log_weight_and_horizon_action';
actionInfo.LowerLimit = -ones(cfg.action.dimension, 1);
actionInfo.UpperLimit = ones(cfg.action.dimension, 1);

episodeCounter = cfg.environment.episodeIndexOffset;
manifestPath = fullfile(runDir, 'episode_manifest.csv');
episodeLogDir = fullfile(runDir, 'episode_logs');
if ~isfolder(episodeLogDir)
    mkdir(episodeLogDir);
end

env = rlFunctionEnv(observationInfo, actionInfo, @step_environment, @reset_environment);

    function [initialObservation, info] = reset_environment()
        episodeCounter = episodeCounter + 1;
        episodeSeed = cfg.environment.baseEpisodeSeed + episodeCounter;
        previousRng = rng;
        cleanup = onCleanup(@() rng(previousRng));
        rng(episodeSeed, 'twister');

        stageIndex = find(episodeCounter <= ...
            cfg.environment.curriculumEndEpisode, 1, 'first');
        [referenceType, referenceOptions, disturbanceType, disturbanceLevel] = ...
            episode_design(stageIndex);
        referenceCount = cfg.environment.stepsPerEpisode + ...
            max(cfg.environment.horizonBank) + 1;
        reference = quad_reference_trajectory(referenceType, ...
            cfg.environment.sampleTime, referenceCount, referenceOptions);
        x0 = reference(:, 1) + cfg.environment.initialStateStd .* randn(12, 1);

        plantSeed = episodeSeed + 100000;
        scenarioSeed = episodeSeed + 200000;
        disturbanceSeed = episodeSeed + 300000;
        thetaPlant = quad_sample_uncertainty(cfg.nmpc.plant, 1, 'train', ...
            plantSeed, cfg.nmpc.scenario.method);
        thetaScenarios = quad_sample_uncertainty(cfg.nmpc.plant, ...
            cfg.environment.scenarioCount, 'train', scenarioSeed, ...
            cfg.nmpc.scenario.method);
        disturbance = quad_generate_disturbance_episode(cfg.nmpc.plant, ...
            cfg.disturbance, disturbanceType, 'train', disturbanceLevel, ...
            cfg.environment.sampleTime, cfg.environment.stepsPerEpisode, ...
            disturbanceSeed);

        info.EpisodeIndex = episodeCounter;
        info.EpisodeSeed = episodeSeed;
        info.StageIndex = stageIndex;
        info.StepIndex = 0;
        info.State = x0;
        info.PreviousInput = quad_hover_input(cfg.nmpc.plant.nominal);
        info.PreviousPrediction = x0;
        info.PreviousSolveTime = 0.0;
        info.PreviousHorizon = cfg.nmpc.predictionHorizon;
        info.WarmStart = [];
        info.Reference = reference;
        info.ReferenceType = referenceType;
        info.ThetaPlant = thetaPlant(1);
        info.ThetaScenarios = thetaScenarios;
        info.Disturbance = disturbance;
        info.DisturbanceType = disturbanceType;
        info.DisturbanceLevel = disturbanceLevel;
        info.Log = initialize_log(cfg.environment.stepsPerEpisode, ...
            cfg.reward.componentNames);
        append_manifest(manifestPath, info, plantSeed, scenarioSeed, disturbanceSeed);
        initialObservation = rl_nmpc_make_observation(info.State, reference(:, 1), ...
            info.PreviousInput, zeros(12, 1), 0.0, ...
            info.PreviousHorizon, cfg);
    end

    function [observation, reward, isDone, info] = step_environment(action, info)
        stepIndex = info.StepIndex + 1;
        action = min(max(double(action(:)), -1.0), 1.0);
        exceptionIdentifier = '';
        try
            [nmpcCfg, mapping] = rl_nmpc_action_to_config(action, cfg);
            warmStart = rl_nmpc_resize_warm_start(info.WarmStart, ...
                nmpcCfg.predictionHorizon, cfg.nmpc.plant.nominal);
            tNow = (stepIndex - 1) * cfg.environment.sampleTime;
            Xref = nmpc_reference_window(info.Reference, stepIndex, tNow, nmpcCfg);
            sol = scenario_nmpc_solve(info.State, Xref, ...
                info.ThetaScenarios, nmpcCfg, warmStart);
            u = sol.u0;
            nextState = quad_step_rk4(tNow, info.State, u, ...
                cfg.environment.sampleTime, info.ThetaPlant, info.Disturbance);
            predictionResidual = nextState - sol.Xpred(:, 2);
            nextReference = info.Reference(:, min(stepIndex + 1, ...
                size(info.Reference, 2)));
            [reward, components, constraintViolation] = compute_reward( ...
                nextState, nextReference, u, info.PreviousInput, sol, nmpcCfg, cfg);
            finiteTransition = all(isfinite(nextState)) && all(isfinite(u)) && ...
                isfinite(reward) && isfinite(sol.solveTime);
            solverFailure = sol.exitflag <= 0;
            invalidEuler = abs(cos(nextState(5))) < ...
                cfg.environment.terminateCosPitchMargin;
            outsideRegion = any(abs(nextState(1:3)) > ...
                cfg.environment.maxAbsolutePosition);
            isDone = stepIndex >= cfg.environment.stepsPerEpisode || ...
                ~finiteTransition || solverFailure || invalidEuler || ...
                outsideRegion || constraintViolation;
            info.WarmStart = nmpc_shift_sequence(sol.U, sol.U(:, end));
            solveTime = sol.solveTime;
            horizon = mapping.horizon;
            exitflag = sol.exitflag;
        catch exception
            exceptionIdentifier = exception.identifier;
            nextState = info.State;
            u = info.PreviousInput;
            predictionResidual = zeros(12, 1);
            reward = -cfg.reward.weights(end);
            components = zeros(numel(cfg.reward.componentNames), 1);
            components(end) = 1.0;
            isDone = true;
            solveTime = 0.0;
            horizon = info.PreviousHorizon;
            exitflag = -999;
        end

        nextReference = info.Reference(:, min(stepIndex + 1, ...
            size(info.Reference, 2)));
        observation = rl_nmpc_make_observation(nextState, nextReference, u, ...
            predictionResidual, solveTime, horizon, cfg);
        info.Log.State(:, stepIndex) = nextState;
        info.Log.Input(:, stepIndex) = u;
        info.Log.Action(:, stepIndex) = action;
        info.Log.Reward(stepIndex) = reward;
        info.Log.RewardComponents(:, stepIndex) = components;
        info.Log.SolveTime(stepIndex) = solveTime;
        info.Log.Horizon(stepIndex) = horizon;
        info.Log.Exitflag(stepIndex) = exitflag;
        info.Log.ExceptionIdentifier{stepIndex} = exceptionIdentifier;
        info.StepIndex = stepIndex;
        info.State = nextState;
        info.PreviousInput = u;
        info.PreviousPrediction = nextState - predictionResidual;
        info.PreviousSolveTime = solveTime;
        info.PreviousHorizon = horizon;

        if isDone
            save_episode_log(episodeLogDir, info, cfg);
        end
    end

    function [referenceType, options, disturbanceType, level] = episode_design(stageIndex)
        switch stageIndex
            case 1
                referenceType = 'hover';
                options.position = [0.0; 0.0; 1.0];
                disturbanceType = 'zero';
                level = 1;
            case 2
                referenceType = 'position_step';
                options.positionBefore = [0.0; 0.0; 1.0];
                options.positionAfter = [0.20 + 0.30 * rand(); ...
                    -0.35 + 0.70 * rand(); 1.0 + 0.20 * rand()];
                options.stepTime = 0.5 + rand();
                candidates = {'constant', 'sinusoidal', 'stochastic'};
                disturbanceType = candidates{randi(numel(candidates))};
                level = 1;
            case 3
                if rand() < 0.7
                    referenceType = 'position_step';
                    options.positionBefore = [0.0; 0.0; 1.0];
                    options.positionAfter = [-0.40 + 0.80 * rand(); ...
                        -0.40 + 0.80 * rand(); 0.9 + 0.30 * rand()];
                    options.stepTime = 0.4 + 1.2 * rand();
                else
                    referenceType = 'circle';
                    options.center = [0.0; 0.0];
                    options.radius = 0.20 + 0.25 * rand();
                    options.altitude = 0.9 + 0.2 * rand();
                    options.angularRate = 0.25 + 0.25 * rand();
                    options.phase = 2 * pi * rand();
                end
                candidates = {'gust', 'sinusoidal', 'stochastic'};
                disturbanceType = candidates{randi(numel(candidates))};
                level = randi(2);
            otherwise
                referenceType = cfg.environment.referenceFamilies{ ...
                    randi(numel(cfg.environment.referenceFamilies))};
                options = diverse_reference_options(referenceType);
                candidates = {'constant', 'gust', 'sinusoidal', 'stochastic'};
                disturbanceType = candidates{randi(numel(candidates))};
                level = randi(3);
        end
    end
end

function options = diverse_reference_options(referenceType)
switch referenceType
    case 'hover'
        options.position = [-0.2 + 0.4 * rand(); -0.2 + 0.4 * rand(); ...
            0.85 + 0.3 * rand()];
    case 'step'
        options.positionBefore = [0.0; 0.0; 1.0];
        options.positionAfter = [-0.5 + rand(); -0.5 + rand(); 0.8 + 0.4 * rand()];
        options.stepTime = 0.3 + 1.5 * rand();
    case 'circle'
        options.center = [-0.1 + 0.2 * rand(); -0.1 + 0.2 * rand()];
        options.radius = 0.15 + 0.40 * rand();
        options.altitude = 0.8 + 0.4 * rand();
        options.angularRate = 0.2 + 0.4 * rand();
        options.phase = 2 * pi * rand();
    otherwise
        error('create_rl_nmpc_environment:BadReferenceType', ...
            'Unknown reference type: %s', referenceType);
end
end

function [reward, components, constraintViolation] = compute_reward( ...
        state, reference, input, previousInput, sol, nmpcCfg, cfg)
error = nmpc_state_error(state, reference);
scaledError = error ./ cfg.observation.errorScale;
inputScale = actuator_ranges(cfg.nmpc.plant.nominal);
uHover = quad_hover_input(cfg.nmpc.plant.nominal);
violations = nmpc_state_bound_violations([state, state], nmpcCfg);
positiveViolation = max(violations, 0.0);
constraintViolation = any(positiveViolation > 1e-8);
components = [sum(scaledError(1:3).^2); ...
              sum(scaledError(4:6).^2); ...
              sum(scaledError(7:9).^2); ...
              sum(scaledError(10:12).^2); ...
              sum(((input - uHover) ./ inputScale).^2); ...
              sum(((input - previousInput) ./ inputScale).^2); ...
              min(sol.solveTime / cfg.observation.solveTimeScale, 10.0); ...
              double(constraintViolation) + sum(positiveViolation.^2); ...
              double(sol.exitflag <= 0)];
reward = -dot(cfg.reward.weights, components);
end

function ranges = actuator_ranges(theta)
ranges = [theta.inputLimits.T(2) - theta.inputLimits.T(1); ...
          theta.inputLimits.tau(:, 2) - theta.inputLimits.tau(:, 1)];
ranges = max(ranges, eps);
end

function log = initialize_log(stepCount, componentNames)
log.State = nan(12, stepCount);
log.Input = nan(4, stepCount);
log.Action = nan(7, stepCount);
log.Reward = nan(1, stepCount);
log.RewardComponents = nan(numel(componentNames), stepCount);
log.RewardComponentNames = componentNames;
log.SolveTime = nan(1, stepCount);
log.Horizon = nan(1, stepCount);
log.Exitflag = nan(1, stepCount);
log.ExceptionIdentifier = repmat({''}, 1, stepCount);
end

function append_manifest(path, info, plantSeed, scenarioSeed, disturbanceSeed)
newFile = ~isfile(path);
[fileId, message] = fopen(path, 'a');
if fileId < 0
    error('create_rl_nmpc_environment:ManifestOpenFailed', '%s', message);
end
cleanup = onCleanup(@() fclose(fileId));
if newFile
    fprintf(fileId, ['episode,episode_seed,stage,reference,disturbance,' ...
        'level,plant_seed,scenario_seed,disturbance_seed\n']);
end
fprintf(fileId, '%d,%d,%d,%s,%s,%d,%d,%d,%d\n', ...
    info.EpisodeIndex, info.EpisodeSeed, info.StageIndex, info.ReferenceType, ...
    info.DisturbanceType, info.DisturbanceLevel, plantSeed, scenarioSeed, ...
    disturbanceSeed);
end

function save_episode_log(directory, info, cfg)
episodeLog = info.Log;
episodeLog.episodeIndex = info.EpisodeIndex;
episodeLog.episodeSeed = info.EpisodeSeed;
episodeLog.stageIndex = info.StageIndex;
episodeLog.referenceType = info.ReferenceType;
episodeLog.disturbanceType = info.DisturbanceType;
episodeLog.disturbanceLevel = info.DisturbanceLevel;
episodeLog.stepCount = info.StepIndex;
episodeLog.thetaPlant = info.ThetaPlant;
episodeLog.thetaScenarios = info.ThetaScenarios;
episodeLog.disturbance = info.Disturbance;
episodeLog.configName = cfg.name;
filePath = fullfile(directory, sprintf('episode_%06d.mat', info.EpisodeIndex));
save(filePath, 'episodeLog', '-v7.3');
end
