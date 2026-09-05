function cfg = targeted_specialist_sac_config()
%TARGETED_SPECIALIST_SAC_CONFIG New SAC run restricted to LQR-weak contexts.

targeted = targeted_lqr_weak_config();
cfg.name = 'targeted_specialist_sac_causal_v2';
cfg.status = ...
    'approved_contract_causal_v2_bank_required_before_training';
cfg.targeted = targeted;
cfg.divergence = targeted_lqr_divergence_config();
cfg.trainingBankPath = fullfile(targeted.resultRoot, ...
    targeted.specialistBank.outputSubfolder, ...
    'training_local_context_bank.mat');
cfg.outputRoot = fullfile(targeted.resultRoot, ...
    'specialist_sac_causal_v2');
cfg.nmpc = step2_nmpc_config();
cfg.nmpc.plant.uncertainty.targeted.rho = ...
    targeted.uncertainty.targeted.rho;
cfg.nmpc.scenario.domain = 'targeted';
cfg.nmpc.scenario.count = 5;
cfg.nmpc.predictionHorizon = 10;
cfg.nmpc.controlHorizon = 10;
cfg.nmpc.rollout.disturbance = [];
cfg.nmpc.solver.maxIterations = 35;
cfg.nmpc.solver.maxFunctionEvaluations = 2500;

cfg.environment.sampleTime = targeted.sampleTime;
cfg.environment.stepsPerEpisode = ...
    targeted.specialistBank.localEpisodeSteps;
cfg.environment.baseEpisodeSeed = 300831501;
cfg.environment.workerSeedStride = 1000000;
cfg.environment.scenarioSeedOffset = 200000;
cfg.environment.terminateCosPitchMargin = 0.15;
cfg.environment.positionTerminationPolicy = 'nmpc_state_bounds_only';
cfg.environment.maxAbsolutePosition = Inf;
cfg.environment.forceCurriculumStage = 0;
cfg.environment.curriculumTransitionFractions = [0.15, 0.35, 0.65];
cfg.environment.curriculumStageNames = { ...
    'parametric_uncertainty_no_disturbance', ...
    'parametric_uncertainty_light_disturbance', ...
    'gust_and_time_varying_disturbance', ...
    'diverse_reference_hard_contexts_only'};
cfg.environment.growthOnlyAuxiliaryStages = [true, true, true, false];
cfg.environment.solverAcceptancePolicy = ...
    'strict_converged_feasible';
cfg.environment.exitflagZeroPolicy = ...
    'strict_pending_closed_loop_validation';
cfg.environment.horizonBank = [5, 10, 15, 20];
cfg.environment.controlHorizonBank = [2, 5, 10, 15, 20];
cfg.environment.scenarioCountBank = 5;

cfg.action.dimension = 8;
cfg.action.weightParameterization = ...
    'absolute_dimensionless_normalized';
cfg.action.groupWeightBounds = [0.10, 100.0; ...
    0.10, 100.0; ...
    0.01, 30.0; ...
    0.01, 30.0; ...
    1e-4, 1.0; ...
    1e-4, 1.0];
cfg.action.terminalWeightMultiplier = 4.0;
cfg.observation.featureVersion = 'causal_feature_v2';
cfg.observation.dimension = 328;
cfg.observation.errorScale = [1.0; 1.0; 1.0; ...
    0.50; 0.50; 0.75; 2.0; 2.0; 2.0; 2.0; 2.0; 2.0];
cfg.observation.residualScale = [0.20; 0.20; 0.20; ...
    0.10; 0.10; 0.10; 0.50; 0.50; 0.50; 0.50; 0.50; 0.50];
cfg.observation.stateScale = max(abs([ ...
    cfg.nmpc.constraints.stateLower(:), ...
    cfg.nmpc.constraints.stateUpper(:)]), [], 2);
cfg.observation.referenceScale = cfg.observation.stateScale;
cfg.observation.inputCenter = [cfg.nmpc.plant.nominal.m * ...
    cfg.nmpc.plant.nominal.g / cfg.nmpc.plant.nominal.alphaT; ...
    0; 0; 0];
cfg.observation.inputScale = [ ...
    diff(cfg.nmpc.plant.nominal.inputLimits.T); ...
    diff(cfg.nmpc.plant.nominal.inputLimits.tau, 1, 2)];
cfg.observation.includeSolveTime = false;
cfg.observation.includeTeacherPrediction = false;
cfg.observation.solveTimeScale = 1.0;
cfg.observation.clip = 5.0;

cfg.reward.componentNames = {'position', 'attitude', 'velocity', ...
    'bodyRate', 'control', 'smoothness', 'solveTime', ...
    'constraint', 'solverFailure'};
cfg.reward.weights = [4.0; 2.0; 0.50; 0.25; 0.05; 0.02; ...
    0.0; 100.0; 200.0];

cfg.agent.discountFactor = 0.99;
cfg.agent.actorLearnRate = 3e-4;
cfg.agent.criticLearnRate = 3e-4;
cfg.agent.entropyLearnRate = 3e-4;
cfg.agent.miniBatchSize = 256;
cfg.agent.experienceBufferLength = 1e6;
cfg.agent.numWarmStartSteps = 256;
cfg.agent.targetSmoothFactor = 0.005;

cfg.training.finalBudgetLocked = false;
cfg.training.budgetUnit = 'episode';
cfg.training.candidateEpisodeCeiling = 1500;
cfg.training.checkpointFrequencyEpisodes = 50;
cfg.training.candidateEvaluationEpisodes = [250, 500, 750, 1000, ...
    1250, 1500];
cfg.training.maxEpisodes = 1000000;
cfg.training.maxStepsPerEpisode = cfg.environment.stepsPerEpisode;
cfg.training.scoreAveragingWindowLength = 20;
cfg.training.parallelMode = 'sync';
cfg.training.workerCount = 3;
cfg.training.workerRandomSeeds = [300831601, 300831602, 300831603];

cfg.probe.episodeCount = 3;
cfg.probe.serialRunId = 'actual_sac_serial_episode_probe_v1';
cfg.probe.parallelRunId = 'actual_sac_sync3_episode_probe_v1';
cfg.probe.forceCurriculumStage = 4;
cfg.probe.maximumContextsPerPositiveFactorialCell = ...
    targeted.specialistBank.maximumHardContextsPerCell;
cfg.logging.enabled = true;
end
