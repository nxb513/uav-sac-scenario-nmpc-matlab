function cfg = targeted_specialist_sac_config()
%TARGETED_SPECIALIST_SAC_CONFIG New SAC run restricted to LQR-weak contexts.

targeted = targeted_lqr_weak_config();
cfg.name = 'targeted_specialist_sac_v1';
cfg.status = 'official_200_step_checkpoint_stream_final_budget_not_locked';
cfg.targeted = targeted;
cfg.trainingBankPath = fullfile(targeted.resultRoot, ...
    targeted.specialistBank.outputSubfolder, ...
    'training_local_context_bank.mat');
cfg.outputRoot = fullfile(targeted.resultRoot, 'specialist_sac_v1');
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
cfg.environment.maxAbsolutePosition = 20.0;
cfg.environment.forceCurriculumStage = 0;
cfg.environment.curriculumTransitionFractions = [0.15, 0.35, 0.65];
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
cfg.observation.dimension = 31;
cfg.observation.errorScale = [1.0; 1.0; 1.0; ...
    0.50; 0.50; 0.75; 2.0; 2.0; 2.0; 2.0; 2.0; 2.0];
cfg.observation.residualScale = [0.20; 0.20; 0.20; ...
    0.10; 0.10; 0.10; 0.50; 0.50; 0.50; 0.50; 0.50; 0.50];
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
cfg.training.candidateTransitionCeiling = 300000;
cfg.training.candidateCheckpoints = [50000, 100000, 150000, 225000, 300000];
cfg.training.transitionBudget = NaN;
cfg.training.cumulativeCheckpoints = [];
cfg.training.segmentSteps = [];
cfg.training.maxEpisodes = 1000000;
cfg.training.maxStepsPerEpisode = cfg.environment.stepsPerEpisode;
cfg.training.scoreAveragingWindowLength = 20;
cfg.training.parallelMode = 'sync';
cfg.training.workerCount = 3;
cfg.training.workerRandomSeeds = [300831601, 300831602, 300831603];

cfg.probe.transitionCount = 100;
cfg.probe.serialRunId = 'actual_sac_serial_100_v1';
cfg.probe.parallelRunId = 'actual_sac_sync3_100_v1';
cfg.probe.forceCurriculumStage = 4;
cfg.probe.contextsPerFactorialCell = 1;
cfg.logging.enabled = true;
end
