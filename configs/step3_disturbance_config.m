function cfg = step3_disturbance_config()
%STEP3_DISTURBANCE_CONFIG Candidate disturbance curriculum and probe setup.
%
% Values are candidate levels only. Final ranges require user approval
% after local runtime and control-response benchmarks.

plantCfg = step1_plant_config();
hoverThrust = plantCfg.nominal.m * plantCfg.nominal.g;
torqueLimits = max(abs(plantCfg.nominal.inputLimits.tau), [], 2);

cfg.name = 'step3_external_disturbance_candidates';
cfg.status = 'candidate_not_final';
cfg.requiresUserApproval = true;

cfg.scale.hoverThrust = hoverThrust;
cfg.scale.torqueLimits = torqueLimits;

cfg.train.forceFractionHover = [0.01, 0.03, 0.05];
cfg.train.torqueFractionLimit = [0.005, 0.01, 0.02];
cfg.train.sinusoidFrequencyHz = [0.25, 1.00];
cfg.train.stochasticCorrelation = [0.85, 0.98];
cfg.train.onsetFraction = [0.05, 0.35];
cfg.train.durationFraction = [0.15, 0.40];

cfg.ood.forceFractionHover = [0.08, 0.12];
cfg.ood.torqueFractionLimit = [0.04, 0.08];
cfg.ood.sinusoidFrequencyHz = [1.25, 2.50];
cfg.ood.stochasticCorrelation = [0.60, 0.995];
cfg.ood.onsetFraction = [0.05, 0.70];
cfg.ood.durationFraction = [0.05, 0.50];

cfg.types = {'zero', 'constant', 'gust', 'sinusoidal', 'stochastic'};

cfg.curriculum(1).name = 'parametric_only';
cfg.curriculum(1).allowedTypes = {'zero'};
cfg.curriculum(1).allowedTrainLevels = 1;
cfg.curriculum(2).name = 'light_disturbance';
cfg.curriculum(2).allowedTypes = {'constant', 'sinusoidal', 'stochastic'};
cfg.curriculum(2).allowedTrainLevels = 1;
cfg.curriculum(3).name = 'gust_and_time_varying';
cfg.curriculum(3).allowedTypes = {'gust', 'sinusoidal', 'stochastic'};
cfg.curriculum(3).allowedTrainLevels = [1, 2];
cfg.curriculum(4).name = 'diverse_reference_and_disturbance';
cfg.curriculum(4).allowedTypes = {'constant', 'gust', 'sinusoidal', 'stochastic'};
cfg.curriculum(4).allowedTrainLevels = [1, 2, 3];

cfg.recovery.positionTolerance = 0.10;
cfg.recovery.relativeToPreEvent = 1.50;
cfg.recovery.dwellTime = 0.15;

cfg.probe.runId = 'step3_disturbance_probe';
cfg.probe.outputRoot = 'results';
cfg.probe.seed = 27082731;
cfg.probe.disturbanceSeed = 27082732;
cfg.probe.scenarioSeed = 27082733;
cfg.probe.stepCount = 20;
cfg.probe.levelIndex = 2;
cfg.probe.domain = 'train';
cfg.probe.referenceType = 'hover';
cfg.probe.referenceOptions.position = [0.0; 0.0; 1.0];
cfg.probe.plantModes = {'nominal', 'uncertain'};
cfg.probe.disturbanceTypes = cfg.types;
cfg.probe.nmpc = step2_nmpc_config();
cfg.probe.nmpc.predictionHorizon = 4;
cfg.probe.nmpc.controlHorizon = 2;
cfg.probe.nmpc.scenario.count = 2;
cfg.probe.nmpc.rollout.disturbance = [];
cfg.probe.nmpc.solver.maxIterations = 35;
cfg.probe.nmpc.solver.maxFunctionEvaluations = 2500;
end
