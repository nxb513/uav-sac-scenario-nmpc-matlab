function cfg = targeted_lqr_weak_config()
%TARGETED_LQR_WEAK_CONFIG Candidate scope for the LQR-weak rebuild.
%
% Candidate values support local coverage/runtime benchmarks only. They are
% not the final training, validation or OOD grid.

plant = step1_plant_config();

cfg.name = 'targeted_lqr_weak_rebuild_v1';
cfg.status = 'candidate_requires_benchmark_and_user_approval';
cfg.resultRoot = fullfile('results', cfg.name);
cfg.sampleTime = 0.05;
cfg.stepCount = 200;
cfg.plant = plant;

cfg.scope.defaultController = 'LQR';
cfg.scope.specialistTeacher = 'SAC_scenario_NMPC';
cfg.scope.specialistStudent = 'direct_control_surrogate_no_residual_nn';
cfg.scope.deploymentController = 'confidence_weighted_surrogate_LQR';
cfg.scope.surrogateTrainingDomain = 'LQR_weak_contexts_only';
cfg.scope.methodsEditable = false;

cfg.reference.seed = 300830101;
cfg.reference.benchmarkCountPerFamily = 50;
cfg.reference.families = {'circle', 'lemniscate', 'vertical_circle', ...
    'spatial_helix', 'smooth_waypoints'};
cfg.reference.targetPeakSpeedRange = [0.60, 2.00];
cfg.reference.preferredHardSpeedRange = [1.00, 2.00];
cfg.reference.peakAccelerationRange = [0.80, 6.00];
cfg.reference.circleRadiusRange = [0.30, 0.90];
cfg.reference.verticalCircleRadiusRange = [0.30, 0.70];
cfg.reference.lemniscateAmplitudeXRange = [0.40, 0.90];
cfg.reference.lemniscateAmplitudeYRange = [0.25, 0.60];
cfg.reference.helixRadiusRange = [0.35, 0.80];
cfg.reference.helixVerticalAmplitudeRange = [0.15, 0.40];
cfg.reference.altitudeRange = [0.80, 1.30];
cfg.reference.maxEquivalentTiltDeg = 35.0;
cfg.reference.maxFeedforwardInputFraction = 1.0;
cfg.reference.maxSampleAttempts = 200;

% Large but in-distribution specialist envelope. The legacy OOD envelope is
% reserved for final confirmation and is never opened during LQR selection.
cfg.uncertainty.targeted.rho = [0.18; ...
    0.10; 0.10; 0.10; ...
    0.45; 0.45; 0.45; ...
    0.45; 0.45; 0.45; ...
    0.22; ...
    0.22; 0.22; 0.22];
cfg.uncertainty.stratumNames = {'low', 'medium', 'high'};
cfg.uncertainty.stratumXiMagnitude = [0, 1/3; 1/3, 2/3; 2/3, 1];

cfg.retune.outputSubfolder = 'lqr_retune_balanced_v3';
cfg.screenBenchmark.replicatesPerCell = 20;
cfg.screenBenchmark.episodeCount = 2700;
cfg.screenBenchmark.seed = 300830201;
cfg.screenBenchmark.disturbanceTypes = ...
    {'zero', 'constant', 'gust', 'sinusoidal', 'stochastic'};
cfg.screenBenchmark.disturbanceLevels = [1, 3];
cfg.screenBenchmark.uncertaintyDomain = 'targeted';
cfg.screenBenchmark.disturbanceDomain = 'train';
cfg.screenBenchmark.evaluationStartStep = 5;
cfg.screenBenchmark.bankOutputSubfolder = 'validation_bank_balanced_v2';
cfg.screenBenchmark.initialStateStd = [0.03 * ones(3, 1); ...
    deg2rad(2.0) * ones(3, 1); 0.05 * ones(3, 1); ...
    deg2rad(5.0) * ones(3, 1)];
cfg.screenBenchmark.outputSubfolder = 'lqr_screen_balanced_v2';
cfg.screenBenchmark.provisionalLqrPath = fullfile('results', cfg.name, ...
    cfg.retune.outputSubfolder, 'selected_lqr.mat');
cfg.screenBenchmark.lqrStatus = ...
    'retuned_targeted_baseline_pending_confirmation';

% The retune stage uses independent validation banks. The values below are
% runtime anchors, not the final study grid.
cfg.retune.replicatesPerCell = 10;
cfg.retune.designEpisodeCount = 1350;
cfg.retune.selectionEpisodeCount = 1350;
cfg.retune.designSeed = 300830801;
cfg.retune.selectionSeed = 300830901;
cfg.retune.uncertaintyDomain = 'targeted';
cfg.retune.disturbanceDomain = 'train';
cfg.retune.disturbanceTypes = cfg.screenBenchmark.disturbanceTypes;
cfg.retune.disturbanceLevels = cfg.screenBenchmark.disturbanceLevels;
cfg.retune.evaluationStartStep = 1;
cfg.retune.initialStateStd = cfg.screenBenchmark.initialStateStd;
cfg.retune.keepTop = 5;
cfg.retune.referenceFeedforward = 'nominal_differential_flatness';
cfg.retune.lqrStructure = ...
    'nominal_hover_lqr_error_feedback_plus_reference_feedforward';
cfg.lqr.statePerturbation = [1e-5 * ones(3, 1); ...
    1e-6 * ones(3, 1); 1e-5 * ones(3, 1); 1e-6 * ones(3, 1)];
cfg.lqr.inputPerturbation = [1e-5; 1e-7; 1e-7; 1e-7];
cfg.retune.grid.positionVelocityScale = [0.5, 1.0, 2.0, 4.0, 8.0];
cfg.retune.grid.attitudeRateScale = [0.25, 0.5, 1.0, 2.0, 4.0];
cfg.retune.grid.inputPenaltyScale = [0.0625, 0.125, 0.25, 0.5, 1.0];
cfg.retune.grid.refinementMultiplier = [0.5, 0.75, 1.0, 1.5, 2.0];
cfg.retune.grid.coarseCandidateCount = 125;
cfg.retune.grid.refinementCandidateCount = 125;
cfg.retune.grid.screenCandidateCount = 250;
cfg.retune.grid.selectionCandidateCount = 5;
cfg.retune.parallel.enabled = true;
cfg.retune.parallel.workerCount = 3;
cfg.retune.status = 'candidate_grid_requires_runtime_check';

% These thresholds seed the LQR-only screening benchmark. They must be
% revisited after observing the candidate-bank distributions.
cfg.weakness.status = 'candidate_not_final';
cfg.weakness.windowSteps = [5, 10, 20];
cfg.weakness.positionRmseM = [0.10, 0.15, 0.20];
cfg.weakness.attitudeRmseDeg = [5.0, 10.0, 15.0];
cfg.weakness.constraintMarginFraction = [0.20, 0.10, 0.05];
cfg.weakness.minimumTeacherPositionGainM = 0.01;
cfg.weakness.minimumTeacherRelativeGain = 0.10;

cfg.specialistBank.sourceSubfolder = 'specialist_local_context_banks_v2';
cfg.specialistBank.outputSubfolder = ...
    'specialist_local_context_banks_200step_v1';
cfg.specialistBank.trainReplicatesPerCell = 20;
cfg.specialistBank.validationReplicatesPerCell = 10;
cfg.specialistBank.rankingWindowSteps = 10;
cfg.specialistBank.sourceStepCount = 400;
cfg.specialistBank.localEpisodeSteps = 200;
cfg.specialistBank.maximumPredictionHorizon = 20;
cfg.specialistBank.trainContextsPerCell = 10;
cfg.specialistBank.validationContextsPerCell = 5;
cfg.specialistBank.cloudContextsPerCell = 1;
cfg.specialistBank.minimumStartStep = 10;
cfg.specialistBank.positionScoreScaleM = 0.10;
cfg.specialistBank.attitudeScoreScaleDeg = 5.0;
cfg.specialistBank.selectionRule = ...
    'top_local_lqr_deficit_per_balanced_factorial_cell';
cfg.specialistBank.absoluteThresholdIsGate = false;

cfg.split.screenSeed = 300830201;
cfg.split.teacherTrainSeed = 300830301;
cfg.split.teacherValidationSeed = 300830401;
cfg.split.surrogateTrainSeed = 300830501;
cfg.split.surrogateValidationSeed = 300830601;
cfg.split.confirmationSeed = 300830701;
cfg.split.oodLocked = true;

cfg.reproducibility.finalGridLocked = false;
cfg.reproducibility.longRunAuthorized = false;
cfg.reproducibility.oldOodUsedForTraining = false;
cfg.reproducibility.requiresRuntimeBenchmark = true;
cfg.reproducibility.requiresUserApprovalAfterBenchmark = true;
end
