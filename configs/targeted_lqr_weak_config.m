function cfg = targeted_lqr_weak_config()
%TARGETED_LQR_WEAK_CONFIG Candidate scope for the LQR-weak rebuild.
%
% Candidate values support local coverage/runtime benchmarks only. They are
% not the final training, validation or OOD grid.

plant = step1_plant_config();

cfg.name = 'targeted_lqr_weak_rebuild_v1';
cfg.status = 'realized_speed_load_coverage_lqr_rerun_authorized';
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
cfg.reference.approvedIdSpeedAnchors = [0.5, 1, 2, 4, 6, 8, 10, 12];
cfg.reference.candidateOodSpeedAnchors = [14, 16];
cfg.reference.approvedLoadNames = {'mild', 'moderate', 'hard'};
cfg.reference.approvedAccelerationTargets = [2.0, 5.0, 9.0];
cfg.reference.approvedLoadFractions = ...
    cfg.reference.approvedAccelerationTargets ./ ...
    max(cfg.reference.approvedAccelerationTargets);
cfg.reference.loadAccelerationRule = ...
    'speed_feasible_fraction_of_full_envelope_target';
cfg.reference.minimumResolvedAcceleration = 0.20;
cfg.reference.accelerationRelativeTolerance = 0.25;
cfg.reference.verticalAccelerationFractionLimit = 0.75;
cfg.reference.verticalCircleMaximumRadius = 45.0;
cfg.reference.minimumGeometryScale = 0.30;
cfg.reference.geometryScaleJitter = [0.90, 1.10];
cfg.reference.speedRelativeTolerance = 0.03;
cfg.reference.physicalMaxTiltDeg = 70.0;
cfg.reference.physicalMaxInputFraction = 1.0;
cfg.reference.robustCandidateMaxTiltDeg = 50.0;
cfg.reference.robustCandidateMaxInputFraction = 0.75;
cfg.reference.dynamicResidualP95Limit = 0.25;
cfg.reference.targetPeakSpeedRange = [0.50, 12.00];
cfg.reference.preferredHardSpeedRange = [8.00, 12.00];
cfg.reference.peakAccelerationRange = [0.15, 10.50];
cfg.reference.altitudeRange = [1.00, 3.00];
cfg.reference.maxEquivalentTiltDeg = ...
    cfg.reference.robustCandidateMaxTiltDeg;
cfg.reference.maxFeedforwardInputFraction = ...
    cfg.reference.robustCandidateMaxInputFraction;
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

cfg.retune.outputSubfolder = 'lqr_retune_realized_coverage_v5';
cfg.retune.selectionRevalidationOutputSubfolder = ...
    'lqr_retune_realized_coverage_v6';
cfg.screenBenchmark.replicatesPerCell = 20;
cfg.screenBenchmark.episodeCount = 2700;
cfg.screenBenchmark.seed = 300830201;
cfg.screenBenchmark.disturbanceTypes = ...
    {'zero', 'constant', 'gust', 'sinusoidal', 'stochastic'};
cfg.screenBenchmark.disturbanceLevels = [1, 3];
cfg.screenBenchmark.uncertaintyDomain = 'targeted';
cfg.screenBenchmark.disturbanceDomain = 'train';
cfg.screenBenchmark.evaluationStartStep = 5;
cfg.screenBenchmark.bankOutputSubfolder = ...
    'validation_bank_realized_coverage_v3';
cfg.screenBenchmark.initialStateStd = [0.03 * ones(3, 1); ...
    deg2rad(2.0) * ones(3, 1); 0.05 * ones(3, 1); ...
    deg2rad(5.0) * ones(3, 1)];
cfg.screenBenchmark.outputSubfolder = 'lqr_screen_realized_coverage_v3';
cfg.screenBenchmark.provisionalLqrPath = fullfile('results', cfg.name, ...
    cfg.retune.selectionRevalidationOutputSubfolder, 'selected_lqr.mat');
cfg.screenBenchmark.lqrStatus = ...
    'selection_revalidated_v6_pending_weakness_screen';

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
cfg.retune.status = 'approved_250_candidate_search';
cfg.retune.longRunAuthorized = true;

% These thresholds seed the LQR-only screening benchmark. They must be
% revisited after observing the candidate-bank distributions.
cfg.weakness.status = 'candidate_not_final';
cfg.weakness.windowSteps = [5, 10, 20];
cfg.weakness.positionRmseM = [0.10, 0.15, 0.20];
cfg.weakness.attitudeRmseDeg = [5.0, 10.0, 15.0];
cfg.weakness.constraintMarginFraction = [0.20, 0.10, 0.05];
cfg.weakness.minimumTeacherPositionGainM = 0.01;
cfg.weakness.minimumTeacherRelativeGain = 0.10;

cfg.predictiveAnalysis.status = 'locked_c012_on_validation_20260904';
cfg.predictiveAnalysis.sourceBankSubfolder = ...
    cfg.screenBenchmark.bankOutputSubfolder;
cfg.predictiveAnalysis.sourceScreenSubfolder = ...
    cfg.screenBenchmark.outputSubfolder;
cfg.predictiveAnalysis.outputSubfolder = ...
    'lqr_predictive_threshold_analysis_v3_cause_coverage';
cfg.predictiveAnalysis.horizonSteps = 20;
cfg.predictiveAnalysis.evaluationStartStep = ...
    cfg.screenBenchmark.evaluationStartStep;
cfg.predictiveAnalysis.representativePositiveFractions = ...
    [0.01, 0.025, 0.05, 0.10];
cfg.predictiveAnalysis.finalThresholdsLocked = true;
cfg.predictiveAnalysis.lockedCandidateId = 'C012';

cfg.specialistBank.legacyOutputSubfolder = ...
    'specialist_predivergence_context_bank_c012_v1';
cfg.specialistBank.outputSubfolder = ...
    'specialist_predivergence_context_bank_c012_causal_v2';
cfg.specialistBank.status = ...
    'approved_c012_causal_v2_rebuild_required';
cfg.specialistBank.featureVersion = 'causal_feature_v2';
cfg.specialistBank.stateHistoryLength = 4;
cfg.specialistBank.appliedInputHistoryLength = 4;
cfg.specialistBank.sourceReplicatesPerCell = 20;
cfg.specialistBank.sourceEpisodeCount = 2700;
cfg.specialistBank.sourceStepCount = 400;
cfg.specialistBank.localEpisodeSteps = 200;
cfg.specialistBank.maximumPredictionHorizon = 20;
cfg.specialistBank.maximumHardContextsPerCell = 10;
cfg.specialistBank.maximumGrowthContextsPerCell = 5;
cfg.specialistBank.maximumSafeContextsPerCell = 1;
cfg.specialistBank.minimumStartStep = 5;
cfg.specialistBank.selectionRule = ...
    ['one_best_context_per_episode_then_balanced_cap_per_cell_' ...
    'hard_only_for_sac_growth_and_safe_retained_separately'];
cfg.specialistBank.absoluteThresholdIsGate = true;

cfg.split.screenSeed = 300830201;
cfg.split.teacherTrainSeed = 300830301;
cfg.split.teacherValidationSeed = 300830401;
cfg.split.surrogateTrainSeed = 300830501;
cfg.split.surrogateValidationSeed = 300830601;
cfg.split.confirmationSeed = 300830701;
cfg.split.oodLocked = true;
cfg.specialistBank.sourceSeed = cfg.split.teacherTrainSeed;

cfg.reproducibility.finalGridLocked = false;
cfg.reproducibility.longRunAuthorized = false;
cfg.reproducibility.oldOodUsedForTraining = false;
cfg.reproducibility.requiresRuntimeBenchmark = true;
cfg.reproducibility.requiresUserApprovalAfterBenchmark = true;
end
