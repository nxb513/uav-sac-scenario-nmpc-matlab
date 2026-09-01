function cfg = targeted_lqr_divergence_config()
%TARGETED_LQR_DIVERGENCE_CONFIG Candidate finite-horizon LQR risk rule.
%
% The 20-step horizon follows the user-defined predictive-intervention
% hypothesis and matches the maximum NMPC prediction horizon. Numerical
% thresholds remain candidates until the LQR-only screening audit is reviewed.

cfg.name = 'targeted_lqr_finite_horizon_divergence_v1';
cfg.status = 'candidate_thresholds_require_screening_and_user_approval';
cfg.finalThresholdsLocked = false;
cfg.horizonSteps = 20;
cfg.horizonSeconds = 1.0;
cfg.sensitivityHorizonSteps = [5, 10, 20];

cfg.threshold.positionM = 0.15;
cfg.threshold.attitudeDeg = 10.0;
cfg.threshold.velocityMps = 0.50;
cfg.threshold.bodyRateRadps = 1.00;
cfg.threshold.warningFraction = 0.50;

cfg.growth.energyFloor = 0.25;
cfg.growth.factor = 1.50;
cfg.growth.minimumRelativeStep = 0.02;
cfg.growth.minimumConsecutiveSteps = 3;

cfg.candidateGrid.positionM = [0.10, 0.15, 0.20, 0.25];
cfg.candidateGrid.attitudeDeg = [5, 10, 15, 20];
cfg.candidateGrid.velocityMps = [0.30, 0.50, 0.75, 1.00];
cfg.candidateGrid.bodyRateRadps = [1.0, 2.0, 3.0, 4.0];
cfg.candidateGrid.growthFactor = [1.25, 1.50, 2.00];
cfg.candidateGrid.minimumConsecutiveSteps = [3, 5];

cfg.selection.minimumLeadSteps = 1;
cfg.selection.preferMaximumLead = true;
cfg.selection.keepOneContextPerFactorialCell = true;
cfg.selection.trainingIncludesBoundaryOnly = false;
cfg.selection.rule = ...
    'counterfactual_event_then_maximum_lead_then_maximum_risk';

cfg.online.hiddenPlantParametersAllowed = false;
cfg.online.hiddenDisturbanceRealizationAllowed = false;
cfg.online.residualDecay = 0.90;
cfg.online.candidateInputs = ...
    ['state_input_history_reference_k_to_k_plus_20_' ...
    'nominal_lqr_rollout_one_step_residual_envelope_margins'];

cfg.surrogate.stateHistoryLength = 4;
cfg.surrogate.previousInputHistoryLength = 4;
cfg.surrogate.referenceLookahead = 20;
cfg.surrogate.predictionResidualDimension = 12;
cfg.surrogate.featureDimension = 328;
cfg.surrogate.legacy208Compatible = false;
end
