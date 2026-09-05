function run_research_contract_p0_tests()
%RUN_RESEARCH_CONTRACT_P0_TESTS Focused tests for the active research contract.

projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(projectRoot, 'configs'));
addpath(genpath(fullfile(projectRoot, 'src')));

test_causal_feature(projectRoot);
test_prediction_error_wrapping();
test_finite_horizon_event_contract();
test_actuator_capacity_issue_is_explicit();

fprintf('Research contract P0 tests passed.\n');
end

function test_causal_feature(~)
plantCfg = step1_plant_config();
dt = 0.05;
count = 30;
X = zeros(12, count);
U = repmat(quad_hover_input(plantCfg.nominal), 1, count - 1);
reference = reshape(1:(12 * count), 12, count) ./ 100;
currentStep = 5;
[feature, parts] = targeted_causal_feature_from_trajectory( ...
    X, U, reference, currentStep, dt, plantCfg.nominal);
assert(isequal(size(feature), [328, 1]));
assert(isequal(parts.StateHistory, X(:, 2:5)));
assert(isequal(parts.InputHistory, U(:, 1:4)));
assert(isequal(parts.ReferenceLookahead, reference(:, 5:25)));
assert(norm(parts.PredictionResidual) < 1e-12, ...
    'Nominal hover must have zero one-step prediction error.');

changedAfterPreview = reference;
changedAfterPreview(:, 26:end) = changedAfterPreview(:, 26:end) + 1e6;
featureAfter = targeted_causal_feature_from_trajectory( ...
    X, U, changedAfterPreview, currentStep, dt, plantCfg.nominal);
assert(isequal(feature, featureAfter), ...
    'Feature changed when reference strictly after k+20 changed.');

assert_throws(@() targeted_causal_feature_from_trajectory( ...
    X, U, reference, 4, dt, plantCfg.nominal), ...
    'MATLAB:notGreaterEqual');
assert_throws(@() targeted_causal_feature_from_trajectory( ...
    X, U, reference(:, 1:24), currentStep, dt, plantCfg.nominal), ...
    'targeted_causal_feature_from_trajectory:MissingPreview');
end

function test_prediction_error_wrapping()
actual = zeros(12, 1);
predicted = zeros(12, 1);
actual(6) = -pi + 0.01;
predicted(6) = pi - 0.01;
residual = quad_state_prediction_error(actual, predicted);
assert(abs(residual(6) - 0.02) < 1e-12, ...
    'Euler prediction error was not wrapped to the shortest branch.');
assert_throws(@() quad_state_prediction_error(actual(1:11), predicted), ...
    'quad_state_prediction_error:BadStateSize');
end

function test_finite_horizon_event_contract()
cfg = targeted_lqr_divergence_config();
nmpcCfg = step2_nmpc_config();
H = cfg.horizonSteps;
X = zeros(12, H + 1);
reference = zeros(12, H + 1);
rawU = repmat(quad_hover_input(nmpcCfg.plant.nominal), 1, H);
saturated = false(1, H);

tracking = X;
tracking(1, 2) = cfg.threshold.positionM;
label = lqr_finite_horizon_divergence_label( ...
    tracking, rawU, saturated, reference, 1, cfg, nmpcCfg);
assert(label.HardEvent && label.TimeToEventSteps == 1);
assert(label.FirstHardEventType == "tracking");
assert(label.FirstTrackingEventOffset == 1);

saturation = saturated;
saturation(1) = true;
label = lqr_finite_horizon_divergence_label( ...
    X, rawU, saturation, reference, 1, cfg, nmpcCfg);
assert(label.HardEvent && label.TimeToEventSteps == 1);
assert(label.FirstHardEventType == "saturation");

outside = X;
outsideReference = reference;
outside(3, :) = nmpcCfg.constraints.stateLower(3) - 1;
outsideReference(3, :) = outside(3, :);
label = lqr_finite_horizon_divergence_label( ...
    outside, rawU, saturated, outsideReference, 1, cfg, nmpcCfg);
assert(label.AlreadyOutsideEnvelope && label.AlreadyOutsideConstraint);
assert(~label.HardEvent && ~label.PreDivergenceEligible, ...
    'A context already outside a state constraint cannot be pre-event safe.');

nonfinite = X;
nonfinite(1, 2) = NaN;
label = lqr_finite_horizon_divergence_label( ...
    nonfinite, rawU, saturated, reference, 1, cfg, nmpcCfg);
assert(label.HardEvent && contains(label.FirstHardEventType, "nonfinite"));

assert_throws(@() lqr_finite_horizon_divergence_label( ...
    X(:, 1:end-1), rawU, saturated, reference, 1, cfg, nmpcCfg), ...
    '');
end

function test_actuator_capacity_issue_is_explicit()
plantCfg = step1_plant_config();
xi = zeros(14, 1);
rho = zeros(14, 1);
xi(1) = 1;
rho(1) = 0.10;
changed = quad_apply_uncertainty(plantCfg.nominal, xi, rho);
assert(isequal(changed.inputLimits, plantCfg.nominal.inputLimits), ...
    'Mass uncertainty must not change fixed hardware input limits.');

shift = plantCfg.modelShift.payload25;
shift.preserveInputLimits = true;
shifted = quad_apply_model_shift(plantCfg.nominal, shift, shift.startTime);
assert(isequal(shifted.inputLimits, plantCfg.nominal.inputLimits), ...
    'The explicit fixed-hardware model-shift path must preserve limits.');
end

function assert_throws(operation, expectedIdentifier)
threw = false;
try
    operation();
catch exception
    threw = true;
    if ~isempty(expectedIdentifier)
        assert(strcmp(exception.identifier, expectedIdentifier), ...
            'Unexpected error identifier: %s.', exception.identifier);
    end
end
assert(threw, 'Expected operation to throw an error.');
end
