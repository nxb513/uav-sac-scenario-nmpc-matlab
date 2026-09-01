function validate_predictive_divergence_implementation()
%VALIDATE_PREDICTIVE_DIVERGENCE_IMPLEMENTATION Deterministic contract tests.

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot, 'configs'));
addpath(fullfile(projectRoot, 'src', 'analysis'));
addpath(fullfile(projectRoot, 'src', 'plant'));
addpath(fullfile(projectRoot, 'src', 'controllers', 'common'));
addpath(fullfile(projectRoot, 'src', 'learning', 'surrogate'));

cfg = targeted_lqr_divergence_config();
nmpcCfg = step2_nmpc_config();
H = cfg.horizonSteps;
reference = zeros(12, H + 1);
X = reference;
hover = quad_hover_input(nmpcCfg.plant.nominal);
rawU = repmat(hover, 1, H);
saturated = false(1, H);

stable = lqr_finite_horizon_divergence_label(X, rawU, saturated, ...
    reference, 1, cfg, nmpcCfg);
assert(~stable.CounterfactualDivergence, ...
    'Zero-error hover must not be labeled as divergence.');

X(1, :) = linspace(0, 0.151, H + 1);
crossing = lqr_finite_horizon_divergence_label(X, rawU, saturated, ...
    reference, 1, cfg, nmpcCfg);
assert(crossing.CounterfactualDivergence && ...
    crossing.PreDivergenceEligible && crossing.TimeToEventSteps == H, ...
    'The synthetic position threshold must be detected at step H.');

X = reference;
X(1, :) = 0.20;
outside = lqr_finite_horizon_divergence_label(X, rawU, saturated, ...
    reference, 1, cfg, nmpcCfg);
assert(outside.CounterfactualDivergence && ...
    outside.AlreadyOutsideEnvelope && ...
    ~outside.PreDivergenceEligible && outside.TimeToEventSteps == 0, ...
    'A state already outside the envelope must not enter pre-divergence training.');

X = reference;
saturated(5) = true;
saturation = lqr_finite_horizon_divergence_label( ...
    X, rawU, saturated, reference, 1, cfg, nmpcCfg);
assert(saturation.CounterfactualDivergence && ...
    saturation.TimeToEventSteps == 5 && ...
    saturation.SaturationForecast, ...
    'The synthetic saturation event must be detected at step 5.');

lqr.K = zeros(4, 12);
hoverReference = zeros(12, H + 1);
hoverReference(3, :) = 1.0;
forecast = lqr_nominal_divergence_forecast( ...
    hoverReference(:, 1), hoverReference, repmat(hover, 1, H), ...
    lqr, nmpcCfg, cfg, zeros(12, 1));
assert(~forecast.CounterfactualDivergence && ...
    isequal(size(forecast.Xpred), [12, H + 1]), ...
    'Nominal hover forecast contract failed.');

feature = targeted_surrogate_build_feature(zeros(12, 4), ...
    zeros(4, 4), zeros(12, 21), zeros(12, 1));
assert(isequal(size(feature), [328, 1]), ...
    'The predictive surrogate feature must be 328-by-1.');

fprintf(['Predictive divergence validation passed: H=%d, ' ...
    'stable/crossing/already-outside/saturation/forecast/328D contracts.\n'], H);
end
