function forecast = lqr_nominal_divergence_forecast(state, reference, ...
        feedforward, lqr, nmpcCfg, divergenceCfg, oneStepResidual)
%LQR_NOMINAL_DIVERGENCE_FORECAST Cheap online 20-step LQR risk forecast.

H = divergenceCfg.horizonSteps;
if nargin < 7 || isempty(oneStepResidual)
    oneStepResidual = zeros(12, 1);
end
assert(isequal(size(state), [12, 1]), 'state must be 12-by-1.');
assert(isequal(size(reference), [12, H + 1]), ...
    'reference must contain the current state and H future states.');
assert(isequal(size(feedforward), [4, H]), ...
    'feedforward must contain H future generalized inputs.');
assert(isequal(size(oneStepResidual), [12, 1]) && ...
    all(isfinite(oneStepResidual)), ...
    'oneStepResidual must be a finite 12-by-1 vector.');

X = nan(12, H + 1);
U = nan(4, H);
rawU = nan(4, H);
saturated = false(1, H);
X(:, 1) = state;
for step = 1:H
    error = nmpc_state_error(X(:, step), reference(:, step));
    rawU(:, step) = feedforward(:, step) - lqr.K * error;
    U(:, step) = quad_saturate_input(rawU(:, step), ...
        nmpcCfg.plant.nominal);
    saturated(step) = any(abs(rawU(:, step) - U(:, step)) > 1e-10);
    nominalNext = quad_step_rk4((step - 1) * nmpcCfg.sampleTime, ...
        X(:, step), U(:, step), nmpcCfg.sampleTime, ...
        nmpcCfg.plant.nominal, []);
    residualWeight = divergenceCfg.online.residualDecay ^ (step - 1);
    X(:, step + 1) = nominalNext + residualWeight .* oneStepResidual;
end

forecast = lqr_finite_horizon_divergence_label(X, rawU, saturated, ...
    reference, 1, divergenceCfg, nmpcCfg);
forecast.PredictionSource = ...
    'nominal_20step_lqr_plus_decaying_measured_one_step_residual';
forecast.ResidualDecay = divergenceCfg.online.residualDecay;
forecast.Xpred = X;
forecast.Upred = U;
forecast.RawUpred = rawU;
end
