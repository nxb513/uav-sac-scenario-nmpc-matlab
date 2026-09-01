function metric = quad_reference_feasibility_metrics( ...
        Xref, Uref, flatness, sampleTime, theta, nmpcCfg)
%QUAD_REFERENCE_FEASIBILITY_METRICS Physical margins of a full reference.

if nargin < 5 || isempty(theta)
    plantCfg = step1_plant_config();
    theta = plantCfg.nominal;
end
if nargin < 6 || isempty(nmpcCfg)
    nmpcCfg = step2_nmpc_config();
end
validateattributes(Xref, {'numeric'}, {'2d', 'nrows', 12, 'finite'});
validateattributes(Uref, {'numeric'}, {'2d', 'nrows', 4, 'finite'});
assert(size(Xref, 2) == size(Uref, 2), ...
    'Xref and Uref must have the same sample count.');

speed = vecnorm(Xref(7:9, :), 2, 1);
acceleration = flatness.acceleration;
accelerationNorm = vecnorm(acceleration, 2, 1);
horizontalAcceleration = vecnorm(acceleration(1:2, :), 2, 1);
tiltDeg = rad2deg(flatness.tilt(:).');

inputScale = [theta.inputLimits.T(2); ...
    max(abs(theta.inputLimits.tau), [], 2)];
inputFraction = abs(Uref) ./ max(inputScale, eps);
stateDerivative = finite_difference(Xref, sampleTime);
modelDerivative = zeros(size(Xref));
for index = 1:size(Xref, 2)
    modelDerivative(:, index) = quad_dynamics(0, Xref(:, index), ...
        Uref(:, index), theta, []);
end
dynamicResidual = stateDerivative - modelDerivative;
if size(Xref, 2) >= 7
    interior = 4:size(Xref, 2) - 3;
else
    interior = 1:size(Xref, 2);
end
residualNorm = vecnorm(dynamicResidual(:, interior), 2, 1);

metric.peakSpeed = max(speed);
metric.p95Speed = percentile_linear(speed, 95);
metric.meanSpeed = mean(speed);
metric.peakAcceleration = max(accelerationNorm);
metric.p95Acceleration = percentile_linear(accelerationNorm, 95);
metric.peakHorizontalAcceleration = max(horizontalAcceleration);
metric.peakTiltDeg = max(tiltDeg);
metric.p95TiltDeg = percentile_linear(tiltDeg, 95);
metric.peakThrustFraction = max(inputFraction(1, :));
metric.peakMomentFraction = max(inputFraction(2:4, :), [], 'all');
metric.peakInputFraction = max(inputFraction, [], 'all');
metric.minimumCosPitch = min(abs(cos(Xref(5, :))));
metric.peakBodyRate = max(vecnorm(Xref(10:12, :), 2, 1));
metric.dynamicResidualRms = sqrt(mean(residualNorm .^ 2));
metric.dynamicResidualP95 = percentile_linear(residualNorm, 95);
metric.dynamicResidualMax = max(residualNorm);
metric.stateBoundViolation = max(nmpc_state_bound_violations(Xref, ...
    nmpcCfg), [], 'all');
metric.finite = all(isfinite([Xref(:); Uref(:); residualNorm(:)]));
end

function derivative = finite_difference(signal, sampleTime)
sampleCount = size(signal, 2);
derivative = zeros(size(signal));
derivative(:, 1) = (signal(:, 2) - signal(:, 1)) ./ sampleTime;
derivative(:, end) = (signal(:, end) - signal(:, end - 1)) ./ sampleTime;
if sampleCount > 2
    derivative(:, 2:end-1) = (signal(:, 3:end) - ...
        signal(:, 1:end-2)) ./ (2 .* sampleTime);
end
end

function value = percentile_linear(values, percentage)
values = sort(values(isfinite(values)));
if isempty(values)
    value = NaN;
    return;
end
rankValue = 1 + (numel(values) - 1) * percentage / 100;
lowerIndex = floor(rankValue);
upperIndex = ceil(rankValue);
weight = rankValue - lowerIndex;
value = (1 - weight) * values(lowerIndex) + ...
    weight * values(upperIndex);
end
