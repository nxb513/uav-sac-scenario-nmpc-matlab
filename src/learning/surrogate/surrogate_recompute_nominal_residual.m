function residual = surrogate_recompute_nominal_residual( ...
        state, input, stepIndex, sampleTime, thetaNominal)
%SURROGATE_RECOMPUTE_NOMINAL_RESIDUAL Build online-available delta x.

if size(state, 1) ~= 12 || size(input, 1) ~= 4
    error('surrogate_recompute_nominal_residual:BadTrajectoryShape', ...
        'State must be 12-by-(K+1) and input must be 4-by-K.');
end
if size(state, 2) ~= size(input, 2) + 1
    error('surrogate_recompute_nominal_residual:BadTrajectoryLength', ...
        'State must contain exactly one more sample than input.');
end
if sampleTime <= 0
    error('surrogate_recompute_nominal_residual:BadSampleTime', ...
        'sampleTime must be positive.');
end

stepIndex = double(stepIndex(:).');
if any(stepIndex < 2) || any(stepIndex > size(input, 2)) || ...
        any(stepIndex ~= floor(stepIndex))
    error('surrogate_recompute_nominal_residual:BadStepIndex', ...
        'stepIndex must contain integer control steps in [2,K].');
end

residual = zeros(12, numel(stepIndex), 'like', state);
for sampleIndex = 1:numel(stepIndex)
    k = stepIndex(sampleIndex);
    previousTime = (k - 2) * sampleTime;
    predicted = quad_step_rk4(previousTime, state(:, k - 1), ...
        input(:, k - 1), sampleTime, thetaNominal, []);
    residual(:, sampleIndex) = state(:, k) - predicted;
end

if any(~isfinite(residual), 'all')
    error('surrogate_recompute_nominal_residual:NonfiniteResidual', ...
        'Nominal residual contains nonfinite values.');
end
end
