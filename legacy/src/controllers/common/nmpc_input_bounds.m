function [lb, ub, uLower, uUpper] = nmpc_input_bounds(theta, horizon)
%NMPC_INPUT_BOUNDS Vectorized generalized-input bounds for fmincon.

if horizon <= 0 || horizon ~= floor(horizon)
    error('nmpc_input_bounds:BadHorizon', ...
          'horizon must be a positive integer.');
end

limits = theta.inputLimits;
uLower = [limits.T(1); limits.tau(:, 1)];
uUpper = [limits.T(2); limits.tau(:, 2)];

lb = repmat(uLower, horizon, 1);
ub = repmat(uUpper, horizon, 1);
end
