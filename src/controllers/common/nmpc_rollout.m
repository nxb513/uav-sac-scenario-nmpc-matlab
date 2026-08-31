function X = nmpc_rollout(x0, U, theta, sampleTime, disturbanceSpec, startTime)
%NMPC_ROLLOUT Propagate the nonlinear quadrotor plant over an input sequence.

if nargin < 5
    disturbanceSpec = [];
end
if nargin < 6 || isempty(startTime)
    startTime = 0.0;
end

x0 = x0(:);
if numel(x0) ~= 12
    error('nmpc_rollout:BadStateSize', 'x0 must have 12 elements.');
end
if size(U, 1) ~= 4
    error('nmpc_rollout:BadInputSize', 'U must be 4-by-N.');
end
if sampleTime <= 0
    error('nmpc_rollout:BadSampleTime', 'sampleTime must be positive.');
end

horizon = size(U, 2);
X = zeros(12, horizon + 1);
X(:, 1) = x0;

for k = 1:horizon
    tk = startTime + (k - 1) * sampleTime;
    X(:, k + 1) = quad_step_rk4(tk, X(:, k), U(:, k), ...
                                sampleTime, theta, disturbanceSpec);
end
end
