function [Xref, Uref, details] = quad_complete_flat_reference( ...
        Xref, sampleTime, theta, yawReference)
%QUAD_COMPLETE_FLAT_REFERENCE Complete position/velocity into a full reference.
%
% The completion uses the desired translational acceleration to construct the
% body z-axis, then obtains Euler angles, body rates, and nominal feedforward
% generalized inputs for the 12-state quadrotor model.

if nargin < 3 || isempty(theta)
    cfg = step1_plant_config();
    theta = cfg.nominal;
end
if nargin < 4 || isempty(yawReference)
    yawReference = 0;
end

validateattributes(Xref, {'numeric'}, {'2d', 'nrows', 12, 'finite'});
validateattributes(sampleTime, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'positive'});
sampleCount = size(Xref, 2);
if sampleCount < 2
    error('quad_complete_flat_reference:TooShort', ...
        'At least two reference samples are required.');
end

yawReference = expand_yaw(yawReference, sampleCount);
velocity = Xref(7:9, :);
acceleration = finite_difference(velocity, sampleTime);
forceVector = theta.m .* (acceleration + [0; 0; theta.g]) + ...
    theta.Dv(:) .* velocity;
forceNorm = vecnorm(forceVector, 2, 1);
if any(forceNorm <= 100 * eps)
    error('quad_complete_flat_reference:DegenerateForce', ...
        'The desired specific-force direction is undefined.');
end

rotation = zeros(3, 3, sampleCount);
for index = 1:sampleCount
    b3 = forceVector(:, index) ./ forceNorm(index);
    heading = [cos(yawReference(index)); sin(yawReference(index)); 0];
    b2 = cross(b3, heading);
    if norm(b2) <= 100 * eps
        heading = [-sin(yawReference(index)); cos(yawReference(index)); 0];
        b2 = cross(b3, heading);
    end
    b2 = b2 ./ norm(b2);
    b1 = cross(b2, b3);
    rotation(:, :, index) = [b1, b2, b3];
end

attitude = zeros(3, sampleCount);
for index = 1:sampleCount
    R = rotation(:, :, index);
    attitude(1, index) = atan2(R(3, 2), R(3, 3));
    attitude(2, index) = asin(max(-1, min(1, -R(3, 1))));
    attitude(3, index) = atan2(R(2, 1), R(1, 1));
end

rotationRate = finite_difference(reshape(rotation, 9, sampleCount), ...
    sampleTime);
bodyRate = zeros(3, sampleCount);
for index = 1:sampleCount
    R = rotation(:, :, index);
    Rdot = reshape(rotationRate(:, index), 3, 3);
    skewRate = 0.5 .* (R.' * Rdot - Rdot.' * R);
    bodyRate(:, index) = [skewRate(3, 2); skewRate(1, 3); ...
        skewRate(2, 1)];
end
bodyAcceleration = finite_difference(bodyRate, sampleTime);

torque = zeros(3, sampleCount);
for index = 1:sampleCount
    omega = bodyRate(:, index);
    torque(:, index) = theta.J * bodyAcceleration(:, index) + ...
        cross(omega, theta.J * omega) + theta.Domega(:) .* omega;
end

Xref(4:6, :) = attitude;
Xref(10:12, :) = bodyRate;
Uref = [forceNorm ./ theta.alphaT; torque ./ theta.alphaTau(:)];

if nargout > 2
    details.acceleration = acceleration;
    details.forceVector = forceVector;
    details.rotation = rotation;
    details.bodyAcceleration = bodyAcceleration;
    details.tilt = acos(max(-1, min(1, squeeze(rotation(3, 3, :)).')));
end
end

function yaw = expand_yaw(yaw, sampleCount)
yaw = yaw(:).';
if isscalar(yaw)
    yaw = repmat(yaw, 1, sampleCount);
elseif numel(yaw) ~= sampleCount
    error('quad_complete_flat_reference:BadYaw', ...
        'yawReference must be scalar or contain one value per sample.');
end
if any(~isfinite(yaw))
    error('quad_complete_flat_reference:BadYaw', ...
        'yawReference must be finite.');
end
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
