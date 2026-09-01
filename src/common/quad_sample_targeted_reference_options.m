function options = quad_sample_targeted_reference_options( ...
        family, cfg, targetPeakSpeed, targetAcceleration)
%QUAD_SAMPLE_TARGETED_REFERENCE_OPTIONS Sample an approved speed/load setup.

if nargin < 3 || isempty(targetPeakSpeed)
    targetPeakSpeed = sample_member(cfg.approvedIdSpeedAnchors);
end
if nargin < 4 || isempty(targetAcceleration)
    targetAcceleration = sample_member( ...
        cfg.approvedAccelerationTargets);
end
validateattributes(targetPeakSpeed, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'positive'});
validateattributes(targetAcceleration, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'positive'});

geometryJitter = uniform(cfg.geometryScaleJitter(1), ...
    cfg.geometryScaleJitter(2));
geometryScale = max(cfg.minimumGeometryScale, ...
    targetPeakSpeed ^ 2 / targetAcceleration) * geometryJitter;
options = struct();
options.phase = uniform(0, 2 * pi);
options.yawReference = 0;
options.altitude = uniform(cfg.altitudeRange(1), cfg.altitudeRange(2));
options.targetPeakSpeed = targetPeakSpeed;
options.targetAcceleration = targetAcceleration;
options.geometryScaleJitter = geometryJitter;
switch family
    case 'circle'
        options.center = [uniform(-0.2, 0.2); uniform(-0.2, 0.2)];
        options.radius = geometryScale;
    case 'lemniscate'
        options.center = [uniform(-0.2, 0.2); uniform(-0.2, 0.2)];
        ratio = uniform(0.55, 0.75);
        options.amplitude = [geometryScale; ratio * geometryScale];
    case 'vertical_circle'
        effectiveAcceleration = min(targetAcceleration, ...
            cfg.verticalAccelerationFractionLimit * 9.81);
        if targetPeakSpeed >= 10 && effectiveAcceleration < 5
            effectiveAcceleration = 5;
        end
        radius = max(cfg.minimumGeometryScale, ...
            targetPeakSpeed ^ 2 / effectiveAcceleration) * ...
            geometryJitter;
        options.effectiveAcceleration = effectiveAcceleration;
        options.center = [uniform(-0.2, 0.2); ...
            radius + uniform(cfg.altitudeRange(1), cfg.altitudeRange(2))];
        options.lateralPosition = uniform(-0.2, 0.2);
        options.radius = radius;
    case 'spatial_helix'
        options.center = [uniform(-0.2, 0.2); uniform(-0.2, 0.2)];
        options.radius = geometryScale;
        options.verticalAmplitude = max(0.15, ...
            uniform(0.18, 0.30) * geometryScale);
        options.verticalRatio = uniform(0.35, 0.75);
        options.verticalPhase = uniform(0, 2 * pi);
        options.altitude = max(options.altitude, ...
            options.verticalAmplitude + 1.0);
    case 'smooth_waypoints'
        segmentDistance = max(0.45, ...
            1.65 * targetPeakSpeed ^ 2 / targetAcceleration) * ...
            geometryJitter;
        halfDiagonal = segmentDistance / sqrt(2);
        zOffset = min(0.50, 0.05 * segmentDistance);
        waypoints = [0, halfDiagonal, 0, -halfDiagonal, 0; ...
            -halfDiagonal, 0, halfDiagonal, 0, -halfDiagonal; ...
            options.altitude, options.altitude + zOffset, ...
            options.altitude, options.altitude - zOffset, ...
            options.altitude];
        heading = uniform(0, 2 * pi);
        rotation = [cos(heading), -sin(heading); ...
            sin(heading), cos(heading)];
        waypoints(1:2, :) = rotation * waypoints(1:2, :);
        segmentLengths = vecnorm(diff(waypoints, 1, 2));
        durations = 1.875 .* segmentLengths ./ targetPeakSpeed;
        options.waypoints = waypoints;
        options.segmentDurations = durations;
    otherwise
        error('quad_sample_targeted_reference_options:UnknownFamily', ...
            'Unknown family %s.', family);
end
end

function value = sample_member(values)
values = values(:).';
assert(~isempty(values) && all(isfinite(values)), ...
    'Approved sampling bank must contain finite values.');
value = values(randi(numel(values)));
end

function value = uniform(lower, upper)
value = lower + (upper - lower) * rand();
end
