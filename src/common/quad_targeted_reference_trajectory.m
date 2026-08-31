function [Xref, time, Uref, flatness] = quad_targeted_reference_trajectory( ...
        kind, sampleTime, sampleCount, options, theta)
%QUAD_TARGETED_REFERENCE_TRAJECTORY Dynamic references for specialist training.

if nargin < 4 || isempty(options)
    options = struct();
end
if nargin < 5 || isempty(theta)
    cfg = step1_plant_config();
    theta = cfg.nominal;
end

switch lower(kind)
    case 'circle'
        radius = positive_scalar(option_or(options, 'radius', 0.5), ...
            'radius');
        peakSpeed = positive_scalar(option_or(options, ...
            'targetPeakSpeed', 1.0), 'targetPeakSpeed');
        base = options;
        base.radius = radius;
        base.angularRate = peakSpeed / radius;
        [Xref, time] = quad_reference_trajectory('circle', sampleTime, ...
            sampleCount, base);

    case {'lemniscate', 'figure_eight'}
        amplitude = vector_option(options, 'amplitude', [0.6; 0.4], 2);
        peakSpeed = positive_scalar(option_or(options, ...
            'targetPeakSpeed', 1.0), 'targetPeakSpeed');
        phaseGrid = linspace(0, 2 * pi, 4097);
        unitSpeed = sqrt((amplitude(1) .* cos(phaseGrid)) .^ 2 + ...
            (amplitude(2) .* cos(2 .* phaseGrid)) .^ 2);
        base = options;
        base.amplitude = amplitude;
        base.angularRate = peakSpeed / max(unitSpeed);
        [Xref, time] = quad_reference_trajectory('lemniscate', ...
            sampleTime, sampleCount, base);

    case {'vertical_circle', 'circle_xz'}
        radius = positive_scalar(option_or(options, 'radius', 0.4), ...
            'radius');
        peakSpeed = positive_scalar(option_or(options, ...
            'targetPeakSpeed', 1.0), 'targetPeakSpeed');
        base = options;
        base.radius = radius;
        base.angularRate = peakSpeed / radius;
        [Xref, time] = quad_reference_trajectory('vertical_circle', ...
            sampleTime, sampleCount, base);

    case {'spatial_helix', 'helix'}
        validate_time(sampleTime, sampleCount);
        time = (0:sampleCount - 1) * sampleTime;
        center = vector_option(options, 'center', [0; 0], 2);
        radius = positive_scalar(option_or(options, 'radius', 0.5), ...
            'radius');
        altitude = finite_scalar(option_or(options, 'altitude', 1.0), ...
            'altitude');
        verticalAmplitude = positive_scalar(option_or(options, ...
            'verticalAmplitude', 0.25), 'verticalAmplitude');
        verticalRatio = positive_scalar(option_or(options, ...
            'verticalRatio', 0.5), 'verticalRatio');
        phase = finite_scalar(option_or(options, 'phase', 0), 'phase');
        verticalPhase = finite_scalar(option_or(options, ...
            'verticalPhase', 0), 'verticalPhase');
        peakSpeed = positive_scalar(option_or(options, ...
            'targetPeakSpeed', 1.0), 'targetPeakSpeed');
        speedFactor = sqrt(radius ^ 2 + ...
            (verticalAmplitude * verticalRatio) ^ 2);
        angularRate = peakSpeed / speedFactor;
        angle = angularRate .* time + phase;
        verticalAngle = verticalRatio .* angle + verticalPhase;
        Xref = zeros(12, sampleCount);
        Xref(1, :) = center(1) + radius .* cos(angle);
        Xref(2, :) = center(2) + radius .* sin(angle);
        Xref(3, :) = altitude + verticalAmplitude .* sin(verticalAngle);
        Xref(7, :) = -radius .* angularRate .* sin(angle);
        Xref(8, :) = radius .* angularRate .* cos(angle);
        Xref(9, :) = verticalAmplitude .* verticalRatio .* ...
            angularRate .* cos(verticalAngle);

    case {'smooth_waypoints', 'minimum_jerk_waypoints'}
        validate_time(sampleTime, sampleCount);
        time = (0:sampleCount - 1) * sampleTime;
        waypoints = option_or(options, 'waypoints', ...
            [0, 0.8, 0.2; 0, 0.3, -0.6; 1.0, 1.2, 0.9]);
        durations = option_or(options, 'segmentDurations', [2.0, 2.0]);
        if size(waypoints, 1) ~= 3 || size(waypoints, 2) < 2 || ...
                any(~isfinite(waypoints), 'all')
            error('quad_targeted_reference_trajectory:BadWaypoints', ...
                'waypoints must be a finite 3-by-N matrix with N >= 2.');
        end
        durations = durations(:).';
        if numel(durations) ~= size(waypoints, 2) - 1 || ...
                any(~isfinite(durations)) || any(durations <= 0)
            error('quad_targeted_reference_trajectory:BadDurations', ...
                'segmentDurations must contain N-1 positive values.');
        end
        [position, velocity] = minimum_jerk_path(time, waypoints, durations);
        Xref = zeros(12, sampleCount);
        Xref(1:3, :) = position;
        Xref(7:9, :) = velocity;

    otherwise
        error('quad_targeted_reference_trajectory:BadKind', ...
            'Unknown targeted reference kind: %s.', kind);
end
[Xref, Uref, flatness] = quad_complete_flat_reference(Xref, sampleTime, ...
    theta, option_or(options, 'yawReference', 0));
end

function [position, velocity] = minimum_jerk_path(time, waypoints, durations)
edges = [0, cumsum(durations)];
position = repmat(waypoints(:, end), 1, numel(time));
velocity = zeros(3, numel(time));
for segment = 1:numel(durations)
    if segment < numel(durations)
        active = time >= edges(segment) & time < edges(segment + 1);
    else
        active = time >= edges(segment) & time <= edges(segment + 1);
    end
    tau = (time(active) - edges(segment)) ./ durations(segment);
    blend = 10 .* tau .^ 3 - 15 .* tau .^ 4 + 6 .* tau .^ 5;
    blendRate = (30 .* tau .^ 2 - 60 .* tau .^ 3 + ...
        30 .* tau .^ 4) ./ durations(segment);
    delta = waypoints(:, segment + 1) - waypoints(:, segment);
    position(:, active) = waypoints(:, segment) + delta .* blend;
    velocity(:, active) = delta .* blendRate;
end
end

function validate_time(sampleTime, sampleCount)
if ~isscalar(sampleTime) || ~isfinite(sampleTime) || sampleTime <= 0
    error('quad_targeted_reference_trajectory:BadSampleTime', ...
        'sampleTime must be positive and finite.');
end
if ~isscalar(sampleCount) || sampleCount < 1 || ...
        sampleCount ~= floor(sampleCount)
    error('quad_targeted_reference_trajectory:BadSampleCount', ...
        'sampleCount must be a positive integer.');
end
end

function value = option_or(options, name, defaultValue)
if isfield(options, name)
    value = options.(name);
else
    value = defaultValue;
end
end

function value = vector_option(options, name, defaultValue, count)
value = option_or(options, name, defaultValue);
value = value(:);
if numel(value) ~= count || any(~isfinite(value))
    error('quad_targeted_reference_trajectory:BadVector', ...
        '%s must contain %d finite values.', name, count);
end
end

function value = positive_scalar(value, name)
value = finite_scalar(value, name);
if value <= 0
    error('quad_targeted_reference_trajectory:BadPositiveScalar', ...
        '%s must be positive.', name);
end
end

function value = finite_scalar(value, name)
if ~isscalar(value) || ~isfinite(value)
    error('quad_targeted_reference_trajectory:BadScalar', ...
        '%s must be a finite scalar.', name);
end
end
