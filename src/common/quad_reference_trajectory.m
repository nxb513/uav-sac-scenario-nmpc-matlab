function [Xref, time] = quad_reference_trajectory(kind, sampleTime, sampleCount, options)
%QUAD_REFERENCE_TRAJECTORY Build reproducible 12-state benchmark references.

if nargin < 4 || isempty(options)
    options = struct();
end
if sampleTime <= 0 || ~isfinite(sampleTime)
    error('quad_reference_trajectory:BadSampleTime', ...
          'sampleTime must be a positive finite scalar.');
end
if sampleCount <= 0 || sampleCount ~= floor(sampleCount)
    error('quad_reference_trajectory:BadSampleCount', ...
          'sampleCount must be a positive integer.');
end

time = (0:sampleCount - 1) * sampleTime;
Xref = zeros(12, sampleCount);

switch lower(kind)
    case 'hover'
        position = option_or(options, 'position', [0.0; 0.0; 1.0]);
        Xref(1:3, :) = repmat(validate_vector(position, 3, 'position'), 1, sampleCount);

    case {'step', 'position_step'}
        positionBefore = option_or(options, 'positionBefore', [0.0; 0.0; 1.0]);
        positionAfter = option_or(options, 'positionAfter', [0.4; -0.3; 1.2]);
        stepTime = option_or(options, 'stepTime', 0.5);
        positionBefore = validate_vector(positionBefore, 3, 'positionBefore');
        positionAfter = validate_vector(positionAfter, 3, 'positionAfter');

        Xref(1:3, :) = repmat(positionBefore, 1, sampleCount);
        Xref(1:3, time >= stepTime) = repmat(positionAfter, 1, nnz(time >= stepTime));

    case 'circle'
        center = validate_vector(option_or(options, 'center', [0.0; 0.0]), 2, 'center');
        radius = option_or(options, 'radius', 0.5);
        altitude = option_or(options, 'altitude', 1.0);
        angularRate = option_or(options, 'angularRate', 0.4);
        phase = option_or(options, 'phase', 0.0);
        if radius <= 0 || angularRate <= 0
            error('quad_reference_trajectory:BadCircle', ...
                  'radius and angularRate must be positive.');
        end

        angle = angularRate * time + phase;
        Xref(1, :) = center(1) + radius * cos(angle);
        Xref(2, :) = center(2) + radius * sin(angle);
        Xref(3, :) = altitude;
        Xref(7, :) = -radius * angularRate * sin(angle);
        Xref(8, :) = radius * angularRate * cos(angle);

    case {'lemniscate', 'figure_eight'}
        center = validate_vector(option_or(options, 'center', [0.0; 0.0]), 2, 'center');
        amplitude = validate_vector(option_or(options, 'amplitude', [0.6; 0.4]), 2, 'amplitude');
        altitude = option_or(options, 'altitude', 1.0);
        angularRate = option_or(options, 'angularRate', 0.83);
        phase = option_or(options, 'phase', 0.0);
        if any(amplitude <= 0) || angularRate <= 0
            error('quad_reference_trajectory:BadLemniscate', ...
                  'amplitude and angularRate must be positive.');
        end

        angle = angularRate * time + phase;
        Xref(1, :) = center(1) + amplitude(1) * sin(angle);
        Xref(2, :) = center(2) + amplitude(2) * sin(angle) .* cos(angle);
        Xref(3, :) = altitude;
        Xref(7, :) = amplitude(1) * angularRate * cos(angle);
        Xref(8, :) = amplitude(2) * angularRate * cos(2 * angle);

    case {'vertical_circle', 'circle_xz'}
        center = validate_vector(option_or(options, 'center', [0.0; 1.0]), 2, 'center');
        lateralPosition = option_or(options, 'lateralPosition', 0.0);
        radius = option_or(options, 'radius', 0.4);
        angularRate = option_or(options, 'angularRate', 2.0);
        phase = option_or(options, 'phase', 0.0);
        if radius <= 0 || angularRate <= 0
            error('quad_reference_trajectory:BadVerticalCircle', ...
                  'radius and angularRate must be positive.');
        end

        angle = angularRate * time + phase;
        Xref(1, :) = center(1) + radius * cos(angle);
        Xref(2, :) = lateralPosition;
        Xref(3, :) = center(2) + radius * sin(angle);
        Xref(7, :) = -radius * angularRate * sin(angle);
        Xref(8, :) = 0.0;
        Xref(9, :) = radius * angularRate * cos(angle);

    otherwise
        error('quad_reference_trajectory:BadKind', ...
              'Unknown reference kind: %s', kind);
end
end

function value = option_or(options, name, defaultValue)
if isfield(options, name)
    value = options.(name);
else
    value = defaultValue;
end
end

function value = validate_vector(value, expectedCount, name)
value = value(:);
if numel(value) ~= expectedCount || any(~isfinite(value))
    error('quad_reference_trajectory:BadOption', ...
          '%s must contain %d finite values.', name, expectedCount);
end
end
