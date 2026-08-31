function dist = quad_disturbance(t, spec)
%QUAD_DISTURBANCE Build additive force/torque disturbance at time t.

dist.force = zeros(3, 1);
dist.torque = zeros(3, 1);

if nargin < 2 || isempty(spec)
    return;
end

if isfield(spec, 'time') && isfield(spec, 'forceSeries') && isfield(spec, 'torqueSeries')
    time = spec.time(:).';
    if size(spec.forceSeries, 1) ~= 3 || size(spec.torqueSeries, 1) ~= 3 || ...
            size(spec.forceSeries, 2) ~= numel(time) || ...
            size(spec.torqueSeries, 2) ~= numel(time)
        error('quad_disturbance:BadTimeSeries', ...
              'forceSeries and torqueSeries must be 3-by-numel(time).');
    end
    sampleIndex = find(time <= t + 10 * eps(max(1.0, abs(t))), 1, 'last');
    if isempty(sampleIndex)
        sampleIndex = 1;
    end
    sampleIndex = min(sampleIndex, numel(time));
    dist.force = spec.forceSeries(:, sampleIndex);
    dist.torque = spec.torqueSeries(:, sampleIndex);
    return;
end

if isfield(spec, 'force')
    dist.force = dist.force + spec.force(:);
end
if isfield(spec, 'torque')
    dist.torque = dist.torque + spec.torque(:);
end

if isfield(spec, 'forceSineAmp') && isfield(spec, 'forceSineFreq')
    amp = spec.forceSineAmp(:);
    freq = spec.forceSineFreq(:);
    dist.force = dist.force + amp .* sin(2 * pi * freq * t);
end

if isfield(spec, 'torqueSineAmp') && isfield(spec, 'torqueSineFreq')
    amp = spec.torqueSineAmp(:);
    freq = spec.torqueSineFreq(:);
    dist.torque = dist.torque + amp .* sin(2 * pi * freq * t);
end
end
