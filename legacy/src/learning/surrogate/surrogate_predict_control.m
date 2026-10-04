function [appliedControl, details] = surrogate_predict_control( ...
        artifact, feature, plant)
%SURROGATE_PREDICT_CONTROL Run one bounded surrogate inference step.

if ~isequal(size(feature), [artifact.inputDimension, 1])
    error('surrogate_predict_control:BadFeature', ...
        'Expected a %d-by-1 feature.', artifact.inputDimension);
end
clock = tic;
normalizedRaw = (single(feature) - artifact.normalization.mean) ./ ...
    artifact.normalization.std;
normalizedClipped = min(max(normalizedRaw, ...
    -artifact.normalization.clip), artifact.normalization.clip);
prediction = predict(artifact.net, normalizedClipped.');
rawControl = double(artifact.targetMidpoint + ...
    artifact.targetHalfRange .* prediction(:));
appliedControl = quad_saturate_input(rawControl, plant);

details.normalizedRaw = double(normalizedRaw);
details.normalizedClipped = double(normalizedClipped);
details.rawControl = rawControl;
details.clipRate = mean(abs(normalizedRaw) > artifact.normalization.clip);
details.saturated = any(abs(appliedControl - rawControl) > 1e-10);
details.latencySeconds = toc(clock);
end
