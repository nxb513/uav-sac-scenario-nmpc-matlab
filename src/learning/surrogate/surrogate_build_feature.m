function feature = surrogate_build_feature(stateHistory, inputHistory, ...
        referenceLookahead, predictionResidual)
%SURROGATE_BUILD_FEATURE Assemble the current 208D surrogate input.

if ~isequal(size(stateHistory), [12, 4])
    error('surrogate_build_feature:BadStateHistory', ...
        'stateHistory must be 12-by-4.');
end
if ~isequal(size(inputHistory), [4, 4])
    error('surrogate_build_feature:BadInputHistory', ...
        'inputHistory must be 4-by-4.');
end
if ~isequal(size(referenceLookahead), [12, 11])
    error('surrogate_build_feature:BadReferenceLookahead', ...
        'referenceLookahead must be 12-by-11.');
end
predictionResidual = predictionResidual(:);
if numel(predictionResidual) ~= 12
    error('surrogate_build_feature:BadPredictionResidual', ...
        'predictionResidual must contain 12 elements.');
end

feature = [stateHistory(:); inputHistory(:); ...
    referenceLookahead(:); predictionResidual];
if numel(feature) ~= 208 || any(~isfinite(feature))
    error('surrogate_build_feature:InvalidFeature', ...
        'The feature must contain 208 finite values.');
end
end
