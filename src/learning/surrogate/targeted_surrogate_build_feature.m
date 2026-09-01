function feature = targeted_surrogate_build_feature(stateHistory, ...
        inputHistory, referenceLookahead, predictionResidual)
%TARGETED_SURROGATE_BUILD_FEATURE Assemble the predictive 328D input.

if ~isequal(size(stateHistory), [12, 4])
    error('targeted_surrogate_build_feature:BadStateHistory', ...
        'stateHistory must be 12-by-4.');
end
if ~isequal(size(inputHistory), [4, 4])
    error('targeted_surrogate_build_feature:BadInputHistory', ...
        'inputHistory must be 4-by-4.');
end
if ~isequal(size(referenceLookahead), [12, 21])
    error('targeted_surrogate_build_feature:BadReferenceLookahead', ...
        'referenceLookahead must be 12-by-21 for k:k+20.');
end
if ~isequal(size(predictionResidual), [12, 1])
    error('targeted_surrogate_build_feature:BadResidual', ...
        'predictionResidual must be 12-by-1.');
end
feature = [stateHistory(:); inputHistory(:); ...
    referenceLookahead(:); predictionResidual];
if numel(feature) ~= 328 || any(~isfinite(feature))
    error('targeted_surrogate_build_feature:InvalidFeature', ...
        'The targeted surrogate feature must contain 328 finite values.');
end
end
