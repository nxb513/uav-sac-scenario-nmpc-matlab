function probability = confidence_apply_calibrator(calibrator, logits)
%CONFIDENCE_APPLY_CALIBRATOR Apply a fitted binary calibrator.

logits = double(logits);
switch calibrator.method
    case 'none'
        probability = confidence_sigmoid(logits);
    case 'temperature'
        probability = confidence_sigmoid(logits ./ ...
            calibrator.temperature);
    case 'platt'
        probability = confidence_sigmoid( ...
            calibrator.scale .* logits + calibrator.bias);
    case 'isotonic'
        rawProbability = confidence_sigmoid(logits);
        probability = zeros(size(rawProbability));
        upperScore = calibrator.upperScore;
        value = calibrator.value;
        for index = 1:numel(rawProbability)
            block = find(rawProbability(index) <= upperScore, 1, 'first');
            if isempty(block)
                block = numel(upperScore);
            end
            probability(index) = value(block);
        end
    otherwise
        error('confidence_apply_calibrator:UnknownMethod', ...
            'Unknown calibration method: %s', calibrator.method);
end
probability = min(max(probability, 0), 1);
end
