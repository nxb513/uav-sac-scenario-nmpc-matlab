function calibrator = confidence_fit_calibrator(method, logits, labels)
%CONFIDENCE_FIT_CALIBRATOR Fit a monotone post-hoc binary calibrator.

method = char(lower(string(method)));
logits = double(logits(:));
labels = double(labels(:));
assert(numel(logits) == numel(labels) && ~isempty(labels));
assert(all(isfinite(logits)) && all(ismember(labels, [0, 1])));
calibrator.method = method;

switch method
    case 'none'
        return;
    case 'temperature'
        objective = @(logTemperature) binary_nll( ...
            logits ./ exp(logTemperature), labels);
        calibrator.logTemperature = fminbnd(objective, -4, 4);
        calibrator.temperature = exp(calibrator.logTemperature);
    case 'platt'
        objective = @(parameters) binary_nll( ...
            exp(clamp(parameters(1), -6, 6)) .* logits + ...
            clamp(parameters(2), -20, 20), labels);
        options = optimset('Display', 'off', 'MaxIter', 1000, ...
            'MaxFunEvals', 3000, 'TolX', 1e-8, 'TolFun', 1e-10);
        parameters = fminsearch(objective, [0, 0], options);
        calibrator.scale = exp(clamp(parameters(1), -6, 6));
        calibrator.bias = clamp(parameters(2), -20, 20);
    case 'isotonic'
        score = confidence_sigmoid(logits);
        [score, order] = sort(score, 'ascend');
        labels = labels(order);
        [uniqueScore, ~, group] = unique(score);
        weight = accumarray(group, 1);
        value = accumarray(group, labels) ./ weight;
        upperScore = uniqueScore;
        blockCount = 0;
        blockWeight = zeros(size(weight));
        blockValue = zeros(size(value));
        blockUpper = zeros(size(upperScore));
        for index = 1:numel(uniqueScore)
            blockCount = blockCount + 1;
            blockWeight(blockCount) = weight(index);
            blockValue(blockCount) = value(index);
            blockUpper(blockCount) = upperScore(index);
            while blockCount > 1 && ...
                    blockValue(blockCount - 1) > blockValue(blockCount)
                totalWeight = blockWeight(blockCount - 1) + ...
                    blockWeight(blockCount);
                blockValue(blockCount - 1) = ( ...
                    blockWeight(blockCount - 1) * ...
                    blockValue(blockCount - 1) + ...
                    blockWeight(blockCount) * blockValue(blockCount)) / ...
                    totalWeight;
                blockWeight(blockCount - 1) = totalWeight;
                blockUpper(blockCount - 1) = blockUpper(blockCount);
                blockCount = blockCount - 1;
            end
        end
        calibrator.upperScore = blockUpper(1:blockCount);
        calibrator.value = blockValue(1:blockCount);
        calibrator.blockWeight = blockWeight(1:blockCount);
    otherwise
        error('confidence_fit_calibrator:UnknownMethod', ...
            'Unknown calibration method: %s', method);
end
end

function value = binary_nll(logit, labels)
value = mean(max(logit, 0) - logit .* labels + log1p(exp(-abs(logit))));
if ~isfinite(value)
    value = realmax('double');
end
end

function output = clamp(input, lower, upper)
output = min(max(input, lower), upper);
end
