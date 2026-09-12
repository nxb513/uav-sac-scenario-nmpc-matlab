function row = confidence_binary_metrics(splitName, predictorName, ...
        labels, probability)
%CONFIDENCE_BINARY_METRICS Compute deterministic binary score metrics.

labels = double(labels(:));
probability = double(probability(:));
assert(numel(labels) == numel(probability) && ~isempty(labels));
finiteFraction = mean(isfinite(probability));
probability(~isfinite(probability)) = 0.5;
clipped = min(max(probability, 1e-7), 1 - 1e-7);
brier = mean((probability - labels) .^ 2);
nll = -mean(labels .* log(clipped) + (1 - labels) .* log(1 - clipped));
[ece10, mce10] = calibration_error(labels, probability, 10);
auroc = roc_auc(labels, probability);
auprc = pr_auc(labels, probability);
row = table(string(splitName), string(predictorName), numel(labels), ...
    mean(labels), mean(probability), brier, nll, ece10, mce10, ...
    auroc, auprc, finiteFraction, ...
    'VariableNames', {'Split', 'Predictor', 'Samples', 'GoodFraction', ...
    'MeanProbability', 'Brier', 'Nll', 'Ece10', 'Mce10', 'Auroc', ...
    'Auprc', 'FiniteFraction'});
end

function [ece, mce] = calibration_error(labels, probability, binCount)
edges = linspace(0, 1, binCount + 1);
ece = 0;
mce = 0;
for index = 1:binCount
    if index < binCount
        mask = probability >= edges(index) & ...
            probability < edges(index + 1);
    else
        mask = probability >= edges(index) & ...
            probability <= edges(index + 1);
    end
    if any(mask)
        gap = abs(mean(probability(mask)) - mean(labels(mask)));
        ece = ece + mean(mask) * gap;
        mce = max(mce, gap);
    end
end
end

function auc = roc_auc(labels, scores)
positiveCount = nnz(labels == 1);
negativeCount = nnz(labels == 0);
if positiveCount == 0 || negativeCount == 0
    auc = NaN;
    return;
end
[sortedScores, order] = sort(scores, 'ascend');
ranks = zeros(size(scores));
startIndex = 1;
while startIndex <= numel(scores)
    endIndex = startIndex;
    while endIndex < numel(scores) && ...
            sortedScores(endIndex + 1) == sortedScores(startIndex)
        endIndex = endIndex + 1;
    end
    ranks(order(startIndex:endIndex)) = 0.5 * (startIndex + endIndex);
    startIndex = endIndex + 1;
end
auc = (sum(ranks(labels == 1)) - ...
    positiveCount * (positiveCount + 1) / 2) / ...
    (positiveCount * negativeCount);
end

function value = pr_auc(labels, scores)
positiveCount = nnz(labels == 1);
if positiveCount == 0
    value = NaN;
    return;
end
[~, order] = sort(scores, 'descend');
sortedLabels = labels(order);
truePositive = cumsum(sortedLabels == 1);
falsePositive = cumsum(sortedLabels == 0);
recall = truePositive / positiveCount;
precision = truePositive ./ max(truePositive + falsePositive, 1);
value = trapz([0; recall], [1; precision]);
end
