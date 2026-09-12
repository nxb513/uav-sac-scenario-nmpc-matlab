function tableOut = confidence_risk_coverage(splitName, predictorName, ...
        labels, confidence, coverageLevels)
%CONFIDENCE_RISK_COVERAGE Compute bad-event risk at top-confidence coverage.

labels = double(labels(:));
confidence = double(confidence(:));
coverageLevels = double(coverageLevels(:));
[~, order] = sort(confidence, 'descend');
bad = 1 - labels(order);
sampleCount = numel(labels);
selectedCount = max(1, min(sampleCount, ...
    round(coverageLevels * sampleCount)));
risk = zeros(size(coverageLevels));
threshold = zeros(size(coverageLevels));
for index = 1:numel(coverageLevels)
    count = selectedCount(index);
    risk(index) = mean(bad(1:count));
    threshold(index) = confidence(order(count));
end
tableOut = table(repmat(string(splitName), numel(coverageLevels), 1), ...
    repmat(string(predictorName), numel(coverageLevels), 1), ...
    coverageLevels, selectedCount, threshold, risk, ...
    'VariableNames', {'Split', 'Predictor', 'Coverage', ...
    'SelectedSamples', 'ConfidenceThreshold', 'BadEventRisk'});
end
