function [labels, grid] = confidence_build_label(X, rawU, reference, ...
        controlIndices, labelCfg)
%CONFIDENCE_BUILD_LABEL Build finite-horizon tracking/safety labels.

horizons = labelCfg.horizons;
positionThresholds = labelCfg.positionThresholds;
attitudeThresholds = deg2rad(labelCfg.attitudeThresholdsDeg);
combinationCount = numel(horizons) * numel(positionThresholds) * ...
    numel(attitudeThresholds);
labels = false(numel(controlIndices), combinationCount);

Horizon = zeros(combinationCount, 1);
PositionThreshold = zeros(combinationCount, 1);
AttitudeThresholdDeg = zeros(combinationCount, 1);
column = 0;
for horizon = horizons
    for positionThreshold = positionThresholds
        for attitudeIndex = 1:numel(attitudeThresholds)
            column = column + 1;
            attitudeThreshold = attitudeThresholds(attitudeIndex);
            Horizon(column) = horizon;
            PositionThreshold(column) = positionThreshold;
            AttitudeThresholdDeg(column) = ...
                labelCfg.attitudeThresholdsDeg(attitudeIndex);
            for sampleIndex = 1:numel(controlIndices)
                k = controlIndices(sampleIndex);
                stateColumns = k + 1:k + horizon;
                inputColumns = k:k + horizon - 1;
                futureX = X(:, stateColumns);
                futureReference = reference(:, stateColumns);
                error = futureX - futureReference;
                error(4:6, :) = atan2(sin(error(4:6, :)), ...
                    cos(error(4:6, :)));
                positionOk = all(vecnorm(error(1:3, :), 2, 1) <= ...
                    positionThreshold);
                attitudeOk = all(vecnorm(error(4:6, :), 2, 1) <= ...
                    attitudeThreshold);
                stateOk = all(futureX >= labelCfg.constraint.stateLower & ...
                    futureX <= labelCfg.constraint.stateUpper, 'all') && ...
                    all(vecnorm(futureX(4:5, :), 2, 1) <= ...
                    labelCfg.constraint.maxTilt);
                futureRawU = rawU(:, inputColumns);
                inputOk = all(futureRawU >= ...
                    labelCfg.constraint.inputLower & futureRawU <= ...
                    labelCfg.constraint.inputUpper, 'all');
                finiteOk = all(isfinite(futureX), 'all') && ...
                    all(isfinite(futureRawU), 'all');
                labels(sampleIndex, column) = positionOk && attitudeOk && ...
                    stateOk && inputOk && finiteOk;
            end
        end
    end
end
grid = table((1:combinationCount).', Horizon, PositionThreshold, ...
    AttitudeThresholdDeg, 'VariableNames', {'Column', 'Horizon', ...
    'PositionThreshold', 'AttitudeThresholdDeg'});
end
