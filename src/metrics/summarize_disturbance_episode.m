function metrics = summarize_disturbance_episode(episode, reference, cfg, disturbanceSpec, recoveryCfg)
%SUMMARIZE_DISTURBANCE_EPISODE Add disturbance response and recovery metrics.

metrics = summarize_control_episode(episode, reference, cfg);
if nargin < 5 || isempty(recoveryCfg)
    recoveryCfg.positionTolerance = 0.10;
    recoveryCfg.relativeToPreEvent = 1.50;
    recoveryCfg.dwellTime = 0.15;
end

stepCount = size(episode.U, 2);
time = (0:stepCount) * episode.sampleTime;
Xref = reference_for_episode(reference, stepCount, episode.sampleTime);
error = nmpc_state_error(episode.X, Xref);
positionNorm = sqrt(sum(error(1:3, :).^2, 1));
attitudeNorm = sqrt(sum(error(4:6, :).^2, 1));

metrics.disturbanceType = disturbanceSpec.type;
metrics.disturbanceDomain = disturbanceSpec.domain;
metrics.disturbanceLevel = disturbanceSpec.levelIndex;
metrics.disturbanceForcePeak = disturbanceSpec.forcePeak;
metrics.disturbanceTorquePeak = disturbanceSpec.torquePeak;
metrics.disturbanceForceRms = sqrt(mean(sum(disturbanceSpec.forceSeries(:, 1:stepCount).^2, 1)));
metrics.disturbanceTorqueRms = sqrt(mean(sum(disturbanceSpec.torqueSeries(:, 1:stepCount).^2, 1)));
metrics.eventStartTime = disturbanceSpec.eventStartTime;
metrics.eventEndTime = disturbanceSpec.eventEndTime;
metrics.peakPositionErrorAfterOnset = NaN;
metrics.peakAttitudeErrorAfterOnset = NaN;
metrics.preEventPositionRmse = NaN;
metrics.recoveryThreshold = NaN;
metrics.recoveryTime = NaN;

if isfinite(disturbanceSpec.eventStartTime)
    preEvent = time < disturbanceSpec.eventStartTime;
    afterOnset = time >= disturbanceSpec.eventStartTime;
    if any(preEvent)
        metrics.preEventPositionRmse = sqrt(mean(positionNorm(preEvent).^2));
    else
        metrics.preEventPositionRmse = positionNorm(1);
    end
    metrics.peakPositionErrorAfterOnset = max(positionNorm(afterOnset));
    metrics.peakAttitudeErrorAfterOnset = max(attitudeNorm(afterOnset));

    episodeEndTime = time(end);
    if isfinite(disturbanceSpec.eventEndTime) && ...
            disturbanceSpec.eventEndTime < episodeEndTime - episode.sampleTime / 2
        threshold = max(recoveryCfg.positionTolerance, ...
                        recoveryCfg.relativeToPreEvent * metrics.preEventPositionRmse);
        metrics.recoveryThreshold = threshold;
        metrics.recoveryTime = find_recovery_time(time, positionNorm, ...
            disturbanceSpec.eventEndTime, threshold, ...
            recoveryCfg.dwellTime, episode.sampleTime);
    end
end
end

function recoveryTime = find_recovery_time(time, errorNorm, eventEndTime, threshold, dwellTime, sampleTime)
dwellSteps = max(1, ceil(dwellTime / sampleTime));
firstCandidate = find(time >= eventEndTime, 1, 'first');
recoveryTime = NaN;
if isempty(firstCandidate)
    return;
end

for index = firstCandidate:numel(time) - dwellSteps + 1
    if all(errorNorm(index:index + dwellSteps - 1) <= threshold)
        recoveryTime = max(0.0, time(index) - eventEndTime);
        return;
    end
end
end

function Xref = reference_for_episode(reference, stepCount, sampleTime)
if isa(reference, 'function_handle')
    Xref = reference((0:stepCount) * sampleTime);
elseif isempty(reference)
    Xref = zeros(12, stepCount + 1);
elseif isvector(reference)
    Xref = repmat(reference(:), 1, stepCount + 1);
else
    if size(reference, 1) ~= 12
        error('summarize_disturbance_episode:BadReference', ...
              'reference must have 12 rows.');
    end
    Xref = reference(:, 1:min(size(reference, 2), stepCount + 1));
    if size(Xref, 2) < stepCount + 1
        Xref = [Xref, repmat(Xref(:, end), 1, stepCount + 1 - size(Xref, 2))];
    end
end
end
