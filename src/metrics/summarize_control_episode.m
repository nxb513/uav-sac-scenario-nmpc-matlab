function metrics = summarize_control_episode(episode, reference, cfg)
%SUMMARIZE_CONTROL_EPISODE Compute scalar closed-loop control metrics.

requiredFields = {'X', 'U', 'solveTime', 'exitflag'};
for i = 1:numel(requiredFields)
    if ~isfield(episode, requiredFields{i})
        error('summarize_control_episode:MissingField', ...
              'episode.%s is required.', requiredFields{i});
    end
end

stepCount = size(episode.U, 2);
if ~isequal(size(episode.X), [12, stepCount + 1])
    error('summarize_control_episode:BadStateSize', ...
          'episode.X must be 12-by-(stepCount+1).');
end
if size(episode.U, 1) ~= 4
    error('summarize_control_episode:BadInputSize', ...
          'episode.U must have four rows.');
end

Xref = reference_for_episode(reference, stepCount, episode.sampleTime);
E = nmpc_state_error(episode.X, Xref);
evaluationError = E(:, 2:end);
positionNorm = sqrt(sum(evaluationError(1:3, :).^2, 1));
attitudeNorm = sqrt(sum(evaluationError(4:6, :).^2, 1));

uHover = quad_hover_input(cfg.plant.nominal);
inputDeviation = episode.U - repmat(uHover, 1, stepCount);
inputIncrement = diff([uHover, episode.U], 1, 2);

constraintValues = nmpc_state_bound_violations(episode.X, cfg);
positiveViolation = max(constraintValues, 0.0);
violationTolerance = 1e-8;

solveTime = episode.solveTime(:);
exitflag = episode.exitflag(:);

metrics.stepCount = stepCount;
metrics.positionRmse = sqrt(mean(positionNorm.^2));
metrics.positionP95 = percentile_linear(positionNorm, 95.0);
metrics.positionMax = max(positionNorm);
metrics.attitudeRmse = sqrt(mean(attitudeNorm.^2));
metrics.attitudeP95 = percentile_linear(attitudeNorm, 95.0);
metrics.attitudeMax = max(attitudeNorm);
metrics.controlDeviationRms = sqrt(mean(sum(inputDeviation.^2, 1)));
metrics.controlIncrementRms = sqrt(mean(sum(inputIncrement.^2, 1)));
metrics.constraintViolationCount = nnz(constraintValues > violationTolerance);
metrics.constraintViolationRate = nnz(constraintValues > violationTolerance) / max(1, numel(constraintValues));
metrics.constraintViolationMax = max([0.0; positiveViolation(:)]);
metrics.solverSuccessRate = mean(exitflag > 0);
metrics.solveTimeMean = mean(solveTime);
metrics.solveTimeMedian = percentile_linear(solveTime, 50.0);
metrics.solveTimeP95 = percentile_linear(solveTime, 95.0);
metrics.solveTimeP99 = percentile_linear(solveTime, 99.0);
metrics.solveTimeMax = max(solveTime);
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
        error('summarize_control_episode:BadReference', ...
              'reference must have 12 rows.');
    end
    Xref = reference(:, 1:min(size(reference, 2), stepCount + 1));
    if size(Xref, 2) < stepCount + 1
        Xref = [Xref, repmat(Xref(:, end), 1, stepCount + 1 - size(Xref, 2))];
    end
end
end

function value = percentile_linear(values, percentage)
values = sort(values(isfinite(values)));
if isempty(values)
    value = NaN;
    return;
end
if numel(values) == 1
    value = values(1);
    return;
end

rank = 1.0 + (numel(values) - 1.0) * percentage / 100.0;
lowerIndex = floor(rank);
upperIndex = ceil(rank);
weight = rank - lowerIndex;
value = (1.0 - weight) * values(lowerIndex) + weight * values(upperIndex);
end
