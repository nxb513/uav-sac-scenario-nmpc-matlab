function estimate = estimate_pipeline_workload(entries, cfg)
%ESTIMATE_PIPELINE_WORKLOAD Extrapolate RL and teacher-dataset workloads.

scenarioMask = strcmp({entries.controller}, 'scenario_nmpc');
scenarioEntries = entries(scenarioMask);
if isempty(scenarioEntries)
    error('estimate_pipeline_workload:NoScenarioData', ...
          'At least one scenario_nmpc benchmark entry is required.');
end

horizons = unique([scenarioEntries.horizon]);
medianSeconds = zeros(size(horizons));
p95Seconds = zeros(size(horizons));
sampleCounts = zeros(size(horizons));

for i = 1:numel(horizons)
    horizonMask = [scenarioEntries.horizon] == horizons(i);
    horizonEntries = scenarioEntries(horizonMask);
    times = [horizonEntries.solveTime];
    medianSeconds(i) = percentile_linear(times, 50.0);
    p95Seconds(i) = percentile_linear(times, 95.0);
    sampleCounts(i) = numel(times);
end

estimate.runtimeByHorizon = table(horizons(:), sampleCounts(:), medianSeconds(:), p95Seconds(:), ...
    'VariableNames', {'Horizon', 'MeasuredSolves', 'MedianSeconds', 'P95Seconds'});
estimate.lowSecondsPerStep = min(medianSeconds);
[~, baseIndex] = min(abs(horizons - cfg.workload.baseHorizon));
estimate.baseSecondsPerStep = medianSeconds(baseIndex);
estimate.baseHorizon = horizons(baseIndex);
estimate.highSecondsPerStep = max(p95Seconds);

rlEpisodes = cfg.workload.rl.episodeCounts(:);
rlStepsPerEpisode = cfg.workload.rl.stepsPerEpisode(:);
rlTransitions = rlEpisodes .* rlStepsPerEpisode;
rlLowHours = rlTransitions * estimate.lowSecondsPerStep / 3600.0;
rlBaseHours = rlTransitions * estimate.baseSecondsPerStep / 3600.0;
rlHighHours = rlTransitions * estimate.highSecondsPerStep / 3600.0;
estimate.rl = table(cfg.workload.rl.stageNames(:), rlEpisodes, rlStepsPerEpisode, ...
                    rlTransitions, rlLowHours, rlBaseHours, rlHighHours, ...
    'VariableNames', {'Stage', 'Episodes', 'StepsPerEpisode', 'Transitions', ...
                      'LowHours', 'BaseHours', 'HighHours'});

datasetCfg = cfg.workload.dataset;
validSamplesPerEpisode = datasetCfg.stepsPerEpisode - datasetCfg.historyLength + 1;
if validSamplesPerEpisode <= 0
    error('estimate_pipeline_workload:BadDatasetWindow', ...
          'historyLength must not exceed stepsPerEpisode.');
end

datasetEpisodes = datasetCfg.episodeCounts(:);
datasetSamples = datasetEpisodes * validSamplesPerEpisode;
datasetGeneratedSteps = datasetEpisodes * datasetCfg.stepsPerEpisode;
datasetLowHours = datasetGeneratedSteps * estimate.lowSecondsPerStep / 3600.0;
datasetBaseHours = datasetGeneratedSteps * estimate.baseSecondsPerStep / 3600.0;
datasetHighHours = datasetGeneratedSteps * estimate.highSecondsPerStep / 3600.0;
estimate.dataset = table(datasetCfg.splitNames(:), datasetEpisodes, ...
                         repmat(datasetCfg.stepsPerEpisode, numel(datasetEpisodes), 1), ...
                         datasetGeneratedSteps, datasetSamples, ...
                         datasetLowHours, datasetBaseHours, datasetHighHours, ...
    'VariableNames', {'Split', 'Episodes', 'StepsPerEpisode', 'GeneratedSteps', 'Samples', ...
                      'LowHours', 'BaseHours', 'HighHours'});

if datasetCfg.useFullStateReference
    referenceDimension = datasetCfg.stateDimension * (datasetCfg.referenceLookahead + 1);
else
    referenceDimension = 6 * (datasetCfg.referenceLookahead + 1);
end
estimate.datasetFeatureDimension = ...
    datasetCfg.stateDimension * datasetCfg.historyLength + ...
    datasetCfg.inputDimension * datasetCfg.historyLength + ...
    referenceDimension + datasetCfg.extraFeatureDimension;
estimate.datasetTargetDimension = datasetCfg.inputDimension;
estimate.datasetTotalGeneratedSteps = sum(datasetGeneratedSteps);
estimate.datasetTotalSamples = sum(datasetSamples);

rawValues = estimate.datasetTotalSamples * ...
            (estimate.datasetFeatureDimension + estimate.datasetTargetDimension);
estimate.datasetSingleMegabytes = rawValues * 4.0 * datasetCfg.storageOverheadFactor / 1024.0^2;
estimate.datasetDoubleMegabytes = rawValues * 8.0 * datasetCfg.storageOverheadFactor / 1024.0^2;

replayValuesPerTransition = 2 * 30 + 7 + 2;
fullTransitions = rlTransitions(end);
estimate.rlReplaySingleMegabytes = fullTransitions * replayValuesPerTransition * 4.0 / 1024.0^2;
estimate.assumption = ['Timing excludes neural-network update, disk I/O and parallel overhead. ' ...
                       'Dataset counts are provisional until step-4 papers are read.'];
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
