function analyze_targeted_lqr_predictive_thresholds()
%ANALYZE_TARGETED_LQR_PREDICTIVE_THRESHOLDS Audit H=20 LQR warning labels.

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot, 'configs'));
addpath(genpath(fullfile(projectRoot, 'src')));
cfg = targeted_lqr_weak_config();
divergenceCfg = targeted_lqr_divergence_config();
nmpcCfg = step2_nmpc_config();
assert(cfg.predictiveAnalysis.finalThresholdsLocked == ...
    divergenceCfg.finalThresholdsLocked, ...
    'Predictive-analysis and divergence lock states differ.');
if divergenceCfg.finalThresholdsLocked
    assert(strcmp(cfg.predictiveAnalysis.lockedCandidateId, 'C012') && ...
        strcmp(divergenceCfg.lockedCandidateId, 'C012'), ...
        'A locked sensitivity reproduction requires approved C012.');
    completionStatus = 'completed_locked_c012_sensitivity_reproduction';
else
    completionStatus = 'completed_diagnostic_thresholds_not_locked';
end
assert(cfg.predictiveAnalysis.horizonSteps == ...
    divergenceCfg.horizonSteps, 'The diagnostic and label horizons differ.');

[bankRoot, screenRoot, outputRoot] = resolve_roots(projectRoot, cfg);
bankPath = fullfile(bankRoot, 'validation_bank.mat');
screenPath = fullfile(screenRoot, 'lqr_validation_screen.mat');
assert(isfile(bankPath) && isfile(screenPath), ...
    'The frozen validation bank and LQR screen are required.');
assert(~isfolder(outputRoot), 'Refusing to overwrite %s.', outputRoot);
temporaryRoot = [char(outputRoot) '.tmp'];
if isfolder(temporaryRoot)
    rmdir(temporaryRoot, 's');
end
mkdir(temporaryRoot);
cleanup = onCleanup(@() cleanup_temporary(temporaryRoot));

clock = tic;
loadedBank = load(bankPath, 'bank');
loadedScreen = load(screenPath, 'traces', 'episodes', 'lqr');
bank = loadedBank.bank;
traces = loadedScreen.traces;
episodes = loadedScreen.episodes;
validate_sources(bank, traces, episodes, cfg);

metric = extract_step_metrics(bank, traces, nmpcCfg, cfg);
startSteps = cfg.predictiveAnalysis.evaluationStartStep: ...
    cfg.stepCount - divergenceCfg.horizonSteps + 1;
windows = future_windows(metric, startSteps, divergenceCfg.horizonSteps);
[cellGroup, familyGroup] = metadata_groups(bank);
[candidates, candidateState] = evaluate_grid(metric, windows, ...
    startSteps, divergenceCfg, cellGroup, familyGroup);
shortlist = representative_shortlist(candidates, ...
    cfg.predictiveAnalysis.representativePositiveFractions);
representative = select_five_percent_candidate(candidates, shortlist);
[~, positive, eligible] = evaluate_candidate(metric, windows, ...
    startSteps, representative, divergenceCfg, cellGroup, familyGroup);
representativeDistribution = distribution_summary(bank, positive, eligible);
futureQuantiles = future_metric_quantiles(windows);
hardEpisodes = episodes(episodes.SaturationFraction > 0 | ...
    episodes.ConstraintViolationCount > 0, :);

wallSeconds = toc(clock);
metadata = analysis_metadata(cfg, divergenceCfg, bankPath, screenPath, ...
    bank, startSteps, candidates, representative, wallSeconds);
writetable(candidates, fullfile(temporaryRoot, ...
    'predictive_threshold_candidate_grid.csv'));
writetable(shortlist, fullfile(temporaryRoot, ...
    'predictive_threshold_representative_shortlist.csv'));
writetable(representativeDistribution, fullfile(temporaryRoot, ...
    'diagnostic_5pct_candidate_distribution.csv'));
writetable(futureQuantiles, fullfile(temporaryRoot, ...
    'future_error_quantiles.csv'));
writetable(hardEpisodes, fullfile(temporaryRoot, ...
    'lqr_hard_episode_manifest.csv'));
save(fullfile(temporaryRoot, 'predictive_threshold_analysis.mat'), ...
    'candidates', 'shortlist', 'representativeDistribution', ...
    'futureQuantiles', 'hardEpisodes', 'metadata', 'candidateState', ...
    'cfg', 'divergenceCfg', '-v7.3');
write_report(temporaryRoot, metadata, candidates, shortlist, ...
    futureQuantiles, hardEpisodes, episodes);
write_text(fullfile(temporaryRoot, 'COMPLETED.txt'), sprintf( ...
    ['task=analyze_targeted_lqr_predictive_thresholds\n' ...
    'status=%s\n' ...
    'episodes=%d\ncandidate_steps=%d\nthreshold_candidates=%d\n' ...
    'hard_episodes=%d\nwall_seconds=%.6f\n'], ...
    completionStatus, numel(bank), metadata.candidateStepCount, ...
    height(candidates), ...
    height(hardEpisodes), wallSeconds));
movefile(temporaryRoot, outputRoot);
clear cleanup;
end

function [bankRoot, screenRoot, outputRoot] = resolve_roots(projectRoot, cfg)
resultRoot = fullfile(projectRoot, cfg.resultRoot);
bankRoot = env_or_default('LQR_BANK_ROOT', fullfile(resultRoot, ...
    cfg.predictiveAnalysis.sourceBankSubfolder));
screenRoot = env_or_default('LQR_SCREEN_ROOT', fullfile(resultRoot, ...
    cfg.predictiveAnalysis.sourceScreenSubfolder));
outputRoot = env_or_default('LQR_ANALYSIS_OUTPUT_ROOT', ...
    fullfile(resultRoot, cfg.predictiveAnalysis.outputSubfolder));
end

function value = env_or_default(name, defaultValue)
value = strtrim(getenv(name));
if isempty(value)
    value = defaultValue;
end
end

function validate_sources(bank, traces, episodes, cfg)
assert(numel(bank) == cfg.screenBenchmark.episodeCount && ...
    numel(traces) == numel(bank) && height(episodes) == numel(bank), ...
    'The bank, trace and episode counts are inconsistent.');
assert(all([bank.EpisodeIndex].' == episodes.EpisodeIndex), ...
    'Bank and screen episode ordering differs.');
assert(all(arrayfun(@(x) size(x.X, 2), traces) == cfg.stepCount + 1), ...
    'Every trace must contain the complete 200-step rollout.');
assert(all(arrayfun(@(x) all(isfinite(x.X), 'all'), traces)), ...
    'The predictive audit requires finite frozen-LQR traces.');
end

function metric = extract_step_metrics(bank, traces, nmpcCfg, cfg)
episodeCount = numel(bank);
stateCount = cfg.stepCount + 1;
metric.position = zeros(episodeCount, stateCount);
metric.attitudeDeg = zeros(episodeCount, stateCount);
metric.velocity = zeros(episodeCount, stateCount);
metric.bodyRate = zeros(episodeCount, stateCount);
metric.constraint = false(episodeCount, stateCount);
metric.finite = true(episodeCount, stateCount);
metric.saturated = false(episodeCount, cfg.stepCount);
for index = 1:episodeCount
    X = traces(index).X;
    error = nmpc_state_error(X, bank(index).Reference);
    metric.position(index, :) = vecnorm(error(1:3, :), 2, 1);
    metric.attitudeDeg(index, :) = ...
        rad2deg(vecnorm(error(4:6, :), 2, 1));
    metric.velocity(index, :) = vecnorm(error(7:9, :), 2, 1);
    metric.bodyRate(index, :) = vecnorm(error(10:12, :), 2, 1);
    metric.finite(index, :) = all(isfinite(X), 1);
    metric.constraint(index, :) = state_crossing(X, nmpcCfg);
    metric.saturated(index, :) = traces(index).saturated;
end
end

function crossing = state_crossing(X, cfg)
lower = cfg.constraints.stateLower(:);
upper = cfg.constraints.stateUpper(:);
crossing = ~all(isfinite(X), 1) | any(X < lower | X > upper, 1);
if isfield(cfg.constraints, 'maxTilt') && ...
        isfinite(cfg.constraints.maxTilt)
    crossing = crossing | ...
        sqrt(X(4, :) .^ 2 + X(5, :) .^ 2) > cfg.constraints.maxTilt;
end
end

function windows = future_windows(metric, starts, horizon)
shape = [size(metric.position, 1), numel(starts)];
windows.position = -inf(shape);
windows.attitudeDeg = -inf(shape);
windows.velocity = -inf(shape);
windows.bodyRate = -inf(shape);
windows.constraintAny = false(shape);
windows.saturationAny = false(shape);
windows.nonfiniteAny = false(shape);
windows.safetyAny = false(shape);
for offset = 1:horizon
    stateIndex = starts + offset;
    inputIndex = starts + offset - 1;
    windows.position = max(windows.position, ...
        metric.position(:, stateIndex));
    windows.attitudeDeg = max(windows.attitudeDeg, ...
        metric.attitudeDeg(:, stateIndex));
    windows.velocity = max(windows.velocity, ...
        metric.velocity(:, stateIndex));
    windows.bodyRate = max(windows.bodyRate, ...
        metric.bodyRate(:, stateIndex));
    windows.constraintAny = windows.constraintAny | ...
        metric.constraint(:, stateIndex);
    windows.saturationAny = windows.saturationAny | ...
        metric.saturated(:, inputIndex);
    windows.nonfiniteAny = windows.nonfiniteAny | ...
        ~metric.finite(:, stateIndex);
    windows.safetyAny = windows.constraintAny | ...
        windows.saturationAny | windows.nonfiniteAny;
end
end

function [cellGroup, familyGroup] = metadata_groups(bank)
[~, ~, cellGroup] = unique(string({bank.CellId}).', 'stable');
[~, ~, familyGroup] = unique(string({bank.Family}).', 'stable');
end

function [result, state] = evaluate_grid(metric, windows, starts, cfg, ...
        cellGroup, familyGroup)
grid = cfg.candidateGrid;
rowCount = numel(grid.positionM) * numel(grid.attitudeDeg) * ...
    numel(grid.velocityMps) * numel(grid.bodyRateRadps) * ...
    numel(grid.growthFactor) * ...
    numel(grid.minimumConsecutiveSteps);
rows = repmat(empty_candidate_row(), rowCount, 1);
cursor = 0;
for position = grid.positionM
    for attitude = grid.attitudeDeg
        for velocity = grid.velocityMps
            for bodyRate = grid.bodyRateRadps
                threshold = struct('positionM', position, ...
                    'attitudeDeg', attitude, 'velocityMps', velocity, ...
                    'bodyRateRadps', bodyRate);
                base = candidate_base(metric, windows, starts, ...
                    threshold, cfg);
                for growthFactor = grid.growthFactor
                    for consecutive = grid.minimumConsecutiveSteps
                        cursor = cursor + 1;
                        rows(cursor) = summarize_candidate(base, ...
                            growthFactor, consecutive, cellGroup, ...
                            familyGroup, cursor, threshold);
                    end
                end
            end
        end
    end
end
result = struct2table(rows);
state = struct('candidateCount', rowCount, ...
    'candidateStepCount', numel(metric.position(:, starts)), ...
    'horizonSteps', cfg.horizonSteps);
end

function base = candidate_base(metric, windows, starts, threshold, cfg)
currentPosition = metric.position(:, starts);
currentAttitude = metric.attitudeDeg(:, starts);
currentVelocity = metric.velocity(:, starts);
currentBodyRate = metric.bodyRate(:, starts);
base.eligible = metric.finite(:, starts) & ...
    currentPosition < threshold.positionM & ...
    currentAttitude < threshold.attitudeDeg & ...
    currentVelocity < threshold.velocityMps & ...
    currentBodyRate < threshold.bodyRateRadps;
base.positionAny = windows.position >= threshold.positionM;
base.attitudeAny = windows.attitudeDeg >= threshold.attitudeDeg;
base.velocityAny = windows.velocity >= threshold.velocityMps;
base.bodyRateAny = windows.bodyRate >= threshold.bodyRateRadps;
base.absoluteAny = base.positionAny | base.attitudeAny | ...
    base.velocityAny | base.bodyRateAny;
base.constraintAny = windows.constraintAny;
base.saturationAny = windows.saturationAny;
base.nonfiniteAny = windows.nonfiniteAny;
base.safetyAny = windows.safetyAny;
base.horizonSteps = cfg.horizonSteps;

energy = (metric.position ./ threshold.positionM) .^ 2 + ...
    (metric.attitudeDeg ./ threshold.attitudeDeg) .^ 2 + ...
    (metric.velocity ./ threshold.velocityMps) .^ 2 + ...
    (metric.bodyRate ./ threshold.bodyRateRadps) .^ 2;
energyStart = energy(:, starts);
energyEnd = energy(:, starts + cfg.horizonSteps);
base.energyGrowth = energyEnd ./ max(energyStart, cfg.growth.energyFloor);
base.warning = max([windows.position(:) ./ threshold.positionM, ...
    windows.attitudeDeg(:) ./ threshold.attitudeDeg, ...
    windows.velocity(:) ./ threshold.velocityMps, ...
    windows.bodyRate(:) ./ threshold.bodyRateRadps], [], 2);
base.warning = reshape(base.warning, size(base.eligible)) >= ...
    cfg.threshold.warningFraction;

currentRun = zeros(size(base.eligible));
base.maximumGrowthRun = zeros(size(base.eligible));
base.firstEvent = zeros(size(base.eligible), 'uint8');
base.firstPosition = false(size(base.eligible));
base.firstAttitude = false(size(base.eligible));
base.firstVelocity = false(size(base.eligible));
base.firstBodyRate = false(size(base.eligible));
base.firstConstraint = false(size(base.eligible));
base.firstSaturation = false(size(base.eligible));
base.firstNonfinite = false(size(base.eligible));
for offset = 1:cfg.horizonSteps
    growthFlag = energy(:, starts + offset) >= ...
        (1 + cfg.growth.minimumRelativeStep) .* ...
        energy(:, starts + offset - 1);
    currentRun = (currentRun + 1) .* growthFlag;
    base.maximumGrowthRun = max(base.maximumGrowthRun, currentRun);
    stateIndex = starts + offset;
    inputIndex = starts + offset - 1;
    positionNow = metric.position(:, stateIndex) >= threshold.positionM;
    attitudeNow = metric.attitudeDeg(:, stateIndex) >= ...
        threshold.attitudeDeg;
    velocityNow = metric.velocity(:, stateIndex) >= threshold.velocityMps;
    bodyRateNow = metric.bodyRate(:, stateIndex) >= ...
        threshold.bodyRateRadps;
    constraintNow = metric.constraint(:, stateIndex);
    saturationNow = metric.saturated(:, inputIndex);
    nonfiniteNow = ~metric.finite(:, stateIndex);
    eventNow = positionNow | attitudeNow | velocityNow | bodyRateNow | ...
        constraintNow | saturationNow | nonfiniteNow;
    first = base.firstEvent == 0 & eventNow;
    base.firstEvent(first) = uint8(offset);
    base.firstPosition(first) = positionNow(first);
    base.firstAttitude(first) = attitudeNow(first);
    base.firstVelocity(first) = velocityNow(first);
    base.firstBodyRate(first) = bodyRateNow(first);
    base.firstConstraint(first) = constraintNow(first);
    base.firstSaturation(first) = saturationNow(first);
    base.firstNonfinite(first) = nonfiniteNow(first);
end
end

function row = summarize_candidate(base, growthFactor, consecutive, ...
        cellGroup, familyGroup, candidateIndex, threshold)
sustained = base.energyGrowth >= growthFactor & ...
    base.maximumGrowthRun >= consecutive & base.warning;
hardEvent = base.eligible & (base.absoluteAny | base.safetyAny);
growthOnly = base.eligible & sustained & ...
    ~base.absoluteAny & ~base.safetyAny;
positive = hardEvent | growthOnly;
lead = double(base.firstEvent);
lead(base.firstEvent == 0 & sustained) = base.horizonSteps;
positiveLead = lead(positive);
episodePositive = any(positive, 2);
hardEpisodePositive = any(hardEvent, 2);
growthOnlyEpisodePositive = any(growthOnly, 2);
onsets = positive & ~[false(size(positive, 1), 1), ...
    positive(:, 1:end - 1)];
hardOnsets = hardEvent & ~[false(size(hardEvent, 1), 1), ...
    hardEvent(:, 1:end - 1)];
row = empty_candidate_row();
row.CandidateId = "C" + compose('%03d', candidateIndex);
row.PositionThresholdM = threshold.positionM;
row.AttitudeThresholdDeg = threshold.attitudeDeg;
row.VelocityThresholdMps = threshold.velocityMps;
row.BodyRateThresholdRadps = threshold.bodyRateRadps;
row.GrowthFactor = growthFactor;
row.MinimumConsecutiveGrowthSteps = consecutive;
row.CandidateStepCount = numel(positive);
row.EligibleStepCount = nnz(base.eligible);
row.EligibleStepFraction = mean(base.eligible, 'all');
row.PositiveStepCount = nnz(positive);
row.PositiveStepFractionAll = mean(positive, 'all');
row.PositiveEligibleFraction = safe_fraction(nnz(positive), ...
    nnz(base.eligible));
row.PositiveOnsetCount = nnz(onsets);
row.PositiveEpisodeCount = nnz(episodePositive);
row.PositiveEpisodeFraction = mean(episodePositive);
row.PositiveCellCount = grouped_positive_count(cellGroup, episodePositive);
row.PositiveFamilyCount = grouped_positive_count(familyGroup, episodePositive);
row.HardEventStepCount = nnz(hardEvent);
row.HardEventEligibleFraction = safe_fraction(nnz(hardEvent), ...
    nnz(base.eligible));
row.HardEventOnsetCount = nnz(hardOnsets);
row.HardEventEpisodeCount = nnz(hardEpisodePositive);
row.HardEventEpisodeFraction = mean(hardEpisodePositive);
row.HardEventCellCount = grouped_positive_count(cellGroup, ...
    hardEpisodePositive);
row.HardEventFamilyCount = grouped_positive_count(familyGroup, ...
    hardEpisodePositive);
row.GrowthOnlyEpisodeCount = nnz(growthOnlyEpisodePositive);
row.GrowthOnlyCellCount = grouped_positive_count(cellGroup, ...
    growthOnlyEpisodePositive);
row.AbsoluteForecastPositiveCount = nnz(positive & base.absoluteAny);
row.SafetyForecastPositiveCount = nnz(positive & base.safetyAny);
row.PositionForecastPositiveCount = nnz(positive & base.positionAny);
row.AttitudeForecastPositiveCount = nnz(positive & base.attitudeAny);
row.VelocityForecastPositiveCount = nnz(positive & base.velocityAny);
row.BodyRateForecastPositiveCount = nnz(positive & base.bodyRateAny);
row.ConstraintForecastPositiveCount = nnz(positive & base.constraintAny);
row.SaturationForecastPositiveCount = nnz(positive & base.saturationAny);
row.NonfiniteForecastPositiveCount = nnz(positive & base.nonfiniteAny);
row.FirstPositionEventCount = nnz(positive & base.firstPosition);
row.FirstAttitudeEventCount = nnz(positive & base.firstAttitude);
row.FirstVelocityEventCount = nnz(positive & base.firstVelocity);
row.FirstBodyRateEventCount = nnz(positive & base.firstBodyRate);
row.FirstConstraintEventCount = nnz(positive & base.firstConstraint);
row.FirstSaturationEventCount = nnz(positive & base.firstSaturation);
row.FirstNonfiniteEventCount = nnz(positive & base.firstNonfinite);
row.GrowthOnlyPositiveCount = nnz(growthOnly);
row.LeadMeanSteps = finite_mean(positiveLead);
row.LeadP50Steps = percentile_linear(positiveLead, 50);
row.LeadP95Steps = percentile_linear(positiveLead, 95);
end

function [row, positive, eligible] = evaluate_candidate(metric, windows, ...
        starts, candidate, cfg, cellGroup, familyGroup)
threshold = struct('positionM', candidate.PositionThresholdM, ...
    'attitudeDeg', candidate.AttitudeThresholdDeg, ...
    'velocityMps', candidate.VelocityThresholdMps, ...
    'bodyRateRadps', candidate.BodyRateThresholdRadps);
base = candidate_base(metric, windows, starts, threshold, cfg);
sustained = base.energyGrowth >= candidate.GrowthFactor & ...
    base.maximumGrowthRun >= ...
    candidate.MinimumConsecutiveGrowthSteps & base.warning;
positive = base.eligible & ...
    (base.absoluteAny | base.safetyAny | sustained);
eligible = base.eligible;
row = summarize_candidate(base, candidate.GrowthFactor, ...
    candidate.MinimumConsecutiveGrowthSteps, cellGroup, familyGroup, ...
    sscanf(char(extractAfter(candidate.CandidateId, 1)), '%d'), threshold);
end

function count = grouped_positive_count(group, episodePositive)
groupCount = max(group);
found = false(groupCount, 1);
for index = 1:groupCount
    found(index) = any(episodePositive(group == index));
end
count = nnz(found);
end

function shortlist = representative_shortlist(candidates, targets)
shortlist = table();
for target = targets
    distance = abs(candidates.PositiveEligibleFraction - target);
    ranking = table((1:height(candidates)).', distance, ...
        candidates.PositiveCellCount, candidates.PositiveEpisodeCount, ...
        'VariableNames', {'Index', 'Distance', 'Cells', 'Episodes'});
    ranking = sortrows(ranking, {'Distance', 'Cells', 'Episodes'}, ...
        {'ascend', 'descend', 'descend'});
    row = candidates(ranking.Index(1), :);
    row.TargetPositiveFraction = target;
    row.AbsoluteFractionDistance = ranking.Distance(1);
    row = movevars(row, {'TargetPositiveFraction', ...
        'AbsoluteFractionDistance'}, 'Before', 'CandidateId');
    shortlist = [shortlist; row]; %#ok<AGROW>
end
end

function candidate = select_five_percent_candidate(candidates, shortlist)
[~, targetIndex] = min(abs(shortlist.TargetPositiveFraction - 0.05));
candidateId = shortlist.CandidateId(targetIndex);
candidate = candidates(candidates.CandidateId == candidateId, :);
assert(height(candidate) == 1, 'The diagnostic representative is ambiguous.');
end

function result = distribution_summary(bank, positive, eligible)
family = string({bank.Family}).';
speed = [bank.TargetReferenceSpeed].';
loadIndex = [bank.LoadIndex].';
[group, familyValue, speedValue, loadValue] = ...
    findgroups(family, speed, loadIndex);
rows = repmat(empty_distribution_row(), max(group), 1);
for index = 1:max(group)
    mask = group == index;
    subsetPositive = positive(mask, :);
    subsetEligible = eligible(mask, :);
    onsets = subsetPositive & ...
        ~[false(nnz(mask), 1), subsetPositive(:, 1:end - 1)];
    rows(index).Family = familyValue(index);
    rows(index).TargetReferenceSpeed = speedValue(index);
    rows(index).LoadIndex = loadValue(index);
    rows(index).EpisodeCount = nnz(mask);
    rows(index).PositiveEpisodeCount = nnz(any(subsetPositive, 2));
    rows(index).PositiveEpisodeFraction = mean(any(subsetPositive, 2));
    rows(index).EligibleStepCount = nnz(subsetEligible);
    rows(index).PositiveStepCount = nnz(subsetPositive);
    rows(index).PositiveEligibleFraction = safe_fraction( ...
        nnz(subsetPositive), nnz(subsetEligible));
    rows(index).PositiveOnsetCount = nnz(onsets);
end
result = struct2table(rows);
end

function result = future_metric_quantiles(windows)
names = ["position_m"; "attitude_deg"; "velocity_mps"; ...
    "body_rate_radps"];
data = {windows.position, windows.attitudeDeg, windows.velocity, ...
    windows.bodyRate};
rows = repmat(empty_quantile_row(), numel(names), 1);
for index = 1:numel(names)
    values = data{index}(:);
    rows(index).Metric = names(index);
    rows(index).Q50 = percentile_linear(values, 50);
    rows(index).Q75 = percentile_linear(values, 75);
    rows(index).Q90 = percentile_linear(values, 90);
    rows(index).Q95 = percentile_linear(values, 95);
    rows(index).Q99 = percentile_linear(values, 99);
    rows(index).Q995 = percentile_linear(values, 99.5);
    rows(index).Q999 = percentile_linear(values, 99.9);
    rows(index).Maximum = max(values);
end
result = struct2table(rows);
end

function metadata = analysis_metadata(cfg, divergenceCfg, bankPath, ...
        screenPath, bank, starts, candidates, representative, wallSeconds)
metadata = struct();
metadata.status = cfg.predictiveAnalysis.status;
metadata.finalThresholdsLocked = false;
metadata.bankPath = bankPath;
metadata.screenPath = screenPath;
metadata.episodeCount = numel(bank);
metadata.factorialCellCount = numel(unique(string({bank.CellId})));
metadata.startStepRange = [starts(1), starts(end)];
metadata.horizonSteps = divergenceCfg.horizonSteps;
metadata.candidateStepCount = numel(bank) * numel(starts);
metadata.thresholdCandidateCount = height(candidates);
metadata.diagnosticRepresentativeId = char(representative.CandidateId);
metadata.wallSeconds = wallSeconds;
metadata.completedAt = char(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd HH:mm:ssXXX'));
end

function write_report(root, metadata, candidates, shortlist, quantiles, ...
        hardEpisodes, episodes)
path = fullfile(root, 'predictive_threshold_analysis_report.md');
fileId = fopen(path, 'w');
assert(fileId >= 0, 'Cannot open %s.', path);
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, '# H=20 predictive LQR threshold analysis\n\n');
fprintf(fileId, '- Status: `%s`.\n', metadata.status);
fprintf(fileId, '- Final thresholds locked: `false`.\n');
fprintf(fileId, '- Episodes/cells: %d/%d.\n', metadata.episodeCount, ...
    metadata.factorialCellCount);
fprintf(fileId, '- Candidate start steps: %d to %d.\n', ...
    metadata.startStepRange(1), metadata.startStepRange(2));
fprintf(fileId, '- Candidate step contexts: %d.\n', ...
    metadata.candidateStepCount);
fprintf(fileId, '- Threshold combinations: %d.\n', height(candidates));
fprintf(fileId, '- Analysis wall time: %.3f s.\n', metadata.wallSeconds);
fprintf(fileId, '- Saturation episodes: %d.\n', ...
    nnz(episodes.SaturationFraction > 0));
fprintf(fileId, '- Constraint-violation episodes: %d.\n', ...
    nnz(episodes.ConstraintViolationCount > 0));
fprintf(fileId, '- Union of hard episodes: %d.\n\n', height(hardEpisodes));
fprintf(fileId, '## Future-error distribution\n\n');
for index = 1:height(quantiles)
    fprintf(fileId, '- %s: q95 %.6g, q99 %.6g, q99.9 %.6g, max %.6g.\n', ...
        char(quantiles.Metric(index)), quantiles.Q95(index), ...
        quantiles.Q99(index), quantiles.Q999(index), ...
        quantiles.Maximum(index));
end
fprintf(fileId, '\n## Class-balance representatives\n\n');
for index = 1:height(shortlist)
    row = shortlist(index, :);
    fprintf(fileId, ['- Target %.1f%% -> %s: p/a/v/r %.3g m / %.3g deg / ' ...
        '%.3g m/s / %.3g rad/s, growth %.3g for %d steps; observed ' ...
        '%.3f%% eligible steps, %d episodes, %d cells.\n'], ...
        100 * row.TargetPositiveFraction, char(row.CandidateId), ...
        row.PositionThresholdM, row.AttitudeThresholdDeg, ...
        row.VelocityThresholdMps, row.BodyRateThresholdRadps, ...
        row.GrowthFactor, row.MinimumConsecutiveGrowthSteps, ...
        100 * row.PositiveEligibleFraction, row.PositiveEpisodeCount, ...
        row.PositiveCellCount);
end
fprintf(fileId, ['\nThe shortlist is diagnostic only. It selects the grid row ' ...
    'nearest each requested class-balance anchor; it does not approve an ' ...
    'LQR-weak threshold. The 5%% row is used only to export a coverage table. ' ...
    'First-event cause counts are nonexclusive when multiple limits cross ' ...
    'at the same first event step. Growth-only labels remain separate. No ' ...
    'NMPC, SAC, surrogate, test or OOD result entered this analysis.\n']);
clear cleanup;
end

function row = empty_candidate_row()
row = struct('CandidateId', "", 'PositionThresholdM', NaN, ...
    'AttitudeThresholdDeg', NaN, 'VelocityThresholdMps', NaN, ...
    'BodyRateThresholdRadps', NaN, 'GrowthFactor', NaN, ...
    'MinimumConsecutiveGrowthSteps', 0, 'CandidateStepCount', 0, ...
    'EligibleStepCount', 0, 'EligibleStepFraction', NaN, ...
    'PositiveStepCount', 0, 'PositiveStepFractionAll', NaN, ...
    'PositiveEligibleFraction', NaN, 'PositiveOnsetCount', 0, ...
    'PositiveEpisodeCount', 0, 'PositiveEpisodeFraction', NaN, ...
    'PositiveCellCount', 0, 'PositiveFamilyCount', 0, ...
    'HardEventStepCount', 0, 'HardEventEligibleFraction', NaN, ...
    'HardEventOnsetCount', 0, 'HardEventEpisodeCount', 0, ...
    'HardEventEpisodeFraction', NaN, 'HardEventCellCount', 0, ...
    'HardEventFamilyCount', 0, 'GrowthOnlyEpisodeCount', 0, ...
    'GrowthOnlyCellCount', 0, ...
    'AbsoluteForecastPositiveCount', 0, ...
    'SafetyForecastPositiveCount', 0, ...
    'PositionForecastPositiveCount', 0, ...
    'AttitudeForecastPositiveCount', 0, ...
    'VelocityForecastPositiveCount', 0, ...
    'BodyRateForecastPositiveCount', 0, ...
    'ConstraintForecastPositiveCount', 0, ...
    'SaturationForecastPositiveCount', 0, ...
    'NonfiniteForecastPositiveCount', 0, ...
    'FirstPositionEventCount', 0, 'FirstAttitudeEventCount', 0, ...
    'FirstVelocityEventCount', 0, 'FirstBodyRateEventCount', 0, ...
    'FirstConstraintEventCount', 0, 'FirstSaturationEventCount', 0, ...
    'FirstNonfiniteEventCount', 0, 'GrowthOnlyPositiveCount', 0, ...
    'LeadMeanSteps', NaN, 'LeadP50Steps', NaN, 'LeadP95Steps', NaN);
end

function row = empty_distribution_row()
row = struct('Family', "", 'TargetReferenceSpeed', NaN, ...
    'LoadIndex', 0, 'EpisodeCount', 0, 'PositiveEpisodeCount', 0, ...
    'PositiveEpisodeFraction', NaN, 'EligibleStepCount', 0, ...
    'PositiveStepCount', 0, 'PositiveEligibleFraction', NaN, ...
    'PositiveOnsetCount', 0);
end

function row = empty_quantile_row()
row = struct('Metric', "", 'Q50', NaN, 'Q75', NaN, 'Q90', NaN, ...
    'Q95', NaN, 'Q99', NaN, 'Q995', NaN, 'Q999', NaN, 'Maximum', NaN);
end

function value = safe_fraction(numerator, denominator)
if denominator == 0
    value = NaN;
else
    value = numerator / denominator;
end
end

function value = finite_mean(values)
values = values(isfinite(values));
if isempty(values)
    value = NaN;
else
    value = mean(values);
end
end

function value = percentile_linear(values, percentage)
values = sort(values(isfinite(values)));
if isempty(values)
    value = NaN;
elseif isscalar(values)
    value = values(1);
else
    rankValue = 1 + (numel(values) - 1) * percentage / 100;
    lowerIndex = floor(rankValue);
    upperIndex = ceil(rankValue);
    weight = rankValue - lowerIndex;
    value = (1 - weight) * values(lowerIndex) + ...
        weight * values(upperIndex);
end
end

function write_text(path, content)
fileId = fopen(path, 'w');
assert(fileId >= 0, 'Cannot open %s.', path);
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, '%s', content);
clear cleanup;
end

function cleanup_temporary(path)
if isfolder(path)
    rmdir(path, 's');
end
end
