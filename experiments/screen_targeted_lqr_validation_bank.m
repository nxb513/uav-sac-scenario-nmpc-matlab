function screen_targeted_lqr_validation_bank()
%SCREEN_TARGETED_LQR_VALIDATION_BANK Screen one frozen bank with pure LQR.

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot, 'configs'));
addpath(genpath(fullfile(projectRoot, 'src')));
cfg = targeted_lqr_weak_config();
nmpcCfg = step2_nmpc_config();
bankRoot = fullfile(projectRoot, cfg.resultRoot, ...
    cfg.screenBenchmark.bankOutputSubfolder);
loadedBank = load(fullfile(bankRoot, 'validation_bank.mat'), ...
    'bank', 'manifest');
lqrRoot = strtrim(getenv('LQR_FROZEN_ROOT'));
if isempty(lqrRoot)
    lqrPath = fullfile(projectRoot, ...
        cfg.screenBenchmark.provisionalLqrPath);
else
    lqrPath = fullfile(lqrRoot, 'selected_lqr.mat');
end
loadedLqr = load(lqrPath, 'selectedLqr');
outputRoot = fullfile(projectRoot, cfg.resultRoot, ...
    cfg.screenBenchmark.outputSubfolder);
if ~isfolder(outputRoot)
    mkdir(outputRoot);
end

bank = loadedBank.bank;
lqr = loadedLqr.selectedLqr;
rows = repmat(empty_episode_row(), numel(bank), 1);
traces = repmat(empty_trace(), numel(bank), 1);
clock = tic;
for episodeIndex = 1:numel(bank)
    rollout = run_lqr(bank(episodeIndex), lqr, cfg);
    traces(episodeIndex) = rollout;
    rows(episodeIndex) = metric_row(rollout, bank(episodeIndex), ...
        cfg, nmpcCfg);
end
wallSeconds = toc(clock);
episodes = struct2table(rows);
prevalence = build_prevalence(episodes, cfg);
byFamily = family_summary(episodes);
bySpeedLoad = speed_load_summary(episodes);
writetable(episodes, fullfile(outputRoot, ...
    'lqr_validation_episode_metrics.csv'));
writetable(byFamily, fullfile(outputRoot, ...
    'lqr_validation_summary_by_family.csv'));
writetable(bySpeedLoad, fullfile(outputRoot, ...
    'lqr_validation_summary_by_speed_load.csv'));
writetable(prevalence, fullfile(outputRoot, ...
    'lqr_weakness_candidate_prevalence.csv'));
save(fullfile(outputRoot, 'lqr_validation_screen.mat'), 'episodes', ...
    'prevalence', 'byFamily', 'bySpeedLoad', 'traces', 'cfg', 'lqr', ...
    'wallSeconds', ...
    '-v7.3');
write_report(outputRoot, cfg, episodes, prevalence, byFamily, ...
    lqr, wallSeconds, lqrPath);
write_text(fullfile(outputRoot, 'COMPLETED.txt'), sprintf( ...
    ['task=screen_targeted_lqr_validation_bank\ncompleted_at=%s\n' ...
    'episode_count=%d\nwall_seconds=%.6f\n' ...
    'lqr_artifact=%s\n'], ...
    char(datetime('now', 'TimeZone', 'local', 'Format', ...
    'yyyy-MM-dd HH:mm:ss.SSSXXX')), height(episodes), wallSeconds, lqrPath));
end

function rollout = run_lqr(scenario, design, cfg)
X = nan(12, cfg.stepCount + 1);
U = nan(4, cfg.stepCount);
controlTime = nan(1, cfg.stepCount);
saturated = false(1, cfg.stepCount);
X(:, 1) = scenario.X0;
for k = 1:cfg.stepCount
    clock = tic;
    error = nmpc_state_error(X(:, k), scenario.Reference(:, k));
    raw = scenario.Feedforward(:, k) - design.K * error;
    U(:, k) = quad_saturate_input(raw, cfg.plant.nominal);
    controlTime(k) = toc(clock);
    saturated(k) = any(abs(raw - U(:, k)) > 1e-10);
    time = (k - 1) * cfg.sampleTime;
    X(:, k + 1) = quad_step_rk4(time, X(:, k), U(:, k), ...
        cfg.sampleTime, scenario.ThetaPlant, scenario.Disturbance);
    if ~all(isfinite(X(:, k + 1)))
        break;
    end
end
rollout.X = X;
rollout.U = U;
rollout.controlTime = controlTime;
rollout.saturated = saturated;
end

function trace = empty_trace()
trace.X = [];
trace.U = [];
trace.controlTime = [];
trace.saturated = [];
end

function row = metric_row(rollout, scenario, cfg, nmpcCfg)
start = cfg.screenBenchmark.evaluationStartStep;
stateIndices = start + 1:cfg.stepCount + 1;
error = nmpc_state_error(rollout.X(:, stateIndices), ...
    scenario.Reference(:, stateIndices));
positionError = vecnorm(error(1:3, :), 2, 1);
attitudeErrorDeg = rad2deg(vecnorm(error(4:6, :), 2, 1));
violation = nmpc_state_bound_violations(rollout.X(:, stateIndices), nmpcCfg);
times = rollout.controlTime(start:end);
times = times(isfinite(times) & times > 0);
if isempty(times)
    meanTime = NaN;
    p95Time = NaN;
else
    meanTime = mean(times);
    p95Time = percentile_linear(times, 95);
end
row = empty_episode_row();
row.EpisodeIndex = scenario.EpisodeIndex;
row.CellId = string(scenario.CellId);
row.Family = string(scenario.Family);
row.DisturbanceType = string(scenario.Disturbance.type);
row.DisturbanceLevel = scenario.Disturbance.levelIndex;
row.TargetReferenceSpeed = scenario.TargetReferenceSpeed;
row.TargetReferenceAcceleration = scenario.TargetReferenceAcceleration;
row.LoadIndex = scenario.LoadIndex;
row.PeakReferenceSpeed = scenario.ReferenceMetric.peakSpeed;
row.PeakReferenceAcceleration = scenario.ReferenceMetric.peakAcceleration;
row.UncertaintyStratum = string(scenario.UncertaintyStratum);
row.MassScale = scenario.ThetaPlant.m / cfg.plant.nominal.m;
row.MinimumInertiaScale = min(diag(scenario.ThetaPlant.J) ./ ...
    diag(cfg.plant.nominal.J));
row.MinimumEffectiveness = min([scenario.ThetaPlant.alphaT; ...
    scenario.ThetaPlant.alphaTau]);
row.PositionRmse = sqrt(mean(positionError .^ 2));
row.AttitudeRmseDeg = sqrt(mean(attitudeErrorDeg .^ 2));
row.PositionMax = max(positionError);
row.AttitudeMaxDeg = max(attitudeErrorDeg);
row.SaturationFraction = mean(rollout.saturated(start:end));
row.ConstraintViolationCount = nnz(violation > 1e-8);
row.ControlTimeMean = meanTime;
row.ControlTimeP95 = p95Time;
row.Finite = all(isfinite(rollout.X), 'all') && ...
    all(isfinite(rollout.U), 'all');
for index = 1:numel(cfg.weakness.windowSteps)
    count = cfg.weakness.windowSteps(index);
    row.(sprintf('PositionWindow%dMax', count)) = ...
        max(rolling_rms(positionError, count));
    row.(sprintf('AttitudeWindow%dMaxDeg', count)) = ...
        max(rolling_rms(attitudeErrorDeg, count));
end
end

function result = build_prevalence(episodes, cfg)
rowCount = numel(cfg.weakness.windowSteps) * ...
    (numel(cfg.weakness.positionRmseM) + ...
    numel(cfg.weakness.attitudeRmseDeg));
rows = repmat(empty_prevalence_row(), rowCount, 1);
cursor = 0;
for windowIndex = 1:numel(cfg.weakness.windowSteps)
    count = cfg.weakness.windowSteps(windowIndex);
    position = episodes.(sprintf('PositionWindow%dMax', count));
    attitude = episodes.(sprintf('AttitudeWindow%dMaxDeg', count));
    for threshold = cfg.weakness.positionRmseM
        cursor = cursor + 1;
        row = empty_prevalence_row();
        row.Metric = "position_window_max_m";
        row.WindowSteps = count;
        row.Threshold = threshold;
        row.EpisodeCount = height(episodes);
        row.WeakEpisodeCount = nnz(position >= threshold);
        row.WeakEpisodeFraction = mean(position >= threshold);
        rows(cursor, 1) = row;
    end
    for threshold = cfg.weakness.attitudeRmseDeg
        cursor = cursor + 1;
        row = empty_prevalence_row();
        row.Metric = "attitude_window_max_deg";
        row.WindowSteps = count;
        row.Threshold = threshold;
        row.EpisodeCount = height(episodes);
        row.WeakEpisodeCount = nnz(attitude >= threshold);
        row.WeakEpisodeFraction = mean(attitude >= threshold);
        rows(cursor, 1) = row;
    end
end
result = struct2table(rows(1:cursor));
end

function result = family_summary(episodes)
families = unique(episodes.Family, 'stable');
rows = repmat(empty_family_row(), numel(families), 1);
for index = 1:numel(families)
    subset = episodes(episodes.Family == families(index), :);
    rows(index).Family = families(index);
    rows(index).EpisodeCount = height(subset);
    rows(index).PositionRmseMean = mean(subset.PositionRmse);
    rows(index).PositionRmseP95 = percentile_linear(subset.PositionRmse, 95);
    rows(index).AttitudeRmseMeanDeg = mean(subset.AttitudeRmseDeg);
    rows(index).AttitudeRmseP95Deg = percentile_linear( ...
        subset.AttitudeRmseDeg, 95);
    rows(index).SaturationEpisodeFraction = mean( ...
        subset.SaturationFraction > 0);
    rows(index).ConstraintViolationEpisodeFraction = mean( ...
        subset.ConstraintViolationCount > 0);
    rows(index).FiniteFraction = mean(subset.Finite);
end
result = struct2table(rows);
end

function result = speed_load_summary(episodes)
[groups, family, speed, loadIndex] = findgroups(episodes.Family, ...
    episodes.TargetReferenceSpeed, episodes.LoadIndex);
rows = repmat(empty_speed_load_row(), max(groups), 1);
for index = 1:max(groups)
    subset = episodes(groups == index, :);
    rows(index).Family = family(index);
    rows(index).TargetReferenceSpeed = speed(index);
    rows(index).LoadIndex = loadIndex(index);
    rows(index).EpisodeCount = height(subset);
    rows(index).RealizedSpeedMean = mean(subset.PeakReferenceSpeed);
    rows(index).RealizedAccelerationMean = ...
        mean(subset.PeakReferenceAcceleration);
    rows(index).PositionRmseMean = mean(subset.PositionRmse);
    rows(index).PositionRmseP95 = percentile_linear( ...
        subset.PositionRmse, 95);
    rows(index).AttitudeRmseMeanDeg = mean(subset.AttitudeRmseDeg);
    rows(index).AttitudeRmseP95Deg = percentile_linear( ...
        subset.AttitudeRmseDeg, 95);
    rows(index).SaturationEpisodeFraction = mean( ...
        subset.SaturationFraction > 0);
    rows(index).ConstraintViolationEpisodeFraction = mean( ...
        subset.ConstraintViolationCount > 0);
    rows(index).FiniteFraction = mean(subset.Finite);
end
result = struct2table(rows);
end

function row = empty_episode_row()
row.EpisodeIndex = 0;
row.CellId = "";
row.Family = "";
row.DisturbanceType = "";
row.DisturbanceLevel = 0;
row.TargetReferenceSpeed = NaN;
row.TargetReferenceAcceleration = NaN;
row.LoadIndex = 0;
row.PeakReferenceSpeed = NaN;
row.PeakReferenceAcceleration = NaN;
row.UncertaintyStratum = "";
row.MassScale = NaN;
row.MinimumInertiaScale = NaN;
row.MinimumEffectiveness = NaN;
row.PositionRmse = NaN;
row.AttitudeRmseDeg = NaN;
row.PositionMax = NaN;
row.AttitudeMaxDeg = NaN;
row.SaturationFraction = NaN;
row.ConstraintViolationCount = 0;
row.ControlTimeMean = NaN;
row.ControlTimeP95 = NaN;
row.Finite = false;
row.PositionWindow5Max = NaN;
row.AttitudeWindow5MaxDeg = NaN;
row.PositionWindow10Max = NaN;
row.AttitudeWindow10MaxDeg = NaN;
row.PositionWindow20Max = NaN;
row.AttitudeWindow20MaxDeg = NaN;
end

function row = empty_prevalence_row()
row.Metric = "";
row.WindowSteps = 0;
row.Threshold = NaN;
row.EpisodeCount = 0;
row.WeakEpisodeCount = 0;
row.WeakEpisodeFraction = NaN;
end

function row = empty_family_row()
row.Family = "";
row.EpisodeCount = 0;
row.PositionRmseMean = NaN;
row.PositionRmseP95 = NaN;
row.AttitudeRmseMeanDeg = NaN;
row.AttitudeRmseP95Deg = NaN;
row.SaturationEpisodeFraction = NaN;
row.ConstraintViolationEpisodeFraction = NaN;
row.FiniteFraction = NaN;
end

function row = empty_speed_load_row()
row.Family = "";
row.TargetReferenceSpeed = NaN;
row.LoadIndex = 0;
row.EpisodeCount = 0;
row.RealizedSpeedMean = NaN;
row.RealizedAccelerationMean = NaN;
row.PositionRmseMean = NaN;
row.PositionRmseP95 = NaN;
row.AttitudeRmseMeanDeg = NaN;
row.AttitudeRmseP95Deg = NaN;
row.SaturationEpisodeFraction = NaN;
row.ConstraintViolationEpisodeFraction = NaN;
row.FiniteFraction = NaN;
end

function values = rolling_rms(data, count)
values = sqrt(conv(data .^ 2, ones(1, count) ./ count, 'valid'));
end

function write_report(outputRoot, ~, episodes, prevalence, byFamily, ...
        lqr, wallSeconds, lqrPath)
path = fullfile(outputRoot, 'lqr_validation_screen_report.md');
fileId = fopen(path, 'w');
assert(fileId >= 0, 'Cannot open %s.', path);
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, '# Targeted LQR validation screen\n\n');
fprintf(fileId, '- LQR artifact: `%s`.\n', lqrPath);
fprintf(fileId, '- Episodes: %d.\n', height(episodes));
fprintf(fileId, '- Wall time: %.6f s (%.6f s/episode).\n', ...
    wallSeconds, wallSeconds / height(episodes));
fprintf(fileId, '- Pure LQR command time: %.6f ms/step.\n', ...
    1000 * mean(episodes.ControlTimeMean));
fprintf(fileId, '- Finite episodes: %.2f%%.\n', 100 * mean(episodes.Finite));
fprintf(fileId, '- Saturation episodes: %.2f%%.\n', ...
    100 * mean(episodes.SaturationFraction > 0));
fprintf(fileId, '- Constraint-violation episodes: %.2f%%.\n\n', ...
    100 * mean(episodes.ConstraintViolationCount > 0));
fprintf(fileId, '- Selected spectral radius: %.8f.\n', ...
    max(abs(lqr.closedLoopPoles)));
fprintf(fileId, '\n## Family summary\n\n');
for index = 1:height(byFamily)
    row = byFamily(index, :);
    fprintf(fileId, ['- %s: position mean/P95 %.4f/%.4f m; attitude ' ...
        'mean/P95 %.3f/%.3f deg.\n'], char(row.Family), ...
        row.PositionRmseMean, row.PositionRmseP95, ...
        row.AttitudeRmseMeanDeg, row.AttitudeRmseP95Deg);
end
fprintf(fileId, '\n## Candidate weak prevalence\n\n');
for index = 1:height(prevalence)
    row = prevalence(index, :);
    fprintf(fileId, '- %s, %d steps, threshold %.3f: %d/%d (%.1f%%).\n', ...
        char(row.Metric), row.WindowSteps, row.Threshold, ...
        row.WeakEpisodeCount, row.EpisodeCount, ...
        100 * row.WeakEpisodeFraction);
end
fprintf(fileId, '\nThe prevalence table is a candidate label grid, not a final ');
fprintf(fileId, 'training selection. No teacher result was used.\n');
clear cleanup;
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
