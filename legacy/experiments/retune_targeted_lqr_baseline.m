function retune_targeted_lqr_baseline()
%RETUNE_TARGETED_LQR_BASELINE Tune and freeze the new targeted LQR baseline.
%
% This stage is independent of the legacy Step 6 manifest. It uses two
% independently seeded validation banks and keeps the plant realization and
% disturbance hidden from the controller. The selected artifact is intended
% to define LQR-weak contexts for the subsequent specialist pipeline.

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot, 'configs'));
addpath(genpath(fullfile(projectRoot, 'src')));

cfg = targeted_lqr_weak_config();
assert(isfield(cfg.retune, 'longRunAuthorized') && ...
    cfg.retune.longRunAuthorized, ...
    'The targeted LQR retune has not been authorized.');
nmpcCfg = step2_nmpc_config();
disturbanceCfg = step3_disturbance_config();
outputRoot = fullfile(projectRoot, cfg.resultRoot, ...
    cfg.retune.outputSubfolder);
if ~isfolder(outputRoot)
    mkdir(outputRoot);
end

previousRng = rng;
cleanup = onCleanup(@() rng(previousRng));
clock = tic;

linearization = linearize_nominal(cfg);
[coarseGrid, coarseCandidates] = build_coarse_candidates( ...
    linearization, cfg, nmpcCfg);

actualWorkerCount = 1;
if cfg.retune.parallel.enabled
    cluster = parcluster;
    actualWorkerCount = min(cfg.retune.parallel.workerCount, ...
        cluster.NumWorkers);
    pool = gcp('nocreate');
    if isempty(pool) || pool.NumWorkers ~= actualWorkerCount
        if ~isempty(pool)
            delete(pool);
        end
        parpool(cluster, actualWorkerCount);
    end
end
cfg.retune.parallel.actualWorkerCount = actualWorkerCount;

selectionSourceRoot = strtrim(getenv('LQR_SELECTION_SOURCE_ROOT'));
if ~isempty(selectionSourceRoot)
    finalize_selection_only(projectRoot, selectionSourceRoot, cfg, ...
        nmpcCfg, disturbanceCfg, clock);
    clear cleanup;
    return;
end

[designBank, designManifest] = quad_build_balanced_targeted_bank(cfg, ...
    disturbanceCfg, cfg.retune.replicatesPerCell, cfg.retune.designSeed);
[selectionBank, selectionManifest] = quad_build_balanced_targeted_bank(cfg, ...
    disturbanceCfg, cfg.retune.replicatesPerCell, cfg.retune.selectionSeed);
save(fullfile(outputRoot, 'validation_banks.mat'), 'designBank', ...
    'selectionBank', 'designManifest', 'selectionManifest', ...
    'linearization', 'coarseGrid', '-v7.3');
writetable(designManifest, fullfile(outputRoot, 'design_bank_manifest.csv'));
writetable(selectionManifest, fullfile(outputRoot, ...
    'selection_bank_manifest.csv'));

[coarseEpisodes, coarseSummary] = evaluate_candidates(coarseCandidates, ...
    designBank, cfg, nmpcCfg);
[~, coarseBestIndex] = min(coarse_rank_score(coarseSummary));
coarseBestId = coarseSummary.CandidateId(coarseBestIndex);
coarseBest = coarseCandidates(strcmp({coarseCandidates.Id}, coarseBestId));
[refinementGrid, refinementCandidates] = build_refinement_candidates( ...
    linearization, cfg, nmpcCfg, coarseBest.Controller.lqr);
[refinementEpisodes, refinementSummary] = evaluate_candidates( ...
    refinementCandidates, designBank, cfg, nmpcCfg);

candidateGrid = [coarseGrid; refinementGrid];
candidates = [coarseCandidates, refinementCandidates];
designEpisodes = [coarseEpisodes; refinementEpisodes];
designSummary = [coarseSummary; refinementSummary];
designRanked = rank_summary(designSummary);
keepCount = min(cfg.retune.keepTop, height(designRanked));
topIds = designRanked.CandidateId(1:keepCount);
topCandidates = candidates(ismember(string({candidates.Id}), topIds));

[selectionEpisodes, selectionSummary] = evaluate_candidates(topCandidates, ...
    selectionBank, cfg, nmpcCfg);
selectionRanked = rank_summary(selectionSummary);
selectedId = selectionRanked.CandidateId(1);
selected = topCandidates(strcmp({topCandidates.Id}, selectedId));
selectedLqr = selected.Controller.lqr;
selectedLqr.artifactVersion = 'targeted_lqr_weak_rebuild_v1_retuned';
selectedLqr.selectionCandidateId = char(selectedId);
selectedLqr.designBankSeed = cfg.retune.designSeed;
selectedLqr.selectionBankSeed = cfg.retune.selectionSeed;
selectedLqr.referenceFamilies = cfg.reference.families;
selectedLqr.referenceFeedforward = cfg.retune.referenceFeedforward;
selectedLqr.lqrStructure = cfg.retune.lqrStructure;

selectedDesignRow = designSummary(designSummary.CandidateId == selectedId, :);
selectedSelectionRow = selectionSummary( ...
    selectionSummary.CandidateId == selectedId, :);

writetable(designEpisodes, fullfile(outputRoot, ...
    'lqr_design_episode_metrics.csv'));
writetable(designSummary, fullfile(outputRoot, ...
    'lqr_design_summary.csv'));
writetable(designRanked, fullfile(outputRoot, ...
    'lqr_design_ranked.csv'));
writetable(selectionEpisodes, fullfile(outputRoot, ...
    'lqr_selection_episode_metrics.csv'));
writetable(selectionSummary, fullfile(outputRoot, ...
    'lqr_selection_summary.csv'));
writetable(selectionRanked, fullfile(outputRoot, ...
    'lqr_selection_ranked.csv'));

wallSeconds = toc(clock);
save(fullfile(outputRoot, 'selected_lqr.mat'), 'selectedLqr', ...
    'linearization', 'coarseGrid', 'refinementGrid', 'candidateGrid', ...
    'designRanked', 'selectionRanked', ...
    'selectedDesignRow', 'selectedSelectionRow', 'cfg', '-v7.3');
save(fullfile(outputRoot, 'retune_results.mat'), 'designEpisodes', ...
    'designSummary', 'designRanked', 'selectionEpisodes', ...
    'selectionSummary', 'selectionRanked', 'candidateGrid', ...
    'coarseGrid', 'refinementGrid', 'linearization', 'wallSeconds', ...
    'cfg', '-v7.3');
write_report(outputRoot, cfg, candidateGrid, designRanked, selectionRanked, ...
    selectedLqr, wallSeconds);
write_text(fullfile(outputRoot, 'COMPLETED.txt'), sprintf( ...
    ['task=retune_targeted_lqr_baseline\ncompleted_at=%s\n' ...
    'design_episodes=%d\nselection_episodes=%d\n' ...
    'candidate_count=%d\nselected_candidate=%s\nwall_seconds=%.6f\n' ...
    'reference_feedforward=%s\nrequested_workers=%d\nactual_workers=%d\n'], ...
    char(datetime('now', 'TimeZone', 'local', 'Format', ...
    'yyyy-MM-dd HH:mm:ss.SSSXXX')), height(designBank), ...
    height(selectionBank), numel(candidates), selectedId, wallSeconds, ...
    cfg.retune.referenceFeedforward, cfg.retune.parallel.workerCount, ...
    cfg.retune.parallel.actualWorkerCount));
clear cleanup;
end

function finalize_selection_only(projectRoot, sourceRoot, cfg, nmpcCfg, ...
        disturbanceCfg, clock)
sourcePath = fullfile(sourceRoot, 'selected_lqr.mat');
assert(isfile(sourcePath), 'Missing source LQR artifact: %s', sourcePath);
source = load(sourcePath, 'linearization', 'candidateGrid', ...
    'designRanked');
required = {'linearization', 'candidateGrid', 'designRanked'};
assert(all(isfield(source, required)), ...
    'Source artifact does not contain the design-selection contract.');

keepCount = min(cfg.retune.keepTop, height(source.designRanked));
topIds = source.designRanked.CandidateId(1:keepCount);
topCandidates = reconstruct_candidates(topIds, source.candidateGrid, ...
    source.linearization, cfg, nmpcCfg);

outputRoot = fullfile(projectRoot, cfg.resultRoot, ...
    cfg.retune.selectionRevalidationOutputSubfolder);
if ~isfolder(outputRoot)
    mkdir(outputRoot);
end
[selectionBank, selectionManifest] = quad_build_balanced_targeted_bank( ...
    cfg, disturbanceCfg, cfg.retune.replicatesPerCell, ...
    cfg.retune.selectionSeed);
writetable(selectionManifest, fullfile(outputRoot, ...
    'selection_bank_manifest.csv'));
save(fullfile(outputRoot, 'selection_bank.mat'), 'selectionBank', ...
    'selectionManifest', '-v7.3');

[selectionEpisodes, selectionSummary] = evaluate_candidates( ...
    topCandidates, selectionBank, cfg, nmpcCfg);
selectionRanked = rank_summary(selectionSummary);
selectedId = selectionRanked.CandidateId(1);
selected = topCandidates(strcmp({topCandidates.Id}, selectedId));
selectedLqr = selected.Controller.lqr;
selectedLqr.artifactVersion = ...
    'targeted_lqr_weak_rebuild_v1_selection_revalidated_v6';
selectedLqr.selectionCandidateId = char(selectedId);
selectedLqr.designBankSeed = cfg.retune.designSeed;
selectedLqr.selectionBankSeed = cfg.retune.selectionSeed;
selectedLqr.referenceFamilies = cfg.reference.families;
selectedLqr.referenceFeedforward = cfg.retune.referenceFeedforward;
selectedLqr.lqrStructure = cfg.retune.lqrStructure;
selectedLqr.sourceDesignRunId = strtrim(getenv('LQR_SOURCE_RUN_ID'));

selectedDesignRow = source.designRanked( ...
    source.designRanked.CandidateId == selectedId, :);
selectedSelectionRow = selectionSummary( ...
    selectionSummary.CandidateId == selectedId, :);
writetable(source.designRanked, fullfile(outputRoot, ...
    'source_design_ranked.csv'));
writetable(selectionEpisodes, fullfile(outputRoot, ...
    'lqr_selection_episode_metrics.csv'));
writetable(selectionSummary, fullfile(outputRoot, ...
    'lqr_selection_summary.csv'));
writetable(selectionRanked, fullfile(outputRoot, ...
    'lqr_selection_ranked.csv'));

wallSeconds = toc(clock);
linearization = source.linearization;
candidateGrid = source.candidateGrid;
designRanked = source.designRanked;
save(fullfile(outputRoot, 'selected_lqr.mat'), 'selectedLqr', ...
    'linearization', 'candidateGrid', 'designRanked', ...
    'selectionRanked', 'selectedDesignRow', 'selectedSelectionRow', ...
    'cfg', '-v7.3');
save(fullfile(outputRoot, 'selection_revalidation_results.mat'), ...
    'selectionEpisodes', 'selectionSummary', 'selectionRanked', ...
    'selectionManifest', 'candidateGrid', 'designRanked', ...
    'linearization', 'wallSeconds', 'cfg', '-v7.3');
write_selection_revalidation_report(outputRoot, cfg, sourceRoot, ...
    selectionRanked, selectedLqr, wallSeconds);
write_text(fullfile(outputRoot, 'COMPLETED.txt'), sprintf( ...
    ['task=finalize_targeted_lqr_selection\ncompleted_at=%s\n' ...
    'source_run_id=%s\ndesign_candidate_count=%d\n' ...
    'selection_candidate_count=%d\nselection_episodes=%d\n' ...
    'selected_candidate=%s\nwall_seconds=%.6f\n' ...
    'requested_workers=%d\nactual_workers=%d\n'], ...
    char(datetime('now', 'TimeZone', 'local', 'Format', ...
    'yyyy-MM-dd HH:mm:ss.SSSXXX')), ...
    strtrim(getenv('LQR_SOURCE_RUN_ID')), height(candidateGrid), ...
    numel(topCandidates), height(selectionBank), selectedId, wallSeconds, ...
    cfg.retune.parallel.workerCount, ...
    cfg.retune.parallel.actualWorkerCount));
end

function candidates = reconstruct_candidates(ids, grid, linearization, ...
        cfg, nmpcCfg)
candidates = empty_candidate_array();
for index = 1:numel(ids)
    row = grid(grid.CandidateId == ids(index), :);
    assert(height(row) == 1, 'Candidate %s is not unique.', ids(index));
    design = build_lqr(linearization, cfg, nmpcCfg, ...
        row.PositionVelocityScale, row.AttitudeRateScale, ...
        row.InputPenaltyScale);
    candidates(index) = candidate(char(ids(index)), design);
end
end

function [grid, candidates] = build_coarse_candidates( ...
        linearization, cfg, nmpcCfg)
grid = empty_grid();
candidates = empty_candidate_array();
cursor = 0;
for pvScale = cfg.retune.grid.positionVelocityScale
    for arScale = cfg.retune.grid.attitudeRateScale
        for rScale = cfg.retune.grid.inputPenaltyScale
            cursor = cursor + 1;
            design = build_lqr(linearization, cfg, nmpcCfg, ...
                pvScale, arScale, rScale);
            id = sprintf('C_%03d', cursor);
            candidates(cursor) = candidate(id, ...
                design);
            grid(cursor, :) = {string(id), "coarse", pvScale, arScale, ...
                rScale};
        end
    end
end
end

function [grid, candidates] = build_refinement_candidates( ...
        linearization, cfg, nmpcCfg, center)
multipliers = cfg.retune.grid.refinementMultiplier;
grid = empty_grid();
candidates = empty_candidate_array();
cursor = 0;
for pvMultiplier = multipliers
    for arMultiplier = multipliers
        for rMultiplier = multipliers
            cursor = cursor + 1;
            pvScale = center.positionVelocityScale * pvMultiplier;
            arScale = center.attitudeRateScale * arMultiplier;
            rScale = center.inputPenaltyScale * rMultiplier;
            design = build_lqr(linearization, cfg, nmpcCfg, ...
                pvScale, arScale, rScale);
            id = sprintf('R_%03d', cursor);
            candidates(cursor) = candidate(id, design);
            grid(cursor, :) = {string(id), "refinement", pvScale, ...
                arScale, rScale};
        end
    end
end
end

function [episodes, summary] = evaluate_candidates(candidates, bank, cfg, nmpcCfg)
candidateRows = cell(numel(candidates), 1);
if cfg.retune.parallel.enabled
    parfor candidateIndex = 1:numel(candidates)
        localRows = cell(numel(bank), 1);
        for episodeIndex = 1:numel(bank)
            rollout = rollout_lqr(bank(episodeIndex), ...
                candidates(candidateIndex).Controller.lqr, cfg);
            localRows{episodeIndex} = metric_row(rollout, ...
                bank(episodeIndex), candidates(candidateIndex).Id, cfg, ...
                nmpcCfg);
        end
        candidateRows{candidateIndex} = vertcat(localRows{:});
    end
else
    for candidateIndex = 1:numel(candidates)
        localRows = cell(numel(bank), 1);
        for episodeIndex = 1:numel(bank)
            rollout = rollout_lqr(bank(episodeIndex), ...
                candidates(candidateIndex).Controller.lqr, cfg);
            localRows{episodeIndex} = metric_row(rollout, ...
                bank(episodeIndex), candidates(candidateIndex).Id, cfg, ...
                nmpcCfg);
        end
        candidateRows{candidateIndex} = vertcat(localRows{:});
    end
end
episodes = vertcat(candidateRows{:});
summary = summarize_candidates(episodes);
end

function score = coarse_rank_score(summary)
ranked = rank_summary(summary);
score = nan(height(summary), 1);
for index = 1:height(ranked)
    score(summary.CandidateId == ranked.CandidateId(index)) = index;
end
end

function grid = empty_grid()
grid = table(strings(0, 1), strings(0, 1), zeros(0, 1), zeros(0, 1), ...
    zeros(0, 1), 'VariableNames', {'CandidateId', 'Phase', ...
    'PositionVelocityScale', 'AttitudeRateScale', 'InputPenaltyScale'});
end

function rollout = rollout_lqr(scenario, design, cfg)
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

function row = metric_row(rollout, scenario, candidateId, cfg, nmpcCfg)
start = cfg.retune.evaluationStartStep;
stateIndices = start + 1:cfg.stepCount + 1;
error = nmpc_state_error(rollout.X(:, stateIndices), ...
    scenario.Reference(:, stateIndices));
positionError = vecnorm(error(1:3, :), 2, 1);
attitudeErrorDeg = rad2deg(vecnorm(error(4:6, :), 2, 1));
validTimes = rollout.controlTime(start:end);
validTimes = validTimes(isfinite(validTimes) & validTimes > 0);
if isempty(validTimes)
    meanControlTime = NaN;
    p95ControlTime = NaN;
else
    meanControlTime = mean(validTimes);
    p95ControlTime = percentile_linear(validTimes, 95);
end
increments = diff(rollout.U(:, start:end), 1, 2);
if isempty(increments)
    incrementRms = NaN;
else
    incrementRms = sqrt(mean(increments(:) .^ 2));
end
violation = nmpc_state_bound_violations(rollout.X(:, stateIndices), nmpcCfg);
row = table(string(candidateId), scenario.EpisodeIndex, ...
    string(scenario.CellId), string(scenario.Family), ...
    string(scenario.Disturbance.type), ...
    scenario.Disturbance.levelIndex, scenario.ReferenceMetric.peakSpeed, ...
    scenario.ReferenceMetric.peakAcceleration, ...
    string(scenario.UncertaintyStratum), ...
    scenario.ThetaPlant.m / cfg.plant.nominal.m, ...
    min(diag(scenario.ThetaPlant.J) ./ diag(cfg.plant.nominal.J)), ...
    min([scenario.ThetaPlant.alphaT; scenario.ThetaPlant.alphaTau]), ...
    sqrt(mean(positionError .^ 2)), sqrt(mean(attitudeErrorDeg .^ 2)), ...
    max(positionError), max(attitudeErrorDeg), ...
    incrementRms, sqrt(mean(rollout.U(:, start:end) .^ 2, 'all')), ...
    mean(rollout.saturated(start:end)), nnz(violation > 1e-8), ...
    meanControlTime, p95ControlTime, ...
    all(isfinite(rollout.X), 'all') && all(isfinite(rollout.U), 'all'), ...
    'VariableNames', {'CandidateId', 'EpisodeIndex', 'CellId', 'Family', ...
    'DisturbanceType', 'DisturbanceLevel', 'PeakReferenceSpeed', ...
    'PeakReferenceAcceleration', 'UncertaintyStratum', 'MassScale', ...
    'MinimumInertiaScale', ...
    'MinimumEffectiveness', 'PositionRmse', 'AttitudeRmseDeg', ...
    'PositionMax', 'AttitudeMaxDeg', 'ControlIncrementRms', ...
    'ControlRms', 'SaturationFraction', 'ConstraintViolationCount', ...
    'ControlTimeMean', 'ControlTimeP95', 'Finite'});
end

function summary = summarize_candidates(episodes)
ids = unique(episodes.CandidateId, 'stable');
rows = cell(numel(ids), 1);
for index = 1:numel(ids)
    subset = episodes(episodes.CandidateId == ids(index), :);
    rows{index} = table(ids(index), height(subset), mean(subset.Finite), ...
        mean(subset.ConstraintViolationCount > 0), ...
        mean(subset.SaturationFraction > 0), mean(subset.PositionRmse), ...
        percentile_linear(subset.PositionRmse, 95), ...
        mean(subset.AttitudeRmseDeg), ...
        percentile_linear(subset.AttitudeRmseDeg, 95), ...
        mean(subset.ControlIncrementRms), mean(subset.ControlRms), ...
        mean(subset.ControlTimeMean), ...
        'VariableNames', {'CandidateId', 'EpisodeCount', 'FiniteRate', ...
        'ViolationEpisodeRate', 'SaturationEpisodeRate', ...
        'PositionRmseMean', 'PositionRmseP95', 'AttitudeRmseMeanDeg', ...
        'AttitudeRmseP95Deg', 'ControlIncrementRmsMean', ...
        'ControlRmsMean', 'ControlTimeMean'});
end
summary = vertcat(rows{:});
end

function ranked = rank_summary(summary)
ranked = sortrows(summary, {'FiniteRate', 'ViolationEpisodeRate', ...
    'SaturationEpisodeRate', 'PositionRmseMean', 'AttitudeRmseMeanDeg', ...
    'ControlIncrementRmsMean', 'ControlTimeMean'}, ...
    {'descend', 'ascend', 'ascend', 'ascend', 'ascend', 'ascend', ...
    'ascend'});
end

function design = build_lqr(linearization, cfg, nmpcCfg, pvScale, ...
        arScale, rScale)
Q = nmpcCfg.weights.Q;
Q([1:3, 7:9], [1:3, 7:9]) = pvScale .* ...
    Q([1:3, 7:9], [1:3, 7:9]);
Q([4:6, 10:12], [4:6, 10:12]) = arScale .* ...
    Q([4:6, 10:12], [4:6, 10:12]);
R = rScale .* nmpcCfg.weights.R;
[K, S, poles] = dlqr(linearization.A, linearization.B, Q, R);
design.K = K;
design.S = S;
design.closedLoopPoles = poles;
design.Q = Q;
design.R = R;
design.uEquilibrium = linearization.uEquilibrium;
design.positionVelocityScale = pvScale;
design.attitudeRateScale = arScale;
design.inputPenaltyScale = rScale;
design.designModel = 'nominal';
design.linearizationPoint = 'hover';
design.sampleTime = cfg.sampleTime;
end

function linearization = linearize_nominal(cfg)
xEquilibrium = zeros(12, 1);
uEquilibrium = quad_hover_input(cfg.plant.nominal);
map = @(x, u) quad_step_rk4(0, x, u, cfg.sampleTime, ...
    cfg.plant.nominal, []);
A = zeros(12, 12);
B = zeros(12, 4);
for index = 1:12
    delta = zeros(12, 1);
    delta(index) = cfg.lqr.statePerturbation(index);
    A(:, index) = (map(xEquilibrium + delta, uEquilibrium) - ...
        map(xEquilibrium - delta, uEquilibrium)) / (2 * delta(index));
end
for index = 1:4
    delta = zeros(4, 1);
    delta(index) = cfg.lqr.inputPerturbation(index);
    B(:, index) = (map(xEquilibrium, uEquilibrium + delta) - ...
        map(xEquilibrium, uEquilibrium - delta)) / (2 * delta(index));
end
linearization.A = A;
linearization.B = B;
linearization.xEquilibrium = xEquilibrium;
linearization.uEquilibrium = uEquilibrium;
end

function values = percentile_linear(values, percentage)
values = sort(values(isfinite(values)));
if isempty(values)
    values = NaN;
    return;
end
if isscalar(values)
    return;
end
rankValue = 1 + (numel(values) - 1) * percentage / 100;
lowerIndex = floor(rankValue);
upperIndex = ceil(rankValue);
weight = rankValue - lowerIndex;
values = (1 - weight) * values(lowerIndex) + ...
    weight * values(upperIndex);
end

function candidates = empty_candidate_array()
prototype = candidate('', struct());
candidates = repmat(prototype, 1, 0);
end

function value = candidate(id, design)
value.Id = id;
value.Controller.lqr = design;
end

function write_report(outputRoot, cfg, grid, ~, selectionRanked, ...
        selected, wallSeconds)
path = fullfile(outputRoot, 'lqr_retune_report.md');
fileId = fopen(path, 'w');
assert(fileId >= 0, 'Cannot open %s.', path);
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, '# Retuned targeted LQR baseline\n\n');
fprintf(fileId, '- Structure: `%s`.\n', cfg.retune.lqrStructure);
fprintf(fileId, '- Reference feedforward: `%s`.\n', ...
    cfg.retune.referenceFeedforward);
fprintf(fileId, ['- Candidate count: %d = %d coarse + %d local ' ...
    'refinement.\n'], ...
    height(grid), cfg.retune.grid.coarseCandidateCount, ...
    cfg.retune.grid.refinementCandidateCount);
fprintf(fileId, '- Design bank: %d episodes, seed `%d`.\n', ...
    cfg.retune.designEpisodeCount, cfg.retune.designSeed);
fprintf(fileId, '- Selection bank: %d episodes, seed `%d`.\n', ...
    cfg.retune.selectionEpisodeCount, cfg.retune.selectionSeed);
fprintf(fileId, '- Total wall time: %.6f s.\n\n', wallSeconds);
fprintf(fileId, '- Parallel workers: %d requested, %d active.\n', ...
    cfg.retune.parallel.workerCount, ...
    cfg.retune.parallel.actualWorkerCount);
fprintf(fileId, '## Frozen candidate\n\n');
fprintf(fileId, '- ID: `%s`.\n', selected.selectionCandidateId);
fprintf(fileId, '- Position/velocity scale: %.4g.\n', ...
    selected.positionVelocityScale);
fprintf(fileId, '- Attitude/rate scale: %.4g.\n', ...
    selected.attitudeRateScale);
fprintf(fileId, '- Input penalty scale: %.4g.\n', ...
    selected.inputPenaltyScale);
fprintf(fileId, '- Spectral radius: %.8f.\n', ...
    max(abs(selected.closedLoopPoles)));
fprintf(fileId, '\n## Selection ranking\n\n');
for index = 1:min(10, height(selectionRanked))
    row = selectionRanked(index, :);
    fprintf(fileId, ['%d. `%s`: finite %.1f%%, violation %.1f%%, ' ...
        'saturation %.1f%%, position RMSE %.4f m, attitude RMSE %.3f deg.\n'], ...
        index, char(row.CandidateId), 100 * row.FiniteRate, ...
        100 * row.ViolationEpisodeRate, ...
        100 * row.SaturationEpisodeRate, row.PositionRmseMean, ...
        row.AttitudeRmseMeanDeg);
end
fprintf(fileId, '\n## Reproducibility guardrails\n\n');
fprintf(fileId, ['- The two banks use independent seeds and are both validation ' ...
    'data; no test/OOD confirmation bank is opened here.\n']);
fprintf(fileId, ['- The controller uses nominal hover linearization and does not ' ...
    'observe the uncertain plant realization or disturbance.\n']);
fprintf(fileId, ['- The artifact defines a retuned baseline; LQR-weak labels and ' ...
    'specialist training must use a fresh LQR-only screen afterward.\n']);
fprintf(fileId, ['- Both 5 x 5 x 5 phases are validation-only model selection; ' ...
    'the final test/OOD grid remains locked.\n']);
clear cleanup;
end

function write_selection_revalidation_report(outputRoot, cfg, sourceRoot, ...
        selectionRanked, selected, wallSeconds)
path = fullfile(outputRoot, 'lqr_selection_revalidation_report.md');
fileId = fopen(path, 'w');
assert(fileId >= 0, 'Cannot open %s.', path);
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, '# Targeted LQR selection revalidation\n\n');
fprintf(fileId, '- Source design artifact: `%s`.\n', sourceRoot);
fprintf(fileId, '- Source run ID: `%s`.\n', ...
    strtrim(getenv('LQR_SOURCE_RUN_ID')));
fprintf(fileId, '- Fresh selection seed: `%d`.\n', ...
    cfg.retune.selectionSeed);
fprintf(fileId, '- Selection episodes: `%d`.\n', ...
    cfg.retune.selectionEpisodeCount);
fprintf(fileId, '- Candidates revalidated: `%d`.\n', ...
    height(selectionRanked));
fprintf(fileId, '- Wall time: `%.6f s`.\n', wallSeconds);
fprintf(fileId, '- Selected candidate: `%s`.\n', ...
    selected.selectionCandidateId);
fprintf(fileId, '- Spectral radius: `%.8f`.\n', ...
    max(abs(selected.closedLoopPoles)));
fprintf(fileId, '- Parallel workers: `%d` requested, `%d` active.\n\n', ...
    cfg.retune.parallel.workerCount, ...
    cfg.retune.parallel.actualWorkerCount);
fprintf(fileId, '## Selection ranking\n\n');
for index = 1:height(selectionRanked)
    row = selectionRanked(index, :);
    fprintf(fileId, ['%d. `%s`: finite %.1f%%, violation %.1f%%, ' ...
        'saturation %.1f%%, position RMSE %.4f m, ' ...
        'attitude RMSE %.3f deg.\n'], index, char(row.CandidateId), ...
        100 * row.FiniteRate, 100 * row.ViolationEpisodeRate, ...
        100 * row.SaturationEpisodeRate, row.PositionRmseMean, ...
        row.AttitudeRmseMeanDeg);
end
fprintf(fileId, '\n## Guardrails\n\n');
fprintf(fileId, ['- The 250-candidate design ranking is reused because its ' ...
    'bank passed the realized speed/load and state-bound audit.\n']);
fprintf(fileId, ['- The selection bank is rebuilt after adding reference and ' ...
    'initial-state bound gates; no invalid v5 selection row is reused.\n']);
fprintf(fileId, ['- OOD remains locked. This artifact is not a test-set ' ...
    'performance claim.\n']);
clear cleanup;
end

function write_text(path, content)
fileId = fopen(path, 'w');
assert(fileId >= 0, 'Cannot open %s.', path);
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, '%s', content);
clear cleanup;
end
