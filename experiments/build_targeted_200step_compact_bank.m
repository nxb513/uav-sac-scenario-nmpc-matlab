function build_targeted_200step_compact_bank()
%BUILD_TARGETED_200STEP_COMPACT_BANK Build reachable 200-step LQR-weak contexts.

projectRoot = fileparts(fileparts(mfilename('fullpath')));
add_project_paths(projectRoot);
cfg = targeted_lqr_weak_config();
disturbanceCfg = step3_disturbance_config();
sourcePath = fullfile(projectRoot, cfg.resultRoot, ...
    cfg.specialistBank.sourceSubfolder, ...
    'training_local_context_bank.mat');
source = load(sourcePath, 'trainContextBank', ...
    'trainContextManifest', 'lqr');

sourceBank = source.trainContextBank;
sourceManifest = source.trainContextManifest;
cellIds = unique(string(sourceManifest.CellId), 'stable');
assert(numel(sourceBank) == 1350 && numel(cellIds) == 135, ...
    'Expected 1,350 source contexts over 135 factorial cells.');
assert(cfg.specialistBank.cloudContextsPerCell == 1, ...
    'This public compact export supports one context per factorial cell.');

outputRoot = fullfile(projectRoot, cfg.resultRoot, ...
    cfg.specialistBank.outputSubfolder);
if isfolder(outputRoot)
    error('build_targeted_200step_compact_bank:OutputExists', ...
        'Refusing to overwrite %s.', outputRoot);
end
mkdir(outputRoot);

candidateBank = sourceBank;
rows = repmat(empty_candidate_row(), numel(sourceBank), 1);
maxStartStep = cfg.specialistBank.sourceStepCount + 1 - ...
    cfg.specialistBank.localEpisodeSteps - ...
    cfg.specialistBank.maximumPredictionHorizon + 1;
assert(maxStartStep >= cfg.specialistBank.minimumStartStep, ...
    'The source trajectory is too short for a 200+20-step window.');

clock = tic;
for index = 1:numel(sourceBank)
    scenario = sourceBank(index);
    [reference, ~, feedforward] = ...
        quad_targeted_reference_trajectory(scenario.Family, ...
        cfg.sampleTime, cfg.specialistBank.sourceStepCount + 1, ...
        scenario.Options, cfg.plant.nominal);
    oldDisturbance = scenario.Disturbance;
    disturbance = quad_generate_disturbance_episode(cfg.plant, ...
        disturbanceCfg, oldDisturbance.type, oldDisturbance.domain, ...
        oldDisturbance.levelIndex, cfg.sampleTime, ...
        cfg.specialistBank.sourceStepCount, oldDisturbance.seed);

    sourceScenario = scenario;
    sourceScenario.X0 = scenario.SourceX0;
    sourceScenario.Reference = reference;
    sourceScenario.Feedforward = feedforward;
    sourceScenario.Disturbance = disturbance;
    sourceRollout = run_lqr_source(sourceScenario, source.lqr, cfg);
    startStep = local_peak_start(sourceRollout.X, reference, cfg, ...
        maxStartStep);

    episodeX0 = sourceRollout.X(:, startStep);
    episodeRollout = run_lqr_episode(episodeX0, startStep, reference, ...
        feedforward, scenario.ThetaPlant, disturbance, source.lqr, cfg);
    [difficulty, positionRms, attitudeRms, peakPosition, peakAttitude] = ...
        score_episode(episodeRollout.X, reference, startStep, cfg);

    scenario.Reference = reference;
    scenario.Feedforward = feedforward;
    scenario.Disturbance = disturbance;
    scenario.StartStep = startStep;
    scenario.X0 = episodeX0;
    if startStep > 1
        scenario.PreviousInput = sourceRollout.U(:, startStep - 1);
    else
        scenario.PreviousInput = feedforward(:, startStep);
    end
    scenario.LqrPositionWindowRms = positionRms;
    scenario.LqrAttitudeWindowRmsDeg = attitudeRms;
    scenario.DifficultyScore = difficulty;
    candidateBank(index) = scenario;

    rows(index).SourceBankIndex = index;
    rows(index).SourceEpisodeIndex = scenario.SourceEpisodeIndex;
    rows(index).CellId = string(scenario.CellId);
    rows(index).Family = string(scenario.Family);
    rows(index).DisturbanceType = string(disturbance.type);
    rows(index).DisturbanceLevel = disturbance.levelIndex;
    rows(index).UncertaintyStratum = string(scenario.UncertaintyStratum);
    rows(index).ReplicateIndex = scenario.ReplicateIndex;
    rows(index).StartStep = startStep;
    rows(index).PositionEpisodeRms = positionRms;
    rows(index).AttitudeEpisodeRmsDeg = attitudeRms;
    rows(index).PeakPositionError = peakPosition;
    rows(index).PeakAttitudeErrorDeg = peakAttitude;
    rows(index).DifficultyScore = difficulty;
    rows(index).InitialPositionError = norm( ...
        episodeX0(1:3) - reference(1:3, startStep));
    rows(index).InitialStateErrorNorm = norm(nmpc_state_error( ...
        episodeX0, reference(:, startStep)));
    rows(index).LqrCompleted = all(isfinite(episodeRollout.X), 'all');
end
candidateMetrics = struct2table(rows);

selectedIndices = zeros(numel(cellIds), 1);
for cellIndex = 1:numel(cellIds)
    indices = find(candidateMetrics.CellId == cellIds(cellIndex));
    assert(numel(indices) == 10, ...
        'Cell %s should contain 10 candidates, found %d.', ...
        cellIds(cellIndex), numel(indices));
    [~, localIndex] = max(candidateMetrics.DifficultyScore(indices));
    selectedIndices(cellIndex) = indices(localIndex);
end

trainContextBank = candidateBank(selectedIndices);
trainContextManifest = candidateMetrics(selectedIndices, :);
trainContextManifest.ContextIndex = (1:height(trainContextManifest)).';
trainContextManifest.SelectedRankInCell = ...
    ones(height(trainContextManifest), 1);
trainContextManifest = movevars(trainContextManifest, ...
    'ContextIndex', 'Before', 'SourceBankIndex');
trainContextManifest = movevars(trainContextManifest, ...
    'SelectedRankInCell', 'After', 'DifficultyScore');
for index = 1:numel(trainContextBank)
    trainContextBank(index).ContextIndex = index;
end
wallSeconds = toc(clock);
validate_bank(trainContextBank, trainContextManifest, cfg, cellIds);

buildMetadata = struct();
buildMetadata.status = ...
    'official_200_step_reachable_lqr_weak_compact_training_bank';
buildMetadata.sourcePath = sourcePath;
buildMetadata.sourceContextCount = numel(sourceBank);
buildMetadata.activeContextCount = numel(trainContextBank);
buildMetadata.factorialCellCount = numel(cellIds);
buildMetadata.sourceStepCount = cfg.specialistBank.sourceStepCount;
buildMetadata.localEpisodeSteps = cfg.specialistBank.localEpisodeSteps;
buildMetadata.maximumPredictionHorizon = ...
    cfg.specialistBank.maximumPredictionHorizon;
buildMetadata.maximumStartStep = maxStartStep;
buildMetadata.initializationRule = ...
    'physically_reachable_lqr_state_at_selected_weak_step';
buildMetadata.selectionRule = ...
    'worst_200_step_lqr_recovery_among_10_candidates_per_cell';
buildMetadata.wallSeconds = wallSeconds;
buildMetadata.builtAt = char(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd HH:mm:ssXXX'));

temporaryPath = fullfile(outputRoot, 'training_local_context_bank.tmp.mat');
finalPath = fullfile(outputRoot, 'training_local_context_bank.mat');
lqr = source.lqr;
save(temporaryPath, 'trainContextBank', 'trainContextManifest', ...
    'candidateMetrics', 'buildMetadata', 'cfg', 'lqr', '-v7.3');
movefile(temporaryPath, finalPath, 'f');
writetable(trainContextManifest, fullfile(outputRoot, ...
    'train_local_context_manifest.csv'));
writetable(candidateMetrics, fullfile(outputRoot, ...
    'candidate_200step_lqr_metrics.csv'));
write_report(outputRoot, buildMetadata, trainContextManifest, cfg);
write_text(fullfile(outputRoot, 'COMPLETED.txt'), sprintf( ...
    ['task=build_targeted_200step_compact_bank\nstatus=completed\n' ...
    'source_contexts=%d\nactive_contexts=%d\nfactorial_cells=%d\n' ...
    'source_steps=%d\nepisode_steps=%d\nmaximum_horizon=%d\n' ...
    'maximum_initial_position_error=%.16g\nwall_seconds=%.6f\n'], ...
    numel(sourceBank), numel(trainContextBank), numel(cellIds), ...
    cfg.specialistBank.sourceStepCount, ...
    cfg.specialistBank.localEpisodeSteps, ...
    cfg.specialistBank.maximumPredictionHorizon, ...
    max(trainContextManifest.InitialPositionError), wallSeconds));
end

function rollout = run_lqr_source(scenario, lqr, cfg)
stepCount = cfg.specialistBank.sourceStepCount;
X = nan(12, stepCount + 1);
U = nan(4, stepCount);
X(:, 1) = scenario.X0;
for step = 1:stepCount
    error = nmpc_state_error(X(:, step), scenario.Reference(:, step));
    raw = scenario.Feedforward(:, step) - lqr.K * error;
    U(:, step) = quad_saturate_input(raw, cfg.plant.nominal);
    X(:, step + 1) = quad_step_rk4((step - 1) * cfg.sampleTime, ...
        X(:, step), U(:, step), cfg.sampleTime, scenario.ThetaPlant, ...
        scenario.Disturbance);
    if ~all(isfinite(X(:, step + 1)))
        break;
    end
end
rollout = struct('X', X, 'U', U);
end

function rollout = run_lqr_episode(x0, startStep, reference, ...
        feedforward, thetaPlant, disturbance, lqr, cfg)
stepCount = cfg.specialistBank.localEpisodeSteps;
X = nan(12, stepCount + 1);
U = nan(4, stepCount);
X(:, 1) = x0;
for localStep = 1:stepCount
    absoluteStep = startStep + localStep - 1;
    error = nmpc_state_error(X(:, localStep), ...
        reference(:, absoluteStep));
    raw = feedforward(:, absoluteStep) - lqr.K * error;
    U(:, localStep) = quad_saturate_input(raw, cfg.plant.nominal);
    time = (absoluteStep - 1) * cfg.sampleTime;
    X(:, localStep + 1) = quad_step_rk4(time, X(:, localStep), ...
        U(:, localStep), cfg.sampleTime, thetaPlant, disturbance);
    if ~all(isfinite(X(:, localStep + 1)))
        break;
    end
end
rollout = struct('X', X, 'U', U);
end

function bestStep = local_peak_start(X, reference, cfg, maxStartStep)
window = cfg.specialistBank.rankingWindowSteps;
bestStep = cfg.specialistBank.minimumStartStep;
bestScore = -inf;
for step = cfg.specialistBank.minimumStartStep:maxStartStep
    indices = step - window + 1:step;
    error = nmpc_state_error(X(:, indices), reference(:, indices));
    if ~all(isfinite(error), 'all')
        continue;
    end
    position = sqrt(mean(vecnorm(error(1:3, :), 2, 1) .^ 2));
    attitude = sqrt(mean(rad2deg(vecnorm(error(4:6, :), 2, 1)) .^ 2));
    score = max(position / cfg.specialistBank.positionScoreScaleM, ...
        attitude / cfg.specialistBank.attitudeScoreScaleDeg);
    if score > bestScore
        bestStep = step;
        bestScore = score;
    end
end
assert(isfinite(bestScore), 'No finite source LQR-deficit window was found.');
end

function [difficulty, positionRms, attitudeRms, ...
        peakPosition, peakAttitude] = ...
        score_episode(X, reference, startStep, cfg)
steps = cfg.specialistBank.localEpisodeSteps;
finiteMask = all(isfinite(X), 1);
assert(finiteMask(1), 'The reachable LQR initial state must be finite.');
firstNonfinite = find(~finiteMask, 1, 'first');
if isempty(firstNonfinite)
    lastFinite = size(X, 2);
else
    lastFinite = firstNonfinite - 1;
end
target = reference(:, startStep + (0:lastFinite - 1));
error = nmpc_state_error(X(:, 1:lastFinite), target);
positionSeries = vecnorm(error(1:3, :), 2, 1);
attitudeSeries = rad2deg(vecnorm(error(4:6, :), 2, 1));
positionRms = sqrt(mean(positionSeries .^ 2));
attitudeRms = sqrt(mean(attitudeSeries .^ 2));
peakPosition = max(positionSeries);
peakAttitude = max(attitudeSeries);

window = cfg.specialistBank.rankingWindowSteps;
difficulty = -inf;
activeWindow = min(window, numel(positionSeries));
for last = activeWindow:numel(positionSeries)
    indices = last - activeWindow + 1:last;
    positionWindow = sqrt(mean(positionSeries(indices) .^ 2));
    attitudeWindow = sqrt(mean(attitudeSeries(indices) .^ 2));
    score = max(positionWindow / ...
        cfg.specialistBank.positionScoreScaleM, attitudeWindow / ...
        cfg.specialistBank.attitudeScoreScaleDeg);
    difficulty = max(difficulty, score);
end
if lastFinite < steps + 1
    difficulty = 1e6 + (steps + 1 - lastFinite);
end
end

function validate_bank(bank, manifest, cfg, expectedCellIds)
assert(numel(bank) == 135 && height(manifest) == 135, ...
    'The compact bank must contain exactly 135 contexts.');
assert(numel(unique(manifest.CellId)) == numel(expectedCellIds), ...
    'The compact bank lost one or more factorial cells.');
assert(all(isfinite(manifest.InitialPositionError)) && ...
    all(isfinite(manifest.InitialStateErrorNorm)), ...
    'Every selected LQR-weak initial-state error must be finite.');
lastReference = [bank.StartStep] + ...
    cfg.specialistBank.localEpisodeSteps + ...
    cfg.specialistBank.maximumPredictionHorizon - 1;
assert(all(lastReference <= cfg.specialistBank.sourceStepCount + 1), ...
    'At least one context lacks the required 200+20 reference window.');
for index = 1:numel(bank)
    assert(isequal(size(bank(index).Reference), ...
        [12, cfg.specialistBank.sourceStepCount + 1]));
    assert(isequal(size(bank(index).Feedforward), ...
        [4, cfg.specialistBank.sourceStepCount + 1]));
    assert(isequal(size(bank(index).Disturbance.forceSeries), ...
        [3, cfg.specialistBank.sourceStepCount + 1]));
    assert(isequal(size(bank(index).Disturbance.torqueSeries), ...
        [3, cfg.specialistBank.sourceStepCount + 1]));
end
end

function row = empty_candidate_row()
row = struct('SourceBankIndex', 0, 'SourceEpisodeIndex', 0, ...
    'CellId', "", 'Family', "", 'DisturbanceType', "", ...
    'DisturbanceLevel', 0, 'UncertaintyStratum', "", ...
    'ReplicateIndex', 0, 'StartStep', 0, ...
    'PositionEpisodeRms', NaN, 'AttitudeEpisodeRmsDeg', NaN, ...
    'PeakPositionError', NaN, 'PeakAttitudeErrorDeg', NaN, ...
    'DifficultyScore', NaN, 'InitialPositionError', NaN, ...
    'InitialStateErrorNorm', NaN, 'LqrCompleted', false);
end

function write_report(outputRoot, metadata, manifest, cfg)
path = fullfile(outputRoot, 'specialist_200step_bank_report.md');
fileId = fopen(path, 'w');
assert(fileId >= 0, 'Cannot open %s.', path);
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, '# Reachable LQR-weak bank for 200-step SAC episodes\n\n');
fprintf(fileId, '- Source candidates inspected: %d.\n', ...
    metadata.sourceContextCount);
fprintf(fileId, '- Active contexts/factorial cells: %d/%d.\n', ...
    metadata.activeContextCount, metadata.factorialCellCount);
fprintf(fileId, '- Source trajectory: %d steps (%.1f s).\n', ...
    metadata.sourceStepCount, metadata.sourceStepCount * cfg.sampleTime);
fprintf(fileId, '- SAC episode: %d steps (%.1f s).\n', ...
    metadata.localEpisodeSteps, ...
    metadata.localEpisodeSteps * cfg.sampleTime);
fprintf(fileId, '- Reserved future reference: %d steps.\n', ...
    metadata.maximumPredictionHorizon);
fprintf(fileId, '- Selected start range: %d to %d.\n', ...
    min(manifest.StartStep), max(manifest.StartStep));
fprintf(fileId, '- Maximum reachable initial position error: %.3g m.\n', ...
    max(manifest.InitialPositionError));
fprintf(fileId, '- Build wall time: %.3f s.\n', metadata.wallSeconds);
fprintf(fileId, ['\nFor every candidate, the source LQR trace proposes a ' ...
    'difficult start. The specialist initial state is the physically ' ...
    'reachable LQR state at that step, and frozen LQR is continued for ' ...
    '200 steps under the stored plant uncertainty and hidden disturbance. ' ...
    'Each of the 135 balanced cells contributes the weakest recovery among ' ...
    'its 10 candidates. NMPC, SAC, surrogate, validation and OOD outcomes ' ...
    'do not enter selection. Fair controller evaluation must separately ' ...
    'use common on-reference or near-reference initial conditions.\n']);
clear cleanup;
end

function write_text(path, content)
fileId = fopen(path, 'w');
assert(fileId >= 0, 'Cannot open %s.', path);
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, '%s', content);
clear cleanup;
end

function add_project_paths(projectRoot)
addpath(fullfile(projectRoot, 'configs'));
addpath(fullfile(projectRoot, 'src', 'common'));
addpath(fullfile(projectRoot, 'src', 'plant'));
addpath(fullfile(projectRoot, 'src', 'controllers', 'common'));
end
