function build_targeted_200step_compact_bank()
%BUILD_TARGETED_200STEP_COMPACT_BANK Build 20-step pre-divergence contexts.

projectRoot = fileparts(fileparts(mfilename('fullpath')));
add_project_paths(projectRoot);
cfg = targeted_lqr_weak_config();
divergenceCfg = targeted_lqr_divergence_config();
nmpcCfg = step2_nmpc_config();
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
assert(divergenceCfg.horizonSteps == ...
    cfg.specialistBank.maximumPredictionHorizon, ...
    'Divergence warning and maximum NMPC horizons must match.');

outputRoot = fullfile(projectRoot, cfg.resultRoot, ...
    cfg.specialistBank.outputSubfolder);
if isfolder(outputRoot)
    error('build_targeted_200step_compact_bank:OutputExists', ...
        'Refusing to overwrite %s.', outputRoot);
end
mkdir(outputRoot);

candidateBank = initialize_candidate_bank(sourceBank);
rows = repmat(empty_candidate_row(), numel(sourceBank), 1);
maxStartStep = cfg.specialistBank.sourceStepCount + 1 - ...
    cfg.specialistBank.localEpisodeSteps - ...
    cfg.specialistBank.maximumPredictionHorizon + 1;
assert(maxStartStep >= cfg.specialistBank.minimumStartStep, ...
    'The source trajectory is too short for a 200+20-step window.');

clock = tic;
for index = 1:numel(sourceBank)
    scenario = extend_scenario(sourceBank(index), cfg, disturbanceCfg);
    rollout = run_lqr_source(scenario, source.lqr, cfg);
    [startStep, risk] = select_intervention(rollout, ...
        scenario.Reference, cfg, divergenceCfg, nmpcCfg, maxStartStep);
    scenario = attach_intervention(scenario, rollout, startStep, risk);
    candidateBank(index) = scenario;
    rows(index) = candidate_row(index, scenario, risk);
end
candidateMetrics = struct2table(rows);

[screeningIndices, cellSelection] = select_per_cell( ...
    candidateMetrics, cellIds);
screeningContextBank = candidateBank(screeningIndices);
screeningContextManifest = candidateMetrics(screeningIndices, :);
screeningContextManifest.ContextIndex = ...
    (1:height(screeningContextManifest)).';
screeningContextManifest = movevars(screeningContextManifest, ...
    'ContextIndex', 'Before', 'SourceBankIndex');
for index = 1:numel(screeningContextBank)
    screeningContextBank(index).ContextIndex = index;
end

trainingMask = screeningContextManifest.PreDivergenceEligible;
trainContextBank = screeningContextBank(trainingMask);
trainContextManifest = screeningContextManifest(trainingMask, :);
trainContextManifest.ContextIndex = (1:height(trainContextManifest)).';
for index = 1:numel(trainContextBank)
    trainContextBank(index).ContextIndex = index;
end
assert(~isempty(trainContextBank), ...
    'No 20-step counterfactual LQR-divergence context was found.');

wallSeconds = toc(clock);
validate_bank(trainContextBank, trainContextManifest, ...
    screeningContextManifest, cfg, divergenceCfg);
buildMetadata = make_metadata(sourcePath, sourceBank, cellIds, ...
    trainContextBank, maxStartStep, wallSeconds, cfg, divergenceCfg);

temporaryPath = fullfile(outputRoot, 'training_local_context_bank.tmp.mat');
finalPath = fullfile(outputRoot, 'training_local_context_bank.mat');
lqr = source.lqr;
save(temporaryPath, 'trainContextBank', 'trainContextManifest', ...
    'screeningContextBank', 'screeningContextManifest', ...
    'candidateMetrics', 'cellSelection', 'buildMetadata', ...
    'cfg', 'divergenceCfg', 'lqr', '-v7.3');
movefile(temporaryPath, finalPath, 'f');
writetable(trainContextManifest, fullfile(outputRoot, ...
    'train_predivergence_context_manifest.csv'));
writetable(screeningContextManifest, fullfile(outputRoot, ...
    'screening_context_manifest.csv'));
writetable(candidateMetrics, fullfile(outputRoot, ...
    'candidate_20step_lqr_risk_metrics.csv'));
writetable(cellSelection, fullfile(outputRoot, ...
    'factorial_cell_selection_summary.csv'));
write_report(outputRoot, buildMetadata, trainContextManifest, ...
    screeningContextManifest, divergenceCfg, cfg);
write_text(fullfile(outputRoot, 'COMPLETED.txt'), sprintf( ...
    ['task=build_targeted_200step_compact_bank\nstatus=completed\n' ...
    'source_contexts=%d\nscreening_cells=%d\n' ...
    'predivergence_training_contexts=%d\nwarning_steps=%d\n' ...
    'source_steps=%d\nepisode_steps=%d\nwall_seconds=%.6f\n'], ...
    numel(sourceBank), numel(screeningContextBank), ...
    numel(trainContextBank), divergenceCfg.horizonSteps, ...
    cfg.specialistBank.sourceStepCount, ...
    cfg.specialistBank.localEpisodeSteps, wallSeconds));
end

function scenario = extend_scenario(scenario, cfg, disturbanceCfg)
[reference, ~, feedforward] = quad_targeted_reference_trajectory( ...
    scenario.Family, cfg.sampleTime, ...
    cfg.specialistBank.sourceStepCount + 1, scenario.Options, ...
    cfg.plant.nominal);
oldDisturbance = scenario.Disturbance;
disturbance = quad_generate_disturbance_episode(cfg.plant, ...
    disturbanceCfg, oldDisturbance.type, oldDisturbance.domain, ...
    oldDisturbance.levelIndex, cfg.sampleTime, ...
    cfg.specialistBank.sourceStepCount, oldDisturbance.seed);
scenario.X0 = scenario.SourceX0;
scenario.Reference = reference;
scenario.Feedforward = feedforward;
scenario.Disturbance = disturbance;
end

function bank = initialize_candidate_bank(sourceBank)
bank = sourceBank;
for index = 1:numel(bank)
    bank(index).InterventionStep = 0;
    bank(index).DivergenceEventStep = Inf;
    bank(index).InterventionLeadSteps = Inf;
    bank(index).CounterfactualDivergence = false;
    bank(index).AlreadyOutsideEnvelope = false;
    bank(index).PreDivergenceEligible = false;
    bank(index).LqrRiskScore = NaN;
    bank(index).LqrLogEnergyGrowthPerStep = NaN;
end
end

function rollout = run_lqr_source(scenario, lqr, cfg)
stepCount = cfg.specialistBank.sourceStepCount;
X = nan(12, stepCount + 1);
U = nan(4, stepCount);
rawU = nan(4, stepCount);
saturated = false(1, stepCount);
X(:, 1) = scenario.X0;
for step = 1:stepCount
    error = nmpc_state_error(X(:, step), scenario.Reference(:, step));
    rawU(:, step) = scenario.Feedforward(:, step) - lqr.K * error;
    U(:, step) = quad_saturate_input(rawU(:, step), cfg.plant.nominal);
    saturated(step) = any(abs(rawU(:, step) - U(:, step)) > 1e-10);
    X(:, step + 1) = quad_step_rk4((step - 1) * cfg.sampleTime, ...
        X(:, step), U(:, step), cfg.sampleTime, scenario.ThetaPlant, ...
        scenario.Disturbance);
    if ~all(isfinite(X(:, step + 1)))
        break;
    end
end
rollout = struct('X', X, 'U', U, 'RawU', rawU, ...
    'Saturated', saturated);
end

function [bestStep, bestRisk] = select_intervention(rollout, reference, ...
        cfg, divergenceCfg, nmpcCfg, maxStartStep)
bestStep = cfg.specialistBank.minimumStartStep;
bestRisk = empty_risk();
foundEvent = false;
for step = cfg.specialistBank.minimumStartStep:maxStartStep
    if ~all(isfinite(rollout.X(:, step)))
        break;
    end
    risk = lqr_finite_horizon_divergence_label(rollout.X, ...
        rollout.RawU, rollout.Saturated, reference, step, ...
        divergenceCfg, nmpcCfg);
    validEvent = risk.PreDivergenceEligible;
    if validEvent && (~foundEvent || better_event(risk, bestRisk))
        bestStep = step;
        bestRisk = risk;
        foundEvent = true;
    elseif ~foundEvent && risk.RiskScore > bestRisk.RiskScore
        bestStep = step;
        bestRisk = risk;
    end
end
assert(isfinite(bestRisk.RiskScore), ...
    'No finite LQR risk score was available for a source context.');
end

function yes = better_event(candidate, incumbent)
if candidate.TimeToEventSteps ~= incumbent.TimeToEventSteps
    yes = candidate.TimeToEventSteps > incumbent.TimeToEventSteps;
else
    yes = candidate.RiskScore > incumbent.RiskScore;
end
end

function scenario = attach_intervention(scenario, rollout, startStep, risk)
scenario.StartStep = startStep;
scenario.InterventionStep = startStep;
scenario.DivergenceEventStep = risk.EventStep;
scenario.InterventionLeadSteps = risk.TimeToEventSteps;
scenario.CounterfactualDivergence = risk.CounterfactualDivergence;
scenario.AlreadyOutsideEnvelope = risk.AlreadyOutsideEnvelope;
scenario.PreDivergenceEligible = risk.PreDivergenceEligible;
scenario.LqrRiskScore = risk.RiskScore;
scenario.LqrLogEnergyGrowthPerStep = risk.LogEnergyGrowthPerStep;
scenario.X0 = rollout.X(:, startStep);
if startStep > 1
    scenario.PreviousInput = rollout.U(:, startStep - 1);
else
    scenario.PreviousInput = scenario.Feedforward(:, startStep);
end
end

function row = candidate_row(sourceIndex, scenario, risk)
row = empty_candidate_row();
row.SourceBankIndex = sourceIndex;
row.SourceEpisodeIndex = scenario.SourceEpisodeIndex;
row.CellId = string(scenario.CellId);
row.Family = string(scenario.Family);
row.DisturbanceType = string(scenario.Disturbance.type);
row.DisturbanceLevel = scenario.Disturbance.levelIndex;
row.UncertaintyStratum = string(scenario.UncertaintyStratum);
row.ReplicateIndex = scenario.ReplicateIndex;
row.InterventionStep = scenario.InterventionStep;
row.DivergenceEventStep = scenario.DivergenceEventStep;
row.InterventionLeadSteps = scenario.InterventionLeadSteps;
row.CounterfactualDivergence = risk.CounterfactualDivergence;
row.AlreadyOutsideEnvelope = risk.AlreadyOutsideEnvelope;
row.PreDivergenceEligible = risk.PreDivergenceEligible;
row.RiskScore = risk.RiskScore;
row.LogEnergyGrowthPerStep = risk.LogEnergyGrowthPerStep;
row.EnergyStart = risk.EnergyStart;
row.EnergyEnd = risk.EnergyEnd;
row.EnergyGrowthFactor = risk.EnergyGrowthFactor;
row.MaximumConsecutiveGrowthSteps = ...
    risk.MaximumConsecutiveGrowthSteps;
row.MaximumPositionErrorM = risk.MaximumPositionErrorM;
row.MaximumAttitudeErrorDeg = risk.MaximumAttitudeErrorDeg;
row.MaximumVelocityErrorMps = risk.MaximumVelocityErrorMps;
row.MaximumBodyRateErrorRadps = risk.MaximumBodyRateErrorRadps;
row.AbsoluteThresholdCrossing = risk.AbsoluteThresholdCrossing;
row.SustainedGrowth = risk.SustainedGrowth;
row.SaturationForecast = risk.SaturationForecast;
row.ConstraintForecast = risk.ConstraintForecast;
row.NonfiniteForecast = risk.NonfiniteForecast;
row.MinimumNormalizedActuatorMargin = ...
    risk.MinimumNormalizedActuatorMargin;
initialError = nmpc_state_error(scenario.X0, ...
    scenario.Reference(:, scenario.StartStep));
row.InitialPositionErrorM = norm(initialError(1:3));
row.InitialStateErrorNorm = norm(initialError);
end

function [selectedIndices, summary] = select_per_cell(metrics, cellIds)
selectedIndices = zeros(numel(cellIds), 1);
rows = repmat(empty_cell_row(), numel(cellIds), 1);
for cellIndex = 1:numel(cellIds)
    indices = find(metrics.CellId == cellIds(cellIndex));
    assert(numel(indices) == 10, ...
        'Cell %s should contain 10 candidates, found %d.', ...
        cellIds(cellIndex), numel(indices));
    eventIndices = indices(metrics.PreDivergenceEligible(indices));
    if isempty(eventIndices)
        pool = indices;
        selectedAs = "boundary_no_event";
    else
        pool = eventIndices;
        selectedAs = "predivergence_event";
    end
    ranking = table(metrics.InterventionLeadSteps(pool), ...
        metrics.RiskScore(pool), pool, 'VariableNames', ...
        {'Lead', 'Risk', 'Index'});
    ranking = sortrows(ranking, {'Lead', 'Risk'}, {'descend', 'descend'});
    selectedIndices(cellIndex) = ranking.Index(1);
    rows(cellIndex).CellId = cellIds(cellIndex);
    rows(cellIndex).CandidateCount = numel(indices);
    rows(cellIndex).DivergenceCandidateCount = numel(eventIndices);
    rows(cellIndex).SelectedSourceBankIndex = ranking.Index(1);
    rows(cellIndex).SelectionClass = selectedAs;
end
summary = struct2table(rows);
end

function metadata = make_metadata(sourcePath, sourceBank, cellIds, ...
        trainBank, maxStartStep, wallSeconds, cfg, divergenceCfg)
metadata = struct();
metadata.status = ...
    'candidate_20_step_predivergence_bank_requires_threshold_approval';
metadata.sourcePath = sourcePath;
metadata.sourceContextCount = numel(sourceBank);
metadata.screeningFactorialCellCount = numel(cellIds);
metadata.trainingContextCount = numel(trainBank);
metadata.sourceStepCount = cfg.specialistBank.sourceStepCount;
metadata.localEpisodeSteps = cfg.specialistBank.localEpisodeSteps;
metadata.warningHorizonSteps = divergenceCfg.horizonSteps;
metadata.maximumStartStep = maxStartStep;
metadata.selectionRule = divergenceCfg.selection.rule;
metadata.thresholdStatus = divergenceCfg.status;
metadata.wallSeconds = wallSeconds;
metadata.builtAt = char(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd HH:mm:ssXXX'));
end

function validate_bank(trainBank, trainManifest, screeningManifest, ...
        cfg, divergenceCfg)
assert(height(screeningManifest) == 135 && ...
    numel(unique(screeningManifest.CellId)) == 135, ...
    'Screening export must retain all 135 factorial cells.');
assert(numel(trainBank) == height(trainManifest), ...
    'Training bank and manifest counts differ.');
assert(all(trainManifest.PreDivergenceEligible), ...
    'Training bank contains a context without a valid future LQR event.');
assert(all(trainManifest.InterventionLeadSteps >= 1 & ...
    trainManifest.InterventionLeadSteps <= divergenceCfg.horizonSteps), ...
    'Training contexts must precede events by 1..H steps.');
lastReference = [trainBank.StartStep] + ...
    cfg.specialistBank.localEpisodeSteps + ...
    cfg.specialistBank.maximumPredictionHorizon - 1;
assert(all(lastReference <= cfg.specialistBank.sourceStepCount + 1), ...
    'At least one context lacks the required 200+20 reference window.');
for index = 1:numel(trainBank)
    assert(all(isfinite(trainBank(index).X0)), ...
        'Every intervention state must be physically finite.');
    assert(trainBank(index).DivergenceEventStep > ...
        trainBank(index).InterventionStep, ...
        'Every training context must begin before its LQR event.');
end
end

function risk = empty_risk()
risk = struct('CounterfactualDivergence', false, ...
    'AlreadyOutsideEnvelope', false, ...
    'PreDivergenceEligible', false, ...
    'TimeToEventSteps', -Inf, 'EventStep', Inf, ...
    'RiskScore', -Inf, 'LogEnergyGrowthPerStep', NaN);
end

function row = empty_candidate_row()
row = struct('SourceBankIndex', 0, 'SourceEpisodeIndex', 0, ...
    'CellId', "", 'Family', "", 'DisturbanceType', "", ...
    'DisturbanceLevel', 0, 'UncertaintyStratum', "", ...
    'ReplicateIndex', 0, 'InterventionStep', 0, ...
    'DivergenceEventStep', Inf, 'InterventionLeadSteps', Inf, ...
    'CounterfactualDivergence', false, ...
    'AlreadyOutsideEnvelope', false, ...
    'PreDivergenceEligible', false, 'RiskScore', NaN, ...
    'LogEnergyGrowthPerStep', NaN, 'EnergyStart', NaN, ...
    'EnergyEnd', NaN, 'EnergyGrowthFactor', NaN, ...
    'MaximumConsecutiveGrowthSteps', 0, ...
    'MaximumPositionErrorM', NaN, 'MaximumAttitudeErrorDeg', NaN, ...
    'MaximumVelocityErrorMps', NaN, ...
    'MaximumBodyRateErrorRadps', NaN, ...
    'AbsoluteThresholdCrossing', false, 'SustainedGrowth', false, ...
    'SaturationForecast', false, 'ConstraintForecast', false, ...
    'NonfiniteForecast', false, ...
    'MinimumNormalizedActuatorMargin', NaN, ...
    'InitialPositionErrorM', NaN, 'InitialStateErrorNorm', NaN);
end

function row = empty_cell_row()
row = struct('CellId', "", 'CandidateCount', 0, ...
    'DivergenceCandidateCount', 0, 'SelectedSourceBankIndex', 0, ...
    'SelectionClass', "");
end

function write_report(outputRoot, metadata, trainManifest, ...
        screeningManifest, divergenceCfg, cfg)
path = fullfile(outputRoot, 'predictive_20step_context_bank_report.md');
fileId = fopen(path, 'w');
assert(fileId >= 0, 'Cannot open %s.', path);
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, '# Predictive 20-step LQR-divergence context bank\n\n');
fprintf(fileId, '- Threshold status: `%s`.\n', ...
    divergenceCfg.status);
fprintf(fileId, '- Source candidates: %d.\n', ...
    metadata.sourceContextCount);
fprintf(fileId, '- Balanced screening cells retained: %d.\n', ...
    height(screeningManifest));
fprintf(fileId, '- Pre-divergence training contexts: %d.\n', ...
    height(trainManifest));
fprintf(fileId, '- Warning horizon: %d steps (%.2f s).\n', ...
    divergenceCfg.horizonSteps, divergenceCfg.horizonSeconds);
fprintf(fileId, '- SAC episode: %d steps (%.2f s).\n', ...
    cfg.specialistBank.localEpisodeSteps, ...
    cfg.specialistBank.localEpisodeSteps * cfg.sampleTime);
fprintf(fileId, '- Intervention lead range: %.0f to %.0f steps.\n', ...
    min(trainManifest.InterventionLeadSteps), ...
    max(trainManifest.InterventionLeadSteps));
fprintf(fileId, '- Build wall time: %.3f s.\n', metadata.wallSeconds);
fprintf(fileId, ['\nEach source trace is generated by the frozen LQR on ' ...
    'the hidden uncertain plant and disturbance. The selected training ' ...
    'state is physically reached before a counterfactual LQR event; it is ' ...
    'not an arbitrary injected error. All 135 cells remain in the screening ' ...
    'manifest, while only event-positive cells enter SAC training. NMPC, ' ...
    'SAC, surrogate, validation and OOD outcomes do not define these labels.\n']);
fprintf(fileId, ['\nThis artifact is a candidate screening result. Review ' ...
    'the continuous metrics and threshold sensitivity before locking the ' ...
    'training bank or launching the final SAC budget.\n']);
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
addpath(fullfile(projectRoot, 'src', 'analysis'));
addpath(fullfile(projectRoot, 'src', 'common'));
addpath(fullfile(projectRoot, 'src', 'plant'));
addpath(fullfile(projectRoot, 'src', 'controllers', 'common'));
end
