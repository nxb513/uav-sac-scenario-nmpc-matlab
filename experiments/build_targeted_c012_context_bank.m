function build_targeted_c012_context_bank()
%BUILD_TARGETED_C012_CONTEXT_BANK Build independent hard-event SAC contexts.

projectRoot = fileparts(fileparts(mfilename('fullpath')));
add_project_paths(projectRoot);
cfg = targeted_lqr_weak_config();
divergenceCfg = targeted_lqr_divergence_config();
nmpcCfg = step2_nmpc_config();
disturbanceCfg = step3_disturbance_config();
validate_locked_contract(cfg, divergenceCfg);

outputRoot = fullfile(projectRoot, cfg.resultRoot, ...
    cfg.specialistBank.outputSubfolder);
assert(~isfolder(outputRoot), 'Refusing to overwrite %s.', outputRoot);
temporaryRoot = [char(outputRoot) '.tmp'];
if isfolder(temporaryRoot)
    rmdir(temporaryRoot, 's');
end
mkdir(temporaryRoot);
cleanup = onCleanup(@() cleanup_temporary(temporaryRoot));

clock = tic;
sourceCfg = cfg;
sourceCfg.stepCount = cfg.specialistBank.sourceStepCount;
[sourceBank, sourceManifest] = quad_build_balanced_targeted_bank( ...
    sourceCfg, disturbanceCfg, ...
    cfg.specialistBank.sourceReplicatesPerCell, ...
    cfg.specialistBank.sourceSeed);
assert(numel(sourceBank) == cfg.specialistBank.sourceEpisodeCount, ...
    'The independent teacher-training source bank has the wrong size.');

[lqr, lqrPath] = load_frozen_lqr(projectRoot, cfg);
episodeCount = numel(sourceBank);
hardChoices = repmat(empty_choice(), episodeCount, 1);
growthChoices = repmat(empty_choice(), episodeCount, 1);
safeChoices = repmat(empty_choice(), episodeCount, 1);
episodeRows = repmat(empty_episode_row(), episodeCount, 1);
maxStartStep = cfg.specialistBank.sourceStepCount + 1 - ...
    cfg.specialistBank.localEpisodeSteps - ...
    cfg.specialistBank.maximumPredictionHorizon;
assert(maxStartStep >= cfg.specialistBank.minimumStartStep, ...
    'The source trajectory cannot provide a 200+20-step continuation.');

for episodeIndex = 1:episodeCount
    rollout = run_lqr_source(sourceBank(episodeIndex), lqr, cfg);
    [hardChoices(episodeIndex), growthChoices(episodeIndex), ...
        safeChoices(episodeIndex), episodeRows(episodeIndex)] = ...
        scan_episode(rollout, sourceBank(episodeIndex), cfg, ...
        divergenceCfg, nmpcCfg, maxStartStep);
end
episodeSummary = struct2table(episodeRows);

[hardEpisodeIndices, cellSummary] = select_balanced_contexts( ...
    sourceBank, hardChoices, growthChoices, safeChoices, cfg);
growthEpisodeIndices = select_class_contexts(sourceBank, growthChoices, ...
    cfg.specialistBank.maximumGrowthContextsPerCell, 'risk');
safeEpisodeIndices = select_class_contexts(sourceBank, safeChoices, ...
    cfg.specialistBank.maximumSafeContextsPerCell, 'risk');

[trainContextBank, trainContextManifest] = materialize_contexts( ...
    sourceBank, hardChoices, hardEpisodeIndices, 'hard_event');
[auxiliaryGrowthContextBank, auxiliaryGrowthContextManifest] = ...
    materialize_contexts(sourceBank, growthChoices, ...
    growthEpisodeIndices, 'growth_only');
[safeBoundaryContextBank, safeBoundaryContextManifest] = ...
    materialize_contexts(sourceBank, safeChoices, ...
    safeEpisodeIndices, 'safe_boundary');

wallSeconds = toc(clock);
validate_outputs(sourceManifest, trainContextBank, ...
    trainContextManifest, auxiliaryGrowthContextBank, ...
    auxiliaryGrowthContextManifest, safeBoundaryContextBank, ...
    safeBoundaryContextManifest, cellSummary, cfg, divergenceCfg);
buildMetadata = make_metadata(cfg, divergenceCfg, lqrPath, ...
    sourceManifest, trainContextManifest, auxiliaryGrowthContextManifest, ...
    safeBoundaryContextManifest, maxStartStep, wallSeconds);

writetable(sourceManifest, fullfile(temporaryRoot, ...
    'teacher_train_source_manifest.csv'));
writetable(episodeSummary, fullfile(temporaryRoot, ...
    'episode_context_candidate_summary.csv'));
writetable(cellSummary, fullfile(temporaryRoot, ...
    'factorial_cell_context_summary.csv'));
writetable(trainContextManifest, fullfile(temporaryRoot, ...
    'hard_context_manifest.csv'));
writetable(auxiliaryGrowthContextManifest, fullfile(temporaryRoot, ...
    'growth_only_context_manifest.csv'));
writetable(safeBoundaryContextManifest, fullfile(temporaryRoot, ...
    'safe_boundary_context_manifest.csv'));
save(fullfile(temporaryRoot, 'training_local_context_bank.mat'), ...
    'trainContextBank', 'trainContextManifest', ...
    'auxiliaryGrowthContextBank', 'auxiliaryGrowthContextManifest', ...
    'safeBoundaryContextBank', 'safeBoundaryContextManifest', ...
    'episodeSummary', 'cellSummary', 'sourceManifest', ...
    'buildMetadata', 'cfg', 'divergenceCfg', 'lqr', '-v7.3');
write_report(temporaryRoot, buildMetadata, cellSummary, ...
    trainContextManifest, auxiliaryGrowthContextManifest, ...
    safeBoundaryContextManifest);
write_text(fullfile(temporaryRoot, 'COMPLETED.txt'), sprintf( ...
    ['task=build_targeted_c012_context_bank\nstatus=completed\n' ...
    'locked_candidate=C012\nsource_seed=%d\nsource_episodes=%d\n' ...
    'source_cells=%d\nhard_contexts=%d\nhard_positive_cells=%d\n' ...
    'growth_only_contexts=%d\nsafe_boundary_contexts=%d\n' ...
    'source_steps=%d\nsac_episode_steps=%d\nhorizon_steps=%d\n' ...
    'wall_seconds=%.6f\n'], cfg.specialistBank.sourceSeed, ...
    height(sourceManifest), numel(unique(sourceManifest.CellId)), ...
    height(trainContextManifest), ...
    numel(unique(trainContextManifest.CellId)), ...
    height(auxiliaryGrowthContextManifest), ...
    height(safeBoundaryContextManifest), ...
    cfg.specialistBank.sourceStepCount, ...
    cfg.specialistBank.localEpisodeSteps, divergenceCfg.horizonSteps, ...
    wallSeconds));
movefile(temporaryRoot, outputRoot);
clear cleanup;
end

function validate_locked_contract(cfg, divergenceCfg)
assert(cfg.predictiveAnalysis.finalThresholdsLocked && ...
    divergenceCfg.finalThresholdsLocked, ...
    'The context bank requires an approved frozen predictive rule.');
assert(strcmp(cfg.predictiveAnalysis.lockedCandidateId, 'C012') && ...
    strcmp(divergenceCfg.lockedCandidateId, 'C012'), ...
    'The context bank is defined only for approved candidate C012.');
assert(divergenceCfg.threshold.positionM == 0.10 && ...
    divergenceCfg.threshold.attitudeDeg == 5.0 && ...
    divergenceCfg.threshold.velocityMps == 0.30 && ...
    divergenceCfg.threshold.bodyRateRadps == 2.0 && ...
    divergenceCfg.growth.factor == 2.0 && ...
    divergenceCfg.growth.minimumConsecutiveSteps == 5, ...
    'The C012 numerical contract has drifted.');
assert(cfg.specialistBank.sourceSeed ~= cfg.split.screenSeed, ...
    'Teacher training contexts must be independent of threshold validation.');
assert(cfg.specialistBank.maximumPredictionHorizon == ...
    divergenceCfg.horizonSteps, 'Context and prediction horizons differ.');
end

function [lqr, path] = load_frozen_lqr(projectRoot, cfg)
root = strtrim(getenv('LQR_FROZEN_ROOT'));
if isempty(root)
    path = fullfile(projectRoot, ...
        cfg.screenBenchmark.provisionalLqrPath);
else
    path = fullfile(root, 'selected_lqr.mat');
end
assert(isfile(path), 'Frozen LQR artifact not found: %s.', path);
loaded = load(path, 'selectedLqr');
assert(isfield(loaded, 'selectedLqr'), ...
    'The frozen artifact does not contain selectedLqr.');
lqr = loaded.selectedLqr;
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
    'Saturated', saturated, 'Reference', scenario.Reference, ...
    'SampleTime', cfg.sampleTime, 'NominalPlant', cfg.plant.nominal);
end

function [hardChoice, growthChoice, safeChoice, row] = scan_episode( ...
        rollout, scenario, cfg, divergenceCfg, nmpcCfg, maxStartStep)
hardChoice = empty_choice();
growthChoice = empty_choice();
safeChoice = empty_choice();
hardCount = 0;
growthCount = 0;
safeCount = 0;
outsideCount = 0;
for step = cfg.specialistBank.minimumStartStep:maxStartStep
    if ~all(isfinite(rollout.X(:, step)))
        break;
    end
    risk = lqr_finite_horizon_divergence_label(rollout.X, ...
        rollout.RawU, rollout.Saturated, scenario.Reference, step, ...
        divergenceCfg, nmpcCfg);
    if risk.PreDivergenceEligible
        hardCount = hardCount + 1;
        hardChoice = update_choice(hardChoice, step, risk, 'hard', rollout);
    elseif risk.GrowthOnlyEligible
        growthCount = growthCount + 1;
        growthChoice = update_choice(growthChoice, step, risk, 'risk', rollout);
    elseif risk.AlreadyOutsideEnvelope
        outsideCount = outsideCount + 1;
    else
        safeCount = safeCount + 1;
        safeChoice = update_choice(safeChoice, step, risk, 'risk', rollout);
    end
end
row = empty_episode_row();
row.SourceEpisodeIndex = scenario.EpisodeIndex;
row.CellId = string(scenario.CellId);
row.Family = string(scenario.Family);
row.DisturbanceType = string(scenario.Disturbance.type);
row.DisturbanceLevel = scenario.Disturbance.levelIndex;
row.UncertaintyStratum = string(scenario.UncertaintyStratum);
row.HardCandidateCount = hardCount;
row.GrowthOnlyCandidateCount = growthCount;
row.SafeCandidateCount = safeCount;
row.AlreadyOutsideCount = outsideCount;
row.HasHardContext = hardChoice.Exists;
row.HardStartStep = hardChoice.StartStep;
row.HardLeadSteps = choice_value(hardChoice, 'TimeToEventSteps');
row.HardRiskScore = choice_value(hardChoice, 'RiskScore');
row.HasGrowthOnlyContext = growthChoice.Exists;
row.GrowthOnlyStartStep = growthChoice.StartStep;
row.GrowthOnlyRiskScore = choice_value(growthChoice, 'RiskScore');
row.HasSafeBoundaryContext = safeChoice.Exists;
row.SafeBoundaryStartStep = safeChoice.StartStep;
row.SafeBoundaryRiskScore = choice_value(safeChoice, 'RiskScore');
end

function choice = update_choice(choice, step, risk, mode, rollout)
replace = ~choice.Exists;
if choice.Exists && strcmp(mode, 'hard')
    replace = risk.TimeToEventSteps > choice.Risk.TimeToEventSteps || ...
        (risk.TimeToEventSteps == choice.Risk.TimeToEventSteps && ...
        risk.RiskScore > choice.Risk.RiskScore);
elseif choice.Exists
    replace = risk.RiskScore > choice.Risk.RiskScore;
end
if replace
    choice.Exists = true;
    choice.StartStep = step;
    choice.Risk = risk;
    choice.State = rollout.X(:, step);
    choice.StateHistory = rollout.X(:, step - 3:step);
    choice.InputHistory = rollout.U(:, step - 4:step - 1);
    [~, featureParts] = targeted_causal_feature_from_trajectory( ...
        rollout.X, rollout.U, rollout.Reference, step, ...
        rollout.SampleTime, rollout.NominalPlant);
    choice.PredictionResidual = featureParts.PredictionResidual;
    if step > 1
        choice.PreviousInput = rollout.U(:, step - 1);
    else
        choice.PreviousInput = rollout.U(:, step);
    end
end
end

function value = choice_value(choice, field)
if choice.Exists
    value = choice.Risk.(field);
else
    value = NaN;
end
end

function [hardIndices, summary] = select_balanced_contexts( ...
        bank, hardChoices, growthChoices, safeChoices, cfg)
hardIndices = select_class_contexts(bank, hardChoices, ...
    cfg.specialistBank.maximumHardContextsPerCell, 'hard');
cellIds = unique(string({bank.CellId}).', 'stable');
rows = repmat(empty_cell_row(), numel(cellIds), 1);
hardSelected = false(numel(bank), 1);
hardSelected(hardIndices) = true;
for index = 1:numel(cellIds)
    inCell = string({bank.CellId}).' == cellIds(index);
    rows(index).CellId = cellIds(index);
    rows(index).SourceEpisodeCount = nnz(inCell);
    rows(index).HardPositiveEpisodeCount = nnz(inCell & [hardChoices.Exists].');
    rows(index).SelectedHardContextCount = nnz(inCell & hardSelected);
    rows(index).GrowthOnlyEpisodeCount = ...
        nnz(inCell & [growthChoices.Exists].');
    rows(index).SafeBoundaryEpisodeCount = ...
        nnz(inCell & [safeChoices.Exists].');
end
summary = struct2table(rows);
end

function selected = select_class_contexts(bank, choices, cap, mode)
cellIds = unique(string({bank.CellId}).', 'stable');
selected = zeros(0, 1);
for cellIndex = 1:numel(cellIds)
    indices = find(string({bank.CellId}).' == cellIds(cellIndex) & ...
        [choices.Exists].');
    if isempty(indices)
        continue;
    end
    indices = indices(:);
    risk = arrayfun(@(index) choices(index).Risk.RiskScore, indices);
    risk = risk(:);
    lead = arrayfun(@(index) ...
        finite_lead(choices(index).Risk.TimeToEventSteps), indices);
    lead = lead(:);
    ranking = table(indices, lead, risk, ...
        'VariableNames', {'Index', 'Lead', 'Risk'});
    if strcmp(mode, 'hard')
        ranking = sortrows(ranking, {'Lead', 'Risk', 'Index'}, ...
            {'descend', 'descend', 'ascend'});
    else
        ranking = sortrows(ranking, {'Risk', 'Index'}, ...
            {'descend', 'ascend'});
    end
    selected = [selected; ranking.Index(1:min(cap, height(ranking)))]; ...
        %#ok<AGROW>
end
end

function value = finite_lead(value)
if ~isfinite(value)
    value = -1;
end
end

function [contexts, manifest] = materialize_contexts( ...
        sourceBank, choices, episodeIndices, className)
assert(~isempty(episodeIndices), 'No %s contexts were selected.', className);
contexts = repmat(attach_context(sourceBank(episodeIndices(1)), ...
    choices(episodeIndices(1)), className), numel(episodeIndices), 1);
rows = repmat(empty_context_row(), numel(episodeIndices), 1);
for index = 1:numel(episodeIndices)
    episodeIndex = episodeIndices(index);
    contexts(index) = attach_context(sourceBank(episodeIndex), ...
        choices(episodeIndex), className);
    contexts(index).ContextIndex = index;
    rows(index) = context_row(index, contexts(index));
end
manifest = struct2table(rows);
end

function context = attach_context(scenario, choice, className)
assert(choice.Exists, 'Cannot materialize a missing context.');
risk = choice.Risk;
context = scenario;
context.ContextIndex = 0;
context.SourceEpisodeIndex = scenario.EpisodeIndex;
context.SourceX0 = scenario.X0;
context.ContextClass = className;
context.StartStep = choice.StartStep;
context.InterventionStep = choice.StartStep;
context.X0 = choice.State;
context.PreviousInput = choice.PreviousInput;
context.StateHistory = choice.StateHistory;
context.InputHistory = choice.InputHistory;
context.PredictionResidual = choice.PredictionResidual;
context.FeatureVersion = 'causal_feature_v2';
context.DivergenceEventStep = risk.EventStep;
context.InterventionLeadSteps = risk.TimeToEventSteps;
context.CounterfactualDivergence = risk.CounterfactualDivergence;
context.HardEvent = risk.HardEvent;
context.GrowthOnly = risk.GrowthOnly;
context.AlreadyOutsideEnvelope = risk.AlreadyOutsideEnvelope;
context.PreDivergenceEligible = risk.PreDivergenceEligible;
context.LqrRiskScore = risk.RiskScore;
context.LqrLogEnergyGrowthPerStep = risk.LogEnergyGrowthPerStep;
context.Label = risk;
end

function row = context_row(index, context)
risk = context.Label;
row = empty_context_row();
row.ContextIndex = index;
row.SourceEpisodeIndex = context.SourceEpisodeIndex;
row.ContextClass = string(context.ContextClass);
row.CellId = string(context.CellId);
row.Family = string(context.Family);
row.DisturbanceType = string(context.Disturbance.type);
row.DisturbanceLevel = context.Disturbance.levelIndex;
row.UncertaintyStratum = string(context.UncertaintyStratum);
row.TargetReferenceSpeed = context.TargetReferenceSpeed;
row.TargetReferenceAcceleration = context.TargetReferenceAcceleration;
row.StartStep = context.StartStep;
row.EventStep = risk.EventStep;
row.LeadSteps = risk.TimeToEventSteps;
row.HardEvent = risk.HardEvent;
row.GrowthOnly = risk.GrowthOnly;
row.PreDivergenceEligible = risk.PreDivergenceEligible;
row.RiskScore = risk.RiskScore;
row.MaximumPositionErrorM = risk.MaximumPositionErrorM;
row.MaximumAttitudeErrorDeg = risk.MaximumAttitudeErrorDeg;
row.MaximumVelocityErrorMps = risk.MaximumVelocityErrorMps;
row.MaximumBodyRateErrorRadps = risk.MaximumBodyRateErrorRadps;
row.SaturationForecast = risk.SaturationForecast;
row.ConstraintForecast = risk.ConstraintForecast;
row.NonfiniteForecast = risk.NonfiniteForecast;
end

function validate_outputs(sourceManifest, trainBank, trainManifest, ...
        growthBank, growthManifest, safeBank, safeManifest, cellSummary, ...
        cfg, divergenceCfg)
assert(height(sourceManifest) == cfg.specialistBank.sourceEpisodeCount && ...
    numel(unique(sourceManifest.CellId)) == 135, ...
    'Source bank must contain 2,700 episodes over 135 cells.');
assert(numel(trainBank) == height(trainManifest) && ...
    all(trainManifest.HardEvent) && ...
    all(trainManifest.PreDivergenceEligible) && ...
    ~any(trainManifest.GrowthOnly), ...
    'SAC bank must contain hard pre-event contexts only.');
assert(all(trainManifest.LeadSteps >= 1 & ...
    trainManifest.LeadSteps <= divergenceCfg.horizonSteps), ...
    'Every SAC context must lead a hard event by 1..H steps.');
assert(numel(growthBank) == height(growthManifest) && ...
    all(growthManifest.GrowthOnly) && ~any(growthManifest.HardEvent), ...
    'Growth-only contexts leaked into the hard target.');
assert(numel(safeBank) == height(safeManifest) && ...
    ~any(safeManifest.HardEvent) && ~any(safeManifest.GrowthOnly), ...
    'Safe-boundary contexts have an invalid label.');
hardCounts = groupcounts(trainManifest.CellId);
assert(all(hardCounts <= ...
    cfg.specialistBank.maximumHardContextsPerCell), ...
    'A hard-positive cell exceeds its context cap.');
assert(height(cellSummary) == 135 && ...
    nnz(cellSummary.SelectedHardContextCount > 0) == ...
    numel(unique(trainManifest.CellId)), ...
    'Hard-positive cell accounting is inconsistent.');
for index = 1:numel(trainBank)
    required = trainBank(index).StartStep + ...
        cfg.specialistBank.localEpisodeSteps + ...
        cfg.specialistBank.maximumPredictionHorizon;
    assert(required <= size(trainBank(index).Reference, 2), ...
        'A SAC context lacks its 200+20 reference continuation.');
    assert(isequal(size(trainBank(index).StateHistory), [12, 4]) && ...
        isequal(size(trainBank(index).InputHistory), [4, 4]) && ...
        isequal(size(trainBank(index).PredictionResidual), [12, 1]) && ...
        all(isfinite(trainBank(index).StateHistory), 'all') && ...
        all(isfinite(trainBank(index).InputHistory), 'all') && ...
        all(isfinite(trainBank(index).PredictionResidual)), ...
        'A retained context lacks finite causal feature history.');
end
end

function metadata = make_metadata(cfg, divergenceCfg, lqrPath, ...
        sourceManifest, hardManifest, growthManifest, safeManifest, ...
        maxStartStep, wallSeconds)
metadata = struct();
metadata.status = 'frozen_c012_independent_teacher_training_context_bank';
metadata.featureVersion = cfg.specialistBank.featureVersion;
metadata.stateHistoryLength = cfg.specialistBank.stateHistoryLength;
metadata.appliedInputHistoryLength = ...
    cfg.specialistBank.appliedInputHistoryLength;
metadata.lockedCandidateId = divergenceCfg.lockedCandidateId;
metadata.sourceSeed = cfg.specialistBank.sourceSeed;
metadata.validationSeed = cfg.split.screenSeed;
metadata.sourceEpisodeCount = height(sourceManifest);
metadata.sourceCellCount = numel(unique(sourceManifest.CellId));
metadata.hardContextCount = height(hardManifest);
metadata.hardPositiveCellCount = numel(unique(hardManifest.CellId));
metadata.growthOnlyContextCount = height(growthManifest);
metadata.safeBoundaryContextCount = height(safeManifest);
metadata.maximumStartStep = maxStartStep;
metadata.lqrPath = lqrPath;
metadata.selectionRule = cfg.specialistBank.selectionRule;
metadata.wallSeconds = wallSeconds;
metadata.builtAt = char(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd HH:mm:ssXXX'));
end

function write_report(root, metadata, cells, hard, growth, safe)
path = fullfile(root, 'c012_context_bank_report.md');
fileId = fopen(path, 'w');
assert(fileId >= 0, 'Cannot open %s.', path);
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, '# C012 independent LQR-weak context bank\n\n');
fprintf(fileId, '- Threshold: `C012` (locked on validation).\n');
fprintf(fileId, '- Source seed: `%d`; validation seed: `%d`.\n', ...
    metadata.sourceSeed, metadata.validationSeed);
fprintf(fileId, '- Source: %d episodes over %d factorial cells.\n', ...
    metadata.sourceEpisodeCount, metadata.sourceCellCount);
fprintf(fileId, '- Hard SAC contexts: %d over %d positive cells.\n', ...
    height(hard), metadata.hardPositiveCellCount);
fprintf(fileId, '- Growth-only auxiliary contexts: %d.\n', height(growth));
fprintf(fileId, '- Safe-boundary audit contexts: %d.\n', height(safe));
fprintf(fileId, '- Cells without a hard training context: %d.\n', ...
    nnz(cells.SelectedHardContextCount == 0));
fprintf(fileId, '- Build wall time: %.6f s.\n', metadata.wallSeconds);
fprintf(fileId, '- Causal feature contract: `%s`.\n', ...
    metadata.featureVersion);
fprintf(fileId, ['\nOnly hard pre-event contexts enter SAC. Growth-only rows ' ...
    'remain a separate auxiliary boundary stratum and safe rows remain ' ...
    'audit-only. The independent teacher-training seed prevents reuse of ' ...
    'the LQR threshold-selection episodes. No NMPC or SAC outcome is used ' ...
    'to define these labels.\n']);
clear cleanup;
end

function choice = empty_choice()
choice = struct('Exists', false, 'StartStep', 0, ...
    'Risk', empty_risk(), 'State', nan(12, 1), ...
    'PreviousInput', nan(4, 1), 'StateHistory', nan(12, 4), ...
    'InputHistory', nan(4, 4), 'PredictionResidual', nan(12, 1));
end

function risk = empty_risk()
risk = struct('CounterfactualDivergence', false, 'HardEvent', false, ...
    'GrowthOnly', false, 'AlreadyOutsideEnvelope', false, ...
    'PreDivergenceEligible', false, 'GrowthOnlyEligible', false, ...
    'TimeToEventSteps', Inf, 'EventStep', Inf, 'RiskScore', -Inf, ...
    'LogEnergyGrowthPerStep', NaN, 'EnergyStart', NaN, ...
    'EnergyEnd', NaN, 'EnergyGrowthFactor', NaN, ...
    'MaximumConsecutiveGrowthSteps', 0, ...
    'MaximumPositionErrorM', NaN, 'MaximumAttitudeErrorDeg', NaN, ...
    'MaximumVelocityErrorMps', NaN, ...
    'MaximumBodyRateErrorRadps', NaN, ...
    'AbsoluteThresholdCrossing', false, 'SustainedGrowth', false, ...
    'SaturationForecast', false, 'ConstraintForecast', false, ...
    'NonfiniteForecast', false, ...
    'MinimumNormalizedActuatorMargin', NaN);
end

function row = empty_episode_row()
row = struct('SourceEpisodeIndex', 0, 'CellId', "", 'Family', "", ...
    'DisturbanceType', "", 'DisturbanceLevel', 0, ...
    'UncertaintyStratum', "", 'HardCandidateCount', 0, ...
    'GrowthOnlyCandidateCount', 0, 'SafeCandidateCount', 0, ...
    'AlreadyOutsideCount', 0, 'HasHardContext', false, ...
    'HardStartStep', 0, 'HardLeadSteps', NaN, 'HardRiskScore', NaN, ...
    'HasGrowthOnlyContext', false, 'GrowthOnlyStartStep', 0, ...
    'GrowthOnlyRiskScore', NaN, 'HasSafeBoundaryContext', false, ...
    'SafeBoundaryStartStep', 0, 'SafeBoundaryRiskScore', NaN);
end

function row = empty_cell_row()
row = struct('CellId', "", 'SourceEpisodeCount', 0, ...
    'HardPositiveEpisodeCount', 0, 'SelectedHardContextCount', 0, ...
    'GrowthOnlyEpisodeCount', 0, 'SafeBoundaryEpisodeCount', 0);
end

function row = empty_context_row()
row = struct('ContextIndex', 0, 'SourceEpisodeIndex', 0, ...
    'ContextClass', "", 'CellId', "", 'Family', "", ...
    'DisturbanceType', "", 'DisturbanceLevel', 0, ...
    'UncertaintyStratum', "", 'TargetReferenceSpeed', NaN, ...
    'TargetReferenceAcceleration', NaN, 'StartStep', 0, ...
    'EventStep', Inf, 'LeadSteps', Inf, 'HardEvent', false, ...
    'GrowthOnly', false, 'PreDivergenceEligible', false, ...
    'RiskScore', NaN, 'MaximumPositionErrorM', NaN, ...
    'MaximumAttitudeErrorDeg', NaN, 'MaximumVelocityErrorMps', NaN, ...
    'MaximumBodyRateErrorRadps', NaN, 'SaturationForecast', false, ...
    'ConstraintForecast', false, 'NonfiniteForecast', false);
end

function cleanup_temporary(path)
if isfolder(path)
    rmdir(path, 's');
end
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
addpath(genpath(fullfile(projectRoot, 'src')));
end
