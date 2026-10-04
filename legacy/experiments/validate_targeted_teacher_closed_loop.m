function audit = validate_targeted_teacher_closed_loop(mode)
%VALIDATE_TARGETED_TEACHER_CLOSED_LOOP Paired H=20 NMPC capability gate.

if nargin < 1
    mode = 'run';
end
mode = string(mode);
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot, 'configs'));
addpath(genpath(fullfile(projectRoot, 'src')));
cfg = targeted_teacher_capability_config();
wallLimit = environment_positive_number('NMPC_MAX_WALL_SECONDS', ...
    cfg.defaultWallLimitSeconds);
requestedWorkers = round(environment_positive_number( ...
    'PARALLEL_WORKERS', cfg.defaultWorkerCount));

bankPath = fullfile(projectRoot, cfg.sac.trainingBankPath);
assert(isfile(bankPath), 'The causal-v2 context bank must be built first.');
loaded = load(bankPath, 'trainContextBank', 'trainContextManifest', ...
    'buildMetadata');
[contexts, selection] = select_contexts(loaded.trainContextBank, ...
    loaded.trainContextManifest, cfg);
[lqr, lqrPath] = load_frozen_lqr(projectRoot, cfg.targeted);
[lqrBaselines, lqrTable] = evaluate_lqr_baselines( ...
    contexts, lqr, cfg);
cases = build_cases(selection, cfg);
caseTable = struct2table(cases);
assert(numel(cases) == 5 * 3 * 3 * 2, ...
    'The capability gate must contain exactly 90 paired rollout tasks.');
audit = struct('ProtocolVersion', cfg.version, 'Mode', mode, ...
    'ContextCount', numel(contexts), 'FamilyCount', ...
    numel(unique(selection.Family)), 'LeadRankStratumCount', ...
    numel(unique(selection.LeadRankStratum)), 'CaseCount', numel(cases), ...
    'RolloutSteps', cfg.rolloutSteps, 'LqrEventCount', ...
    nnz(lqrTable.LeadSteps >= 1), 'FullSacPairGridRetained', ...
    cfg.fullSacPairGridRetained);
if mode == "contract_only"
    return;
end
if mode == "single_case_probe"
    probe = execute_case(cases(1), contexts, lqrBaselines, cfg, wallLimit);
    audit.ProbeRow = probe.Row;
    audit.ProbeTrace = probe.Trace;
    return;
end
assert(mode == "run", 'Unknown execution mode: %s.', mode);

bankSha256 = quad_sha256_file(bankPath);
lqrSha256 = quad_sha256_file(lqrPath);

attemptId = environment_text('TEACHER_ATTEMPT_ID', ...
    char(datetime('now', 'Format', 'yyyyMMdd_HHmmss')));
runRoot = fullfile(projectRoot, cfg.outputRoot, ...
    ['run_' sanitize_identifier(attemptId)]);
assert(~isfolder(runRoot), 'Refusing to overwrite %s.', runRoot);
taskRoot = fullfile(runRoot, 'tasks');
mkdir(taskRoot);
write_text(fullfile(runRoot, 'RUNNING.txt'), sprintf( ...
    ['status=running\nprotocol_version=%s\ncase_count=%d\n' ...
    'rollout_steps=%d\n'], cfg.version, numel(cases), cfg.rolloutSteps));
save(fullfile(runRoot, 'resolved_config.mat'), 'cfg', 'wallLimit', ...
    'requestedWorkers', 'lqrPath', 'bankSha256', 'lqrSha256', '-v7.3');
buildMetadata = loaded.buildMetadata;
save(fullfile(runRoot, 'selected_contexts.mat'), 'contexts', 'selection', ...
    'lqrBaselines', 'lqrTable', 'buildMetadata', '-v7.3');
writetable(selection, fullfile(runRoot, 'selected_contexts.csv'));
writetable(lqrTable, fullfile(runRoot, 'lqr_baseline_measurements.csv'));
writetable(caseTable, fullfile(runRoot, 'case_manifest.csv'));

resumeRoot = strtrim(getenv('TEACHER_RESUME_ROOT'));
resumeCount = import_resume_tasks( ...
    resumeRoot, taskRoot, cases, cfg.version);
pending = find(~arrayfun(@(item) isfile(task_path(taskRoot, ...
    item.CaseIndex)), cases));
pendingCases = cases(pending);

pool = gcp('nocreate');
if ~isempty(pool)
    delete(pool);
end
cluster = parcluster('Processes');
workerCount = min(requestedWorkers, cluster.NumWorkers);
pool = parpool(cluster, workerCount);
poolCleanup = onCleanup(@() delete_pool(pool));
clock = tic;
parfor pendingIndex = 1:numel(pendingCases)
    result = execute_case(pendingCases(pendingIndex), contexts, ...
        lqrBaselines, cfg, wallLimit);
    save_task_atomic(taskRoot, result);
end
wallSeconds = toc(clock);
clear poolCleanup;
delete_pool(pool);

[measurements, traces] = load_results(taskRoot, cases, cfg.version);
summary = summarize_results(measurements);
writetable(measurements, fullfile(runRoot, ...
    'teacher_closed_loop_measurements.csv'));
writetable(summary, fullfile(runRoot, ...
    'teacher_closed_loop_summary.csv'));
save(fullfile(runRoot, 'teacher_closed_loop_results.mat'), ...
    'measurements', 'summary', 'traces', 'lqrBaselines', ...
    'selection', 'caseTable', 'cfg', '-v7.3');
metadata = struct('ProtocolVersion', cfg.version, ...
    'CaseCount', height(measurements), 'RolloutSteps', cfg.rolloutSteps, ...
    'WorkerCount', workerCount, 'ResumedTaskCount', resumeCount, ...
    'NewTaskCount', numel(pending), 'WallLimitSeconds', wallLimit, ...
    'WallSeconds', wallSeconds, 'ContextSourceSeed', ...
    loaded.buildMetadata.sourceSeed, 'LqrPath', lqrPath);
metadata.ContextBankSha256 = bankSha256;
metadata.LqrSha256 = lqrSha256;
save(fullfile(runRoot, 'run_metadata.mat'), 'metadata');
write_report(runRoot, measurements, summary, metadata, cfg);
write_hashes(runRoot, {'resolved_config.mat', 'selected_contexts.csv', ...
    'lqr_baseline_measurements.csv', 'case_manifest.csv', ...
    'teacher_closed_loop_measurements.csv', ...
    'teacher_closed_loop_summary.csv', ...
    'teacher_closed_loop_results.mat', 'run_metadata.mat', ...
    'capability_report.md'});
delete(fullfile(runRoot, 'RUNNING.txt'));
write_text(fullfile(runRoot, 'COMPLETED.txt'), sprintf( ...
    ['status=completed\nprotocol_version=%s\ncases=%d\n' ...
    'rollout_steps=%d\nworkers=%d\nresumed_tasks=%d\n' ...
    'new_tasks=%d\nwall_seconds=%.6f\n'], cfg.version, ...
    height(measurements), cfg.rolloutSteps, workerCount, resumeCount, ...
    numel(pending), wallSeconds));
end

function [contexts, selection] = select_contexts(bank, manifest, cfg)
familyOrder = string(cfg.targeted.reference.families(:));
strata = string(cfg.context.stratumNames(:));
indices = zeros(numel(familyOrder) * numel(strata), 1);
stratumColumn = strings(size(indices));
cursor = 0;
for familyIndex = 1:numel(familyOrder)
    candidates = find(manifest.Family == familyOrder(familyIndex));
    assert(numel(candidates) >= 2 * numel(strata), ...
        'Insufficient hard contexts for family %s.', familyOrder(familyIndex));
    subset = manifest(candidates, :);
    [~, order] = sortrows([subset.LeadSteps, subset.RiskScore], [1, 2]);
    candidates = candidates(order);
    edges = round(linspace(0, numel(candidates), numel(strata) + 1));
    for stratumIndex = 1:numel(strata)
        segment = edges(stratumIndex) + 1:edges(stratumIndex + 1);
        selectedRank = segment(ceil(numel(segment) / 2));
        cursor = cursor + 1;
        indices(cursor) = candidates(selectedRank);
        stratumColumn(cursor) = strata(stratumIndex);
    end
end
assert(numel(unique(indices)) == numel(indices), ...
    'Context selection must not contain duplicates.');
contexts = bank(indices);
selection = manifest(indices, :);
selection.LocalContextIndex = (1:height(selection)).';
selection.LeadRankStratum = stratumColumn;
selection = movevars(selection, {'LocalContextIndex', ...
    'LeadRankStratum'}, 'Before', 1);
end

function [lqr, path] = load_frozen_lqr(projectRoot, cfg)
path = fullfile(projectRoot, cfg.screenBenchmark.provisionalLqrPath);
assert(isfile(path), 'Frozen LQR artifact not found: %s.', path);
loaded = load(path, 'selectedLqr');
assert(isfield(loaded, 'selectedLqr'), ...
    'Frozen LQR artifact does not contain selectedLqr.');
lqr = loaded.selectedLqr;
end

function [baselines, tableOut] = evaluate_lqr_baselines(contexts, lqr, cfg)
baselines = repmat(empty_baseline(), numel(contexts), 1);
rows = repmat(empty_lqr_row(), numel(contexts), 1);
for index = 1:numel(contexts)
    baselines(index) = run_lqr(contexts(index), lqr, cfg);
    label = baselines(index).Label;
    assert(label.HardEvent && ~label.AlreadyOutsideEnvelope, ...
        'Selected LQR context no longer reproduces a future hard event.');
    assert(label.TimeToEventSteps == contexts(index).InterventionLeadSteps, ...
        'Recomputed LQR lead time differs from bank provenance.');
    rows(index).LocalContextIndex = index;
    rows(index).SourceEpisodeIndex = contexts(index).SourceEpisodeIndex;
    rows(index).Family = string(contexts(index).Family);
    rows(index).LeadSteps = label.TimeToEventSteps;
    rows(index).FirstEventType = label.FirstHardEventType;
    rows(index).PeakNormalizedError = baselines(index).PeakNormalizedError;
    rows(index).SaturationForecast = label.SaturationForecast;
    rows(index).ConstraintForecast = label.ConstraintForecast;
end
tableOut = struct2table(rows);
end

function baseline = run_lqr(context, lqr, cfg)
H = cfg.rolloutSteps;
X = nan(12, H + 1);
U = nan(4, H);
rawU = nan(4, H);
saturated = false(1, H);
X(:, 1) = context.X0;
for step = 1:H
    absoluteStep = context.StartStep + step - 1;
    reference = context.Reference(:, absoluteStep);
    error = nmpc_state_error(X(:, step), reference);
    rawU(:, step) = context.Feedforward(:, absoluteStep) - lqr.K * error;
    U(:, step) = quad_saturate_input(rawU(:, step), ...
        cfg.targeted.plant.nominal);
    saturated(step) = any(abs(U(:, step) - rawU(:, step)) > 1e-10);
    time = (absoluteStep - 1) * cfg.targeted.sampleTime;
    X(:, step + 1) = quad_step_rk4(time, X(:, step), U(:, step), ...
        cfg.targeted.sampleTime, context.ThetaPlant, context.Disturbance);
end
reference = context.Reference(:, ...
    context.StartStep:context.StartStep + H);
label = lqr_finite_horizon_divergence_label(X, rawU, saturated, ...
    reference, 1, cfg.divergence, cfg.sac.nmpc);
baseline = struct('X', X, 'U', U, 'RawU', rawU, ...
    'Saturated', saturated, 'Reference', reference, 'Label', label, ...
    'PeakNormalizedError', peak_normalized_error(X, reference, ...
    cfg.divergence));
end

function cases = build_cases(selection, cfg)
count = height(selection) * numel(cfg.protocols) * numel(cfg.algorithms);
cases = repmat(empty_case(), count, 1);
cursor = 0;
for contextIndex = 1:height(selection)
    for protocolIndex = 1:numel(cfg.protocols)
        for algorithmIndex = 1:numel(cfg.algorithms)
            cursor = cursor + 1;
            protocol = cfg.protocols(protocolIndex);
            cases(cursor).CaseIndex = cursor;
            cases(cursor).CaseKey = string(sprintf('C%02d_P%02d_A%02d', ...
                contextIndex, protocolIndex, algorithmIndex));
            cases(cursor).ContextIndex = contextIndex;
            cases(cursor).SourceEpisodeIndex = ...
                selection.SourceEpisodeIndex(contextIndex);
            cases(cursor).Family = selection.Family(contextIndex);
            cases(cursor).LeadRankStratum = ...
                selection.LeadRankStratum(contextIndex);
            cases(cursor).LqrLeadSteps = selection.LeadSteps(contextIndex);
            cases(cursor).ProtocolIndex = protocolIndex;
            cases(cursor).Protocol = string(protocol.Name);
            cases(cursor).AlgorithmIndex = algorithmIndex;
            cases(cursor).Algorithm = string(cfg.algorithms{algorithmIndex});
            cases(cursor).N = protocol.N;
            cases(cursor).Nc = protocol.Nc;
            cases(cursor).WeightAction = protocol.WeightAction;
        end
    end
end
end

function result = execute_case(spec, contexts, baselines, cfg, wallLimit)
row = empty_result();
row = copy_case_fields(row, spec);
row.M = cfg.sac.nmpc.scenario.count;
context = contexts(spec.ContextIndex);
baseline = baselines(spec.ContextIndex);
trace = initialize_trace(cfg.rolloutSteps);
try
    action = zeros(cfg.sac.action.dimension, 1);
    action(1:6) = spec.WeightAction;
    [nmpcCfg, mapping] = rl_nmpc_action_to_config(action, cfg.sac);
    nmpcCfg.predictionHorizon = spec.N;
    nmpcCfg.controlHorizon = spec.Nc;
    nmpcCfg.solver.algorithm = char(spec.Algorithm);
    nmpcCfg.solver.maxWallSeconds = wallLimit;
    nmpcCfg.scenario.count = 5;
    nmpcCfg.rollout.disturbance = [];
    acceptanceCfg = cfg.sac;
    acceptanceCfg.environment.solverAcceptancePolicy = cfg.acceptancePolicy;
    scenarioSeed = 300833000 + 100 * spec.ContextIndex + ...
        spec.ProtocolIndex;
    row.ScenarioSeed = scenarioSeed;
    scenarios = quad_sample_uncertainty(nmpcCfg.plant, 5, 'targeted', ...
        scenarioSeed, nmpcCfg.scenario.method);
    trace.X(:, 1) = context.X0;
    previousInput = context.PreviousInput;
    warmStart = [];
    attempted = 0;
    completed = 0;
    for step = 1:cfg.rolloutSteps
        attempted = step;
        absoluteStep = context.StartStep + step - 1;
        time = (absoluteStep - 1) * cfg.targeted.sampleTime;
        reference = nmpc_reference_window(context.Reference, ...
            absoluteStep, time, nmpcCfg);
        solution = scenario_nmpc_solve(trace.X(:, step), reference, ...
            scenarios, nmpcCfg, warmStart, previousInput);
        [accepted, reason] = targeted_nmpc_accept_solution( ...
            solution, acceptanceCfg);
        strictAccepted = solution.converged && solution.feasible && ...
            ~solution.timedOut;
        trace.Accepted(step) = accepted;
        trace.StrictAccepted(step) = strictAccepted;
        trace.AcceptanceReason(step) = reason;
        trace.Exitflag(step) = solution.exitflag;
        trace.Converged(step) = solution.converged;
        trace.Feasible(step) = solution.feasible;
        trace.FeasibleSuboptimal(step) = solution.feasibleSuboptimal;
        trace.TimedOut(step) = solution.timedOut;
        trace.SolveSeconds(step) = solution.solveTime;
        trace.MaximumConstraintViolation(step) = ...
            solution.maxConstraintViolation;
        trace.RelativeCostImprovement(step) = relative_improvement( ...
            solution.warmStartCost, solution.cost);
        if ~accepted
            break;
        end
        trace.RawU(:, step) = solution.u0;
        trace.U(:, step) = quad_saturate_input( ...
            solution.u0, cfg.targeted.plant.nominal);
        trace.Saturated(step) = any(abs( ...
            trace.RawU(:, step) - trace.U(:, step)) > 1e-10);
        trace.X(:, step + 1) = quad_step_rk4(time, trace.X(:, step), ...
            trace.U(:, step), cfg.targeted.sampleTime, ...
            context.ThetaPlant, context.Disturbance);
        previousInput = trace.U(:, step);
        warmStart = nmpc_shift_sequence(solution.U, solution.U(:, end));
        completed = step;
        if ~all(isfinite(trace.X(:, step + 1)))
            break;
        end
    end
    trace.Reference = context.Reference(:, ...
        context.StartStep:context.StartStep + cfg.rolloutSteps);
    label = lqr_finite_horizon_divergence_label(trace.X, trace.RawU, ...
        trace.Saturated, trace.Reference, 1, cfg.divergence, nmpcCfg);
    row.AttemptedSteps = attempted;
    row.CompletedSteps = completed;
    row.AllCandidateAccepted = completed == cfg.rolloutSteps && ...
        all(trace.Accepted);
    row.AllStrictAccepted = row.AllCandidateAccepted && ...
        all(trace.StrictAccepted);
    row.ConvergedStepFraction = mean(trace.Converged(1:attempted));
    row.FeasibleSuboptimalStepFraction = ...
        mean(trace.FeasibleSuboptimal(1:attempted));
    row.TimedOutAny = any(trace.TimedOut(1:attempted));
    row.MinimumRelativeCostImprovement = min( ...
        trace.RelativeCostImprovement(1:attempted));
    row.MeanSolveSeconds = mean(trace.SolveSeconds(1:attempted));
    row.MedianSolveSeconds = median(trace.SolveSeconds(1:attempted));
    row.MaximumSolveSeconds = max(trace.SolveSeconds(1:attempted));
    row.TeacherHardEvent = label.HardEvent;
    row.TeacherFirstEventType = label.FirstHardEventType;
    row.TeacherTrackingEvent = label.AbsoluteThresholdCrossing;
    row.TeacherConstraintEvent = label.ConstraintForecast;
    row.TeacherSaturationEvent = label.SaturationForecast;
    row.EventAvoided = row.AllCandidateAccepted && ...
        baseline.Label.HardEvent && ~label.HardEvent;
    row.TrackingEventAvoided = row.AllCandidateAccepted && ...
        baseline.Label.AbsoluteThresholdCrossing && ...
        ~label.AbsoluteThresholdCrossing;
    row.TeacherPeakNormalizedError = peak_normalized_error( ...
        trace.X, trace.Reference, cfg.divergence);
    row.LqrPeakNormalizedError = baseline.PeakNormalizedError;
    row.PeakErrorRatio = row.TeacherPeakNormalizedError / ...
        max(row.LqrPeakNormalizedError, eps);
    row.QualityUsable = row.EventAvoided && ...
        row.MinimumRelativeCostImprovement > 0 && ...
        ~row.TeacherConstraintEvent;
    row.MinimumGroupWeight = min(mapping.groupWeights);
    row.MaximumGroupWeight = max(mapping.groupWeights);
catch exception
    row.ErrorIdentifier = string(exception.identifier);
    row.ErrorMessage = string(strrep(exception.message, newline, ' '));
end
result = struct('ProtocolVersion', cfg.version, ...
    'CaseKey', spec.CaseKey, 'Row', row, 'Trace', trace);
end

function trace = initialize_trace(stepCount)
trace = struct('X', nan(12, stepCount + 1), ...
    'U', nan(4, stepCount), 'RawU', nan(4, stepCount), ...
    'Saturated', false(1, stepCount), ...
    'Accepted', false(1, stepCount), ...
    'StrictAccepted', false(1, stepCount), ...
    'AcceptanceReason', strings(1, stepCount), ...
    'Exitflag', nan(1, stepCount), 'Converged', false(1, stepCount), ...
    'Feasible', false(1, stepCount), ...
    'FeasibleSuboptimal', false(1, stepCount), ...
    'TimedOut', false(1, stepCount), ...
    'SolveSeconds', nan(1, stepCount), ...
    'MaximumConstraintViolation', inf(1, stepCount), ...
    'RelativeCostImprovement', nan(1, stepCount), ...
    'Reference', nan(12, stepCount + 1));
end

function result = copy_case_fields(result, spec)
names = fieldnames(spec);
for index = 1:numel(names)
    result.(names{index}) = spec.(names{index});
end
end

function count = import_resume_tasks(root, taskRoot, cases, version)
count = 0;
if isempty(root) || ~isfolder(root)
    return;
end
expectedKeys = string({cases.CaseKey});
files = dir(fullfile(root, '**', 'teacher_case_*.mat'));
for index = 1:numel(files)
    source = fullfile(files(index).folder, files(index).name);
    try
        loaded = load(source, 'taskResult');
    catch
        continue;
    end
    if ~isfield(loaded, 'taskResult') || ...
            ~strcmp(loaded.taskResult.ProtocolVersion, version)
        continue;
    end
    match = find(expectedKeys == string(loaded.taskResult.CaseKey));
    if ~isscalar(match) || ...
            ~task_matches_case(loaded.taskResult, cases(match))
        continue;
    end
    destination = task_path(taskRoot, cases(match).CaseIndex);
    if ~isfile(destination)
        copyfile(source, destination);
        count = count + 1;
    end
end
end

function save_task_atomic(taskRoot, taskResult)
destination = task_path(taskRoot, taskResult.Row.CaseIndex);
temporary = [tempname(taskRoot) '.mat'];
save(temporary, 'taskResult', '-v7.3');
movefile(temporary, destination);
end

function path = task_path(taskRoot, caseIndex)
path = fullfile(taskRoot, sprintf('teacher_case_%03d.mat', caseIndex));
end

function [measurements, traces] = load_results(taskRoot, cases, version)
rows = repmat(empty_result(), numel(cases), 1);
traces = repmat(initialize_trace(1), numel(cases), 1);
for index = 1:numel(cases)
    path = task_path(taskRoot, cases(index).CaseIndex);
    assert(isfile(path), 'Missing completed task file: %s.', path);
    loaded = load(path, 'taskResult');
    assert(strcmp(loaded.taskResult.ProtocolVersion, version) && ...
        string(loaded.taskResult.CaseKey) == cases(index).CaseKey && ...
        task_matches_case(loaded.taskResult, cases(index)), ...
        'Task provenance mismatch: %s.', path);
    rows(index) = loaded.taskResult.Row;
    if index == 1
        traces = repmat(loaded.taskResult.Trace, numel(cases), 1);
    end
    traces(index) = loaded.taskResult.Trace;
end
measurements = struct2table(rows);
end

function yes = task_matches_case(taskResult, spec)
row = taskResult.Row;
yes = row.CaseIndex == spec.CaseIndex && ...
    string(row.CaseKey) == spec.CaseKey && ...
    row.ContextIndex == spec.ContextIndex && ...
    row.SourceEpisodeIndex == spec.SourceEpisodeIndex && ...
    string(row.Protocol) == spec.Protocol && ...
    string(row.Algorithm) == spec.Algorithm && ...
    row.N == spec.N && row.Nc == spec.Nc && ...
    abs(row.WeightAction - spec.WeightAction) < eps;
end

function summary = summarize_results(measurements)
[groups, algorithm, protocol] = findgroups( ...
    measurements.Algorithm, measurements.Protocol);
rows = repmat(empty_summary(), max(groups), 1);
for index = 1:max(groups)
    subset = measurements(groups == index, :);
    rows(index).Algorithm = algorithm(index);
    rows(index).Protocol = protocol(index);
    rows(index).CaseCount = height(subset);
    rows(index).CompleteFraction = mean(subset.AllCandidateAccepted);
    rows(index).StrictCompleteFraction = mean(subset.AllStrictAccepted);
    rows(index).EventAvoidanceFraction = mean(subset.EventAvoided);
    rows(index).TrackingAvoidanceFraction = ...
        mean(subset.TrackingEventAvoided);
    rows(index).QualityUsableFraction = mean(subset.QualityUsable);
    rows(index).MedianPeakErrorRatio = median( ...
        subset.PeakErrorRatio, 'omitnan');
    rows(index).MedianSolveSeconds = median( ...
        subset.MedianSolveSeconds, 'omitnan');
    rows(index).P95MaximumSolveSeconds = nearest_percentile( ...
        subset.MaximumSolveSeconds, 0.95);
    rows(index).TimeoutFraction = mean(subset.TimedOutAny);
    rows(index).ErrorFraction = mean(strlength( ...
        subset.ErrorIdentifier) > 0);
end
summary = sortrows(struct2table(rows), {'Algorithm', 'Protocol'});
end

function value = peak_normalized_error(X, reference, cfg)
error = nmpc_state_error(X, reference);
components = [vecnorm(error(1:3, :), 2, 1) ./ ...
    cfg.threshold.positionM; ...
    rad2deg(vecnorm(error(4:6, :), 2, 1)) ./ ...
    cfg.threshold.attitudeDeg; ...
    vecnorm(error(7:9, :), 2, 1) ./ cfg.threshold.velocityMps; ...
    vecnorm(error(10:12, :), 2, 1) ./ ...
    cfg.threshold.bodyRateRadps];
finite = components(isfinite(components));
if isempty(finite)
    value = Inf;
else
    value = max(finite);
end
end

function value = relative_improvement(initial, final)
value = (initial - final) / max(abs(initial), eps);
end

function value = nearest_percentile(data, probability)
data = sort(data(isfinite(data)));
if isempty(data)
    value = NaN;
else
    value = data(max(1, ceil(probability * numel(data))));
end
end

function row = empty_case()
row = struct('CaseIndex', 0, 'CaseKey', "", 'ContextIndex', 0, ...
    'SourceEpisodeIndex', 0, 'Family', "", 'LeadRankStratum', "", ...
    'LqrLeadSteps', NaN, 'ProtocolIndex', 0, 'Protocol', "", ...
    'AlgorithmIndex', 0, 'Algorithm', "", 'N', 0, 'Nc', 0, ...
    'WeightAction', NaN);
end

function row = empty_result()
row = empty_case();
row.M = 0;
row.ScenarioSeed = 0;
row.AttemptedSteps = 0;
row.CompletedSteps = 0;
row.AllCandidateAccepted = false;
row.AllStrictAccepted = false;
row.ConvergedStepFraction = NaN;
row.FeasibleSuboptimalStepFraction = NaN;
row.TimedOutAny = false;
row.MinimumRelativeCostImprovement = NaN;
row.MeanSolveSeconds = NaN;
row.MedianSolveSeconds = NaN;
row.MaximumSolveSeconds = NaN;
row.TeacherHardEvent = true;
row.TeacherFirstEventType = "not_run";
row.TeacherTrackingEvent = true;
row.TeacherConstraintEvent = true;
row.TeacherSaturationEvent = false;
row.EventAvoided = false;
row.TrackingEventAvoided = false;
row.TeacherPeakNormalizedError = Inf;
row.LqrPeakNormalizedError = NaN;
row.PeakErrorRatio = Inf;
row.QualityUsable = false;
row.MinimumGroupWeight = NaN;
row.MaximumGroupWeight = NaN;
row.ErrorIdentifier = "";
row.ErrorMessage = "";
end

function value = empty_baseline()
trace = initialize_trace(1);
value = struct('X', trace.X, 'U', trace.U, 'RawU', trace.RawU, ...
    'Saturated', trace.Saturated, 'Reference', trace.Reference, ...
    'Label', struct(), 'PeakNormalizedError', NaN);
end

function row = empty_lqr_row()
row = struct('LocalContextIndex', 0, 'SourceEpisodeIndex', 0, ...
    'Family', "", 'LeadSteps', NaN, 'FirstEventType', "", ...
    'PeakNormalizedError', NaN, 'SaturationForecast', false, ...
    'ConstraintForecast', false);
end

function row = empty_summary()
row = struct('Algorithm', "", 'Protocol', "", 'CaseCount', 0, ...
    'CompleteFraction', NaN, 'StrictCompleteFraction', NaN, ...
    'EventAvoidanceFraction', NaN, 'TrackingAvoidanceFraction', NaN, ...
    'QualityUsableFraction', NaN, 'MedianPeakErrorRatio', NaN, ...
    'MedianSolveSeconds', NaN, 'P95MaximumSolveSeconds', NaN, ...
    'TimeoutFraction', NaN, 'ErrorFraction', NaN);
end

function write_report(root, measurements, summary, metadata, cfg)
fileId = fopen(fullfile(root, 'capability_report.md'), 'w');
assert(fileId >= 0, 'Cannot create capability report.');
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, '# Targeted teacher paired closed-loop capability gate\n\n');
fprintf(fileId, '- Protocol: `%s`.\n', cfg.version);
fprintf(fileId, '- Cases: %d = 5 families x 3 lead-rank strata x 3 anchors x 2 solvers.\n', ...
    height(measurements));
fprintf(fileId, '- Continuation: H=%d steps; scenario count M=5.\n', ...
    cfg.rolloutSteps);
fprintf(fileId, '- Candidate acceptance under test: `%s`.\n', ...
    cfg.acceptancePolicy);
fprintf(fileId, '- Workers: %d; new/resumed tasks: %d/%d.\n', ...
    metadata.WorkerCount, metadata.NewTaskCount, metadata.ResumedTaskCount);
fprintf(fileId, '- Wall work completed in %.6f s.\n', metadata.WallSeconds);
fprintf(fileId, '- Complete candidate-policy rollouts: %d/%d.\n', ...
    nnz(measurements.AllCandidateAccepted), height(measurements));
fprintf(fileId, '- Complete strict-policy rollouts: %d/%d.\n', ...
    nnz(measurements.AllStrictAccepted), height(measurements));
fprintf(fileId, '- LQR events avoided: %d/%d.\n', ...
    nnz(measurements.EventAvoided), height(measurements));
fprintf(fileId, '- Quality-usable rollouts: %d/%d.\n', ...
    nnz(measurements.QualityUsable), height(measurements));
fprintf(fileId, '\nSummary rows: %d.\n\n', height(summary));
fprintf(fileId, ['This development gate does not select the final solver, ' ...
    'change the active strict acceptance policy, reduce the 14-pair SAC ' ...
    'grid, train SAC, open test/OOD data, or edit Methods. A positive result ' ...
    'only establishes closed-loop capability on the declared paired cases.\n']);
clear cleanup;
end

function write_hashes(root, names)
lines = strings(numel(names), 1);
for index = 1:numel(names)
    path = fullfile(root, names{index});
    lines(index) = string(quad_sha256_file(path)) + "  " + ...
        string(names{index});
end
write_text(fullfile(root, 'SHA256SUMS.txt'), ...
    char(strjoin(lines, newline) + newline));
end

function value = environment_positive_number(name, defaultValue)
raw = strtrim(getenv(name));
if isempty(raw)
    value = defaultValue;
else
    value = str2double(raw);
    assert(isfinite(value) && value > 0, ...
        '%s must be a positive finite number.', name);
end
end

function value = environment_text(name, defaultValue)
value = strtrim(getenv(name));
if isempty(value)
    value = defaultValue;
end
end

function value = sanitize_identifier(value)
value = regexprep(char(value), '[^A-Za-z0-9_-]', '_');
end

function delete_pool(pool)
if ~isempty(pool) && isvalid(pool)
    delete(pool);
end
end

function write_text(path, content)
[fileId, message] = fopen(path, 'w');
if fileId < 0
    error('validate_targeted_teacher_closed_loop:OpenFailed', '%s', message);
end
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, '%s', content);
clear cleanup;
end
