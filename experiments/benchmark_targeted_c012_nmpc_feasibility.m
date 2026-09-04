function benchmark_targeted_c012_nmpc_feasibility()
%BENCHMARK_TARGETED_C012_NMPC_FEASIBILITY Screen 14 pairs on C012 contexts.

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot, 'configs'));
addpath(genpath(fullfile(projectRoot, 'src')));
cfg = targeted_specialist_sac_config();
wallLimit = environment_positive_number('NMPC_MAX_WALL_SECONDS', 60);
requestedWorkers = round(environment_positive_number( ...
    'PARALLEL_WORKERS', 2));

bankPath = fullfile(projectRoot, cfg.trainingBankPath);
assert(isfile(bankPath), 'The C012 context bank must be built first.');
loaded = load(bankPath, 'trainContextBank', 'trainContextManifest', ...
    'buildMetadata');
[contexts, selection] = select_family_contexts( ...
    loaded.trainContextBank, loaded.trainContextManifest);
pairs = reachable_pairs(cfg);
profiles = weight_profiles();
cases = build_cases(contexts, selection, pairs, profiles);
assert(numel(cases) == 5 * 3 * 14, ...
    'The bounded screen must contain exactly 210 solves.');

outputRoot = fullfile(projectRoot, cfg.targeted.resultRoot, ...
    'specialist_nmpc_c012_feasibility_screen_v1');
assert(~isfolder(outputRoot), 'Refusing to overwrite %s.', outputRoot);
temporaryRoot = [char(outputRoot) '.tmp'];
if isfolder(temporaryRoot)
    rmdir(temporaryRoot, 's');
end
mkdir(temporaryRoot);
cleanup = onCleanup(@() cleanup_temporary(temporaryRoot));

pool = gcp('nocreate');
if ~isempty(pool)
    delete(pool);
end
cluster = parcluster('Processes');
workerCount = min(requestedWorkers, cluster.NumWorkers);
pool = parpool(cluster, workerCount);
poolCleanup = onCleanup(@() delete_pool(pool));

rows = repmat(empty_result(), numel(cases), 1);
clock = tic;
parfor caseIndex = 1:numel(cases)
    rows(caseIndex) = execute_case( ...
        cases(caseIndex), contexts, cfg, wallLimit);
end
wallSeconds = toc(clock);
measurements = struct2table(rows);
summary = summarize_measurements(measurements);
validate_measurements(measurements, summary, wallLimit);

writetable(selection, fullfile(temporaryRoot, ...
    'selected_family_contexts.csv'));
writetable(pairs, fullfile(temporaryRoot, ...
    'reachable_horizon_pairs.csv'));
writetable(profiles, fullfile(temporaryRoot, ...
    'weight_profiles.csv'));
writetable(measurements, fullfile(temporaryRoot, ...
    'feasibility_measurements.csv'));
writetable(summary, fullfile(temporaryRoot, ...
    'feasibility_summary_by_profile_pair.csv'));
metadata = struct('status', ...
    'bounded_runtime_screen_not_training_or_final_grid_selection', ...
    'caseCount', height(measurements), 'workerCount', workerCount, ...
    'wallLimitSeconds', wallLimit, 'wallSeconds', wallSeconds, ...
    'contextSourceSeed', loaded.buildMetadata.sourceSeed, ...
    'builtAt', char(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd HH:mm:ssXXX')));
save(fullfile(temporaryRoot, 'feasibility_screen.mat'), ...
    'measurements', 'summary', 'selection', 'pairs', 'profiles', ...
    'metadata', '-v7.3');
write_report(temporaryRoot, measurements, summary, metadata);
write_text(fullfile(temporaryRoot, 'COMPLETED.txt'), sprintf( ...
    ['task=benchmark_targeted_c012_nmpc_feasibility\nstatus=completed\n' ...
    'cases=%d\ncontexts=5\nweight_profiles=3\nhorizon_pairs=14\n' ...
    'scenario_count=5\nworkers=%d\nsolve_wall_limit_seconds=%.6g\n' ...
    'wall_seconds=%.6f\n'], height(measurements), workerCount, ...
    wallLimit, wallSeconds));
clear poolCleanup;
delete_pool(pool);
movefile(temporaryRoot, outputRoot);
clear cleanup;
end

function [contexts, selection] = select_family_contexts(bank, manifest)
families = unique(manifest.Family, 'stable');
assert(numel(families) == 5, 'Expected five hard-context families.');
indices = zeros(numel(families), 1);
for familyIndex = 1:numel(families)
    candidates = find(manifest.Family == families(familyIndex));
    [~, order] = sort(manifest.RiskScore(candidates));
    indices(familyIndex) = candidates(order(ceil(numel(order) / 2)));
end
contexts = bank(indices);
selection = manifest(indices, :);
selection.ScreenContextIndex = (1:height(selection)).';
selection = movevars(selection, 'ScreenContextIndex', 'Before', 1);
end

function pairs = reachable_pairs(cfg)
[N, Nc] = ndgrid(cfg.environment.horizonBank(:), ...
    cfg.environment.controlHorizonBank(:));
mask = Nc <= N;
pairs = table(N(mask), Nc(mask), 'VariableNames', {'N', 'Nc'});
pairs = sortrows(pairs, {'N', 'Nc'});
assert(height(pairs) == 14, 'Expected 14 feasible horizon pairs.');
end

function profiles = weight_profiles()
Name = ["low"; "mid"; "high"];
ActionValue = [-0.5; 0.0; 0.5];
profiles = table(Name, ActionValue);
end

function cases = build_cases(contexts, selection, pairs, profiles)
count = numel(contexts) * height(profiles) * height(pairs);
cases = repmat(empty_case(), count, 1);
cursor = 0;
for contextIndex = 1:numel(contexts)
    for profileIndex = 1:height(profiles)
        for pairIndex = 1:height(pairs)
            cursor = cursor + 1;
            cases(cursor).CaseIndex = cursor;
            cases(cursor).ContextIndex = contextIndex;
            cases(cursor).SourceEpisodeIndex = ...
                selection.SourceEpisodeIndex(contextIndex);
            cases(cursor).ProfileIndex = profileIndex;
            cases(cursor).Profile = profiles.Name(profileIndex);
            cases(cursor).WeightAction = profiles.ActionValue(profileIndex);
            cases(cursor).N = pairs.N(pairIndex);
            cases(cursor).Nc = pairs.Nc(pairIndex);
        end
    end
end
end

function row = execute_case(spec, contexts, cfg, wallLimit)
row = empty_result();
row.CaseIndex = spec.CaseIndex;
row.ContextIndex = spec.ContextIndex;
row.SourceEpisodeIndex = spec.SourceEpisodeIndex;
row.Profile = spec.Profile;
row.WeightAction = spec.WeightAction;
row.N = spec.N;
row.Nc = spec.Nc;
row.M = cfg.nmpc.scenario.count;
context = contexts(spec.ContextIndex);
row.Family = string(context.Family);
row.TargetReferenceSpeed = context.TargetReferenceSpeed;
row.TargetReferenceAcceleration = context.TargetReferenceAcceleration;
row.DisturbanceType = string(context.Disturbance.type);
row.DisturbanceLevel = context.Disturbance.levelIndex;
row.UncertaintyStratum = string(context.UncertaintyStratum);

try
    action = zeros(cfg.action.dimension, 1);
    action(1:6) = spec.WeightAction;
    [nmpcCfg, mapping] = rl_nmpc_action_to_config(action, cfg);
    nmpcCfg.predictionHorizon = spec.N;
    nmpcCfg.controlHorizon = spec.Nc;
    nmpcCfg.scenario.count = cfg.nmpc.scenario.count;
    nmpcCfg.solver.maxWallSeconds = wallLimit;
    row.MinimumGroupWeight = min(mapping.groupWeights);
    row.MaximumGroupWeight = max(mapping.groupWeights);

    scenarioSeed = 300832100 + ...
        1000 * spec.ContextIndex + 10 * spec.ProfileIndex;
    thetaScenarios = quad_sample_uncertainty(nmpcCfg.plant, ...
        nmpcCfg.scenario.count, 'targeted', scenarioSeed, ...
        nmpcCfg.scenario.method);
    step = context.StartStep;
    time = (step - 1) * cfg.environment.sampleTime;
    reference = nmpc_reference_window( ...
        context.Reference, step, time, nmpcCfg);
    wallClock = tic;
    solution = scenario_nmpc_solve(context.X0, reference, ...
        thetaScenarios, nmpcCfg, [], context.PreviousInput);
    row.WallSeconds = toc(wallClock);
    row.SolveSeconds = solution.solveTime;
    row.Exitflag = solution.exitflag;
    row.TimedOut = solution.timedOut;
    row.Converged = solution.converged;
    row.Feasible = solution.feasible;
    row.FeasibleSuboptimal = solution.feasibleSuboptimal;
    row.StrictEnvironmentAccepted = ...
        solution.converged && solution.feasible && ~solution.timedOut;
    row.CandidateFeasibleAccepted = ...
        (solution.converged || solution.feasibleSuboptimal) && ...
        solution.feasible && ~solution.timedOut;
    row.MaximumConstraintViolation = solution.maxConstraintViolation;
    row.Cost = solution.cost;
    row.WarmStartCost = solution.warmStartCost;
    row.RelativeCostImprovement = relative_improvement( ...
        solution.warmStartCost, solution.cost);
    if isfield(solution.output, 'iterations')
        row.Iterations = solution.output.iterations;
    end
    if isfield(solution.output, 'funcCount')
        row.FunctionCount = solution.output.funcCount;
    end
catch exception
    row.ErrorIdentifier = string(exception.identifier);
    row.ErrorMessage = string(strrep(exception.message, newline, ' '));
end
end

function summary = summarize_measurements(measurements)
[group, profile, N, Nc] = findgroups( ...
    measurements.Profile, measurements.N, measurements.Nc);
rows = repmat(empty_summary(), max(group), 1);
for index = 1:max(group)
    subset = measurements(group == index, :);
    rows(index).Profile = profile(index);
    rows(index).N = N(index);
    rows(index).Nc = Nc(index);
    rows(index).CaseCount = height(subset);
    rows(index).ConvergedFraction = mean(subset.Converged);
    rows(index).FeasibleFraction = mean(subset.Feasible);
    rows(index).FeasibleSuboptimalFraction = ...
        mean(subset.FeasibleSuboptimal);
    rows(index).StrictEnvironmentAcceptedFraction = ...
        mean(subset.StrictEnvironmentAccepted);
    rows(index).CandidateFeasibleAcceptedFraction = ...
        mean(subset.CandidateFeasibleAccepted);
    rows(index).TimedOutFraction = mean(subset.TimedOut);
    rows(index).ErrorFraction = mean(strlength(subset.ErrorIdentifier) > 0);
    rows(index).SolveSecondsMean = mean(subset.SolveSeconds, 'omitnan');
    rows(index).SolveSecondsMedian = median(subset.SolveSeconds, 'omitnan');
    rows(index).SolveSecondsMax = max(subset.SolveSeconds, [], 'omitnan');
    rows(index).MaximumConstraintViolation = ...
        max(subset.MaximumConstraintViolation, [], 'omitnan');
end
summary = struct2table(rows);
summary = sortrows(summary, {'Profile', 'N', 'Nc'});
end

function validate_measurements(measurements, summary, wallLimit)
assert(height(measurements) == 210 && height(summary) == 42, ...
    'The bounded screen output is incomplete.');
assert(numel(unique(measurements.Family)) == 5 && ...
    numel(unique(measurements.Profile)) == 3, ...
    'Family or weight-profile coverage is incomplete.');
assert(all(measurements.M == 5), 'Scenario count changed during the screen.');
assert(wallLimit > 0, 'The wall guard must remain positive.');
end

function value = relative_improvement(initial, final)
value = (initial - final) / max(abs(initial), eps);
end

function row = empty_case()
row = struct('CaseIndex', 0, 'ContextIndex', 0, ...
    'SourceEpisodeIndex', 0, 'ProfileIndex', 0, 'Profile', "", ...
    'WeightAction', NaN, 'N', 0, 'Nc', 0);
end

function row = empty_result()
row = struct('CaseIndex', 0, 'ContextIndex', 0, ...
    'SourceEpisodeIndex', 0, 'Family', "", ...
    'TargetReferenceSpeed', NaN, 'TargetReferenceAcceleration', NaN, ...
    'DisturbanceType', "", 'DisturbanceLevel', 0, ...
    'UncertaintyStratum', "", 'Profile', "", 'WeightAction', NaN, ...
    'MinimumGroupWeight', NaN, 'MaximumGroupWeight', NaN, ...
    'N', 0, 'Nc', 0, 'M', 0, 'SolveSeconds', NaN, ...
    'WallSeconds', NaN, 'Exitflag', NaN, 'TimedOut', false, ...
    'Converged', false, 'Feasible', false, ...
    'FeasibleSuboptimal', false, 'StrictEnvironmentAccepted', false, ...
    'CandidateFeasibleAccepted', false, ...
    'MaximumConstraintViolation', Inf, 'Cost', NaN, ...
    'WarmStartCost', NaN, 'RelativeCostImprovement', NaN, ...
    'Iterations', NaN, 'FunctionCount', NaN, ...
    'ErrorIdentifier', "", 'ErrorMessage', "");
end

function row = empty_summary()
row = struct('Profile', "", 'N', 0, 'Nc', 0, 'CaseCount', 0, ...
    'ConvergedFraction', NaN, 'FeasibleFraction', NaN, ...
    'FeasibleSuboptimalFraction', NaN, ...
    'StrictEnvironmentAcceptedFraction', NaN, ...
    'CandidateFeasibleAcceptedFraction', NaN, ...
    'TimedOutFraction', NaN, 'ErrorFraction', NaN, ...
    'SolveSecondsMean', NaN, 'SolveSecondsMedian', NaN, ...
    'SolveSecondsMax', NaN, 'MaximumConstraintViolation', NaN);
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

function write_report(root, measurements, summary, metadata)
fileId = fopen(fullfile(root, 'feasibility_screen_report.md'), 'w');
assert(fileId >= 0, 'Cannot open feasibility-screen report.');
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, '# C012 NMPC bounded feasibility screen\n\n');
fprintf(fileId, '- Cases: %d = 5 families x 3 Q/R profiles x 14 pairs.\n', ...
    height(measurements));
fprintf(fileId, '- Scenario count: M=5 fixed.\n');
fprintf(fileId, '- Workers: %d.\n', metadata.workerCount);
fprintf(fileId, '- Per-solve wall guard: %.6g s.\n', ...
    metadata.wallLimitSeconds);
fprintf(fileId, '- Total wall time: %.6f s.\n', metadata.wallSeconds);
fprintf(fileId, '- Converged fraction: %.3f.\n', ...
    mean(measurements.Converged));
fprintf(fileId, '- Feasible fraction: %.3f.\n', ...
    mean(measurements.Feasible));
fprintf(fileId, '- Feasible-suboptimal fraction: %.3f.\n', ...
    mean(measurements.FeasibleSuboptimal));
fprintf(fileId, '- Timed-out fraction: %.3f.\n', ...
    mean(measurements.TimedOut));
fprintf(fileId, '- Strict current-policy acceptance: %.3f.\n', ...
    mean(measurements.StrictEnvironmentAccepted));
fprintf(fileId, '- Candidate feasible acceptance: %.3f.\n', ...
    mean(measurements.CandidateFeasibleAccepted));
finiteTimes = measurements.SolveSeconds(isfinite(measurements.SolveSeconds));
fprintf(fileId, '- Solve time mean/median/max: %.6f/%.6f/%.6f s.\n', ...
    mean(finiteTimes), median(finiteTimes), max(finiteTimes));
fprintf(fileId, ['\nThis is an execution/feasibility screen, not SAC ' ...
    'training, final Q/R selection, final horizon selection or a performance ' ...
    'comparison. exitflag=0 policy remains unchanged until these paired ' ...
    'convergence and feasibility results are reviewed.\n']);
fprintf(fileId, '\nSummary rows: %d.\n', height(summary));
clear cleanup;
end

function delete_pool(pool)
if ~isempty(pool) && isvalid(pool)
    delete(pool);
end
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
