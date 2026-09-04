function benchmark_targeted_c012_nmpc_solver_algorithms()
%BENCHMARK_TARGETED_C012_NMPC_SOLVER_ALGORITHMS Screen solver algorithms.

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot, 'configs'));
addpath(genpath(fullfile(projectRoot, 'src')));
cfg = targeted_specialist_sac_config();
wallLimit = environment_positive_number('NMPC_MAX_WALL_SECONDS', 60);
requestedWorkers = round(environment_positive_number( ...
    'PARALLEL_WORKERS', 2));

loaded = load(fullfile(projectRoot, cfg.trainingBankPath), ...
    'trainContextBank');
[contexts, contextTable] = selected_contexts(loaded.trainContextBank);
algorithms = ["sqp"; "interior-point"; "active-set"; "sqp-legacy"];
pairs = table([10; 15; 20], [10; 15; 20], ...
    'VariableNames', {'N', 'Nc'});
cases = build_cases(contexts, algorithms, pairs);
assert(numel(cases) == 24, 'Expected exactly 24 solver-screen cases.');

outputRoot = fullfile(projectRoot, cfg.targeted.resultRoot, ...
    'specialist_nmpc_c012_solver_algorithm_screen_v1');
assert(~isfolder(outputRoot), 'Refusing to overwrite %s.', outputRoot);
temporaryRoot = [outputRoot '.tmp'];
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
    rows(caseIndex) = execute_case(cases(caseIndex), contexts, cfg, ...
        wallLimit);
end
wallSeconds = toc(clock);
measurements = struct2table(rows);
summary = summarize_measurements(measurements);
assert(height(measurements) == 24 && all(measurements.M == 5), ...
    'Solver-screen output is incomplete.');

writetable(contextTable, fullfile(temporaryRoot, ...
    'selected_contexts.csv'));
writetable(pairs, fullfile(temporaryRoot, 'selected_pairs.csv'));
writetable(table(algorithms), fullfile(temporaryRoot, ...
    'solver_algorithms.csv'));
writetable(measurements, fullfile(temporaryRoot, ...
    'solver_measurements.csv'));
writetable(summary, fullfile(temporaryRoot, 'solver_summary.csv'));
metadata = struct('status', 'solver_screen_not_training_or_grid_change', ...
    'caseCount', height(measurements), 'workerCount', workerCount, ...
    'wallLimitSeconds', wallLimit, 'wallSeconds', wallSeconds, ...
    'builtAt', char(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd HH:mm:ssXXX')));
save(fullfile(temporaryRoot, 'solver_screen.mat'), 'measurements', ...
    'summary', 'contextTable', 'pairs', 'algorithms', 'metadata', '-v7.3');
write_report(temporaryRoot, measurements, metadata);
write_text(fullfile(temporaryRoot, 'COMPLETED.txt'), sprintf( ...
    ['task=benchmark_targeted_c012_nmpc_solver_algorithms\n' ...
    'status=completed\ncases=%d\ncontexts=2\nalgorithms=4\n' ...
    'pairs=3\nscenario_count=5\nworkers=%d\n' ...
    'solve_wall_limit_seconds=%.6g\nwall_seconds=%.6f\n'], ...
    height(measurements), workerCount, wallLimit, wallSeconds));
clear poolCleanup;
delete_pool(pool);
movefile(temporaryRoot, outputRoot);
clear cleanup;
end

function [contexts, selection] = selected_contexts(bank)
family = string({bank.Family});
sourceEpisode = [bank.SourceEpisodeIndex];
circle = find(family == "circle" & sourceEpisode == 1805);
lemniscate = find(family == "lemniscate" & sourceEpisode == 807);
assert(isscalar(circle) && isscalar(lemniscate), ...
    'Expected exact static-screen contexts were not found.');
indices = [circle, lemniscate];
contexts = bank(indices);
selection = table((1:2).', indices(:), sourceEpisode(indices).', ...
    family(indices).', 'VariableNames', {'ScreenContextIndex', ...
    'ContextIndex', 'SourceEpisodeIndex', 'Family'});
end

function cases = build_cases(contexts, algorithms, pairs)
count = numel(contexts) * numel(algorithms) * height(pairs);
cases = repmat(empty_case(), count, 1);
cursor = 0;
for contextIndex = 1:numel(contexts)
    for algorithmIndex = 1:numel(algorithms)
        for pairIndex = 1:height(pairs)
            cursor = cursor + 1;
            cases(cursor).CaseIndex = cursor;
            cases(cursor).ContextIndex = contextIndex;
            cases(cursor).Algorithm = algorithms(algorithmIndex);
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
row.Algorithm = spec.Algorithm;
row.N = spec.N;
row.Nc = spec.Nc;
row.M = 5;
context = contexts(spec.ContextIndex);
row.SourceEpisodeIndex = context.SourceEpisodeIndex;
row.Family = string(context.Family);

try
    action = zeros(cfg.action.dimension, 1);
    action(1:6) = 0.5;
    [nmpcCfg, ~] = rl_nmpc_action_to_config(action, cfg);
    nmpcCfg.predictionHorizon = spec.N;
    nmpcCfg.controlHorizon = spec.Nc;
    nmpcCfg.scenario.count = 5;
    nmpcCfg.solver.algorithm = char(spec.Algorithm);
    nmpcCfg.solver.maxWallSeconds = wallLimit;
    originalScreenIndex = [1, 5];
    scenarioSeed = 300832100 + ...
        1000 * originalScreenIndex(spec.ContextIndex) + 30;
    row.ScenarioSeed = scenarioSeed;
    scenarios = quad_sample_uncertainty(nmpcCfg.plant, 5, ...
        'targeted', scenarioSeed, nmpcCfg.scenario.method);
    step = context.StartStep;
    time = (step - 1) * cfg.environment.sampleTime;
    reference = nmpc_reference_window(context.Reference, step, time, ...
        nmpcCfg);
    wallClock = tic;
    solution = scenario_nmpc_solve(context.X0, reference, scenarios, ...
        nmpcCfg, [], context.PreviousInput);
    row.WallSeconds = toc(wallClock);
    row.SolveSeconds = solution.solveTime;
    row.Exitflag = solution.exitflag;
    row.TimedOut = solution.timedOut;
    row.Converged = solution.converged;
    row.Feasible = solution.feasible;
    row.FeasibleSuboptimal = solution.feasibleSuboptimal;
    row.CandidateAccepted = (solution.converged || ...
        solution.feasibleSuboptimal) && solution.feasible && ...
        ~solution.timedOut;
    row.MaximumConstraintViolation = solution.maxConstraintViolation;
    row.Cost = solution.cost;
    row.WarmStartCost = solution.warmStartCost;
    row.RelativeCostImprovement = (solution.warmStartCost - ...
        solution.cost) / max(abs(solution.warmStartCost), eps);
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
[group, algorithm, family, N, Nc] = findgroups( ...
    measurements.Algorithm, measurements.Family, ...
    measurements.N, measurements.Nc);
rows = repmat(empty_summary(), max(group), 1);
for index = 1:max(group)
    subset = measurements(group == index, :);
    rows(index).Algorithm = algorithm(index);
    rows(index).Family = family(index);
    rows(index).N = N(index);
    rows(index).Nc = Nc(index);
    rows(index).Converged = mean(subset.Converged);
    rows(index).Feasible = mean(subset.Feasible);
    rows(index).CandidateAccepted = mean(subset.CandidateAccepted);
    rows(index).TimedOut = mean(subset.TimedOut);
    rows(index).SolveSeconds = mean(subset.SolveSeconds, 'omitnan');
    rows(index).MaximumConstraintViolation = ...
        max(subset.MaximumConstraintViolation, [], 'omitnan');
    rows(index).Error = mean(strlength(subset.ErrorIdentifier) > 0);
end
summary = sortrows(struct2table(rows), {'Algorithm', 'Family', 'N'});
end

function row = empty_case()
row = struct('CaseIndex', 0, 'ContextIndex', 0, 'Algorithm', "", ...
    'N', 0, 'Nc', 0);
end

function row = empty_result()
row = struct('CaseIndex', 0, 'ContextIndex', 0, ...
    'SourceEpisodeIndex', 0, 'Family', "", 'Algorithm', "", ...
    'N', 0, 'Nc', 0, 'M', 0, 'ScenarioSeed', 0, ...
    'SolveSeconds', NaN, 'WallSeconds', NaN, 'Exitflag', NaN, ...
    'TimedOut', false, 'Converged', false, 'Feasible', false, ...
    'FeasibleSuboptimal', false, 'CandidateAccepted', false, ...
    'MaximumConstraintViolation', Inf, 'Cost', NaN, ...
    'WarmStartCost', NaN, 'RelativeCostImprovement', NaN, ...
    'Iterations', NaN, 'FunctionCount', NaN, ...
    'ErrorIdentifier', "", 'ErrorMessage', "");
end

function row = empty_summary()
row = struct('Algorithm', "", 'Family', "", 'N', 0, 'Nc', 0, ...
    'Converged', NaN, 'Feasible', NaN, 'CandidateAccepted', NaN, ...
    'TimedOut', NaN, 'SolveSeconds', NaN, ...
    'MaximumConstraintViolation', NaN, 'Error', NaN);
end

function write_report(root, measurements, metadata)
fileId = fopen(fullfile(root, 'solver_screen_report.md'), 'w');
assert(fileId >= 0, 'Cannot create solver-screen report.');
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, '# C012 bounded NMPC solver-algorithm screen\n\n');
fprintf(fileId, '- Cases: 24 = 2 contexts x 4 algorithms x 3 pairs.\n');
fprintf(fileId, '- Q/R profile: high; scenario count: M=5.\n');
fprintf(fileId, '- Workers: %d; wall guard: %.6g s.\n', ...
    metadata.workerCount, metadata.wallLimitSeconds);
fprintf(fileId, '- Total wall time: %.6f s.\n', metadata.wallSeconds);
fprintf(fileId, '- Candidate acceptance: %.3f.\n', ...
    mean(measurements.CandidateAccepted));
fprintf(fileId, '- Timeout fraction: %.3f.\n', ...
    mean(measurements.TimedOut));
fprintf(fileId, '- Error fraction: %.3f.\n\n', ...
    mean(strlength(measurements.ErrorIdentifier) > 0));
fprintf(fileId, ['This bounded screen compares solver implementations only. ' ...
    'It does not train SAC or change Q/R, M, the 14-pair grid, reward, ' ...
    'acceptance policy, test/OOD data or Methods.\n']);
clear cleanup;
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
