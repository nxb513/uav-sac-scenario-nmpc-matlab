function audit_targeted_c012_nmpc_feasibility_result(inputFolder, outputFolder)
%AUDIT_TARGETED_C012_NMPC_FEASIBILITY_RESULT Audit the 210-solve CSV result.

if nargin < 2
    error('Provide the raw screen folder and a new audit output folder.');
end
inputFolder = char(inputFolder);
outputFolder = char(outputFolder);
assert(isfolder(inputFolder), 'Missing input folder: %s.', inputFolder);
assert(~isfolder(outputFolder), 'Refusing to overwrite %s.', outputFolder);

measurements = readtable(fullfile(inputFolder, ...
    'feasibility_measurements.csv'), 'TextType', 'string');
assert(height(measurements) == 210, 'Expected exactly 210 solves.');
assert(all(measurements.M == 5), 'Scenario count must remain fixed at M=5.');
assert(all(measurements.FeasibleSuboptimal <= measurements.Feasible));
assert(all(measurements.FeasibleSuboptimal <= ~measurements.TimedOut));
assert(all(measurements.FeasibleSuboptimal <= ...
    (measurements.Exitflag == 0)));

candidate = logical(measurements.CandidateFeasibleAccepted);
assert(all(measurements.MaximumConstraintViolation(candidate) == 0));
assert(all(measurements.RelativeCostImprovement(candidate) > 0));

family = grouped_summary(measurements, "Family");
pairKey = "N" + string(measurements.N) + "_Nc" + ...
    string(measurements.Nc);
pairMeasurements = addvars(measurements, pairKey, ...
    'NewVariableNames', 'Pair');
pair = grouped_summary(pairMeasurements, "Pair");
profile = grouped_summary(measurements, "Profile");

temporaryFolder = [outputFolder '.tmp'];
if isfolder(temporaryFolder)
    rmdir(temporaryFolder, 's');
end
mkdir(temporaryFolder);
cleanup = onCleanup(@() cleanup_temporary(temporaryFolder));
writetable(family, fullfile(temporaryFolder, 'summary_by_family.csv'));
writetable(pair, fullfile(temporaryFolder, 'summary_by_pair.csv'));
writetable(profile, fullfile(temporaryFolder, 'summary_by_profile.csv'));
write_report(temporaryFolder, measurements);
movefile(temporaryFolder, outputFolder);
clear cleanup;
end

function summary = grouped_summary(measurements, variable)
[groups, labels] = findgroups(measurements.(variable));
rows = repmat(empty_row(), max(groups), 1);
for index = 1:max(groups)
    subset = measurements(groups == index, :);
    rows(index).Group = string(labels(index));
    rows(index).Cases = height(subset);
    rows(index).Converged = nnz(subset.Converged);
    rows(index).Feasible = nnz(subset.Feasible);
    rows(index).FeasibleSuboptimal = nnz(subset.FeasibleSuboptimal);
    rows(index).CandidateAccepted = nnz(subset.CandidateFeasibleAccepted);
    rows(index).TimedOut = nnz(subset.TimedOut);
    rows(index).MedianSolveSeconds = median(subset.SolveSeconds);
    rows(index).P95SolveSeconds = nearest_percentile( ...
        subset.SolveSeconds, 0.95);
    rows(index).MaximumSolveSeconds = max(subset.SolveSeconds);
end
summary = struct2table(rows);
end

function row = empty_row()
row = struct('Group', "", 'Cases', 0, 'Converged', 0, ...
    'Feasible', 0, 'FeasibleSuboptimal', 0, ...
    'CandidateAccepted', 0, 'TimedOut', 0, ...
    'MedianSolveSeconds', NaN, 'P95SolveSeconds', NaN, ...
    'MaximumSolveSeconds', NaN);
end

function value = nearest_percentile(values, probability)
values = sort(values(:));
index = max(1, ceil(probability * numel(values)));
value = values(index);
end

function write_report(folder, measurements)
candidate = logical(measurements.CandidateFeasibleAccepted);
feasibleSuboptimal = logical(measurements.FeasibleSuboptimal);
fileId = fopen(fullfile(folder, 'independent_audit.md'), 'w');
assert(fileId >= 0, 'Cannot create audit report.');
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, '# Independent audit of C012 NMPC feasibility screen\n\n');
fprintf(fileId, '- Cases: %d.\n', height(measurements));
fprintf(fileId, '- Converged: %d (%.3f).\n', ...
    nnz(measurements.Converged), mean(measurements.Converged));
fprintf(fileId, '- Feasible: %d (%.3f).\n', ...
    nnz(measurements.Feasible), mean(measurements.Feasible));
fprintf(fileId, '- Feasible-suboptimal: %d (%.3f).\n', ...
    nnz(feasibleSuboptimal), mean(feasibleSuboptimal));
fprintf(fileId, '- Candidate accepted: %d (%.3f).\n', ...
    nnz(candidate), mean(candidate));
fprintf(fileId, '- Timed out: %d (%.3f).\n', ...
    nnz(measurements.TimedOut), mean(measurements.TimedOut));
fprintf(fileId, '- Solve seconds mean/median/p95/max: ');
fprintf(fileId, '%.6f/%.6f/%.6f/%.6f.\n', ...
    mean(measurements.SolveSeconds), median(measurements.SolveSeconds), ...
    nearest_percentile(measurements.SolveSeconds, 0.95), ...
    max(measurements.SolveSeconds));
fprintf(fileId, '- Minimum candidate cost improvement: %.6f.\n', ...
    min(measurements.RelativeCostImprovement(candidate)));
fprintf(fileId, '- Maximum candidate constraint violation: %.6g.\n\n', ...
    max(measurements.MaximumConstraintViolation(candidate)));
fprintf(fileId, ['The static screen supports closed-loop validation of ' ...
    '`converged_or_feasible_suboptimal`; it does not by itself authorize ' ...
    'changing the training default. Timed-out and infeasible iterates remain ' ...
    'rejected.\n']);
clear cleanup;
end

function cleanup_temporary(folder)
if isfolder(folder)
    rmdir(folder, 's');
end
end
