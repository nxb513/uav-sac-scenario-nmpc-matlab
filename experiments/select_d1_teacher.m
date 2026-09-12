function selection = select_d1_teacher(varargin)
%SELECT_D1_TEACHER Stage S3 aggregator: pick g* from teacher-grid task files.
%
% Loads every results/d1_teacher_grid/tasks/cfg*_*.mat, aggregates per-config
% episode-violation rate over the 24 teacher-dev cases, and picks the frozen
% teacher g* = argmin violation rate, tie-broken by mean cumulative normalized
% tracking error, then total solve time. Writes selected_teacher.mat + a report.
% Requires every (config,case) task present, else it errors (no silent partial).

opts = parse_args(varargin{:});
add_project_paths();
projectRoot = fileparts(fileparts(mfilename('fullpath')));
taskDir = fullfile(projectRoot, 'results', 'd1_teacher_grid', 'tasks');
if ~isempty(opts.resumeRoot) && exist(fullfile(opts.resumeRoot, 'tasks'), 'dir')
    import_tasks(fullfile(opts.resumeRoot, 'tasks'), taskDir);
end
files = dir(fullfile(taskDir, 'cfg*_*.mat'));
if isempty(files)
    error('select_d1_teacher:NoTasks', 'No task files in %s.', taskDir);
end

results = cell(numel(files), 1);
for i = 1:numel(files)
    d = load(fullfile(taskDir, files(i).name), 'result');
    results{i} = d.result;
end
results = [results{:}];

configIdx = unique([results.configIndex]);
nExpectedCases = numel(unique({results.groupId}));
agg = struct('index', {}, 'label', {}, 'nCases', {}, 'violationRate', {}, ...
    'meanCumNorm', {}, 'totalSolveTime', {}, 'convergedFraction', {});
for c = configIdx
    sub = results([results.configIndex] == c);
    agg(end + 1) = struct('index', c, 'label', sub(1).configLabel, ...
        'nCases', numel(sub), ...
        'violationRate', mean([sub.episodeViolation]), ...
        'meanCumNorm', mean([sub.cumulativeNormError]), ...
        'totalSolveTime', sum([sub.totalSolveTime]), ...
        'convergedFraction', mean([sub.convergedFraction])); %#ok<AGROW>
end

% Completeness check.
for c = configIdx
    if agg([agg.index] == c).nCases ~= nExpectedCases
        error('select_d1_teacher:Incomplete', ...
            'Config %d has %d/%d cases; run remaining shards first.', ...
            c, agg([agg.index] == c).nCases, nExpectedCases);
    end
end

% argmin violationRate, tie-break meanCumNorm, then totalSolveTime.
key = [ [agg.violationRate].', [agg.meanCumNorm].', [agg.totalSolveTime].' ];
[~, order] = sortrows(key, [1 2 3]);
best = agg(order(1));

selection = struct();
selection.selectedIndex = best.index;
selection.selectedLabel = best.label;
selection.selectedViolationRate = best.violationRate;
selection.ranking = agg(order);
selection.nCases = nExpectedCases;
selection.generatedAt = datestr(now, 'yyyy-mm-dd HH:MM:SS');

outDir = fullfile(projectRoot, 'results', 'd1_teacher_grid');
save(fullfile(outDir, 'selected_teacher.mat'), 'selection');
write_report(fullfile(outDir, 'teacher_selection_report.md'), selection, agg(order));

fprintf('\n== S3 teacher selection ==\n');
fprintf('%-24s %6s %8s %10s %8s\n', 'config', 'viol%', 'cumNorm', 'solve(s)', 'conv%');
for i = 1:numel(order)
    a = agg(order(i));
    marker = ''; if a.index == best.index; marker = '  <== g*'; end
    fprintf('%-24s %6.1f %8.1f %10.0f %8.0f%s\n', a.label, ...
        100 * a.violationRate, a.meanCumNorm, a.totalSolveTime, ...
        100 * a.convergedFraction, marker);
end
fprintf('\nFROZEN teacher g* = %s (index %d), violation %.1f%% over %d dev cases.\n', ...
    best.label, best.index, 100 * best.violationRate, nExpectedCases);
fprintf('Saved: %s\n', fullfile(outDir, 'selected_teacher.mat'));

% Kill-gate hint (teacher must be meaningfully good on hard dev cases).
if best.violationRate >= 0.5
    fprintf(['\n[KILL-GATE WARNING] best teacher violates on >=50%% of dev ' ...
        'cases. Investigate before S5/S6 (teacher may not beat LQR).\n']);
end
end

% ------------------------------------------------------------------------
function import_tasks(srcDir, dstDir)
if ~exist(dstDir, 'dir'); mkdir(dstDir); end
f = dir(fullfile(srcDir, 'cfg*_*.mat'));
for i = 1:numel(f)
    dst = fullfile(dstDir, f(i).name);
    if ~exist(dst, 'file'); copyfile(fullfile(srcDir, f(i).name), dst); end
end
end

function write_report(path, selection, ranked)
fid = fopen(path, 'w'); if fid < 0; return; end
fprintf(fid, '# D1 teacher selection (S3)\n\nGenerated: %s\n\n', ...
    selection.generatedAt);
fprintf(fid, 'Frozen g* = **%s** (index %d), violation %.1f%% over %d dev cases.\n\n', ...
    selection.selectedLabel, selection.selectedIndex, ...
    100 * selection.selectedViolationRate, selection.nCases);
fprintf(fid, '| rank | config | violation%% | meanCumNorm | solve(s) | conv%% |\n');
fprintf(fid, '|---|---|---|---|---|---|\n');
for i = 1:numel(ranked)
    a = ranked(i);
    fprintf(fid, '| %d | %s | %.1f | %.1f | %.0f | %.0f |\n', i, a.label, ...
        100 * a.violationRate, a.meanCumNorm, a.totalSolveTime, ...
        100 * a.convergedFraction);
end
fclose(fid);
end

function opts = parse_args(varargin)
opts = struct('resumeRoot', getenv_default('D1_RESUME_ROOT', ''));
for k = 1:2:numel(varargin); opts.(varargin{k}) = varargin{k + 1}; end
end

function s = getenv_default(name, default)
s = getenv(name); if isempty(s); s = default; end
end

function add_project_paths()
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(genpath(fullfile(root, 'src')));
addpath(fullfile(root, 'configs'));
end
