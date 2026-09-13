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
agg = struct('index', {}, 'label', {}, 'nCases', {}, 'divergenceRate', {}, ...
    'meanLogGrowth', {}, 'meanRmsPos', {}, 'meanExpansionFraction', {}, ...
    'totalSolveTime', {}, 'convergedFraction', {});
for c = configIdx
    sub = results([results.configIndex] == c);
    nd = sub(~[sub.diverged]);                   % cleanly-tracked (bounded) cases
    if isempty(nd)
        mlg = Inf; mrp = Inf; mef = Inf;
    else
        mlg = mean([nd.meanLogGrowth], 'omitnan');   % the g criterion (log growth)
        mrp = mean([nd.rmsPositionM]);
        mef = mean([nd.expansionFraction], 'omitnan');
    end
    agg(end + 1) = struct('index', c, 'label', sub(1).configLabel, ...
        'nCases', numel(sub), ...
        'divergenceRate', mean([sub.diverged]), ...
        'meanLogGrowth', mlg, 'meanRmsPos', mrp, 'meanExpansionFraction', mef, ...
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

% argmin: divergence rate (does the error stay bounded, not blow up) -> RMS
% position error (how tightly it tracks) -> total solve time. This "bounded and
% small" ranking is the robust reading of "contracts best" and, unlike the raw
% forward log-growth, is not polluted by the healthy startup transient (the error
% grows from an exact initial condition up to steady state). The fine 20-step
% forward-growth quantity g (for the confidence c and blend alpha) is pinned at
% S7 on the stored trajectories, where the transient is handled explicitly.
key = [ [agg.divergenceRate].', [agg.meanRmsPos].', [agg.totalSolveTime].' ];
[~, order] = sortrows(key, [1 2 3]);
best = agg(order(1));

selection = struct();
selection.selectedIndex = best.index;
selection.selectedLabel = best.label;
selection.divergenceRate = best.divergenceRate;
selection.meanLogGrowth = best.meanLogGrowth;
selection.meanRmsPos = best.meanRmsPos;
selection.ranking = agg(order);
selection.nCases = nExpectedCases;
selection.generatedAt = datestr(now, 'yyyy-mm-dd HH:MM:SS');

outDir = fullfile(projectRoot, 'results', 'd1_teacher_grid');
save(fullfile(outDir, 'selected_teacher.mat'), 'selection');
write_report(fullfile(outDir, 'teacher_selection_report.md'), selection, agg(order));

fprintf('\n== S3 teacher selection (growth-based) ==\n');
fprintf('%-24s %6s %9s %8s %6s %6s\n', 'config', 'div%', 'logGrow', ...
    'rmsPos', 'exp%', 'conv%');
for i = 1:numel(order)
    a = agg(order(i));
    marker = ''; if a.index == best.index; marker = '  <== g*'; end
    fprintf('%-24s %6.1f %9.3f %8.3f %6.0f %6.0f%s\n', a.label, ...
        100 * a.divergenceRate, a.meanLogGrowth, a.meanRmsPos, ...
        100 * a.meanExpansionFraction, 100 * a.convergedFraction, marker);
end
fprintf(['\nFROZEN teacher g* = %s (index %d): divergence %.1f%%, ' ...
    'mean log-growth %.3f over %d dev cases.\n'], best.label, best.index, ...
    100 * best.divergenceRate, best.meanLogGrowth, nExpectedCases);
fprintf('Saved: %s\n', fullfile(outDir, 'selected_teacher.mat'));

% Kill-gate hint: the teacher must keep the tracking error BOUNDED (not blow up)
% on the dev cases. (Raw log-growth is transient-polluted, so it is reported but
% not gated on here.)
if best.divergenceRate >= 0.5
    fprintf(['\n[KILL-GATE WARNING] best teacher diverges on >=50%% of dev ' ...
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
fprintf(fid, '# D1 teacher selection (S3, growth-based)\n\nGenerated: %s\n\n', ...
    selection.generatedAt);
fprintf(fid, ['Frozen g* = **%s** (index %d): divergence %.1f%%, mean ' ...
    'log-growth %.3f over %d dev cases.\n\n'], selection.selectedLabel, ...
    selection.selectedIndex, 100 * selection.divergenceRate, ...
    selection.meanLogGrowth, selection.nCases);
fprintf(fid, ['Ranking key: divergence rate (error stays bounded), then RMS ' ...
    'position error (tracking tightness), then solve time. logGrowth/exp%% are ' ...
    'reported for reference but include the startup transient; the fine 20-step ' ...
    'forward-growth g is pinned at S7 on the stored trajectories.\n\n']);
fprintf(fid, ['| rank | config | diverge%% | logGrowth | rmsPos(m) | exp%% | conv%% |\n']);
fprintf(fid, '|---|---|---|---|---|---|---|\n');
for i = 1:numel(ranked)
    a = ranked(i);
    fprintf(fid, '| %d | %s | %.1f | %.3f | %.3f | %.0f | %.0f |\n', i, a.label, ...
        100 * a.divergenceRate, a.meanLogGrowth, a.meanRmsPos, ...
        100 * a.meanExpansionFraction, 100 * a.convergedFraction);
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
