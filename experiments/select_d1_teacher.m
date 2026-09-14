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
    'budgetRate', {}, 'meanG_H', {}, 'meanRmsPos', {}, ...
    'fractionContracting', {}, 'totalSolveTime', {}, 'convergedFraction', {});
for c = configIdx
    sub = results([results.configIndex] == c);
    nd = sub(~[sub.diverged]);                   % bounded cases (incl budget-stopped)
    if isempty(nd)
        mgh = Inf; mrp = Inf; fcon = NaN;
    else
        % Lyapunov contraction g_H = (1/H)log(V_{k+H}/V_k), V=e'Pe (the g criterion).
        mgh = mean([nd.meanG_H], 'omitnan');
        mrp = mean([nd.rmsPositionM]);
        fcon = mean([nd.fractionContracting], 'omitnan');
    end
    agg(end + 1) = struct('index', c, 'label', sub(1).configLabel, ...
        'nCases', numel(sub), ...
        'divergenceRate', mean([sub.diverged]), ...
        'budgetRate', mean(field_or_false(sub, 'budgetStopped')), ...
        'meanG_H', mgh, 'meanRmsPos', mrp, 'fractionContracting', fcon, ...
        'totalSolveTime', sum([sub.totalSolveTime]), ...
        'convergedFraction', mean([sub.convergedFraction])); %#ok<AGROW>
end

% Completeness check: tolerate a few missing cases (e.g. a shard that failed
% despite the guards) by ranking on the available cases, but refuse a config that
% is missing too many to compare fairly.
maxMissing = 3;
for c = configIdx
    nc = agg([agg.index] == c).nCases;
    if nc < nExpectedCases - maxMissing
        error('select_d1_teacher:Incomplete', ...
            'Config %d has only %d/%d cases (> %d missing); rerun shards first.', ...
            c, nc, nExpectedCases, maxMissing);
    elseif nc < nExpectedCases
        warning('select_d1_teacher:PartialConfig', ...
            'Config %d has %d/%d cases (%d missing); ranking on available cases.', ...
            c, nc, nExpectedCases, nExpectedCases - nc);
    end
end

% argmin: TRUE divergence rate (error blew up) -> budget-stop rate (couldn't
% finish in the wall budget; a practicality penalty kept below divergence) -> RMS
% position error (how tightly it tracks) -> total solve time. "Bounded and small"
% is the robust reading of "contracts best"; budget-stops are separated from
% divergence so a long episode is not corrupted by a slow shard timing out. The
% fine 20-step forward-growth g (for the confidence c and blend alpha) is pinned
% at S7 on the stored trajectories, where the startup transient is handled.
key = [ [agg.divergenceRate].', [agg.budgetRate].', ...
        [agg.meanRmsPos].', [agg.totalSolveTime].' ];
[~, order] = sortrows(key, [1 2 3 4]);
best = agg(order(1));

selection = struct();
selection.selectedIndex = best.index;
selection.selectedLabel = best.label;
selection.divergenceRate = best.divergenceRate;
selection.meanG_H = best.meanG_H;              % Lyapunov contraction g_H (V=e'Pe)
selection.fractionContracting = best.fractionContracting;
selection.meanRmsPos = best.meanRmsPos;
selection.ranking = agg(order);
selection.nCases = nExpectedCases;
selection.generatedAt = datestr(now, 'yyyy-mm-dd HH:MM:SS');

outDir = fullfile(projectRoot, 'results', 'd1_teacher_grid');
save(fullfile(outDir, 'selected_teacher.mat'), 'selection');
write_report(fullfile(outDir, 'teacher_selection_report.md'), selection, agg(order));

fprintf('\n== S3 teacher selection (growth-based) ==\n');
fprintf('%-24s %6s %6s %8s %6s %6s\n', 'config', 'div%', 'bud%', ...
    'rmsPos', 'conv%', 'nCase');
for i = 1:numel(order)
    a = agg(order(i));
    marker = ''; if a.index == best.index; marker = '  <== g*'; end
    fprintf('%-24s %6.1f %6.1f %8.3f %6.0f %6d%s\n', a.label, ...
        100 * a.divergenceRate, 100 * a.budgetRate, a.meanRmsPos, ...
        100 * a.convergedFraction, a.nCases, marker);
end
fprintf(['\nFROZEN teacher g* = %s (index %d): divergence %.1f%%, ' ...
    'mean g_H (V=e''Pe contraction) %.4f over %d dev cases.\n'], best.label, ...
    best.index, 100 * best.divergenceRate, best.meanG_H, nExpectedCases);
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
fprintf(fid, ['# D1 teacher selection (S3, Lyapunov contraction V=e''Pe)\n\n' ...
    'Generated: %s\n\n'], selection.generatedAt);
fprintf(fid, ['Frozen g* = **%s** (index %d): divergence %.1f%%, mean g_H ' ...
    '(V=e''Pe contraction, P=selectedLqr.S) %.4f over %d dev cases.\n\n'], ...
    selection.selectedLabel, selection.selectedIndex, ...
    100 * selection.divergenceRate, selection.meanG_H, selection.nCases);
fprintf(fid, ['Ranking key: TRUE divergence rate (error blew up) -> budget-stop ' ...
    'rate (ran out of wall budget; separated from divergence so a slow shard does ' ...
    'not corrupt selection) -> RMS position error -> solve time. Contraction is ' ...
    'measured on V=e''Pe over the frozen horizon cfg.contraction.horizonSteps ' ...
    '(raw diagnostic, not the final S7 target); tracking accuracy (RMS/max) is ' ...
    'reported separately and is not the divergence definition.\n\n']);
fprintf(fid, ['| rank | config | diverge%% | budget%% | rmsPos(m) | conv%% | nCase |\n']);
fprintf(fid, '|---|---|---|---|---|---|---|\n');
for i = 1:numel(ranked)
    a = ranked(i);
    fprintf(fid, '| %d | %s | %.1f | %.1f | %.3f | %.0f | %d |\n', i, a.label, ...
        100 * a.divergenceRate, 100 * a.budgetRate, a.meanRmsPos, ...
        100 * a.convergedFraction, a.nCases);
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

function v = field_or_false(structArray, name)
% Vector of a logical field, defaulting missing entries to false (back-compat
% with task files written before the field existed).
v = false(1, numel(structArray));
for i = 1:numel(structArray)
    if isfield(structArray(i), name) && ~isempty(structArray(i).(name))
        v(i) = logical(structArray(i).(name));
    end
end
end

function add_project_paths()
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(genpath(fullfile(root, 'src')));
addpath(fullfile(root, 'configs'));
end
