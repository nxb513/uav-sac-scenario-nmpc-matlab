function manifest = build_d1_reference_bank(varargin)
%BUILD_D1_REFERENCE_BANK Stage S0: pre-registered reference bank + splits.
%
% Enumerates the (family x speed x accel) cell grid, generates a fixed number of
% seeded reference cases per cell through the SAME pipeline the controllers use,
% classifies each case into a physical-feasibility tier (quad_reference_
% feasibility_tier) using timewise maxima, and assigns every case a one-way
% train/calib/gate-dev/test split by a deterministic HASH of its trajectory-group
% id (so a group always lands in the same split; regeneration cannot leak a case
% across splits). Also selects the 24 teacher-development cases as a stratified
% subset of the Train x Tier-A cases. No controller is run and no NMPC is solved
% here; membership is controller-independent.
%
% Usage:
%   manifest = build_d1_reference_bank();                       % defaults, saves
%   manifest = build_d1_reference_bank('casesPerCell', 30, 'save', true);
%
% Frozen parameters live in the returned manifest.params and are documented in
% docs/notes/d1_preregistration_20260912.md.

opts = parse_options(varargin{:});
add_project_paths();

cfg = targeted_lqr_weak_config();
ref = cfg.reference;
theta = cfg.plant.nominal;
nmpcCfg = step2_nmpc_config();
sampleTime = cfg.sampleTime;
sampleCount = cfg.stepCount;

caps = struct( ...
    'tiltRobustDeg', ref.robustCandidateMaxTiltDeg, ...
    'tiltPhysicalDeg', ref.physicalMaxTiltDeg, ...
    'inputRobustFraction', ref.robustCandidateMaxInputFraction, ...
    'inputPhysicalFraction', ref.physicalMaxInputFraction, ...
    'residualMaxLimit', opts.residualMaxLimit);

splitEdges = struct('train', opts.fracTrain, ...
    'calib', opts.fracTrain + opts.fracCalib, ...
    'gatedev', opts.fracTrain + opts.fracCalib + opts.fracGateDev);
% test = remainder.

rng(opts.seed, 'twister');

fprintf('== S0 build_d1_reference_bank ==\n');
fprintf('grid: %d families x %d speeds x %d accels x %d cases = %d cases\n', ...
    numel(ref.families), numel(ref.approvedIdSpeedAnchors), ...
    numel(ref.approvedAccelerationTargets), opts.casesPerCell, ...
    numel(ref.families) * numel(ref.approvedIdSpeedAnchors) * ...
    numel(ref.approvedAccelerationTargets) * opts.casesPerCell);

cases = generate_cases(ref, theta, nmpcCfg, sampleTime, sampleCount, ...
    caps, splitEdges, opts);

teacherDevIds = select_teacher_dev(cases, opts.teacherDevCount, ...
    ref.approvedAccelerationTargets);
for k = 1:numel(cases)
    cases(k).isTeacherDev = any(strcmp(cases(k).groupId, teacherDevIds));
end

manifest = struct();
manifest.version = 'd1_reference_bank_v1';
manifest.generatedAt = datestr(now, 'yyyy-mm-dd HH:MM:SS');
manifest.params = struct('seed', opts.seed, 'casesPerCell', opts.casesPerCell, ...
    'families', {ref.families}, 'speeds', ref.approvedIdSpeedAnchors, ...
    'accels', ref.approvedAccelerationTargets, 'caps', caps, ...
    'splitFractions', struct('train', opts.fracTrain, 'calib', opts.fracCalib, ...
        'gatedev', opts.fracGateDev, 'test', ...
        1 - opts.fracTrain - opts.fracCalib - opts.fracGateDev), ...
    'teacherDevCount', opts.teacherDevCount, ...
    'sampleTime', sampleTime, 'sampleCount', sampleCount);
manifest.cases = cases;
manifest.summary = summarize(cases, ref);

print_summary(manifest.summary, cases);

if opts.save
    outDir = fullfile('results', 'd1_bank');
    if ~exist(outDir, 'dir'); mkdir(outDir); end
    stamp = datestr(now, 'yyyymmdd_HHMMSS');
    save(fullfile(outDir, sprintf('reference_bank_%s.mat', stamp)), 'manifest');
    writeJson(fullfile(outDir, sprintf('reference_bank_%s.json', stamp)), ...
        rmfield(manifest, 'cases'));
    write_cases_csv(fullfile(outDir, sprintf('reference_bank_%s.csv', stamp)), ...
        cases);
    fprintf('\nSaved bank to %s (reference_bank_%s.*)\n', outDir, stamp);
end
end

% ------------------------------------------------------------------------
function cases = generate_cases(ref, theta, nmpcCfg, sampleTime, ...
        sampleCount, caps, splitEdges, opts)
families = ref.families;
speeds = ref.approvedIdSpeedAnchors;
accels = ref.approvedAccelerationTargets;
total = numel(families) * numel(speeds) * numel(accels) * opts.casesPerCell;
cases = repmat(empty_case(), total, 1);
idx = 0;
for fi = 1:numel(families)
    for si = 1:numel(speeds)
        for ai = 1:numel(accels)
            for rep = 1:opts.casesPerCell
                idx = idx + 1;
                cases(idx) = one_case(families{fi}, speeds(si), accels(ai), ...
                    rep, ref, theta, nmpcCfg, sampleTime, sampleCount, ...
                    caps, splitEdges);
            end
        end
    end
end
end

function c = one_case(family, speed, accel, rep, ref, theta, nmpcCfg, ...
        sampleTime, sampleCount, caps, splitEdges)
c = empty_case();
c.family = family; c.speed = speed; c.accel = accel; c.rep = rep;
c.groupId = sprintf('%s|v%g|a%g|r%d', family, speed, accel, rep);
c.split = assign_split(c.groupId, splitEdges);
try
    options = quad_sample_targeted_reference_options(family, ref, speed, accel);
    if isfield(options, 'radius'); c.radius = options.radius;
    elseif isfield(options, 'amplitude'); c.radius = max(options.amplitude(:)); end
    [Xref, ~, Uref, flatness] = quad_targeted_reference_trajectory( ...
        family, sampleTime, sampleCount, options, theta);
    metric = quad_reference_feasibility_metrics(Xref, Uref, flatness, ...
        sampleTime, theta, nmpcCfg);
    c.peakTiltDeg = metric.peakTiltDeg;
    c.peakInputFraction = metric.peakInputFraction;
    c.peakBodyRate = metric.peakBodyRate;
    c.peakAccel = metric.peakAcceleration;
    c.residualMax = metric.dynamicResidualMax;
    c.stateBound = metric.stateBoundViolation;
    c.finite = logical(metric.finite);
    [c.tier, reasons] = quad_reference_feasibility_tier(metric, caps);
    c.reasons = strjoin(cellstr(reasons), '+');
catch err
    c.tier = 'C_infeasible';
    c.reasons = ['genError:' err.identifier];
    c.finite = false;
end
end

function c = empty_case()
c = struct('family', '', 'speed', NaN, 'accel', NaN, 'rep', NaN, ...
    'groupId', '', 'split', '', 'isTeacherDev', false, 'radius', NaN, ...
    'peakTiltDeg', NaN, 'peakInputFraction', NaN, 'peakBodyRate', NaN, ...
    'peakAccel', NaN, 'residualMax', NaN, 'stateBound', NaN, ...
    'finite', false, 'tier', '', 'reasons', '');
end

% ------------------------------------------------------------------------
function split = assign_split(groupId, edges)
u = hash_unit(groupId);
if u < edges.train
    split = 'train';
elseif u < edges.calib
    split = 'calib';
elseif u < edges.gatedev
    split = 'gatedev';
else
    split = 'test';
end
end

function u = hash_unit(str)
% Deterministic, machine-independent hash of a string to [0,1) via MD5.
md = java.security.MessageDigest.getInstance('MD5');
raw = mod(double(md.digest(uint8(char(str)))), 256);
val = raw(1) + raw(2) * 256 + raw(3) * 65536 + raw(4) * 16777216;
u = val / 2^32;
end

% ------------------------------------------------------------------------
function ids = select_teacher_dev(cases, count, accels)
% Stratified deterministic pick from Train x Tier-A: spread over families and
% acceleration levels, easy+hard. Ordered by hash for reproducibility.
eligible = cases(strcmp({cases.split}, 'train') & ...
    strcmp({cases.tier}, 'A_primary'));
if numel(eligible) < count
    warning('build_d1_reference_bank:TeacherDev', ...
        'Only %d eligible teacher-dev cases (< %d requested).', ...
        numel(eligible), count);
    ids = {eligible.groupId};
    return;
end
families = unique({eligible.family}, 'stable');
buckets = {};
for fi = 1:numel(families)
    for ai = 1:numel(accels)
        mask = strcmp({eligible.family}, families{fi}) & ...
            [eligible.accel] == accels(ai);
        sub = eligible(mask);
        if isempty(sub); continue; end
        h = arrayfun(@(c) hash_unit([c.groupId '|tdev']), sub);
        [~, order] = sort(h);
        buckets{end+1} = {sub(order).groupId}; %#ok<AGROW>
    end
end
% Round-robin across buckets until we have `count`.
ids = {};
bi = 1; guard = 0;
while numel(ids) < count && guard < 100000
    b = buckets{mod(bi - 1, numel(buckets)) + 1};
    if ~isempty(b)
        ids{end+1} = b{1}; %#ok<AGROW>
        buckets{mod(bi - 1, numel(buckets)) + 1} = b(2:end);
    end
    bi = bi + 1; guard = guard + 1;
end
ids = unique(ids, 'stable');
ids = ids(1:min(count, numel(ids)));
end

% ------------------------------------------------------------------------
function summary = summarize(cases, ref)
tiers = {'A_primary', 'B_boundary', 'C_infeasible', 'U_unresolved'};
summary.tierCounts = struct();
for t = 1:numel(tiers)
    summary.tierCounts.(matlab.lang.makeValidName(tiers{t})) = ...
        sum(strcmp({cases.tier}, tiers{t}));
end
summary.total = numel(cases);
% Tier-A split sizes.
splits = {'train', 'calib', 'gatedev', 'test'};
summary.primarySplitCounts = struct();
isA = strcmp({cases.tier}, 'A_primary');
for s = 1:numel(splits)
    summary.primarySplitCounts.(splits{s}) = ...
        sum(isA & strcmp({cases.split}, splits{s}));
end
summary.teacherDevCount = sum([cases.isTeacherDev]);
% Per-family Tier-A fraction.
fams = ref.families;
summary.perFamilyPrimaryFrac = struct();
for fi = 1:numel(fams)
    mask = strcmp({cases.family}, fams{fi});
    summary.perFamilyPrimaryFrac.(fams{fi}) = ...
        mean(strcmp({cases(mask).tier}, 'A_primary'));
end
% Residual distribution (to justify residualMaxLimit).
res = [cases.residualMax];
res = res(isfinite(res));
summary.residualP50 = median(res);
summary.residualP95 = quantile_local(res, 0.95);
summary.residualP99 = quantile_local(res, 0.99);
summary.residualMax = max(res);
end

function print_summary(summary, cases)
fprintf('\n-- Tier counts (of %d) --\n', summary.total);
fn = fieldnames(summary.tierCounts);
for i = 1:numel(fn)
    fprintf('  %-14s %5d (%.1f%%)\n', fn{i}, summary.tierCounts.(fn{i}), ...
        100 * summary.tierCounts.(fn{i}) / summary.total);
end
fprintf('\n-- Tier-A (primary) split sizes --\n');
fn = fieldnames(summary.primarySplitCounts);
for i = 1:numel(fn)
    fprintf('  %-8s %5d\n', fn{i}, summary.primarySplitCounts.(fn{i}));
end
fprintf('  teacher-dev (subset of train): %d\n', summary.teacherDevCount);
fprintf('\n-- Per-family Tier-A fraction --\n');
fn = fieldnames(summary.perFamilyPrimaryFrac);
for i = 1:numel(fn)
    fprintf('  %-18s %.1f%%\n', fn{i}, 100 * summary.perFamilyPrimaryFrac.(fn{i}));
end
fprintf('\n-- Residual (max per case) distribution --\n');
fprintf('  p50=%.3f  p95=%.3f  p99=%.3f  max=%.3f\n', ...
    summary.residualP50, summary.residualP95, summary.residualP99, ...
    summary.residualMax);
nA = sum(strcmp({cases.tier}, 'A_primary'));
fprintf('\nTier-A total = %d eligible cases.\n', nA);
end

% ------------------------------------------------------------------------
function opts = parse_options(varargin)
opts = struct('casesPerCell', 30, 'seed', 30120901, 'save', true, ...
    'residualMaxLimit', 1.0, 'teacherDevCount', 24, ...
    'fracTrain', 0.40, 'fracCalib', 0.15, 'fracGateDev', 0.15);
for k = 1:2:numel(varargin)
    if ~isfield(opts, varargin{k})
        error('build_d1_reference_bank:BadOption', 'Unknown option %s.', ...
            varargin{k});
    end
    opts.(varargin{k}) = varargin{k + 1};
end
end

function add_project_paths()
here = fileparts(mfilename('fullpath'));
projectRoot = fileparts(here);
addpath(genpath(fullfile(projectRoot, 'src')));
addpath(fullfile(projectRoot, 'configs'));
end

function value = quantile_local(data, p)
data = sort(data(isfinite(data)));
if isempty(data); value = NaN; return; end
r = 1 + (numel(data) - 1) * p;
lo = floor(r); hi = ceil(r); w = r - lo;
value = (1 - w) * data(lo) + w * data(hi);
end

function write_cases_csv(path, cases)
fid = fopen(path, 'w');
if fid < 0; return; end
fprintf(fid, ['groupId,family,speed,accel,rep,split,isTeacherDev,tier,' ...
    'radius,peakTiltDeg,peakInputFraction,peakBodyRate,peakAccel,' ...
    'residualMax,stateBound,finite,reasons\n']);
for k = 1:numel(cases)
    c = cases(k);
    fprintf(fid, '%s,%s,%g,%g,%d,%s,%d,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.5f,%.5f,%d,%s\n', ...
        c.groupId, c.family, c.speed, c.accel, c.rep, c.split, c.isTeacherDev, ...
        c.tier, c.radius, c.peakTiltDeg, c.peakInputFraction, c.peakBodyRate, ...
        c.peakAccel, c.residualMax, c.stateBound, c.finite, c.reasons);
end
fclose(fid);
end

function writeJson(path, data)
fid = fopen(path, 'w');
if fid < 0; return; end
fwrite(fid, jsonencode(data));
fclose(fid);
end
