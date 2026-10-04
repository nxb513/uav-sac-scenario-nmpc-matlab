function results = audit_reference_and_inertia_feasibility(varargin)
%AUDIT_REFERENCE_AND_INERTIA_FEASIBILITY Pre-flight physical audit of the bank.
%
% Reproducible, local, no training and no NMPC solves. For every trajectory
% family x speed x acceleration cell it regenerates cases through the SAME
% pipeline the controllers use
%   quad_sample_targeted_reference_options -> quad_targeted_reference_trajectory
%   -> quad_reference_feasibility_metrics
% and records the realized peak tilt / thrust / moment / body-rate / dynamic
% residual, flagging any case that exceeds the declared robust or physical caps.
% It also audits whether independent per-axis inertia sampling can break the
% rigid-body inertia triangle inequality (Ji <= Jj + Jk), the concrete issue
% raised in docs/notes/d1_physical_setup_literature_audit_20260911.md.
%
% Usage:
%   results = audit_reference_and_inertia_feasibility();
%   results = audit_reference_and_inertia_feasibility('numPerCell', 40, ...
%       'seed', 3009, 'save', true);
%
% Returns a struct with per-case records, per-family/per-cell aggregates and
% the inertia-consistency report. Writes JSON + MAT when 'save' is true.

opts = parse_options(varargin{:});
add_project_paths();

cfg = targeted_lqr_weak_config();
ref = cfg.reference;
theta = cfg.plant.nominal;
nmpcCfg = step2_nmpc_config();
sampleTime = cfg.sampleTime;
sampleCount = cfg.stepCount;

rng(opts.seed, 'twister');

fprintf('== Reference feasibility audit ==\n');
fprintf('families=%d  speeds=%d  accels=%d  perCell=%d  steps=%d  dt=%.3f\n', ...
    numel(ref.families), numel(ref.approvedIdSpeedAnchors), ...
    numel(ref.approvedAccelerationTargets), opts.numPerCell, ...
    sampleCount, sampleTime);

caps = struct( ...
    'tiltRobustDeg', ref.robustCandidateMaxTiltDeg, ...
    'tiltPhysicalDeg', ref.physicalMaxTiltDeg, ...
    'inputRobustFraction', ref.robustCandidateMaxInputFraction, ...
    'inputPhysicalFraction', ref.physicalMaxInputFraction, ...
    'residualP95Limit', ref.dynamicResidualP95Limit);

records = reference_records(ref, theta, nmpcCfg, sampleTime, ...
    sampleCount, caps, opts);
[familyTable, cellTable] = summarize_references(records, caps);
inertia = inertia_consistency_report(opts);

print_reference_summary(familyTable, cellTable, caps);
print_inertia_summary(inertia);

results = struct();
results.generatedAt = datestr(now, 'yyyy-mm-dd HH:MM:SS');
results.seed = opts.seed;
results.numPerCell = opts.numPerCell;
results.sampleTime = sampleTime;
results.sampleCount = sampleCount;
results.caps = caps;
results.records = records;
results.familyTable = familyTable;
results.cellTable = cellTable;
results.inertia = inertia;

if opts.save
    outDir = fullfile('results', 'reference_feasibility_audit');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    stamp = datestr(now, 'yyyymmdd_HHMMSS');
    matPath = fullfile(outDir, sprintf('audit_%s.mat', stamp));
    jsonPath = fullfile(outDir, sprintf('audit_%s.json', stamp));
    save(matPath, 'results');
    writeJson(jsonPath, results);
    fprintf('\nSaved:\n  %s\n  %s\n', matPath, jsonPath);
end
end

% ------------------------------------------------------------------------
function records = reference_records(ref, theta, nmpcCfg, sampleTime, ...
        sampleCount, caps, opts)
families = ref.families;
speeds = ref.approvedIdSpeedAnchors;
accels = ref.approvedAccelerationTargets;
records = struct([]);
count = 0;
for fi = 1:numel(families)
    family = families{fi};
    for si = 1:numel(speeds)
        for ai = 1:numel(accels)
            for rep = 1:opts.numPerCell
                count = count + 1;
                rec = one_reference_case(family, speeds(si), accels(ai), ...
                    ref, theta, nmpcCfg, sampleTime, sampleCount, caps);
                if isempty(records)
                    records = rec;
                else
                    records(count) = rec; %#ok<AGROW>
                end
            end
        end
    end
end
end

% ------------------------------------------------------------------------
function rec = one_reference_case(family, speed, accel, ref, theta, ...
        nmpcCfg, sampleTime, sampleCount, caps)
rec = struct( ...
    'family', family, 'speed', speed, 'accel', accel, ...
    'radius', NaN, 'loopCoverage', NaN, ...
    'tiltDeg', NaN, 'thrustFraction', NaN, 'momentFraction', NaN, ...
    'inputFraction', NaN, 'bodyRate', NaN, 'peakAccel', NaN, ...
    'residualP95', NaN, 'stateBound', NaN, 'finite', false, ...
    'genError', '', 'robustOK', false, 'physicalOK', false);
try
    options = quad_sample_targeted_reference_options(family, ref, ...
        speed, accel);
    rec.radius = geometry_scale(options);
    [Xref, ~, Uref, flatness] = quad_targeted_reference_trajectory( ...
        family, sampleTime, sampleCount, options, theta);
    metric = quad_reference_feasibility_metrics(Xref, Uref, flatness, ...
        sampleTime, theta, nmpcCfg);

    rec.tiltDeg = metric.peakTiltDeg;
    rec.thrustFraction = metric.peakThrustFraction;
    rec.momentFraction = metric.peakMomentFraction;
    rec.inputFraction = metric.peakInputFraction;
    rec.bodyRate = metric.peakBodyRate;
    rec.peakAccel = metric.peakAcceleration;
    rec.residualP95 = metric.dynamicResidualP95;
    rec.stateBound = metric.stateBoundViolation;
    rec.finite = logical(metric.finite);
    rec.loopCoverage = loop_coverage(family, rec.radius, speed, ...
        sampleTime, sampleCount);

    rec.robustOK = rec.finite && ...
        rec.tiltDeg <= caps.tiltRobustDeg && ...
        rec.inputFraction <= caps.inputRobustFraction && ...
        rec.residualP95 <= caps.residualP95Limit && ...
        rec.stateBound <= 0;
    rec.physicalOK = rec.finite && ...
        rec.tiltDeg <= caps.tiltPhysicalDeg && ...
        rec.inputFraction <= caps.inputPhysicalFraction && ...
        rec.stateBound <= 0;
catch err
    rec.genError = err.identifier;
end
end

% ------------------------------------------------------------------------
function scale = geometry_scale(options)
if isfield(options, 'radius')
    scale = options.radius;
elseif isfield(options, 'amplitude')
    scale = max(options.amplitude(:));
else
    scale = NaN;
end
end

function coverage = loop_coverage(family, radius, speed, sampleTime, ...
        sampleCount)
% Fraction of a full closed loop covered within the episode (circles only).
coverage = NaN;
if any(strcmpi(family, {'circle', 'vertical_circle'})) && ...
        isfinite(radius) && radius > 0 && speed > 0
    period = 2 * pi * radius / speed;
    coverage = (sampleCount - 1) * sampleTime / period;
end
end

% ------------------------------------------------------------------------
function [familyTable, cellTable] = summarize_references(records, caps)
families = unique({records.family}, 'stable');
familyTable = struct([]);
for fi = 1:numel(families)
    mask = strcmp({records.family}, families{fi});
    sub = records(mask);
    familyTable(fi).family = families{fi}; %#ok<AGROW>
    familyTable(fi).count = numel(sub); %#ok<AGROW>
    familyTable(fi).genErrorFrac = mean(~cellfun(@isempty, ...
        {sub.genError})); %#ok<AGROW>
    familyTable(fi).robustFeasibleFrac = mean([sub.robustOK]); %#ok<AGROW>
    familyTable(fi).physicalFeasibleFrac = mean([sub.physicalOK]); %#ok<AGROW>
    familyTable(fi).medianTiltDeg = nanmedian_local([sub.tiltDeg]); %#ok<AGROW>
    familyTable(fi).p95TiltDeg = p95_local([sub.tiltDeg]); %#ok<AGROW>
    familyTable(fi).medianThrustFrac = nanmedian_local( ...
        [sub.thrustFraction]); %#ok<AGROW>
    familyTable(fi).p95InputFrac = p95_local([sub.inputFraction]); %#ok<AGROW>
    familyTable(fi).p95ResidualP95 = p95_local([sub.residualP95]); %#ok<AGROW>
    familyTable(fi).p95BodyRate = p95_local([sub.bodyRate]); %#ok<AGROW>
end

% Per-cell aggregation, keep only cells that are not fully robust-feasible.
key = arrayfun(@(r) sprintf('%s|%g|%g', r.family, r.speed, r.accel), ...
    records, 'UniformOutput', false);
uniqueKeys = unique(key, 'stable');
cellTable = struct([]);
n = 0;
for ki = 1:numel(uniqueKeys)
    mask = strcmp(key, uniqueKeys{ki});
    sub = records(mask);
    robustFrac = mean([sub.robustOK]);
    if robustFrac >= 1 - eps
        continue;
    end
    n = n + 1;
    cellTable(n).family = sub(1).family; %#ok<AGROW>
    cellTable(n).speed = sub(1).speed; %#ok<AGROW>
    cellTable(n).accel = sub(1).accel; %#ok<AGROW>
    cellTable(n).count = numel(sub); %#ok<AGROW>
    cellTable(n).robustFeasibleFrac = robustFrac; %#ok<AGROW>
    cellTable(n).physicalFeasibleFrac = mean([sub.physicalOK]); %#ok<AGROW>
    cellTable(n).genErrorFrac = mean(~cellfun(@isempty, ...
        {sub.genError})); %#ok<AGROW>
    cellTable(n).p95TiltDeg = p95_local([sub.tiltDeg]); %#ok<AGROW>
    cellTable(n).p95InputFrac = p95_local([sub.inputFraction]); %#ok<AGROW>
    cellTable(n).medianLoopCoverage = nanmedian_local( ...
        [sub.loopCoverage]); %#ok<AGROW>
end
end

% ------------------------------------------------------------------------
function report = inertia_consistency_report(opts)
% Does independent per-axis inertia scaling break Ji <= Jj + Jk?
plant = step1_plant_config();
weak = targeted_lqr_weak_config();
Jnom = diag(plant.nominal.J);

envelopes = struct( ...
    'name', {'train', 'targeted', 'ood'}, ...
    'rho', {plant.uncertainty.train.rho(2:4), ...
            weak.uncertainty.targeted.rho(2:4), ...
            plant.uncertainty.ood.rho(2:4)}, ...
    'rhoFull', {plant.uncertainty.train.rho, ...
                weak.uncertainty.targeted.rho, ...
                plant.uncertainty.ood.rho});

domainName = {'train', 'targeted', 'ood'};
report = struct([]);
for ei = 1:numel(envelopes)
    rho = envelopes(ei).rho(:);
    % Deterministic worst corner for Jz <= Jx + Jy: Jz up, Jx & Jy down.
    worstMargin = (Jnom(1) * (1 - rho(1)) + Jnom(2) * (1 - rho(2))) - ...
        Jnom(3) * (1 + rho(3));
    % (A) Naive independent sampling (the pre-fix problem, for reference).
    N = opts.inertiaSamples;
    xi = 2 * rand(3, N) - 1;
    J = Jnom .* (1 + rho .* xi);
    violatedRaw = (J(3, :) > J(1, :) + J(2, :)) | ...
        (J(1, :) > J(2, :) + J(3, :)) | ...
        (J(2, :) > J(1, :) + J(3, :));
    % (B) Through the fixed sampler (rejection). Should be exactly 0.
    [viaViolationFraction, viaMeanAttempts] = via_sampler_violation( ...
        domainName{ei}, envelopes(ei).rhoFull, opts);
    report(ei).name = envelopes(ei).name; %#ok<AGROW>
    report(ei).rhoInertia = rho.'; %#ok<AGROW>
    report(ei).worstCornerMargin = worstMargin; %#ok<AGROW>
    report(ei).worstCornerViolated = worstMargin < 0; %#ok<AGROW>
    report(ei).mcViolationFractionRaw = mean(violatedRaw); %#ok<AGROW>
    report(ei).viaSamplerViolationFraction = viaViolationFraction; %#ok<AGROW>
    report(ei).viaSamplerMeanAttempts = viaMeanAttempts; %#ok<AGROW>
    report(ei).nominalJ = Jnom.'; %#ok<AGROW>
end
end

function [violationFraction, meanAttempts] = via_sampler_violation( ...
        domain, rho14, opts)
% Draw through the fixed quad_sample_uncertainty and confirm consistency.
plant = step1_plant_config();
plant.uncertainty.(domain).rho = rho14(:);
N = min(opts.inertiaSamples, 5000);
[thetaSamples, ~, info] = quad_sample_uncertainty(plant, N, domain, ...
    opts.seed + 991, 'uniform');
bad = 0;
for k = 1:N
    if ~quad_inertia_consistent(thetaSamples(k).J)
        bad = bad + 1;
    end
end
violationFraction = bad / N;
meanAttempts = info.inertiaMeanAttempts;
end

% ------------------------------------------------------------------------
function print_reference_summary(familyTable, cellTable, caps)
fprintf('\n-- Per-family feasibility (robust caps: tilt<=%.0f deg, ', ...
    caps.tiltRobustDeg);
fprintf('input<=%.2f, residualP95<=%.2f) --\n', ...
    caps.inputRobustFraction, caps.residualP95Limit);
fprintf('%-18s %6s %8s %8s %8s %8s %9s %9s\n', 'family', 'n', ...
    'robust%', 'phys%', 'genErr%', 'medTilt', 'p95Tilt', 'p95Input');
for fi = 1:numel(familyTable)
    t = familyTable(fi);
    fprintf('%-18s %6d %8.1f %8.1f %8.1f %8.1f %9.1f %9.2f\n', ...
        t.family, t.count, 100 * t.robustFeasibleFrac, ...
        100 * t.physicalFeasibleFrac, 100 * t.genErrorFrac, ...
        t.medianTiltDeg, t.p95TiltDeg, t.p95InputFrac);
end

if isempty(cellTable)
    fprintf('\nAll cells are 100%% robust-feasible.\n');
    return;
end
fprintf('\n-- Cells below 100%% robust feasibility (worst first) --\n');
[~, order] = sort([cellTable.robustFeasibleFrac], 'ascend');
fprintf('%-16s %6s %6s %8s %8s %8s %8s %9s\n', 'family', 'v(m/s)', ...
    'a', 'robust%', 'phys%', 'p95Tilt', 'p95Inp', 'loopCov');
shown = 0;
for idx = order
    c = cellTable(idx);
    fprintf('%-16s %6.1f %6.1f %8.1f %8.1f %8.1f %8.2f %9.2f\n', ...
        c.family, c.speed, c.accel, 100 * c.robustFeasibleFrac, ...
        100 * c.physicalFeasibleFrac, c.p95TiltDeg, c.p95InputFrac, ...
        c.medianLoopCoverage);
    shown = shown + 1;
    if shown >= 25
        fprintf('  ... (%d more cells)\n', numel(cellTable) - shown);
        break;
    end
end
end

% ------------------------------------------------------------------------
function print_inertia_summary(report)
fprintf('\n-- Inertia triangle-inequality audit (Ji <= Jj + Jk) --\n');
fprintf('nominal J = [%.4e %.4e %.4e]  (Jz-(Jx+Jy)=%.3e)\n', ...
    report(1).nominalJ(1), report(1).nominalJ(2), report(1).nominalJ(3), ...
    report(1).nominalJ(3) - (report(1).nominalJ(1) + report(1).nominalJ(2)));
fprintf('%-10s %8s %14s %10s %12s %12s\n', 'envelope', 'rhoJ', ...
    'worstMargin', 'rawViol%', 'fixedViol%', 'meanTries');
for ei = 1:numel(report)
    r = report(ei);
    fprintf('%-10s %8.2f %14.3e %10.1f %12.2f %12.2f\n', r.name, ...
        r.rhoInertia(1), r.worstCornerMargin, ...
        100 * r.mcViolationFractionRaw, ...
        100 * r.viaSamplerViolationFraction, r.viaSamplerMeanAttempts);
end
fprintf(['(rawViol%% = naive independent sampling; fixedViol%% = through the ' ...
    'rejection sampler, must be 0)\n']);
end

% ------------------------------------------------------------------------
function opts = parse_options(varargin)
opts = struct('numPerCell', 20, 'seed', 30093011, 'save', true, ...
    'inertiaSamples', 200000);
for k = 1:2:numel(varargin)
    name = varargin{k};
    value = varargin{k + 1};
    if ~isfield(opts, name)
        error('audit_reference_and_inertia_feasibility:BadOption', ...
            'Unknown option %s.', name);
    end
    opts.(name) = value;
end
end

function add_project_paths()
here = fileparts(mfilename('fullpath'));
projectRoot = fileparts(here);
addpath(genpath(fullfile(projectRoot, 'src')));
addpath(fullfile(projectRoot, 'configs'));
end

function value = nanmedian_local(data)
data = data(isfinite(data));
if isempty(data)
    value = NaN;
else
    value = median(data);
end
end

function value = p95_local(data)
data = sort(data(isfinite(data)));
if isempty(data)
    value = NaN;
    return;
end
rankValue = 1 + (numel(data) - 1) * 0.95;
lowerIndex = floor(rankValue);
upperIndex = ceil(rankValue);
weight = rankValue - lowerIndex;
value = (1 - weight) * data(lowerIndex) + weight * data(upperIndex);
end

function out = ternary(condition, a, b)
if condition
    out = a;
else
    out = b;
end
end

function writeJson(path, data)
fid = fopen(path, 'w');
if fid < 0
    warning('audit_reference_and_inertia_feasibility:Json', ...
        'Could not open %s for writing.', path);
    return;
end
fwrite(fid, jsonencode(data));
fclose(fid);
end
