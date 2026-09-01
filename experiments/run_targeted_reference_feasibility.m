function run_targeted_reference_feasibility()
%RUN_TARGETED_REFERENCE_FEASIBILITY Screen five families over speed/load.

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot, 'configs'));
addpath(genpath(fullfile(projectRoot, 'src')));
cfg = targeted_lqr_weak_config();
nmpcCfg = step2_nmpc_config();
outputRoot = fullfile(projectRoot, cfg.resultRoot, ...
    'reference_feasibility_v1');
if isfolder(outputRoot) && numel(dir(outputRoot)) > 2
    error('run_targeted_reference_feasibility:OutputExists', ...
        'Refusing to overwrite %s.', outputRoot);
end
if ~isfolder(outputRoot)
    mkdir(outputRoot);
end

rows = repmat(empty_row(), 0, 1);
clock = tic;
caseIndex = 0;
for familyIndex = 1:numel(cfg.reference.families)
    family = cfg.reference.families{familyIndex};
    for speed = cfg.reference.approvedIdSpeedAnchors
        for loadIndex = 1:numel( ...
                cfg.reference.approvedAccelerationTargets)
            caseIndex = caseIndex + 1;
            accelerationTarget = ...
                cfg.reference.approvedAccelerationTargets(loadIndex);
            options = build_options(family, speed, accelerationTarget, ...
                cfg.reference, cfg.plant.nominal.g);
            [Xref, ~, Uref, flatness] = ...
                quad_targeted_reference_trajectory(family, ...
                cfg.sampleTime, cfg.stepCount + 1, options, ...
                cfg.plant.nominal);
            metric = quad_reference_feasibility_metrics(Xref, Uref, ...
                flatness, cfg.sampleTime, cfg.plant.nominal, nmpcCfg);
            row = make_row(caseIndex, family, speed, loadIndex, ...
                accelerationTarget, options, metric, cfg.reference);
            rows(end + 1, 1) = row; %#ok<AGROW>
        end
    end
end
wallSeconds = toc(clock);
manifest = struct2table(rows);
summary = summarize(manifest);
writetable(manifest, fullfile(outputRoot, 'feasibility_manifest.csv'));
writetable(summary, fullfile(outputRoot, 'feasibility_summary.csv'));
save(fullfile(outputRoot, 'feasibility_results.mat'), 'cfg', 'manifest', ...
    'summary', 'wallSeconds', '-v7.3');
write_report(outputRoot, cfg, manifest, summary, wallSeconds);
write_text(fullfile(outputRoot, 'COMPLETED.txt'), sprintf( ...
    ['task=run_targeted_reference_feasibility\ncompleted_at=%s\n' ...
    'case_count=%d\nwall_seconds=%.6f\nfigures=false\n' ...
    'training_grid_locked=false\n'], ...
    char(datetime('now', 'TimeZone', 'local', 'Format', ...
    'yyyy-MM-dd HH:mm:ss.SSSXXX')), height(manifest), wallSeconds));
end

function options = build_options(family, speed, acceleration, refCfg, gravity)
scale = max(refCfg.minimumGeometryScale, speed ^ 2 / acceleration);
options = struct('targetPeakSpeed', speed, 'phase', 0, ...
    'yawReference', 0);
switch family
    case 'circle'
        options.center = [0; 0];
        options.altitude = 2.0;
        options.radius = scale;
    case 'lemniscate'
        options.center = [0; 0];
        options.altitude = 2.0;
        options.amplitude = [scale; 0.65 * scale];
    case 'vertical_circle'
        verticalAcceleration = min(acceleration, ...
            refCfg.verticalAccelerationFractionLimit * gravity);
        radius = max(refCfg.minimumGeometryScale, ...
            speed ^ 2 / verticalAcceleration);
        options.center = [0; radius + 1.0];
        options.lateralPosition = 0;
        options.radius = radius;
    case 'spatial_helix'
        options.center = [0; 0];
        options.radius = scale;
        options.altitude = max(2.0, 0.30 * scale + 1.0);
        options.verticalAmplitude = max(0.15, 0.25 * scale);
        options.verticalRatio = 0.5;
        options.verticalPhase = 0;
    case 'smooth_waypoints'
        segmentDistance = max(0.45, 1.65 * speed ^ 2 / acceleration);
        options.waypoints = [0, segmentDistance, segmentDistance, 0; ...
            0, 0, segmentDistance, segmentDistance; ...
            2, 2 + 0.15 * segmentDistance, 2, ...
            2 - 0.10 * segmentDistance];
        segmentLengths = vecnorm(diff(options.waypoints, 1, 2));
        options.segmentDurations = 1.875 .* segmentLengths ./ speed;
    otherwise
        error('run_targeted_reference_feasibility:UnknownFamily', ...
            'Unknown family %s.', family);
end
end

function row = make_row(caseIndex, family, targetSpeed, loadIndex, ...
        accelerationTarget, options, metric, refCfg)
speedError = abs(metric.peakSpeed - targetSpeed) / targetSpeed;
physical = metric.finite && speedError <= refCfg.speedRelativeTolerance && ...
    metric.peakTiltDeg <= refCfg.physicalMaxTiltDeg && ...
    metric.peakInputFraction <= refCfg.physicalMaxInputFraction && ...
    metric.stateBoundViolation <= 1e-8 && ...
    metric.dynamicResidualP95 <= refCfg.dynamicResidualP95Limit;
robust = physical && ...
    metric.peakTiltDeg <= refCfg.robustCandidateMaxTiltDeg && ...
    metric.peakInputFraction <= refCfg.robustCandidateMaxInputFraction;
row = empty_row();
row.CaseIndex = caseIndex;
row.Family = string(family);
row.LoadName = string(refCfg.approvedLoadNames{loadIndex});
row.TargetSpeed = targetSpeed;
row.AccelerationTarget = accelerationTarget;
row.GeometryScale = geometry_scale(options);
row.PeakSpeed = metric.peakSpeed;
row.SpeedRelativeError = speedError;
row.PeakAcceleration = metric.peakAcceleration;
row.PeakHorizontalAcceleration = metric.peakHorizontalAcceleration;
row.PeakTiltDeg = metric.peakTiltDeg;
row.PeakThrustFraction = metric.peakThrustFraction;
row.PeakMomentFraction = metric.peakMomentFraction;
row.PeakInputFraction = metric.peakInputFraction;
row.PeakBodyRate = metric.peakBodyRate;
row.MinimumCosPitch = metric.minimumCosPitch;
row.DynamicResidualP95 = metric.dynamicResidualP95;
row.DynamicResidualMax = metric.dynamicResidualMax;
row.StateBoundViolation = metric.stateBoundViolation;
row.PhysicalFeasible = physical;
row.RobustTrainCandidate = robust;
end

function value = geometry_scale(options)
if isfield(options, 'radius')
    value = options.radius;
elseif isfield(options, 'amplitude')
    value = max(options.amplitude);
else
    value = max(vecnorm(diff(options.waypoints, 1, 2)));
end
end

function summary = summarize(manifest)
families = unique(manifest.Family, 'stable');
rows = repmat(struct('Family', "", 'CaseCount', 0, ...
    'PhysicalFeasibleCount', 0, 'RobustTrainCandidateCount', 0, ...
    'MaximumPhysicalSpeed', NaN, 'MaximumRobustCandidateSpeed', NaN, ...
    'PeakTiltDeg', NaN, 'PeakInputFraction', NaN, ...
    'MaximumDynamicResidualP95', NaN), numel(families), 1);
for index = 1:numel(families)
    subset = manifest(manifest.Family == families(index), :);
    rows(index).Family = families(index);
    rows(index).CaseCount = height(subset);
    rows(index).PhysicalFeasibleCount = nnz(subset.PhysicalFeasible);
    rows(index).RobustTrainCandidateCount = ...
        nnz(subset.RobustTrainCandidate);
    rows(index).MaximumPhysicalSpeed = maximum_eligible_speed(subset, ...
        subset.PhysicalFeasible);
    rows(index).MaximumRobustCandidateSpeed = maximum_eligible_speed( ...
        subset, subset.RobustTrainCandidate);
    rows(index).PeakTiltDeg = max(subset.PeakTiltDeg);
    rows(index).PeakInputFraction = max(subset.PeakInputFraction);
    rows(index).MaximumDynamicResidualP95 = ...
        max(subset.DynamicResidualP95);
end
summary = struct2table(rows);
end

function value = maximum_eligible_speed(subset, mask)
if any(mask)
    value = max(subset.TargetSpeed(mask));
else
    value = NaN;
end
end

function write_report(outputRoot, cfg, manifest, summary, wallSeconds)
fileId = fopen(fullfile(outputRoot, 'feasibility_report.md'), 'w');
assert(fileId >= 0, 'Cannot open feasibility report.');
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, '# Targeted reference feasibility screen\n\n');
fprintf(fileId, '- Cases: `%d`.\n', height(manifest));
fprintf(fileId, '- Wall time: `%.6f s`.\n', wallSeconds);
fprintf(fileId, '- Speed anchors: `%s m/s`.\n', ...
    strjoin(string(cfg.reference.approvedIdSpeedAnchors), ', '));
fprintf(fileId, ['- Physical gate: tilt <= %.1f deg, input fraction ' ...
    '<= %.2f, state bounds valid, p95 dynamics residual <= %.3f.\n'], ...
    cfg.reference.physicalMaxTiltDeg, ...
    cfg.reference.physicalMaxInputFraction, ...
    cfg.reference.dynamicResidualP95Limit);
fprintf(fileId, ['- Robust-train candidate gate additionally requires ' ...
    'tilt <= %.1f deg and input fraction <= %.2f.\n\n'], ...
    cfg.reference.robustCandidateMaxTiltDeg, ...
    cfg.reference.robustCandidateMaxInputFraction);
fprintf(fileId, '## Family summary\n\n');
for index = 1:height(summary)
    row = summary(index, :);
    fprintf(fileId, ['- `%s`: physical `%d/%d`, robust candidate `%d/%d`, ' ...
        'maximum physical/robust speed `%.2f/%.2f m/s`.\n'], ...
        row.Family, row.PhysicalFeasibleCount, row.CaseCount, ...
        row.RobustTrainCandidateCount, row.CaseCount, ...
        row.MaximumPhysicalSpeed, row.MaximumRobustCandidateSpeed);
end
fprintf(fileId, '\n## Interpretation guardrails\n\n');
fprintf(fileId, ['- This screen changes geometry with speed so that speed ' ...
    'is not confounded with an impossible centripetal acceleration.\n']);
fprintf(fileId, ['- A passing row is only a nominal-reference feasibility ' ...
    'candidate; it is not yet an approved training condition.\n']);
fprintf(fileId, ['- Uncertainty, disturbance and closed-loop tracking are ' ...
    'evaluated after the reference envelope is approved.\n']);
fprintf(fileId, ['- OOD speed anchors remain locked and were not evaluated ' ...
    'by this run.\n']);
clear cleanup;
end

function row = empty_row()
row = struct('CaseIndex', 0, 'Family', "", 'LoadName', "", ...
    'TargetSpeed', NaN, 'AccelerationTarget', NaN, ...
    'GeometryScale', NaN, 'PeakSpeed', NaN, ...
    'SpeedRelativeError', NaN, 'PeakAcceleration', NaN, ...
    'PeakHorizontalAcceleration', NaN, 'PeakTiltDeg', NaN, ...
    'PeakThrustFraction', NaN, 'PeakMomentFraction', NaN, ...
    'PeakInputFraction', NaN, 'PeakBodyRate', NaN, ...
    'MinimumCosPitch', NaN, 'DynamicResidualP95', NaN, ...
    'DynamicResidualMax', NaN, 'StateBoundViolation', NaN, ...
    'PhysicalFeasible', false, 'RobustTrainCandidate', false);
end

function write_text(path, content)
fileId = fopen(path, 'w');
assert(fileId >= 0, 'Cannot open %s.', path);
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, '%s', content);
clear cleanup;
end
