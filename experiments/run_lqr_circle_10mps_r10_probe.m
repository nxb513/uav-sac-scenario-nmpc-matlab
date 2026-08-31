function run_lqr_circle_10mps_r10_probe()
%RUN_LQR_CIRCLE_10MPS_R10_PROBE Probe the frozen LQR on a fast circle.

projectRoot = fileparts(fileparts(mfilename('fullpath')));
add_project_paths(projectRoot);

cfg = targeted_lqr_weak_config();
plant = cfg.plant.nominal;
sampleTime = cfg.sampleTime;
stepCount = cfg.stepCount;
outputRoot = fullfile(projectRoot, cfg.resultRoot, ...
    'lqr_circle_10mps_r10_probe');
if isfolder(outputRoot)
    error('run_lqr_circle_10mps_r10_probe:OutputExists', ...
        'Refusing to overwrite %s.', outputRoot);
end
mkdir(outputRoot);

sourcePath = fullfile(projectRoot, cfg.resultRoot, ...
    cfg.specialistBank.sourceSubfolder, ...
    'training_local_context_bank.mat');
artifact = load(sourcePath, 'lqr');
assert(isfield(artifact, 'lqr') && isfield(artifact.lqr, 'K'), ...
    'The frozen selected LQR was not found in %s.', sourcePath);
lqr = artifact.lqr;

options = struct();
options.radius = 10.0;
options.targetPeakSpeed = 10.0;
options.center = [0; 0];
options.altitude = 1.0;
options.phase = 0.0;
options.yawReference = 0.0;
[reference, time, feedforward, flatness] = ...
    quad_targeted_reference_trajectory('circle', sampleTime, ...
    stepCount + 1, options, plant);

X = nan(12, stepCount + 1);
U = nan(4, stepCount);
rawU = nan(4, stepCount);
saturated = false(1, stepCount);
X(:, 1) = reference(:, 1);

completedSteps = 0;
for step = 1:stepCount
    trackingError = nmpc_state_error(X(:, step), reference(:, step));
    rawU(:, step) = feedforward(:, step) - lqr.K * trackingError;
    U(:, step) = quad_saturate_input(rawU(:, step), plant);
    saturated(step) = any(abs(rawU(:, step) - U(:, step)) > 1e-10);
    X(:, step + 1) = quad_step_rk4((step - 1) * sampleTime, ...
        X(:, step), U(:, step), sampleTime, plant, []);
    completedSteps = step;
    if ~all(isfinite(X(:, step + 1)))
        break;
    end
end

sampleCount = completedSteps + 1;
metrics = compute_metrics(X(:, 1:sampleCount), ...
    U(:, 1:completedSteps), rawU(:, 1:completedSteps), ...
    saturated(1:completedSteps), reference(:, 1:sampleCount), ...
    feedforward(:, 1:sampleCount), flatness, plant, ...
    sampleTime, stepCount, completedSteps, options);

save(fullfile(outputRoot, 'lqr_circle_10mps_r10_trace.mat'), ...
    'X', 'U', 'rawU', 'saturated', 'reference', 'feedforward', ...
    'flatness', 'time', 'metrics', 'options', 'lqr', 'cfg', '-v7.3');
writetable(struct2table(metrics), ...
    fullfile(outputRoot, 'lqr_circle_10mps_r10_metrics.csv'));
write_report(outputRoot, metrics, lqr, sourcePath);
plot_xy(outputRoot, reference(:, 1:sampleCount), X(:, 1:sampleCount));
plot_timeseries(outputRoot, time(1:sampleCount), ...
    reference(:, 1:sampleCount), X(:, 1:sampleCount), ...
    U(:, 1:completedSteps), plant);
write_text(fullfile(outputRoot, 'COMPLETED.txt'), sprintf( ...
    'status=%s\ncompleted_steps=%d\nrequested_steps=%d\n', ...
    metrics.Status, completedSteps, stepCount));

fprintf(['LQR speed probe: status=%s, position RMSE=%.6f m, ' ...
    'peak position error=%.6f m, saturation=%.3f%%.\n'], ...
    metrics.Status, metrics.PositionRmseM, ...
    metrics.PeakPositionErrorM, 100 * metrics.SaturationFraction);
end

function metrics = compute_metrics(X, U, rawU, saturated, reference, ...
        feedforward, flatness, plant, sampleTime, requestedSteps, ...
        completedSteps, options)
error = nmpc_state_error(X, reference);
positionError = vecnorm(error(1:3, :), 2, 1);
velocityError = vecnorm(error(7:9, :), 2, 1);
attitudeErrorDeg = rad2deg(vecnorm(error(4:6, :), 2, 1));
referenceSpeed = vecnorm(reference(7:9, :), 2, 1);
actualSpeed = vecnorm(X(7:9, :), 2, 1);
actualTilt = zeros(1, size(X, 2));
for index = 1:size(X, 2)
    rotation = quad_rotm_zyx(X(4:6, index));
    actualTilt(index) = acos(max(-1, min(1, rotation(3, 3))));
end

if completedSteps == requestedSteps && all(isfinite(X), 'all')
    status = "completed";
else
    status = "terminated_nonfinite";
end

metrics = struct();
metrics.Status = status;
metrics.SampleTimeS = sampleTime;
metrics.RequestedSteps = requestedSteps;
metrics.CompletedSteps = completedSteps;
metrics.DurationS = completedSteps * sampleTime;
metrics.RadiusM = options.radius;
metrics.TargetSpeedMps = options.targetPeakSpeed;
metrics.RealizedReferencePeakSpeedMps = max(referenceSpeed);
metrics.PositionRmseM = sqrt(mean(positionError .^ 2));
metrics.PeakPositionErrorM = max(positionError);
metrics.VelocityRmseMps = sqrt(mean(velocityError .^ 2));
metrics.PeakVelocityErrorMps = max(velocityError);
metrics.AttitudeRmseDeg = sqrt(mean(attitudeErrorDeg .^ 2));
metrics.PeakAttitudeErrorDeg = max(attitudeErrorDeg);
metrics.PeakReferenceTiltDeg = rad2deg(max(flatness.tilt(1:size(X, 2))));
metrics.PeakActualTiltDeg = rad2deg(max(actualTilt));
metrics.PeakActualSpeedMps = max(actualSpeed);
metrics.SaturationFraction = mean(saturated);
metrics.PeakThrustFraction = max(U(1, :)) / plant.inputLimits.T(2);
metrics.PeakRawThrustFraction = max(rawU(1, :)) / plant.inputLimits.T(2);
metrics.PeakFeedforwardThrustFraction = ...
    max(feedforward(1, :)) / plant.inputLimits.T(2);
metrics.CentripetalAccelerationMps2 = ...
    options.targetPeakSpeed ^ 2 / options.radius;
metrics.NominalWeightN = plant.m * plant.g;
end

function plot_xy(outputRoot, reference, X)
figureHandle = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [100, 100, 900, 760]);
cleanup = onCleanup(@() close(figureHandle));
plot(reference(1, :), reference(2, :), 'k--', 'LineWidth', 2.0);
hold on;
plot(X(1, :), X(2, :), 'b-', 'LineWidth', 1.6);
plot(reference(1, 1), reference(2, 1), 'ko', ...
    'MarkerFaceColor', 'k', 'MarkerSize', 7);
plot(X(1, end), X(2, end), 'bs', ...
    'MarkerFaceColor', 'b', 'MarkerSize', 7);
axis equal;
grid on;
xlabel('x (m)');
ylabel('y (m)');
title('Frozen LQR: circle R=10 m, v=10 m/s');
legend({'Reference', 'LQR', 'Start', 'End'}, 'Location', 'best');
exportgraphics(figureHandle, fullfile(outputRoot, ...
    'lqr_circle_10mps_r10_xy.png'), 'Resolution', 180);
clear cleanup;
end

function plot_timeseries(outputRoot, time, reference, X, U, plant)
error = nmpc_state_error(X, reference);
positionError = vecnorm(error(1:3, :), 2, 1);
referenceSpeed = vecnorm(reference(7:9, :), 2, 1);
actualSpeed = vecnorm(X(7:9, :), 2, 1);
actualTilt = zeros(1, size(X, 2));
for index = 1:size(X, 2)
    rotation = quad_rotm_zyx(X(4:6, index));
    actualTilt(index) = acos(max(-1, min(1, rotation(3, 3))));
end

figureHandle = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [100, 100, 1000, 900]);
cleanup = onCleanup(@() close(figureHandle));
tiledlayout(4, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
plot(time, positionError, 'r-', 'LineWidth', 1.5);
grid on;
ylabel('||e_p|| (m)');
title('Tracking and control history');

nexttile;
plot(time, referenceSpeed, 'k--', 'LineWidth', 1.5);
hold on;
plot(time, actualSpeed, 'b-', 'LineWidth', 1.5);
grid on;
ylabel('speed (m/s)');
legend({'Reference', 'LQR'}, 'Location', 'best');

nexttile;
plot(time, rad2deg(actualTilt), 'b-', 'LineWidth', 1.5);
grid on;
ylabel('tilt (deg)');

nexttile;
stairs(time(1:end-1), U(1, :) ./ (plant.m * plant.g), ...
    'Color', [0.2, 0.45, 0.2], 'LineWidth', 1.3);
grid on;
xlabel('time (s)');
ylabel('T/(mg)');

exportgraphics(figureHandle, fullfile(outputRoot, ...
    'lqr_circle_10mps_r10_timeseries.png'), 'Resolution', 180);
clear cleanup;
end

function write_report(outputRoot, metrics, lqr, sourcePath)
path = fullfile(outputRoot, 'lqr_circle_10mps_r10_report.md');
fileId = fopen(path, 'w');
assert(fileId >= 0, 'Cannot open %s.', path);
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, '# LQR circle speed probe\n\n');
fprintf(fileId, '- Frozen LQR source: `%s`.\n', sourcePath);
fprintf(fileId, '- Status: `%s`.\n', metrics.Status);
fprintf(fileId, '- Circle: radius `%.3f m`, speed `%.3f m/s`.\n', ...
    metrics.RadiusM, metrics.TargetSpeedMps);
fprintf(fileId, '- Duration: `%.3f s` (`%d/%d` steps).\n', ...
    metrics.DurationS, metrics.CompletedSteps, metrics.RequestedSteps);
fprintf(fileId, '- Position RMSE/peak: `%.6f / %.6f m`.\n', ...
    metrics.PositionRmseM, metrics.PeakPositionErrorM);
fprintf(fileId, '- Velocity RMSE/peak: `%.6f / %.6f m/s`.\n', ...
    metrics.VelocityRmseMps, metrics.PeakVelocityErrorMps);
fprintf(fileId, '- Attitude RMSE/peak: `%.6f / %.6f deg`.\n', ...
    metrics.AttitudeRmseDeg, metrics.PeakAttitudeErrorDeg);
fprintf(fileId, '- Reference/actual peak tilt: `%.6f / %.6f deg`.\n', ...
    metrics.PeakReferenceTiltDeg, metrics.PeakActualTiltDeg);
fprintf(fileId, '- Saturation fraction: `%.6f`.\n', ...
    metrics.SaturationFraction);
fprintf(fileId, '- Peak thrust: `%.6f` of configured maximum.\n', ...
    metrics.PeakThrustFraction);
if isfield(lqr, 'selectionCandidateId')
    fprintf(fileId, '- Frozen candidate: `%s`.\n', ...
        char(string(lqr.selectionCandidateId)));
end
fprintf(fileId, ['\nThis is a nominal-plant feasibility probe only. It ' ...
    'does not establish uncertainty, disturbance or OOD performance.\n']);
clear cleanup;
end

function write_text(path, content)
fileId = fopen(path, 'w');
assert(fileId >= 0, 'Cannot open %s.', path);
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, '%s', content);
clear cleanup;
end

function add_project_paths(projectRoot)
addpath(fullfile(projectRoot, 'configs'));
addpath(fullfile(projectRoot, 'src', 'common'));
addpath(fullfile(projectRoot, 'src', 'plant'));
addpath(fullfile(projectRoot, 'src', 'controllers', 'common'));
end
