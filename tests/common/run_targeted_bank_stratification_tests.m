function run_targeted_bank_stratification_tests()
%RUN_TARGETED_BANK_STRATIFICATION_TESTS Check approved speed/load coverage.

projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(projectRoot, 'configs'));
addpath(genpath(fullfile(projectRoot, 'src')));
cfg = targeted_lqr_weak_config();
disturbanceCfg = step3_disturbance_config();
[bank, manifest] = quad_build_balanced_targeted_bank( ...
    cfg, disturbanceCfg, 10, 26090102);

assert(numel(bank) == 1350 && height(manifest) == 1350);
cellIds = unique(manifest.CellId, 'stable');
assert(numel(cellIds) == 135);
for index = 1:numel(cellIds)
    subset = manifest(manifest.CellId == cellIds(index), :);
    assert(numel(unique(subset.TargetReferenceSpeed)) == ...
        numel(cfg.reference.approvedIdSpeedAnchors));
    assert(numel(unique(subset.LoadIndex)) == ...
        numel(cfg.reference.approvedLoadNames));
end
assert(min(manifest.PeakReferenceSpeed) >= 0.5 - 1e-6);
assert(max(manifest.PeakReferenceSpeed) <= 12.0 + 1e-6);
speedError = abs(manifest.PeakReferenceSpeed - ...
    manifest.TargetReferenceSpeed) ./ manifest.TargetReferenceSpeed;
accelerationError = abs(manifest.PeakReferenceAcceleration - ...
    manifest.TargetReferenceAcceleration) ./ ...
    manifest.TargetReferenceAcceleration;
assert(max(speedError) <= cfg.reference.speedRelativeTolerance + eps);
assert(max(accelerationError) <= ...
    cfg.reference.accelerationRelativeTolerance + eps);
for row = 1:height(manifest)
    expectedAcceleration = quad_targeted_load_acceleration( ...
        manifest.TargetReferenceSpeed(row), manifest.LoadIndex(row), ...
        cfg.reference, manifest.Family(row));
    assert(abs(manifest.TargetReferenceAcceleration(row) - ...
        expectedAcceleration) < 1e-12);
end
fprintf(['Targeted bank stratification passed: %d episodes, %d cells, ' ...
    'eight speeds and three feasible load levels per cell.\n'], ...
    height(manifest), numel(cellIds));
end
