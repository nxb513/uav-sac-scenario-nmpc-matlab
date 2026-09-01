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
    assert(numel(unique(subset.TargetReferenceAcceleration)) == ...
        numel(cfg.reference.approvedAccelerationTargets));
end
assert(min(manifest.PeakReferenceSpeed) >= 0.5 - 1e-6);
assert(max(manifest.PeakReferenceSpeed) <= 12.0 + 1e-6);
fprintf(['Targeted bank stratification passed: %d episodes, %d cells, ' ...
    'eight speeds and three load targets per cell.\n'], ...
    height(manifest), numel(cellIds));
end
