function [bank, manifest] = quad_build_balanced_targeted_bank( ...
        cfg, disturbanceCfg, replicatesPerCell, seed)
%QUAD_BUILD_BALANCED_TARGETED_BANK Build a randomized full-factorial bank.
%
% The bank crosses every reference family, nine disturbance conditions, and
% three uncertainty strata. Latin-hypercube magnitudes and random signs are
% sampled independently inside each cell before the rows are randomized.

validateattributes(replicatesPerCell, {'numeric'}, ...
    {'scalar', 'integer', 'positive'});
previousRng = rng;
cleanup = onCleanup(@() rng(previousRng));
rng(seed, 'twister');
nmpcCfg = step2_nmpc_config();

conditions = disturbance_conditions(cfg.retune.disturbanceLevels);
familyCount = numel(cfg.reference.families);
conditionCount = height(conditions);
stratumCount = numel(cfg.uncertainty.stratumNames);
speedAnchors = cfg.reference.approvedIdSpeedAnchors(:).';
accelerationTargets = cfg.reference.approvedAccelerationTargets(:).';
speedCount = numel(speedAnchors);
loadCount = numel(accelerationTargets);
episodeCount = familyCount * conditionCount * stratumCount * ...
    replicatesPerCell;
parameterCount = numel(cfg.plant.uncertainty.names);

specifications = repmat(empty_specification(), episodeCount, 1);
cursor = 0;
for familyIndex = 1:familyCount
    for conditionIndex = 1:conditionCount
        for stratumIndex = 1:stratumCount
            bounds = cfg.uncertainty.stratumXiMagnitude(stratumIndex, :);
            magnitudes = bounds(1) + diff(bounds) .* lhsdesign( ...
                replicatesPerCell, parameterCount, ...
                'Criterion', 'maximin', 'Iterations', 20);
            signs = 2 .* (rand(replicatesPerCell, parameterCount) >= 0.5) - 1;
            xiValues = signs .* magnitudes;
            speedOffset = mod(familyIndex + 2 * conditionIndex + ...
                3 * stratumIndex - 6, speedCount);
            loadOffset = mod(2 * familyIndex + conditionIndex + ...
                stratumIndex - 4, loadCount);
            for replicateIndex = 1:replicatesPerCell
                cursor = cursor + 1;
                specifications(cursor).FamilyIndex = familyIndex;
                specifications(cursor).ConditionIndex = conditionIndex;
                specifications(cursor).StratumIndex = stratumIndex;
                specifications(cursor).ReplicateIndex = replicateIndex;
                specifications(cursor).Xi = xiValues(replicateIndex, :).';
                speedIndex = mod(speedOffset + replicateIndex - 1, ...
                    speedCount) + 1;
                loadIndex = mod(loadOffset + replicateIndex - 1, ...
                    loadCount) + 1;
                specifications(cursor).SpeedAnchorIndex = speedIndex;
                specifications(cursor).LoadIndex = loadIndex;
                specifications(cursor).TargetSpeed = speedAnchors(speedIndex);
                specifications(cursor).TargetAcceleration = ...
                    quad_targeted_load_acceleration( ...
                    speedAnchors(speedIndex), loadIndex, cfg.reference, ...
                    cfg.reference.families{familyIndex});
            end
        end
    end
end
specifications = specifications(randperm(episodeCount));

bank = repmat(empty_scenario(), episodeCount, 1);
rows = repmat(empty_manifest_row(parameterCount), episodeCount, 1);
rho = cfg.uncertainty.targeted.rho(:);
for episodeIndex = 1:episodeCount
    spec = specifications(episodeIndex);
    family = cfg.reference.families{spec.FamilyIndex};
    [options, reference, feedforward, metric] = accepted_reference( ...
        family, spec.TargetSpeed, spec.TargetAcceleration, cfg, nmpcCfg);
    thetaPlant = quad_apply_uncertainty(cfg.plant.nominal, spec.Xi, rho);
    thetaPlant.inputLimits = cfg.plant.nominal.inputLimits;
    disturbanceType = char(conditions.Type(spec.ConditionIndex));
    disturbanceLevel = conditions.Level(spec.ConditionIndex);
    disturbanceSeed = seed + 100000 + episodeIndex;
    disturbance = quad_generate_disturbance_episode(cfg.plant, ...
        disturbanceCfg, disturbanceType, cfg.retune.disturbanceDomain, ...
        disturbanceLevel, cfg.sampleTime, cfg.stepCount, disturbanceSeed);
    x0 = accepted_initial_state(reference(:, 1), ...
        cfg.retune.initialStateStd, nmpcCfg, ...
        cfg.reference.maxSampleAttempts);

    bank(episodeIndex).EpisodeIndex = episodeIndex;
    bank(episodeIndex).CellId = sprintf('F%d_D%d_U%d', ...
        spec.FamilyIndex, spec.ConditionIndex, spec.StratumIndex);
    bank(episodeIndex).Family = family;
    bank(episodeIndex).Options = options;
    bank(episodeIndex).Reference = reference;
    bank(episodeIndex).Feedforward = feedforward;
    bank(episodeIndex).ReferenceMetric = metric;
    bank(episodeIndex).Xi = spec.Xi;
    bank(episodeIndex).UncertaintyStratum = ...
        cfg.uncertainty.stratumNames{spec.StratumIndex};
    bank(episodeIndex).ReplicateIndex = spec.ReplicateIndex;
    bank(episodeIndex).SpeedAnchorIndex = spec.SpeedAnchorIndex;
    bank(episodeIndex).LoadIndex = spec.LoadIndex;
    bank(episodeIndex).TargetReferenceSpeed = spec.TargetSpeed;
    bank(episodeIndex).TargetReferenceAcceleration = ...
        spec.TargetAcceleration;
    bank(episodeIndex).X0 = x0;
    bank(episodeIndex).ThetaPlant = thetaPlant;
    bank(episodeIndex).Disturbance = disturbance;

    row = empty_manifest_row(parameterCount);
    row.EpisodeIndex = episodeIndex;
    row.CellId = string(bank(episodeIndex).CellId);
    row.Family = string(family);
    row.DisturbanceType = string(disturbanceType);
    row.DisturbanceLevel = disturbanceLevel;
    row.UncertaintyStratum = string(bank(episodeIndex).UncertaintyStratum);
    row.ReplicateIndex = spec.ReplicateIndex;
    row.SpeedAnchorIndex = spec.SpeedAnchorIndex;
    row.LoadIndex = spec.LoadIndex;
    row.TargetReferenceSpeed = spec.TargetSpeed;
    row.TargetReferenceAcceleration = spec.TargetAcceleration;
    row.PeakReferenceSpeed = metric.peakSpeed;
    row.PeakReferenceAcceleration = metric.peakAcceleration;
    row.PeakReferenceTiltDeg = metric.peakTiltDeg;
    row.PeakFeedforwardFraction = metric.peakFeedforwardFraction;
    row.ReferenceStateBoundViolation = metric.stateBoundViolation;
    row.InitialStateBoundViolation = max([0; ...
        nmpc_state_bound_violations([x0, x0], nmpcCfg)]);
    row.MassScale = thetaPlant.m / cfg.plant.nominal.m;
    row.MinimumInertiaScale = min(diag(thetaPlant.J) ./ ...
        diag(cfg.plant.nominal.J));
    row.MinimumEffectiveness = min([thetaPlant.alphaT; thetaPlant.alphaTau]);
    row.DisturbanceForcePeak = disturbance.forcePeak;
    row.DisturbanceTorquePeak = disturbance.torquePeak;
    row.DisturbanceSeed = disturbanceSeed;
    row.OptionsJson = string(jsonencode(options));
    row.Xi = spec.Xi.';
    row.X0 = x0.';
    rows(episodeIndex) = row;
end
manifest = struct2table(rows);
validate_reference_stratification(manifest, cfg, replicatesPerCell);
clear cleanup;
end

function conditions = disturbance_conditions(levels)
types = ["zero"; "constant"; "constant"; "gust"; "gust"; ...
    "sinusoidal"; "sinusoidal"; "stochastic"; "stochastic"];
level = [1; levels(1); levels(2); levels(1); levels(2); ...
    levels(1); levels(2); levels(1); levels(2)];
conditions = table(types, level, 'VariableNames', {'Type', 'Level'});
end

function [options, reference, feedforward, metric] = ...
        accepted_reference(family, targetSpeed, targetAcceleration, cfg, ...
        nmpcCfg)
for attempt = 1:cfg.reference.maxSampleAttempts
    options = quad_sample_targeted_reference_options(family, cfg.reference, ...
        targetSpeed, targetAcceleration);
    [reference, ~, feedforward, flatness] = ...
        quad_targeted_reference_trajectory(family, cfg.sampleTime, ...
        cfg.stepCount + 1, options, cfg.plant.nominal);
    metric = reference_metric(reference, feedforward, flatness, cfg, ...
        nmpcCfg);
    speedRelativeError = abs(metric.peakSpeed - targetSpeed) / targetSpeed;
    accelerationRelativeError = abs(metric.peakAcceleration - ...
        targetAcceleration) / targetAcceleration;
    if speedRelativeError <= cfg.reference.speedRelativeTolerance && ...
            accelerationRelativeError <= ...
            cfg.reference.accelerationRelativeTolerance && ...
            metric.peakSpeed >= cfg.reference.targetPeakSpeedRange(1) - 1e-6 && ...
            metric.peakSpeed <= cfg.reference.targetPeakSpeedRange(2) + 1e-6 && ...
            metric.peakAcceleration >= cfg.reference.peakAccelerationRange(1) && ...
            metric.peakAcceleration <= cfg.reference.peakAccelerationRange(2) && ...
            metric.peakTiltDeg <= cfg.reference.maxEquivalentTiltDeg && ...
            metric.peakFeedforwardFraction <= ...
            cfg.reference.maxFeedforwardInputFraction && ...
            metric.stateBoundViolation <= 1e-8
        return;
    end
end
error('quad_build_balanced_targeted_bank:SamplingFailed', ...
    ['Could not sample an accepted %s reference at target speed %.6g ' ...
    'm/s and target acceleration %.6g m/s^2.'], ...
    family, targetSpeed, targetAcceleration);
end

function x0 = accepted_initial_state(referenceState, standardDeviation, ...
        nmpcCfg, maximumAttempts)
for attempt = 1:maximumAttempts
    x0 = referenceState + standardDeviation .* randn(12, 1);
    violation = nmpc_state_bound_violations([x0, x0], nmpcCfg);
    if isempty(violation) || max(violation) <= 1e-8
        return;
    end
end
error('quad_build_balanced_targeted_bank:InitialStateSamplingFailed', ...
    'Could not sample an initial state inside the declared bounds.');
end

function metric = reference_metric(reference, feedforward, flatness, cfg, ...
        nmpcCfg)
metric.peakSpeed = max(vecnorm(reference(7:9, :), 2, 1));
metric.peakAcceleration = max(vecnorm(flatness.acceleration, 2, 1));
metric.peakTiltDeg = max(rad2deg(flatness.tilt));
thrustFraction = feedforward(1, :) ./ cfg.plant.nominal.inputLimits.T(2);
torqueLimits = max(abs(cfg.plant.nominal.inputLimits.tau), [], 2);
torqueFraction = abs(feedforward(2:4, :)) ./ torqueLimits;
metric.peakFeedforwardFraction = max([thrustFraction(:); torqueFraction(:)]);
metric.stateBoundViolation = max([0; ...
    nmpc_state_bound_violations(reference, nmpcCfg)]);
end

function value = empty_specification()
value.FamilyIndex = 0;
value.ConditionIndex = 0;
value.StratumIndex = 0;
value.ReplicateIndex = 0;
value.Xi = [];
value.SpeedAnchorIndex = 0;
value.LoadIndex = 0;
value.TargetSpeed = NaN;
value.TargetAcceleration = NaN;
end

function value = empty_scenario()
value.EpisodeIndex = 0;
value.CellId = '';
value.Family = '';
value.Options = struct();
value.Reference = [];
value.Feedforward = [];
value.ReferenceMetric = struct();
value.Xi = [];
value.UncertaintyStratum = '';
value.ReplicateIndex = 0;
value.SpeedAnchorIndex = 0;
value.LoadIndex = 0;
value.TargetReferenceSpeed = NaN;
value.TargetReferenceAcceleration = NaN;
value.X0 = [];
value.ThetaPlant = struct();
value.Disturbance = struct();
end

function value = empty_manifest_row(parameterCount)
value.EpisodeIndex = 0;
value.CellId = "";
value.Family = "";
value.DisturbanceType = "";
value.DisturbanceLevel = 0;
value.UncertaintyStratum = "";
value.ReplicateIndex = 0;
value.SpeedAnchorIndex = 0;
value.LoadIndex = 0;
value.TargetReferenceSpeed = NaN;
value.TargetReferenceAcceleration = NaN;
value.PeakReferenceSpeed = NaN;
value.PeakReferenceAcceleration = NaN;
value.PeakReferenceTiltDeg = NaN;
value.PeakFeedforwardFraction = NaN;
value.ReferenceStateBoundViolation = NaN;
value.InitialStateBoundViolation = NaN;
value.MassScale = NaN;
value.MinimumInertiaScale = NaN;
value.MinimumEffectiveness = NaN;
value.DisturbanceForcePeak = NaN;
value.DisturbanceTorquePeak = NaN;
value.DisturbanceSeed = 0;
value.OptionsJson = "";
value.Xi = nan(1, parameterCount);
value.X0 = nan(1, 12);
end

function validate_reference_stratification(manifest, cfg, replicatesPerCell)
cellIds = unique(manifest.CellId, 'stable');
requiredSpeeds = min(numel(cfg.reference.approvedIdSpeedAnchors), ...
    replicatesPerCell);
requiredLoads = min(numel(cfg.reference.approvedAccelerationTargets), ...
    replicatesPerCell);
for index = 1:numel(cellIds)
    subset = manifest(manifest.CellId == cellIds(index), :);
    assert(height(subset) == replicatesPerCell, ...
        'Unexpected replicate count in cell %s.', cellIds(index));
    assert(numel(unique(subset.TargetReferenceSpeed)) >= requiredSpeeds, ...
        'Cell %s does not cover the approved speed anchors.', cellIds(index));
    assert(numel(unique(subset.LoadIndex)) >= requiredLoads, ...
        'Cell %s does not cover the approved load targets.', cellIds(index));
end
speedError = abs(manifest.PeakReferenceSpeed - ...
    manifest.TargetReferenceSpeed) ./ manifest.TargetReferenceSpeed;
accelerationError = abs(manifest.PeakReferenceAcceleration - ...
    manifest.TargetReferenceAcceleration) ./ ...
    manifest.TargetReferenceAcceleration;
assert(all(speedError <= cfg.reference.speedRelativeTolerance + eps), ...
    'Bank contains references outside the realized-speed tolerance.');
assert(all(accelerationError <= ...
    cfg.reference.accelerationRelativeTolerance + eps), ...
    'Bank contains references outside the realized-acceleration tolerance.');
assert(all(manifest.ReferenceStateBoundViolation <= 1e-8), ...
    'Bank contains references outside the declared state bounds.');
assert(all(manifest.InitialStateBoundViolation <= 1e-8), ...
    'Bank contains initial states outside the declared state bounds.');
end
