function label = lqr_finite_horizon_divergence_label( ...
        X, rawU, saturated, reference, startStep, cfg, nmpcCfg)
%LQR_FINITE_HORIZON_DIVERGENCE_LABEL Label future LQR tracking loss.

H = cfg.horizonSteps;
validate_inputs(X, rawU, saturated, reference, startStep, H);
stateIndices = startStep:startStep + H;
futureStateIndices = stateIndices(2:end);
inputIndices = startStep:startStep + H - 1;

error = nmpc_state_error(X(:, stateIndices), reference(:, stateIndices));
position = vecnorm(error(1:3, :), 2, 1);
attitudeDeg = rad2deg(vecnorm(error(4:6, :), 2, 1));
velocity = vecnorm(error(7:9, :), 2, 1);
bodyRate = vecnorm(error(10:12, :), 2, 1);
normalized = [position ./ cfg.threshold.positionM; ...
    attitudeDeg ./ cfg.threshold.attitudeDeg; ...
    velocity ./ cfg.threshold.velocityMps; ...
    bodyRate ./ cfg.threshold.bodyRateRadps];
energy = sum(normalized .^ 2, 1);

finiteFuture = all(isfinite(X(:, futureStateIndices)), 1);
absoluteCross = position(2:end) >= cfg.threshold.positionM | ...
    attitudeDeg(2:end) >= cfg.threshold.attitudeDeg | ...
    velocity(2:end) >= cfg.threshold.velocityMps | ...
    bodyRate(2:end) >= cfg.threshold.bodyRateRadps;
constraintCross = state_constraint_crossing( ...
    X(:, futureStateIndices), nmpcCfg);
saturationCross = saturated(inputIndices);
safetyCross = ~finiteFuture | constraintCross | saturationCross;
alreadyOutsideEnvelope = any(normalized(:, 1) >= 1) || ...
    ~all(isfinite(X(:, startStep)));

growthFlags = energy(2:end) >= ...
    (1 + cfg.growth.minimumRelativeStep) .* energy(1:end-1);
growthRun = maximum_true_run(growthFlags);
energyBase = max(energy(1), cfg.growth.energyFloor);
growthFactor = energy(end) / energyBase;
warningCross = max(normalized(:, 2:end), [], 'all') >= ...
    cfg.threshold.warningFraction;
sustainedGrowth = growthFactor >= cfg.growth.factor && ...
    growthRun >= cfg.growth.minimumConsecutiveSteps && warningCross;

eventMask = absoluteCross | safetyCross;
firstHardEvent = find(eventMask, 1, 'first');
hardEvent = ~alreadyOutsideEnvelope && ~isempty(firstHardEvent);
growthOnly = ~alreadyOutsideEnvelope && isempty(firstHardEvent) && ...
    sustainedGrowth;
if hardEvent
    timeToEvent = firstHardEvent;
    hardEventOffset = firstHardEvent;
else
    timeToEvent = Inf;
    hardEventOffset = [];
end

finiteSeverity = normalized(:, 2:end);
finiteSeverity = finiteSeverity(isfinite(finiteSeverity));
if isempty(finiteSeverity)
    severity = 0;
else
    severity = max(finiteSeverity);
end
growthSeverity = growthFactor / cfg.growth.factor;
if ~isfinite(growthSeverity)
    growthSeverity = 0;
end
riskScore = max(severity, growthSeverity);
if any(safetyCross)
    riskScore = max(riskScore, ...
        1e6 + H - find(safetyCross, 1, 'first'));
end
actuatorMargin = minimum_actuator_margin( ...
    rawU(:, inputIndices), nmpcCfg.plant.nominal);

label = struct();
label.CounterfactualDivergence = hardEvent;
label.HardEvent = hardEvent;
label.GrowthOnly = growthOnly;
label.AlreadyOutsideEnvelope = alreadyOutsideEnvelope;
label.PreDivergenceEligible = hardEvent && ...
    timeToEvent >= cfg.selection.minimumLeadSteps && ...
    timeToEvent <= H;
label.GrowthOnlyEligible = growthOnly;
label.TimeToEventSteps = timeToEvent;
label.EventStep = event_step(startStep, hardEventOffset);
label.RiskScore = riskScore;
label.LogEnergyGrowthPerStep = log_growth(energy(1), energy(end), H, ...
    cfg.growth.energyFloor);
label.EnergyStart = energy(1);
label.EnergyEnd = energy(end);
label.EnergyGrowthFactor = growthFactor;
label.MaximumConsecutiveGrowthSteps = growthRun;
label.MaximumPositionErrorM = finite_max(position(2:end));
label.MaximumAttitudeErrorDeg = finite_max(attitudeDeg(2:end));
label.MaximumVelocityErrorMps = finite_max(velocity(2:end));
label.MaximumBodyRateErrorRadps = finite_max(bodyRate(2:end));
label.AbsoluteThresholdCrossing = any(absoluteCross);
label.SustainedGrowth = sustainedGrowth;
label.SaturationForecast = any(saturationCross);
label.ConstraintForecast = any(constraintCross);
label.NonfiniteForecast = any(~finiteFuture);
label.MinimumNormalizedActuatorMargin = actuatorMargin;
end

function crossing = state_constraint_crossing(X, nmpcCfg)
crossing = ~all(isfinite(X), 1);
if ~isfield(nmpcCfg, 'constraints') || ...
        ~nmpcCfg.constraints.enableStateBounds
    return;
end
lower = nmpcCfg.constraints.stateLower(:);
upper = nmpcCfg.constraints.stateUpper(:);
crossing = crossing | any(X < lower | X > upper, 1);
if isfield(nmpcCfg.constraints, 'maxTilt') && ...
        isfinite(nmpcCfg.constraints.maxTilt)
    tilt = sqrt(X(4, :) .^ 2 + X(5, :) .^ 2);
    crossing = crossing | tilt > nmpcCfg.constraints.maxTilt;
end
end

function margin = minimum_actuator_margin(rawU, theta)
lower = [theta.inputLimits.T(1); theta.inputLimits.tau(:, 1)];
upper = [theta.inputLimits.T(2); theta.inputLimits.tau(:, 2)];
range = max(upper - lower, eps);
lowerMargin = (rawU - lower) ./ range;
upperMargin = (upper - rawU) ./ range;
margin = min([lowerMargin(:); upperMargin(:)]);
if ~isfinite(margin)
    margin = -Inf;
end
end

function run = maximum_true_run(flags)
run = 0;
current = 0;
for index = 1:numel(flags)
    if flags(index)
        current = current + 1;
        run = max(run, current);
    else
        current = 0;
    end
end
end

function value = finite_max(data)
data = data(isfinite(data));
if isempty(data)
    value = Inf;
else
    value = max(data);
end
end

function value = log_growth(initialEnergy, finalEnergy, horizon, floorValue)
value = log((finalEnergy + floorValue) / ...
    (initialEnergy + floorValue)) / horizon;
end

function step = event_step(startStep, offset)
if isempty(offset)
    step = Inf;
else
    step = startStep + offset;
end
end

function validate_inputs(X, rawU, saturated, reference, startStep, H)
assert(size(X, 1) == 12 && size(reference, 1) == 12, ...
    'State and reference arrays must have 12 rows.');
assert(size(rawU, 1) == 4 && isvector(saturated), ...
    'rawU must have four rows and saturated must be a vector.');
assert(startStep >= 1 && startStep == floor(startStep), ...
    'startStep must be a positive integer.');
assert(startStep + H <= size(X, 2) && ...
    startStep + H <= size(reference, 2), ...
    'State/reference arrays do not contain the requested future horizon.');
assert(startStep + H - 1 <= size(rawU, 2) && ...
    startStep + H - 1 <= numel(saturated), ...
    'Input arrays do not contain the requested future horizon.');
end
