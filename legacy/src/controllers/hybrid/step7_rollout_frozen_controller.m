function rollout = step7_rollout_frozen_controller(scenario, controllerName, context)
%STEP7_ROLLOUT_FROZEN_CONTROLLER Run one frozen lightweight controller.

cfg = context.cfg;
steps = cfg.stepCount;
warmup = cfg.warmupSteps;
X = nan(12, steps + 1);
U = nan(4, steps);
rawU = nan(4, steps);
uSurrogate = nan(4, steps);
uFallback = nan(4, steps);
confidence = nan(1, steps);
alpha = nan(1, steps);
controlTime = zeros(1, steps);
saturated = false(1, steps);
fallbackSaturated = false(1, steps);
exitflag = ones(1, steps);
X(:, 1) = scenario.x0;

for k = 1:warmup
    raw = lqr_command(X(:, k), scenario.reference(:, k), ...
        context.selection.selectedLqr);
    U(:, k) = quad_saturate_input(raw, cfg.plant.nominal);
    rawU(:, k) = raw;
    uFallback(:, k) = U(:, k);
    alpha(k) = 0;
    [X(:, k + 1), ~] = plant_step(scenario, X(:, k), U(:, k), k, cfg);
end

startStep = warmup + 1;
stateHistory = X(:, startStep - 3:startStep);
inputHistory = U(:, startStep - 4:startStep - 1);
predictionResidual = recompute_residual(X, U, startStep, cfg);
pidState = initialize_pid_state();
arbState = initialize_arbitration_state();

for k = startStep:steps
    clock = tic;
    if ~all(isfinite(X(:, k))) || ~all(isfinite(stateHistory), 'all') || ...
            ~all(isfinite(inputHistory), 'all') || ...
            ~all(isfinite(predictionResidual))
        controlTime(k) = toc(clock);
        exitflag(k:end) = 0;
        break;
    end
    feature = surrogate_build_feature(stateHistory, inputHistory, ...
        scenario.reference(:, k:k + cfg.referenceLookahead), ...
        predictionResidual);
    if ~all(isfinite(feature))
        controlTime(k) = toc(clock);
        exitflag(k:end) = 0;
        break;
    end
    [uSurrogate(:, k), surrogateDetails] = surrogate_predict_control( ...
        context.surrogate, feature, cfg.plant.nominal);
    if ~all(isfinite(uSurrogate(:, k))) || ...
            ~all(isfinite(surrogateDetails.normalizedClipped)) || ...
            ~all(isfinite(surrogateDetails.normalizedRaw)) || ...
            ~all(isfinite(surrogateDetails.rawControl))
        controlTime(k) = toc(clock);
        exitflag(k:end) = 0;
        break;
    end
    confidenceFeature = confidence_build_feature( ...
        surrogateDetails.normalizedClipped, surrogateDetails.normalizedRaw, ...
        surrogateDetails.rawControl, predictionResidual, cfg.plant.nominal, ...
        cfg.confidenceFeatureSet);
    confidence(k) = predict_confidence(confidenceFeature, context.confidence);
    if ~isfinite(confidence(k))
        controlTime(k) = toc(clock);
        exitflag(k:end) = 0;
        break;
    end

    [kind, fallbackKind, mapping] = controller_contract(controllerName, ...
        context.selection);
    if strcmp(fallbackKind, 'pid')
        [uFallback(:, k), fallbackRaw, pidState] = pid_command( ...
            X(:, k), scenario.reference(:, k), ...
            context.selection.selectedPid, pidState, cfg);
    elseif strcmp(fallbackKind, 'lqr')
        fallbackRaw = lqr_command(X(:, k), scenario.reference(:, k), ...
            context.selection.selectedLqr);
        uFallback(:, k) = quad_saturate_input(fallbackRaw, cfg.plant.nominal);
    else
        fallbackRaw = uSurrogate(:, k);
        uFallback(:, k) = uSurrogate(:, k);
    end
    if ~all(isfinite(fallbackRaw)) || ~all(isfinite(uFallback(:, k)))
        controlTime(k) = toc(clock);
        exitflag(k:end) = 0;
        break;
    end
    fallbackSaturated(k) = any(abs(fallbackRaw - uFallback(:, k)) > 1e-10);

    if strcmp(kind, 'surrogate')
        alpha(k) = 1;
    elseif strcmp(kind, 'fallback')
        alpha(k) = 0;
    else
        [alpha(k), arbState] = arbitration_step(confidence(k), mapping, ...
            arbState, cfg.sampleTime);
    end
    rawU(:, k) = alpha(k) .* uSurrogate(:, k) + ...
        (1 - alpha(k)) .* uFallback(:, k);
    U(:, k) = quad_saturate_input(rawU(:, k), cfg.plant.nominal);
    if ~isfinite(alpha(k)) || ~all(isfinite(rawU(:, k))) || ...
            ~all(isfinite(U(:, k)))
        controlTime(k) = toc(clock);
        exitflag(k:end) = 0;
        break;
    end
    saturated(k) = any(abs(U(:, k) - rawU(:, k)) > 1e-10);
    controlTime(k) = toc(clock);

    time = (k - 1) * cfg.sampleTime;
    nominalPrediction = quad_step_rk4(time, X(:, k), U(:, k), ...
        cfg.sampleTime, cfg.plant.nominal, []);
    [X(:, k + 1), ~] = plant_step(scenario, X(:, k), U(:, k), k, cfg);
    if ~all(isfinite(X(:, k + 1)))
        exitflag(k:end) = 0;
        break;
    end
    predictionResidual = X(:, k + 1) - nominalPrediction;
    stateHistory = [stateHistory(:, 2:end), X(:, k + 1)];
    inputHistory = [inputHistory(:, 2:end), U(:, k)];
end

rollout = pack_rollout(X, U, rawU, uSurrogate, uFallback, confidence, ...
    alpha, controlTime, saturated, fallbackSaturated, exitflag, cfg);
end

function [kind, fallbackKind, mapping] = controller_contract(name, selection)
mapping = struct();
switch name
    case 'H0_surrogate'
        kind = 'surrogate'; fallbackKind = 'surrogate';
    case 'H1_pid'
        kind = 'fallback'; fallbackKind = 'pid';
    case 'H2_lqr'
        kind = 'fallback'; fallbackKind = 'lqr';
    case 'H3_hard_pid'
        kind = 'hybrid'; fallbackKind = 'pid';
        mapping = selection.selectedHardPidMapping;
    case 'H4_weighted_pid'
        kind = 'hybrid'; fallbackKind = 'pid';
        mapping = selection.selectedWeightedPidMapping;
    case 'H5_hard_lqr'
        kind = 'hybrid'; fallbackKind = 'lqr';
        mapping = selection.selectedHardLqrMapping;
    case 'H6_weighted_lqr'
        kind = 'hybrid'; fallbackKind = 'lqr';
        mapping = selection.selectedWeightedLqrMapping;
    otherwise
        error('step7_rollout_frozen_controller:UnknownController', ...
            'Unknown controller: %s', name);
end
end

function probability = predict_confidence(feature, artifact)
normalized = (single(feature) - artifact.normalization.mean) ./ ...
    artifact.normalization.std;
normalized = min(max(normalized, -artifact.normalization.clip), ...
    artifact.normalization.clip);
logit = double(predict(artifact.network, normalized.'));
probability = double(confidence_apply_calibrator( ...
    artifact.calibrator, logit));
probability = probability(1);
end

function state = initialize_pid_state()
state.positionIntegral = zeros(3, 1);
state.attitudeIntegral = zeros(3, 1);
end

function state = initialize_arbitration_state()
state.initialized = false;
state.filteredConfidence = 0;
state.previousFilteredConfidence = 0;
state.alpha = 0;
state.hardMode = 0;
end

function [bounded, raw, state] = pid_command(x, reference, pid, state, cfg)
dt = cfg.sampleTime;
positionError = reference(1:3) - x(1:3);
velocityError = reference(7:9) - x(7:9);
positionIntegral = clamp_vector(state.positionIntegral + ...
    dt .* positionError, pid.positionIntegralLimit);
acceleration = pid.kpPosition .* positionError + ...
    pid.kdVelocity .* velocityError + pid.kiPosition .* positionIntegral;
horizontalNorm = norm(acceleration(1:2));
if horizontalNorm > pid.maximumHorizontalAcceleration
    acceleration(1:2) = acceleration(1:2) .* ...
        pid.maximumHorizontalAcceleration ./ horizontalNorm;
end
acceleration(3) = min(max(acceleration(3), ...
    -pid.maximumVerticalAcceleration), pid.maximumVerticalAcceleration);

yawReference = reference(6);
rollReference = (acceleration(1) * sin(yawReference) - ...
    acceleration(2) * cos(yawReference)) / cfg.plant.nominal.g;
pitchReference = (acceleration(1) * cos(yawReference) + ...
    acceleration(2) * sin(yawReference)) / cfg.plant.nominal.g;
rollReference = min(max(rollReference, ...
    -pid.maximumCommandedTilt), pid.maximumCommandedTilt);
pitchReference = min(max(pitchReference, ...
    -pid.maximumCommandedTilt), pid.maximumCommandedTilt);
attitudeReference = [rollReference; pitchReference; yawReference];
attitudeError = wrap_angles(attitudeReference - x(4:6));
rateError = reference(10:12) - x(10:12);
attitudeIntegral = clamp_vector(state.attitudeIntegral + ...
    dt .* attitudeError, pid.attitudeIntegralLimit);

denominator = max(0.30, cos(x(4)) * cos(x(5)));
thrust = cfg.plant.nominal.m * ...
    (cfg.plant.nominal.g + acceleration(3)) / denominator;
torque = pid.kpAttitude .* attitudeError + ...
    pid.kdRate .* rateError + pid.kiAttitude .* attitudeIntegral + ...
    cross(x(10:12), cfg.plant.nominal.J * x(10:12));
raw = [thrust; torque];
bounded = quad_saturate_input(raw, cfg.plant.nominal);

delta = bounded - raw;
if pid.antiWindupGain > 0
    if pid.kiPosition(3) > 0
        positionIntegral(3) = positionIntegral(3) + dt * ...
            pid.antiWindupGain * delta(1) / ...
            (cfg.plant.nominal.m * pid.kiPosition(3));
    end
    activeAttitude = pid.kiAttitude > 0;
    correction = zeros(3, 1);
    activeIndices = find(activeAttitude);
    correction(activeAttitude) = delta(activeIndices + 1) ./ ...
        pid.kiAttitude(activeAttitude);
    attitudeIntegral = attitudeIntegral + ...
        dt * pid.antiWindupGain .* correction;
end
state.positionIntegral = clamp_vector(positionIntegral, ...
    pid.positionIntegralLimit);
state.attitudeIntegral = clamp_vector(attitudeIntegral, ...
    pid.attitudeIntegralLimit);
end

function raw = lqr_command(x, reference, design)
error = nmpc_state_error(x, reference);
raw = design.uEquilibrium - design.K * error;
end

function [alpha, state] = arbitration_step(confidence, mapping, state, dt)
if ~isfinite(confidence)
    confidence = 0;
end
if ~state.initialized
    state.filteredConfidence = confidence;
    state.previousFilteredConfidence = confidence;
    state.initialized = true;
else
    beta = dt / (mapping.filterTimeConstant + dt);
    state.previousFilteredConfidence = state.filteredConfidence;
    state.filteredConfidence = state.filteredConfidence + ...
        beta * (confidence - state.filteredConfidence);
end
c = state.filteredConfidence;
if strcmp(mapping.kind, 'hard')
    if state.hardMode == 0 && c >= mapping.thresholdHigh
        state.hardMode = 1;
    elseif state.hardMode == 1 && c <= mapping.thresholdLow
        state.hardMode = 0;
    end
    target = double(state.hardMode);
else
    rising = c >= state.previousFilteredConfidence;
    shift = mapping.hysteresisHalfWidth * (2 * double(rising) - 1);
    low = min(max(mapping.thresholdLow + shift, 0), 1);
    high = min(max(mapping.thresholdHigh + shift, low + eps), 1);
    z = min(max((c - low) / (high - low), 0), 1);
    target = z * z * (3 - 2 * z);
end
delta = min(max(target - state.alpha, -mapping.alphaFallRate * dt), ...
    mapping.alphaRiseRate * dt);
state.alpha = min(max(state.alpha + delta, 0), 1);
alpha = state.alpha;
end

function [nextState, theta] = plant_step(scenario, x, u, stepIndex, cfg)
time = (stepIndex - 1) * cfg.sampleTime;
theta = quad_apply_model_shift(scenario.thetaPlant, scenario.shift, time);
theta.inputLimits = cfg.plant.nominal.inputLimits;
nextState = quad_step_rk4(time, x, u, cfg.sampleTime, theta, ...
    scenario.disturbance);
end

function residual = recompute_residual(X, U, stepIndex, cfg)
time = (stepIndex - 2) * cfg.sampleTime;
predicted = quad_step_rk4(time, X(:, stepIndex - 1), ...
    U(:, stepIndex - 1), cfg.sampleTime, cfg.plant.nominal, []);
residual = X(:, stepIndex) - predicted;
end

function value = clamp_vector(value, limit)
value = min(max(value, -limit), limit);
end

function value = wrap_angles(value)
value = atan2(sin(value), cos(value));
end

function rollout = pack_rollout(X, U, rawU, uSurrogate, uFallback, ...
        confidence, alpha, controlTime, saturated, fallbackSaturated, ...
        exitflag, cfg)
rollout.X = X;
rollout.U = U;
rollout.rawU = rawU;
rollout.uSurrogate = uSurrogate;
rollout.uFallback = uFallback;
rollout.confidence = confidence;
rollout.alpha = alpha;
rollout.controlTime = controlTime;
rollout.solveTime = controlTime;
rollout.saturated = saturated;
rollout.fallbackSaturated = fallbackSaturated;
rollout.exitflag = exitflag;
rollout.sampleTime = cfg.sampleTime;
rollout.evaluationStartStep = cfg.warmupSteps + 1;
end
