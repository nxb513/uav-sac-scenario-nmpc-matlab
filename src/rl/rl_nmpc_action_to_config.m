function [nmpcCfg, mapping] = rl_nmpc_action_to_config(action, cfg)
%RL_NMPC_ACTION_TO_CONFIG Map SAC action to weights and NMPC structure.

action = double(action(:));
if numel(action) ~= cfg.action.dimension || any(~isfinite(action))
    error('rl_nmpc_action_to_config:BadAction', ...
        'Action has the wrong dimension or contains nonfinite values.');
end
action = min(max(action, -1.0), 1.0);

nmpcCfg = cfg.nmpc;
if isfield(cfg.action, 'weightParameterization') && ...
        strcmp(cfg.action.weightParameterization, ...
        'absolute_dimensionless_normalized')
    groupWeights = map_log_groups(action(1:6), ...
        cfg.action.groupWeightBounds);
    errorScale = cfg.observation.errorScale(:);
    inputScale = actuator_ranges(cfg.nmpc.plant.nominal);
    Q = [repmat(groupWeights(1), 3, 1); ...
        repmat(groupWeights(2), 3, 1); ...
        repmat(groupWeights(3), 3, 1); ...
        repmat(groupWeights(4), 3, 1)] ./ (errorScale .^ 2);
    R = [groupWeights(5); repmat(groupWeights(6), 3, 1)] ./ ...
        (inputScale .^ 2);
    nmpcCfg.weights.Q = diag(Q);
    nmpcCfg.weights.Qf = cfg.action.terminalWeightMultiplier * ...
        nmpcCfg.weights.Q;
    nmpcCfg.weights.R = diag(R);
    mapping.groupWeights = groupWeights;
    mapping.multipliers = nan(6, 1);
else
    bounds = cfg.action.logMultiplierBounds;
    logMultiplier = log(bounds(1)) + 0.5 * (action(1:6) + 1.0) * ...
        (log(bounds(2)) - log(bounds(1)));
    multipliers = exp(logMultiplier);
    Q0 = diag(cfg.nmpc.weights.Q);
    R0 = diag(cfg.nmpc.weights.R);
    Q = Q0;
    Q(1:3) = Q0(1:3) * multipliers(1);
    Q(4:6) = Q0(4:6) * multipliers(2);
    Q(7:9) = Q0(7:9) * multipliers(3);
    Q(10:12) = Q0(10:12) * multipliers(4);
    R = R0;
    R(1) = R0(1) * multipliers(5);
    R(2:4) = R0(2:4) * multipliers(6);
    nmpcCfg.weights.Q = diag(Q);
    nmpcCfg.weights.Qf = 4.0 * nmpcCfg.weights.Q;
    nmpcCfg.weights.R = diag(R);
    mapping.groupWeights = nan(6, 1);
    mapping.multipliers = multipliers;
end

horizonBank = cfg.environment.horizonBank(:).';
horizonIndex = bounded_index(action(7), numel(horizonBank));
nmpcCfg.predictionHorizon = horizonBank(horizonIndex);

if cfg.action.dimension >= 8
    controlBank = cfg.environment.controlHorizonBank(:).';
    feasibleControl = controlBank(controlBank <= ...
        nmpcCfg.predictionHorizon);
    controlIndex = bounded_index(action(8), numel(feasibleControl));
    nmpcCfg.controlHorizon = feasibleControl(controlIndex);
else
    nmpcCfg.controlHorizon = min(cfg.environment.controlHorizon, ...
        nmpcCfg.predictionHorizon);
    controlIndex = NaN;
end

if cfg.action.dimension >= 9
    scenarioBank = cfg.environment.scenarioCountBank(:).';
    scenarioIndex = bounded_index(action(9), numel(scenarioBank));
    nmpcCfg.scenario.count = scenarioBank(scenarioIndex);
else
    scenarioIndex = NaN;
end

mapping.action = action;
mapping.horizonIndex = horizonIndex;
mapping.horizon = nmpcCfg.predictionHorizon;
mapping.controlHorizonIndex = controlIndex;
mapping.controlHorizon = nmpcCfg.controlHorizon;
mapping.scenarioCountIndex = scenarioIndex;
mapping.scenarioCount = nmpcCfg.scenario.count;
end

function weights = map_log_groups(action, bounds)
assert(isequal(size(bounds), [6, 2]), ...
    'groupWeightBounds must be a 6-by-2 matrix.');
assert(all(bounds(:, 1) > 0) && all(bounds(:, 2) > bounds(:, 1)), ...
    'Every group-weight interval must be positive and increasing.');
fraction = 0.5 * (action(:) + 1.0);
weights = exp(log(bounds(:, 1)) + fraction .* ...
    (log(bounds(:, 2)) - log(bounds(:, 1))));
end

function index = bounded_index(value, count)
assert(count >= 1, 'A discrete action bank cannot be empty.');
index = min(count, floor(0.5 * (value + 1.0) * count) + 1);
end

function ranges = actuator_ranges(theta)
ranges = [theta.inputLimits.T(2) - theta.inputLimits.T(1); ...
    theta.inputLimits.tau(:, 2) - theta.inputLimits.tau(:, 1)];
ranges = max(ranges, eps);
end
