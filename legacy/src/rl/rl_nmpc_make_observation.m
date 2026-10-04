function observation = rl_nmpc_make_observation(state, reference, ...
        previousInput, residual, solveTime, horizon, cfg, varargin)
%RL_NMPC_MAKE_OBSERVATION Build SAC observation consistently.

error = nmpc_state_error(state, reference);
uHover = quad_hover_input(cfg.nmpc.plant.nominal);
inputScale = actuator_ranges(cfg.nmpc.plant.nominal);
horizonBank = cfg.environment.horizonBank;
if isscalar(horizonBank)
    normalizedHorizon = 0.0;
else
    normalizedHorizon = 2.0 * (horizon - min(horizonBank)) / ...
        (max(horizonBank) - min(horizonBank)) - 1.0;
end
observation = [error ./ cfg.observation.errorScale; ...
    (previousInput - uHover) ./ inputScale; ...
    residual ./ cfg.observation.residualScale; ...
    solveTime / cfg.observation.solveTimeScale; normalizedHorizon];
if cfg.observation.dimension >= 31
    requiredArguments = cfg.observation.dimension - 30;
    assert(numel(varargin) == requiredArguments, ...
        'Expanded observation has the wrong structural arguments.');
    normalizedControl = normalize_bank_value(varargin{1}, ...
        cfg.environment.controlHorizonBank);
    observation = [observation; normalizedControl];
end
if cfg.observation.dimension >= 32
    assert(numel(varargin) == 2, ...
        'Expanded observation requires control horizon and scenario count.');
    normalizedScenario = normalize_bank_value(varargin{2}, ...
        cfg.environment.scenarioCountBank);
    observation = [observation; normalizedScenario];
end
assert(numel(observation) == cfg.observation.dimension, ...
    'Observation dimension does not match its numeric specification.');
observation = min(max(observation, -cfg.observation.clip), ...
    cfg.observation.clip);
end

function normalized = normalize_bank_value(value, bank)
bank = bank(:).';
if isscalar(bank)
    normalized = 0.0;
else
    normalized = 2.0 * (value - min(bank)) / ...
        (max(bank) - min(bank)) - 1.0;
end
end

function ranges = actuator_ranges(theta)
ranges = [theta.inputLimits.T(2) - theta.inputLimits.T(1); ...
          theta.inputLimits.tau(:, 2) - theta.inputLimits.tau(:, 1)];
ranges = max(ranges, eps);
end
