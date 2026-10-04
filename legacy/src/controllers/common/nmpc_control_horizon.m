function controlHorizon = nmpc_control_horizon(cfg)
%NMPC_CONTROL_HORIZON Validate and return the number of optimized moves.

predictionHorizon = cfg.predictionHorizon;
if isfield(cfg, 'controlHorizon') && ~isempty(cfg.controlHorizon)
    controlHorizon = cfg.controlHorizon;
else
    controlHorizon = predictionHorizon;
end

if predictionHorizon <= 0 || predictionHorizon ~= floor(predictionHorizon)
    error('nmpc_control_horizon:BadPredictionHorizon', ...
          'predictionHorizon must be a positive integer.');
end
if controlHorizon <= 0 || controlHorizon ~= floor(controlHorizon) || ...
        controlHorizon > predictionHorizon
    error('nmpc_control_horizon:BadControlHorizon', ...
          'controlHorizon must be an integer in [1, predictionHorizon].');
end
end
