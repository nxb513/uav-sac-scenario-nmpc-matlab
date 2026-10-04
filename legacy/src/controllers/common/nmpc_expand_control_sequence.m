function U = nmpc_expand_control_sequence(Ucontrol, predictionHorizon)
%NMPC_EXPAND_CONTROL_SEQUENCE Hold the final move after control horizon.

if size(Ucontrol, 1) ~= 4 || isempty(Ucontrol)
    error('nmpc_expand_control_sequence:BadInput', ...
          'Ucontrol must be a nonempty 4-by-Nc matrix.');
end
controlHorizon = size(Ucontrol, 2);
if predictionHorizon < controlHorizon || predictionHorizon ~= floor(predictionHorizon)
    error('nmpc_expand_control_sequence:BadHorizon', ...
          'predictionHorizon must be an integer at least as large as Nc.');
end

U = [Ucontrol, repmat(Ucontrol(:, end), 1, predictionHorizon - controlHorizon)];
end
