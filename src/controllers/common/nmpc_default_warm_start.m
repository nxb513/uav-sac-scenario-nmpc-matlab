function U = nmpc_default_warm_start(theta, horizon)
%NMPC_DEFAULT_WARM_START Repeat hover input over the prediction horizon.

if horizon <= 0 || horizon ~= floor(horizon)
    error('nmpc_default_warm_start:BadHorizon', ...
          'horizon must be a positive integer.');
end

uHover = quad_hover_input(theta);
U = repmat(uHover, 1, horizon);
end
