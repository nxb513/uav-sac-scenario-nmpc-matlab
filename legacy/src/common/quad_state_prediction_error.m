function residual = quad_state_prediction_error(actualState, predictedState)
%QUAD_STATE_PREDICTION_ERROR State error with wrapped Euler components.

actualState = actualState(:);
predictedState = predictedState(:);
if numel(actualState) ~= 12 || numel(predictedState) ~= 12
    error('quad_state_prediction_error:BadStateSize', ...
        'Actual and predicted states must each contain 12 elements.');
end
if any(~isfinite(actualState)) || any(~isfinite(predictedState))
    error('quad_state_prediction_error:NonfiniteState', ...
        'Actual and predicted states must be finite.');
end

residual = actualState - predictedState;
residual(4:6) = mod(residual(4:6) + pi, 2 * pi) - pi;
end
