function Xref = nmpc_reference_window(reference, stepIndex, currentTime, cfg)
%NMPC_REFERENCE_WINDOW Extract a 12-by-(N+1) reference window for MPC.

if stepIndex <= 0 || stepIndex ~= floor(stepIndex)
    error('nmpc_reference_window:BadStepIndex', ...
          'stepIndex must be a positive integer.');
end

horizon = cfg.predictionHorizon;

if isa(reference, 'function_handle')
    tGrid = currentTime + (0:horizon) * cfg.sampleTime;
    Xref = nmpc_prepare_reference(reference(tGrid), horizon);
    return;
end

if nargin < 1 || isempty(reference) || isvector(reference) || size(reference, 2) == 1
    Xref = nmpc_prepare_reference(reference, horizon);
    return;
end

if size(reference, 1) ~= 12
    error('nmpc_reference_window:BadReferenceSize', ...
          'reference must have 12 state rows.');
end

lastIndex = min(stepIndex + horizon, size(reference, 2));
window = reference(:, stepIndex:lastIndex);
if size(window, 2) < horizon + 1
    window = [window, repmat(window(:, end), 1, horizon + 1 - size(window, 2))];
end

Xref = window;
end
