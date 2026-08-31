function cost = nmpc_tracking_cost(X, U, Xref, theta, cfg, previousInput)
%NMPC_TRACKING_COST Quadratic tracking, control-deviation and smoothness cost.

if nargin < 6
    previousInput = [];
end

horizon = size(U, 2);
if size(X, 2) ~= horizon + 1 || size(Xref, 2) ~= horizon + 1
    error('nmpc_tracking_cost:BadHorizon', ...
          'X and Xref must have N+1 columns matching U.');
end

Q = cfg.weights.Q;
Qf = cfg.weights.Qf;
R = cfg.weights.R;
dU = cfg.weights.dU;

switch lower(cfg.weights.inputReference)
    case 'hover'
        uRef = repmat(quad_hover_input(theta), 1, horizon);
    case 'zero'
        uRef = zeros(4, horizon);
    otherwise
        error('nmpc_tracking_cost:BadInputReference', ...
              'Unknown input reference mode: %s', cfg.weights.inputReference);
end

cost = 0.0;
for k = 1:horizon
    e = nmpc_state_error(X(:, k + 1), Xref(:, k + 1));
    uErr = U(:, k) - uRef(:, k);
    cost = cost + e.' * Q * e + uErr.' * R * uErr;

    if ~isempty(dU)
        if k == 1 && ~isempty(previousInput)
            du = U(:, k) - previousInput(:);
        elseif k > 1
            du = U(:, k) - U(:, k - 1);
        else
            du = zeros(4, 1);
        end
        cost = cost + du.' * dU * du;
    end
end

eTerminal = nmpc_state_error(X(:, end), Xref(:, end));
cost = cost + eTerminal.' * Qf * eTerminal;
end
