function cost = nmpc_tracking_cost(X, U, Xref, theta, cfg, previousInput, ...
        uRefWindow)
%NMPC_TRACKING_COST Quadratic tracking, control-deviation and smoothness cost.
%
% The input-deviation term penalizes (u - uRef): with inputReference 'feedforward'
% uRef is the time-varying differential-flatness feedforward uRefWindow (4-by-N,
% one column per prediction step), so the teacher is penalized only for deviating
% from the input needed to fly the trajectory, not for the necessary non-hover
% actuation. 'hover'/'zero' keep uRef constant (legacy modes).

if nargin < 6
    previousInput = [];
end
if nargin < 7
    uRefWindow = [];
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
    case 'feedforward'
        if isempty(uRefWindow)
            error('nmpc_tracking_cost:MissingFeedforward', ...
                  ['inputReference ''feedforward'' requires the uRefWindow ' ...
                   'argument (4-by-N flatness feedforward).']);
        end
        if size(uRefWindow, 1) ~= 4 || size(uRefWindow, 2) < horizon
            error('nmpc_tracking_cost:BadFeedforward', ...
                  'uRefWindow must be 4-by-M with M >= horizon.');
        end
        uRef = uRefWindow(:, 1:horizon);
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
