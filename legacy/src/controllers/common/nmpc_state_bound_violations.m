function c = nmpc_state_bound_violations(X, cfg)
%NMPC_STATE_BOUND_VIOLATIONS Return c <= 0 inequality violations.

c = [];
if ~isfield(cfg, 'constraints') || ~cfg.constraints.enableStateBounds
    return;
end

lower = cfg.constraints.stateLower(:);
upper = cfg.constraints.stateUpper(:);
if numel(lower) ~= 12 || numel(upper) ~= 12
    error('nmpc_state_bound_violations:BadBounds', ...
          'stateLower and stateUpper must have 12 elements.');
end

Xpred = X(:, 2:end);
lowerMat = repmat(lower, 1, size(Xpred, 2));
upperMat = repmat(upper, 1, size(Xpred, 2));
c = [c; Xpred(:) - upperMat(:); lowerMat(:) - Xpred(:)];

if isfield(cfg.constraints, 'maxTilt') && isfinite(cfg.constraints.maxTilt)
    tilt = sqrt(Xpred(4, :).^2 + Xpred(5, :).^2);
    c = [c; tilt(:) - cfg.constraints.maxTilt];
end
end
