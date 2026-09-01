function acceleration = quad_targeted_load_acceleration( ...
        targetSpeed, loadIndex, cfg, family)
%QUAD_TARGETED_LOAD_ACCELERATION Resolve a feasible speed-dependent load.

validateattributes(targetSpeed, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'positive'});
validateattributes(loadIndex, {'numeric'}, ...
    {'scalar', 'integer', 'positive', ...
    '<=', numel(cfg.approvedLoadFractions)});

fullEnvelopeMaximum = max(cfg.approvedAccelerationTargets);
geometryLimitedMaximum = targetSpeed ^ 2 / cfg.minimumGeometryScale;
availableMaximum = min(fullEnvelopeMaximum, geometryLimitedMaximum);
acceleration = cfg.approvedLoadFractions(loadIndex) * availableMaximum;
acceleration = max(cfg.minimumResolvedAcceleration, acceleration);
if nargin >= 4 && strcmpi(char(family), 'vertical_circle')
    workspaceMinimum = targetSpeed ^ 2 / ...
        cfg.verticalCircleMaximumRadius;
    acceleration = max(acceleration, workspaceMinimum);
end
end
