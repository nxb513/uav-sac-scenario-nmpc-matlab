function theta = quad_apply_model_shift(theta0, shift, t)
%QUAD_APPLY_MODEL_SHIFT Apply deterministic parameter shift after startTime.

theta = theta0;
originalInputLimits = theta0.inputLimits;

if nargin < 2 || isempty(shift) || ~isfield(shift, 'enabled') || ~shift.enabled
    return;
end
if nargin < 3
    error('quad_apply_model_shift:MissingTime', 'Time t is required.');
end
if t < shift.startTime
    return;
end

theta.m = theta.m * get_scale(shift, 'mScale', 1.0);
theta.J = diag(diag(theta.J) .* get_scale(shift, 'JScale', ones(3, 1)));
theta.Dv = theta.Dv .* get_scale(shift, 'DvScale', ones(3, 1));
theta.Domega = theta.Domega .* get_scale(shift, 'DomegaScale', ones(3, 1));
theta.alphaT = theta.alphaT * get_scale(shift, 'alphaTScale', 1.0);
theta.alphaTau = theta.alphaTau .* get_scale(shift, 'alphaTauScale', ones(3, 1));
theta.inputLimits = originalInputLimits;
end

function value = get_scale(s, field, defaultValue)
if isfield(s, field)
    value = s.(field);
else
    value = defaultValue;
end
value = value(:);
end
