function uSat = quad_saturate_input(u, theta)
%QUAD_SATURATE_INPUT Apply generalized input bounds.

u = u(:);
if numel(u) ~= 4
    error('quad_saturate_input:BadInputSize', 'Input u must have 4 elements.');
end

if nargin < 2 || isempty(theta) || ~isfield(theta, 'inputLimits')
    uSat = u;
    return;
end

uSat = u;
uSat(1) = min(max(uSat(1), theta.inputLimits.T(1)), theta.inputLimits.T(2));

for i = 1:3
    uSat(i + 1) = min(max(uSat(i + 1), theta.inputLimits.tau(i, 1)), ...
                      theta.inputLimits.tau(i, 2));
end
end
