function U = nmpc_saturate_sequence(U, theta)
%NMPC_SATURATE_SEQUENCE Apply generalized input saturation column-wise.

if size(U, 1) ~= 4
    error('nmpc_saturate_sequence:BadInputSize', 'U must be 4-by-N.');
end

for k = 1:size(U, 2)
    U(:, k) = quad_saturate_input(U(:, k), theta);
end
end
