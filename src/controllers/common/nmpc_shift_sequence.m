function Ushift = nmpc_shift_sequence(U, tailInput)
%NMPC_SHIFT_SEQUENCE Warm-start the next receding-horizon solve.

if size(U, 1) ~= 4
    error('nmpc_shift_sequence:BadInputSize', 'U must be 4-by-N.');
end

if size(U, 2) == 1
    Ushift = U;
    return;
end

if nargin < 2 || isempty(tailInput)
    tailInput = U(:, end);
else
    tailInput = tailInput(:);
end

if numel(tailInput) ~= 4
    error('nmpc_shift_sequence:BadTailInput', 'tailInput must have 4 elements.');
end

Ushift = [U(:, 2:end), tailInput];
end
