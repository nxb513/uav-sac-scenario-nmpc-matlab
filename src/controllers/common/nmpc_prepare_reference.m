function Xref = nmpc_prepare_reference(reference, horizon)
%NMPC_PREPARE_REFERENCE Normalize reference into 12-by-(N+1).

if horizon <= 0 || horizon ~= floor(horizon)
    error('nmpc_prepare_reference:BadHorizon', ...
          'horizon must be a positive integer.');
end

if nargin < 1 || isempty(reference)
    Xref = zeros(12, horizon + 1);
    return;
end

if isstruct(reference)
    if isfield(reference, 'X')
        reference = reference.X;
    elseif isfield(reference, 'x')
        reference = reference.x;
    else
        error('nmpc_prepare_reference:BadReferenceStruct', ...
              'reference struct must contain field X or x.');
    end
end

if isvector(reference)
    reference = reference(:);
end

if size(reference, 1) ~= 12
    error('nmpc_prepare_reference:BadReferenceSize', ...
          'reference must have 12 state rows.');
end

if size(reference, 2) == 1
    Xref = repmat(reference, 1, horizon + 1);
elseif size(reference, 2) >= horizon + 1
    Xref = reference(:, 1:horizon + 1);
else
    error('nmpc_prepare_reference:ShortReference', ...
          'reference must contain at least N+1 columns.');
end
end
