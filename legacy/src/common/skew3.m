function S = skew3(a)
%SKEW3 Return the 3-by-3 skew-symmetric matrix for cross products.

a = a(:);
if numel(a) ~= 3
    error('skew3:BadInputSize', 'Input must have 3 elements.');
end

S = [0, -a(3), a(2);
     a(3), 0, -a(1);
     -a(2), a(1), 0];
end
