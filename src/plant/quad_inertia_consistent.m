function tf = quad_inertia_consistent(J, tol)
%QUAD_INERTIA_CONSISTENT Physical validity of a diagonal inertia.
%
% Returns true iff the principal moments J form a physically realizable rigid
% body: all strictly positive AND satisfying the triangle inequalities
%   J_i <= J_j + J_k   for every axis
% which are necessary for a positive-semidefinite second-moment (pseudo-inertia)
% matrix. See Wensing, Kim, Slotine, IEEE RA-L 2018,
% https://doi.org/10.1109/LRA.2017.2729659.
%
% J may be a 3-vector of principal moments or a 3x3 diagonal inertia matrix.
% tol (default 0) is an absolute slack added to the triangle inequalities to
% absorb declared machine-roundoff only; keep it at or near 0 for real checks.

if nargin < 2 || isempty(tol)
    tol = 0;
end
if isequal(size(J), [3, 3])
    Jd = diag(J);
else
    Jd = J(:);
end
if numel(Jd) ~= 3
    error('quad_inertia_consistent:BadInput', ...
        'J must be a 3-vector or a 3x3 diagonal inertia matrix.');
end

tf = all(Jd > 0) && ...
    Jd(1) <= Jd(2) + Jd(3) + tol && ...
    Jd(2) <= Jd(1) + Jd(3) + tol && ...
    Jd(3) <= Jd(1) + Jd(2) + tol;
end
