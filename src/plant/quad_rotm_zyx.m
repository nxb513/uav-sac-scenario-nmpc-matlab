function R = quad_rotm_zyx(eta)
%QUAD_ROTM_ZYX Body-to-inertial rotation matrix for ZYX Euler angles.
%
% eta = [phi; theta; psi].

eta = eta(:);
if numel(eta) ~= 3
    error('quad_rotm_zyx:BadInputSize', 'eta must have 3 elements.');
end

phi = eta(1);
theta = eta(2);
psi = eta(3);

cphi = cos(phi); sphi = sin(phi);
cth = cos(theta); sth = sin(theta);
cpsi = cos(psi); spsi = sin(psi);

R = [cpsi * cth, cpsi * sth * sphi - spsi * cphi, cpsi * sth * cphi + spsi * sphi;
     spsi * cth, spsi * sth * sphi + cpsi * cphi, spsi * sth * cphi - cpsi * sphi;
     -sth,       cth * sphi,                         cth * cphi];
end
