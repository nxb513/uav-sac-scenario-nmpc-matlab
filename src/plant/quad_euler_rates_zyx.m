function etaDot = quad_euler_rates_zyx(eta, omega)
%QUAD_EULER_RATES_ZYX Convert body rates to ZYX Euler angle rates.
%
% eta = [phi; theta; psi], omega = [p; q; r].

eta = eta(:);
omega = omega(:);
if numel(eta) ~= 3 || numel(omega) ~= 3
    error('quad_euler_rates_zyx:BadInputSize', ...
          'eta and omega must each have 3 elements.');
end

phi = eta(1);
theta = eta(2);
cth = cos(theta);

if abs(cth) < 1e-6
    error('quad_euler_rates_zyx:Singularity', ...
          'ZYX Euler rates are singular when cos(theta) is near zero.');
end

W = [1, sin(phi) * tan(theta),  cos(phi) * tan(theta);
     0, cos(phi),             -sin(phi);
     0, sin(phi) / cth,        cos(phi) / cth];

etaDot = W * omega;
end
