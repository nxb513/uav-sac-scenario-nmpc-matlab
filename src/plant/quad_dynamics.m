function [xdot, aux] = quad_dynamics(t, x, u, theta, disturbanceSpec)
%QUAD_DYNAMICS Nonlinear 12-state quadrotor plant with uncertainty hooks.
%
% State: x = [px; py; pz; phi; theta; psi; vx; vy; vz; p; q; r]
% Input: u = [T; tau_phi; tau_theta; tau_psi]

if nargin < 4 || isempty(theta)
    cfg = step1_plant_config();
    theta = cfg.nominal;
end
if nargin < 5
    disturbanceSpec = [];
end

x = x(:);
u = u(:);
if numel(x) ~= 12
    error('quad_dynamics:BadStateSize', 'State x must have 12 elements.');
end
if numel(u) ~= 4
    error('quad_dynamics:BadInputSize', 'Input u must have 4 elements.');
end

u = quad_saturate_input(u, theta);

pos = x(1:3);
eta = x(4:6);
vel = x(7:9);
omega = x(10:12);

if isa(disturbanceSpec, 'function_handle')
    dist = disturbanceSpec(t, x, u, theta);
else
    dist = quad_disturbance(t, disturbanceSpec);
end

R = quad_rotm_zyx(eta);
e3 = [0; 0; 1];

T = theta.alphaT * u(1);
tau = theta.alphaTau(:) .* u(2:4);

posDot = vel;
etaDot = quad_euler_rates_zyx(eta, omega);
velDot = [0; 0; -theta.g] ...
         + (T / theta.m) * (R * e3) ...
         - theta.Dv(:) .* vel / theta.m ...
         + dist.force(:) / theta.m;
omegaDot = theta.J \ (tau + dist.torque(:) ...
                      - cross(omega, theta.J * omega) ...
                      - theta.Domega(:) .* omega);

xdot = [posDot; etaDot; velDot; omegaDot];

if nargout > 1
    aux.pos = pos;
    aux.eta = eta;
    aux.vel = vel;
    aux.omega = omega;
    aux.R = R;
    aux.T = T;
    aux.tau = tau;
    aux.disturbance = dist;
    aux.u = u;
end
end
