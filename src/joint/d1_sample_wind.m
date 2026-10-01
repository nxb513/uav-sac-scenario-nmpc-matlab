function ds = d1_sample_wind(cfg, T, rs)
%D1_SAMPLE_WIND One random training-wind realization for a flight of T steps (synthetic;
% NO measured wind data is used in training).
%  mean   : U ~ Uniform[windMin, windMax] m/s, horizontal, azimuth ~ Uniform[0, 2*pi)
%  gusts  : Dryden low-altitude turbulence (d1_dryden), each component a first-order
%           Gauss-Markov process with time constant L/U (turbulence advected by U)
%  force  : linear rotor drag, mass-normalized (Faessler et al., RA-L 2018, identified on a
%           0.61 kg quadrotor): F = m R diag(windDrag) R' w, world frame (m = nominal mass;
%           an aerodynamic force, the plant divides by its true mass)
% Returns the disturbance handle ds(t, x, u, theta) used by quad_dynamics.
% rs (optional): a RandStream to draw from instead of the global stream (validation winds).
if nargin < 3, urand = @() rand; nrand = @() randn;
else, urand = @() rand(rs); nrand = @() randn(rs); end
U = cfg.windMin + (cfg.windMax - cfg.windMin)*urand();
psi = 2*pi*urand();
[sg, L] = d1_dryden(U, cfg);
tau = L/U;
dt = cfg.Ts/2; n = 2*T + 5; tt = (0:n-1)*dt;
g = zeros(3, n);
for i = 1:3
    ph = exp(-dt/tau(i)); q = sg(i)*sqrt(1 - ph^2);
    g(i,1) = sg(i)*nrand();
    for j = 2:n, g(i,j) = ph*g(i,j-1) + q*nrand(); end
end
c = cos(psi); s = sin(psi);
W = [c*(U + g(1,:)) - s*g(2,:); s*(U + g(1,:)) + c*g(2,:); g(3,:)];   % (u,v,w) -> world
Gi = {griddedInterpolant(tt, W(1,:), 'linear', 'nearest'), ...
      griddedInterpolant(tt, W(2,:), 'linear', 'nearest'), ...
      griddedInterpolant(tt, W(3,:), 'linear', 'nearest')};
D = diag(cfg.windDrag); m = cfg.plant.m;
ds = @(t, x, u, th) wind_force(t, x, Gi, D, m);
end

function d = wind_force(t, x, Gi, D, m)
w = [Gi{1}(t); Gi{2}(t); Gi{3}(t)];
R = quad_rotm_zyx(x(4:6));
d = struct('force', m*(R*D*R.')*w, 'torque', zeros(3,1));
end
