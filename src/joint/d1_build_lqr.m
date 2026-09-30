function lqr = d1_build_lqr(cfg)
%D1_BUILD_LQR Bryson LQR on the nominal hover linearization (discrete, Ts):
% K, Riccati P, hover input uh, (Ad, Bd), (Q, R).
P = cfg.plant; theta = P.nominal;
xh = zeros(12,1); uh = cfg.uh;
[A, B] = num_linearize(@(x,u) quad_dynamics(0, x, u, theta, []), xh, uh);
sysd = c2d(ss(A, B, eye(12), zeros(12,4)), cfg.Ts);
[Q0, R0] = d1_bryson_weights(P);
[K, Sr] = dlqr(sysd.A, sysd.B, Q0, R0);
lqr.K = K; lqr.P = Sr; lqr.uh = uh; lqr.Ad = sysd.A; lqr.Bd = sysd.B;
lqr.Q = Q0; lqr.R = R0;                               % for alpha_bar (Prop 1)
end

function [A, B] = num_linearize(f, x0, u0)
n = numel(x0); m = numel(u0); h = 1e-6;
A = zeros(n); B = zeros(n, m);
for i = 1:n
    dx = zeros(n,1); dx(i) = h;
    A(:,i) = (f(x0+dx,u0) - f(x0-dx,u0)) / (2*h);
end
for j = 1:m
    du = zeros(m,1); du(j) = h;
    B(:,j) = (f(x0,u0+du) - f(x0,u0-du)) / (2*h);
end
end
