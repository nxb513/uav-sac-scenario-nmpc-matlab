function [rhoMax, ok, rho] = d1_student_stability(W, lqr, cfg)
%D1_STUDENT_STABILITY Linearized (nominal hover, no disturbance, x_ref = 0) closed loop of the
% deployed law u = uh - K x + alpha*W*phi for frozen alpha. With x_ref = 0 the reference
% features vanish and phi = [x; F_hat; 0]; F_hat_k = L1 x_k + L0 x_{k-1} + Lu du_{k-1}
% (Jacobians of d1_fhat at hover, central differences). With z_k = [x_k; x_{k-1}; du_{k-1}]:
%   z_{k+1} = A_alpha z_k,  A_alpha = [Ad+Bd*G1, Bd*G0, Bd*Gu; I, 0, 0; G1, G0, Gu],
%   G1 = -K + alpha*(We + WF*L1),  G0 = alpha*WF*L0,  Gu = alpha*WF*Lu.
% ok = spectral radius < 1 for every alpha on cfg.alphaGrid (alpha = 0 is the LQR loop).
% A necessary check for frozen alpha only: it does not cover time-varying alpha, saturation
% or an off-nominal plant.
xh = zeros(12,1); uh = cfg.uh; h = 1e-6;
L1 = zeros(3,12); L0 = zeros(3,12); Lu = zeros(3,4);
for i = 1:12
    d = zeros(12,1); d(i) = h;
    L1(:,i) = (d1_fhat(xh, uh, xh+d, cfg) - d1_fhat(xh, uh, xh-d, cfg))/(2*h);
    L0(:,i) = (d1_fhat(xh+d, uh, xh, cfg) - d1_fhat(xh-d, uh, xh, cfg))/(2*h);
end
for j = 1:4
    d = zeros(4,1); d(j) = h;
    Lu(:,j) = (d1_fhat(xh, uh+d, xh, cfg) - d1_fhat(xh, uh-d, xh, cfg))/(2*h);
end
We = W(:,1:12); WF = W(:,13:15);
Ad = lqr.Ad; Bd = lqr.Bd; K = lqr.K;
rho = zeros(size(cfg.alphaGrid));
for ia = 1:numel(cfg.alphaGrid)
    a = cfg.alphaGrid(ia);
    G1 = -K + a*(We + WF*L1); G0 = a*WF*L0; Gu = a*WF*Lu;
    A = [Ad + Bd*G1, Bd*G0, Bd*Gu; eye(12), zeros(12), zeros(12,4); G1, G0, Gu];
    rho(ia) = max(abs(eig(A)));
end
rhoMax = max(rho); ok = rhoMax < 1;
end
