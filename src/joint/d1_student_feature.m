function phi = d1_student_feature(s, x, k, Xref, Uref, cfg)
%D1_STUDENT_FEATURE The 27 features of the linear student Du = W*phi at step k:
%   e_k = x_k - x_ref,k (12, Euler angles wrapped)        -> feedback gain of the teacher
%   F_hat_k (3, d1_fhat; zero right after a (re)start)      -> wind compensation
%   ub_k = u_ref,k - u_h (4, flat feed-forward of the reference)
%   ub_{k+Nc} - ub_k, ub_{k+N} - ub_k (8)                  -> preview at the teacher's two horizons
% No constant feature: at the hover equilibrium without wind phi = 0 and the teacher equals
% the LQR (Du* = 0), so Du(0) = 0 must hold.
e = x - Xref(:,k); e(4:6) = mod(e(4:6) + pi, 2*pi) - pi;
if s.valid, Fh = d1_fhat(s.xPrev, s.uPrev, x, cfg); else, Fh = zeros(3,1); end
ub = Uref(:,k) - cfg.uh;
phi = [e; Fh; ub; Uref(:,k+cfg.Nc) - Uref(:,k); Uref(:,k+cfg.N) - Uref(:,k)];
end
