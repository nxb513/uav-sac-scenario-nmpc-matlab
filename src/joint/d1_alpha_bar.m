function ab = d1_alpha_bar(e, d, lqr)
%D1_ALPHA_BAR Linearized one-step contraction budget alpha_bar(e, d) (Prop. 1 of
% docs/D1_method): the largest alpha >= 0 with V(A_cl e + alpha B d) <= V(e) on the hover
% linearization, V = e'Pe (A_cl'PA_cl - P = -(Q + K'RK)); d = residual Delta_u.
Acl = lqr.Ad - lqr.Bd*lqr.K; S = lqr.P;
m = e.'*(lqr.Q + lqr.K.'*lqr.R*lqr.K)*e;
b = 2 * e.'*Acl.'*S*lqr.Bd*d;
q = d.'*lqr.Bd.'*S*lqr.Bd*d;
if q > 1e-12
    ab = (-b + sqrt(max(b^2 + 4*q*m,0)))/(2*q);
elseif b > 1e-12
    ab = m/b;
else
    ab = inf;
end
ab = max(ab, 0);
end
