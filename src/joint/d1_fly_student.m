function R = d1_fly_student(stu, conf, Xref, Uref, ds, theta, lqr, cfg, mode)
%D1_FLY_STUDENT Teacher-free flight of the student on one reference with the common flight
% rules (d1_case_len steps, restart on the reference after a divergence, real time):
%   mode 'alpha1' : u = sat(sat(u_LQR) + W*phi)            (pure student, alpha = 1)
%   mode 'blend'  : u = d1_blend_control(...)             (deployed controller)
%   mode 'lqr'    : u = sat(u_LQR)
% R.X (12 x T, state after step k, NaN when diverged), R.Xr = Xref(:,2:T+1), R.alpha (1 x T),
% R.psi (18 x T, c_S features at every step), R.nDiv.
T = d1_case_len(Xref, cfg); Ts = cfg.Ts; uh = cfg.uh;
R.X = nan(12,T); R.Xr = Xref(:,2:T+1); R.alpha = nan(1,T); R.psi = nan(18,T); R.nDiv = 0;
x = Xref(:,1); s = d1_student_init(x, cfg);
for k = 1:T
    e = x - Xref(:,k);
    phi = d1_student_feature(s, x, k, Xref, Uref, cfg);
    R.psi(:,k) = d1_cs_feature(e, phi);
    uLs = d1_sat(uh - lqr.K*e, cfg);
    switch mode
        case 'alpha1'
            du = stu.W*phi; if ~all(isfinite(du)), du = zeros(4,1); end
            u = d1_sat(uLs + du, cfg); R.alpha(k) = 1;
        case 'blend'
            [u, R.alpha(k)] = d1_blend_control(uh - lqr.K*e, e, phi, stu, conf, lqr, cfg);
        case 'lqr'
            u = uLs; R.alpha(k) = 0;
        otherwise
            error('d1_fly_student:mode', 'unknown mode %s', mode);
    end
    [xNew, div] = d1_plant_step((k-1)*Ts, x, u, Ts, theta, ds);
    if div
        R.nDiv = R.nDiv + 1; x = Xref(:,k+1); s = d1_student_init(x, cfg);
    else
        s = d1_student_push(s, x, u); x = xNew; R.X(:,k) = x;
    end
end
end
