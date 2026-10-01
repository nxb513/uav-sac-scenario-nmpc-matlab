function [st, info] = d1_dagger_case(teacher, W, kase, ds, lqr, cfg)
%D1_DAGGER_CASE One DAgger data flight on the nominal plant in the training wind.
%   W = []  : the TEACHER flies (DAgger iteration 1, beta_1 = 1).
%   W given : the STUDENT flies at alpha = 1, u = sat(sat(u_LQR) + W*phi); the teacher is only
%             QUERIED at every visited state (that state, the student's previous input as
%             u_prev, the exact current wind force) and its command is never applied.
% Label at every usable teacher solve inside the flight envelope (position error < 5 m,
% |roll|, |pitch| <= 1.35 rad): y = (u_T - sat(u_LQR)) ./ resHalf with the student features
% phi of that state. info.usable = share of steps that gave a label. Returns the per-flight sufficient statistics st.G/H/Q/n and
% diagnostics info (usable/timeout shares, solve times, restarts, capped position RMSE).
% Common flight rules: d1_case_len steps, restart on the reference after a divergence
% (student memory, u_prev and the teacher solver reset).
Xref = kase.Xref; Uref = kase.Uref; T = d1_case_len(Xref, cfg); Ts = cfg.Ts; uh = cfg.uh;
theta = cfg.plant.nominal; P = cfg.nPhi; studentFlies = ~isempty(W);
st = struct('G', zeros(P), 'H', zeros(P,4), 'Q', zeros(4), 'n', 0, 'fold', 0, 'id', kase.groupId);
x = Xref(:,1); uprev = uh; s = d1_student_init(x, cfg);
d1_teacher_reset(teacher, Xref, 1, cfg);
pe = zeros(1,T); tsol = zeros(1,T); nUse = 0; nTo = 0; nDiv = 0; t0 = tic;
for k = 1:T
    t = (k-1)*Ts; e = x - Xref(:,k);
    phi = d1_student_feature(s, x, k, Xref, Uref, cfg);
    F = d1_wind_now(ds, t, x, uprev, theta);
    [uT, status, usable, tsol(k)] = d1_teacher_step(teacher, x, uprev, Xref, k, F, cfg);
    uT = d1_sat(uT, cfg); nTo = nTo + (status == 7);
    uLs = d1_sat(uh - lqr.K*e, cfg);
    % label only inside the flight envelope of the metrics: position error below the 5 m
    % cap of d1_track_err and |roll|, |pitch| <= 1.35 rad (attitude-violation rule of the
    % reward); outside it (e.g. an unstable early student) the flight is already a failure
    inEnv = norm(e(1:3)) < 5 && all(abs(x(4:5)) <= 1.35);
    if usable && inEnv && all(isfinite(phi))
        y = (uT - uLs)./cfg.resHalf; Gk = phi*phi.';
        if all(isfinite(Gk(:))) && all(isfinite(y))
            st.G = st.G + Gk; st.H = st.H + phi*y.'; st.Q = st.Q + y*y.'; st.n = st.n + 1;
            nUse = nUse + 1;
        end
    end
    if studentFlies
        du = W*phi; if ~all(isfinite(du)), du = zeros(4,1); end
        u = d1_sat(uLs + du, cfg);
    else
        u = uT;
    end
    [xNew, div] = d1_plant_step(t, x, u, Ts, theta, ds);
    if div
        nDiv = nDiv + 1; pe(k) = 5;
        x = Xref(:,k+1); uprev = uh; s = d1_student_init(x, cfg);
        d1_teacher_reset(teacher, Xref, k+1, cfg);
    else
        s = d1_student_push(s, x, u); x = xNew; uprev = u;
        pe(k) = min(norm(x(1:3) - Xref(1:3,k+1)), 5);
    end
end
info = struct('usable', nUse/T, 'timeouts', nTo, 'restarts', nDiv, 'posRmse', sqrt(mean(pe.^2)), ...
    'tmax', max(tsol), 'tp99', prctile(tsol, 99), 'time', toc(t0));
end
