function conf = d1_consolidate(cfg, lqr, cases, stu)
%D1_CONSOLIDATE Confidences of the deployed controller, both learned in the training wind
% on the nominal plant (global random stream):
%   c_LQR : logistic P(V_{k+H} < V_k | e) on cLqrCases LQR flights (V = e'Pe, LQR Riccati P),
%           features d1_conf_feature(e) (16); a divergence leaves a NaN column that breaks
%           every window across it.
%   c_S   : soft-label logistic predicting s_k = exp(-(E20(k)/epsP)^2), E20 = RMS of the capped
%           position error over the PAST H steps of a student flight at alpha = 1, on csCases
%           flights; features d1_cs_feature (18).
theta = cfg.plant.nominal; H = cfg.H;
% ---- c_LQR ------------------------------------------------------------------------
sel = unique(round(linspace(1, numel(cases), min(cfg.cLqrCases, numel(cases)))));
XL = zeros(16,0); yL = zeros(1,0);
for ci = 1:numel(sel)
    kase = cases(sel(ci)); T = d1_case_len(kase.Xref, cfg);
    if cfg.windOn, ds = d1_sample_wind(cfg, T); else, ds = []; end
    R = d1_fly_student([], [], kase.Xref, kase.Uref, ds, theta, lqr, cfg, 'lqr');
    E = R.X - R.Xr;                                   % NaN columns at diverged steps
    if size(E,2) <= H, continue; end
    o = d1_finite_horizon_contraction(E, lqr.P, H, struct());
    m = o.validMask & o.windowFinite;
    F = d1_conf_feature(E);
    XL = [XL, F(:,m)]; yL = [yL, double(o.isContracting(m))]; %#ok<AGROW>
end
conf.LQR = d1_fit_logistic(XL.', yL.', false);
% ---- c_S --------------------------------------------------------------------------
sel = unique(round(linspace(1, numel(cases), min(cfg.csCases, numel(cases)))));
XS = zeros(18,0); yS = zeros(1,0);
for ci = 1:numel(sel)
    kase = cases(sel(ci)); T = d1_case_len(kase.Xref, cfg);
    if cfg.windOn, ds = d1_sample_wind(cfg, T); else, ds = []; end
    R = d1_fly_student(stu, [], kase.Xref, kase.Uref, ds, theta, lqr, cfg, 'alpha1');
    pe = d1_track_err(R.X, R.Xr);                     % capped, diverged step = 5 m
    for k = H:T
        E20 = sqrt(mean(pe(k-H+1:k).^2));
        XS(:,end+1) = R.psi(:,k); yS(end+1) = exp(-(E20/cfg.epsP)^2); %#ok<AGROW>
    end
end
conf.S = d1_fit_logistic(XS.', yS.', true);
conf.epsP = cfg.epsP; conf.H = H;
conf.def = ['c_S = soft-label logistic of s_k = exp(-(RMS_pastH_pos/epsP)^2) on d1_cs_feature; ' ...
    'c_LQR = logistic P(V_{k+H} < V_k) on d1_conf_feature'];
fprintf('CONF_DONE c_LQR(acc=%.2f base=%.2f n=%d) c_S(meanS=%.3f n=%d)\n', ...
    conf.LQR.acc, conf.LQR.base, conf.LQR.n, conf.S.base, conf.S.n);
end
