function [u, alpha, du] = d1_blend_control(uBase, e, phi, stu, conf, lqr, cfg)
%D1_BLEND_CONTROL The deployed (proposed) controller:
%   u = sat( sat(uBase) + alpha*Du ),  Du = W*phi (linear student),
%   alpha = c_S * g_L(c_LQR),  g_L = clip((c_high - c_LQR)/(c_high - c_low), 0, 1),
%   optional alpha_safe cap.
% uBase = uh - K e (LQR). The base is saturated BEFORE the residual is added, exactly as in
% the label Du* = u_teacher - sat(u_LQR). c_S = logistic on d1_cs_feature, c_LQR = logistic on
% d1_conf_feature(e). Without a trained c_S (no conf) alpha = 0 (pure base); a non-finite
% feature also gives Du = 0.
du = zeros(4,1); cS = 0;
if ~isempty(stu) && all(isfinite(phi))
    du = stu.W * phi;
    if ~all(isfinite(du)), du = zeros(4,1); end
    if ~isempty(conf) && isfield(conf, 'S') && ~isempty(conf.S.w)
        cS = d1_predict_logistic(conf.S, d1_cs_feature(e, phi).');
    end
end
haveL = ~isempty(conf) && isfield(conf,'LQR') && ~isempty(conf.LQR.w);
if haveL, cLp = d1_predict_logistic(conf.LQR, d1_conf_feature(e).'); else, cLp = 0; end
gL = min(max((cfg.cHigh - cLp)/(cfg.cHigh - cfg.cLow), 0), 1);
alpha = cS * gL;
if cfg.alphaSafe
    alpha = min(alpha, (1-cfg.epsSafe)*d1_alpha_bar(e, du, lqr));
end
u = d1_sat(d1_sat(uBase, cfg) + alpha*du, cfg);
end
