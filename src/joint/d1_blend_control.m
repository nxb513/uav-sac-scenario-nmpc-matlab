function [u, alpha] = d1_blend_control(uBase, e, h, XrefLook, sur, conf, lqr, cfg)
%D1_BLEND_CONTROL The deployed (proposed) controller:
%   u = sat( sat(uBase) + alpha*Du ),  alpha = c_S * g_L(c_LQR),
%   g_L = clip((c_high - c_LQR)/(c_high - c_low), 0, 1),   optional alpha_safe cap.
% uBase = uh - K e (LQR). The base is saturated BEFORE the residual is added, exactly as
% in the training label Du* = u_teacher - sat(u_LQR), so alpha = 1 with an exact Du
% reproduces the teacher. Du, c_S from the surrogate on the history feature
% (d1_hist_feature); c_LQR from the logistic on the LQR error e. No conf (c_LQR not
% trained) -> c_LQR = 0 -> g_L = 1 (alpha = c_S). Non-finite feature or network output
% -> Du = 0, c_S = 0 (pure base).
feat = d1_hist_feature(h, XrefLook);
du = zeros(4,1); cS = 0;
if all(isfinite(feat))
    du = d1_sur_predict_du(sur, feat); cS = d1_sur_predict_cs(sur, feat);
    if ~all(isfinite([du; cS])), du = zeros(4,1); cS = 0; end
end
haveConf = ~isempty(conf) && isfield(conf,'LQR') && ~isempty(conf.LQR.w);
if haveConf, cLp = d1_predict_logistic(conf.LQR, d1_conf_feature(e).'); else, cLp = 0; end
gL = min(max((cfg.cHigh - cLp)/(cfg.cHigh - cfg.cLow), 0), 1);
alpha = cS * gL;
if cfg.alphaSafe
    alpha = min(alpha, (1-cfg.epsSafe)*d1_alpha_bar(e, du, lqr));
end
u = d1_sat(d1_sat(uBase, cfg) + alpha*du, cfg);
end
