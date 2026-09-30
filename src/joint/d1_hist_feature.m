function feat = d1_hist_feature(h, XrefLook)
%D1_HIST_FEATURE Raw 208-D surrogate feature at step k:
% [x_{k-3..k} (48); u_{k-4..k-1} (16); x_ref,k..k+10 (132); r_k (12)]
% (surrogate_build_feature). Non-finite input -> NaN feature (caller falls back to the base).
v = [h.X(:); h.U(:); XrefLook(:); h.r(:)];
if all(isfinite(v))
    feat = surrogate_build_feature(h.X, h.U, XrefLook, h.r);
else
    feat = nan(208, 1);
end
end
