function h = d1_hist_push(h, u, xNew, cfg)
%D1_HIST_PUSH Advance the surrogate history after u_k was applied and the plant reached
% x_{k+1} (not diverged): append x_{k+1} and u_k and set the nominal one-step prediction
% residual r_{k+1} = x_{k+1} - Phi_nom(x_k, u_k) (RK4 of the nominal model without wind,
% wrapped Euler angles). r carries what the nominal model does not explain -- the wind
% and any model error -- so the deployed surrogate can infer the disturbance the
% privileged teacher is told.
xPrev = h.X(:, end); r = zeros(12, 1);
try
    xp = quad_step_rk4(0, xPrev, u, cfg.Ts, cfg.plant.nominal, []);
    if all(isfinite(xp)) && all(isfinite(xNew))
        r = quad_state_prediction_error(xNew, xp);
    end
catch
end
h.X = [h.X(:, 2:end), xNew]; h.U = [h.U(:, 2:end), u]; h.r = r;
end
