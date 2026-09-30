function h = d1_hist_init(x, cfg)
%D1_HIST_INIT Surrogate input history at the start of a flight or after a divergence
% restart: current state repeated (h.X, 12x4, last column = current state x_k), hover
% inputs (h.U, 4x4, last column = u_{k-1}) and a zero prediction residual (h.r).
h.X = repmat(x, 1, 4); h.U = repmat(cfg.uh, 1, 4); h.r = zeros(12, 1);
end
