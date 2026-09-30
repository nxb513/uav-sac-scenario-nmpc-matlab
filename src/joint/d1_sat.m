function u = d1_sat(u, cfg)
%D1_SAT Actuator box [cfg.uLo, cfg.uHi] (fixed nominal hardware limits).
u = min(max(u, cfg.uLo), cfg.uHi);
end
