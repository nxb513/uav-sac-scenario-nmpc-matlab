function s = d1_feature_scale(cfg)
%D1_FEATURE_SCALE Fixed normalization of the 208-D surrogate feature (d1_hist_feature).
%  states / reference : declared envelope (position 100 m, roll/pitch 1.35 rad, yaw pi,
%                       speed 25 m/s, body rate 10 rad/s)
%  inputs             : actuator bounds max(|uLo|, |uHi|)
%  residual r_k       : one-step response to the largest declared disturbance,
%                       translational: largest training-wind acceleration
%                       a_max = max(windDrag) (windMax + 3 sigma_u(windMax)) -> position
%                       Ts^2/2 a_max, velocity Ts a_max;
%                       rotational: full actuator angular acceleration
%                       alphaTau tau_max / J -> angle Ts^2/2, rate Ts times that.
ss = [100;100;100;1.35;1.35;pi;25;25;25;10;10;10];
us = max(abs(cfg.uLo), abs(cfg.uHi));
Ts = cfg.Ts; nom = cfg.plant.nominal;
sg = d1_dryden(cfg.windMax, cfg);
aMax = max(cfg.windDrag) * (cfg.windMax + 3*sg(1));
wdMax = nom.alphaTau(:) .* us(2:4) ./ diag(nom.J);
rs = [Ts^2/2*aMax*ones(3,1); Ts^2/2*wdMax; Ts*aMax*ones(3,1); Ts*wdMax];
s = [repmat(ss,4,1); repmat(us,4,1); repmat(ss,11,1); rs];   % 48+16+132+12 = 208
end
