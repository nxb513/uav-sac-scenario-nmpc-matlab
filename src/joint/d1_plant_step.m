function [x, div] = d1_plant_step(t, x, u, Ts, theta, ds)
%D1_PLANT_STEP One RK4 plant step at real time t (common flight rule). Divergence =
% integration error (e.g. singular ZYX Euler rates at pitch = +-90 deg), a non-finite
% state, or |p| > 1e4 m. A diverged flight is NOT stopped: the caller charges the step,
% restarts the plant on the reference, resets its controller's internal state and keeps
% flying to the end.
try
    x = quad_step_rk4(t, x, u, Ts, theta, ds);
catch
    x = nan(12,1);
end
div = ~all(isfinite(x)) || norm(x(1:3)) > 1e4;
end
