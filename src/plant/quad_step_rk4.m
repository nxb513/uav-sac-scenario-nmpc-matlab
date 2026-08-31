function xNext = quad_step_rk4(t, x, u, dt, theta, disturbanceSpec)
%QUAD_STEP_RK4 One fixed-step RK4 integration step.

if dt <= 0
    error('quad_step_rk4:BadStep', 'dt must be positive.');
end

f = @(tt, xx) quad_dynamics(tt, xx, u, theta, disturbanceSpec);

k1 = f(t, x);
k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1);
k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2);
k4 = f(t + dt, x + dt * k3);

xNext = x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4);
xNext(4:6) = wrap_euler(xNext(4:6));
end

function eta = wrap_euler(eta)
eta = mod(eta + pi, 2*pi) - pi;
end
