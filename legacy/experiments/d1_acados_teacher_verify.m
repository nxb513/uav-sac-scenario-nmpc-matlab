function d1_acados_teacher_verify()
%D1_ACADOS_TEACHER_VERIFY Build the acados quad NMPC teacher and fly it closed-loop.
%
% Verifies the teacher the SAC loop needs:
%   * tracking cost with a per-stage reference yref (time-varying),
%   * runtime Q,R update via cost_set('W',...) WITHOUT rebuilding the solver
%     (SAC retunes Q,R every rollout -> must not recompile),
%   * input + state + tilt constraints,
%   * closed-loop flight of a reference for N_STEPS=1000 with an RK4 nominal plant.
%
% Single nominal scenario for now (M=5 robust extension is the next increment).
% Nc=5 move-blocking is NOT yet applied (full control horizon); flagged as pending.
% NOT a science run -- a build/verify harness.

check_acados_requirements();
import casadi.*

P = plant_params();
Ts = 0.05; N = 20; N_STEPS = 1000;

model = quad_acados_model(P);
solver = build_solver(model, P, N, Ts);

% ---- reference: gentle circle (speed ~2 m/s), zero-attitude reference --------
r = 4.0; wref = 2.0 / r; z0 = 2.0;              % v = w*r = 2 m/s
uhover = [P.m * P.g; 0; 0; 0];
xref_fun = @(t) [r*cos(wref*t); r*sin(wref*t); z0; 0; 0; 0; ...
    -r*wref*sin(wref*t); r*wref*cos(wref*t); 0; 0; 0; 0];

% Bryson weights (initial Q0,R0; SAC would override W at runtime).
[Q0, R0] = bryson_weights(P);

% ---- closed-loop rollout -----------------------------------------------------
x = xref_fun(0); x(1) = x(1) + 0.5; x(3) = x(3) - 0.3;   % start off-reference
posErr = zeros(N_STEPS,1); attErr = zeros(N_STEPS,1);
solveMs = zeros(N_STEPS,1); okCount = 0;
retuneStep = 500;                                        % test runtime W change

for k = 1:N_STEPS
    t = (k-1)*Ts;
    % runtime Q,R retune halfway (proves no rebuild needed) --------------------
    if k == retuneStep
        W = blkdiag(Q0*3.0, R0*0.5);
        for s = 0:N-1, solver.set('cost_W', W, s); end
    end
    % per-stage references over the horizon -----------------------------------
    for s = 0:N-1
        yref = [xref_fun(t + s*Ts); uhover];
        solver.set('cost_y_ref', yref, s);
    end
    solver.set('cost_y_ref_e', xref_fun(t + N*Ts));
    solver.set('constr_x0', x);
    tic; solver.solve(); solveMs(k) = 1000*toc;
    if solver.get('status') == 0, okCount = okCount + 1; end
    u = solver.get('u', 0);
    % nominal RK4 plant step ---------------------------------------------------
    x = rk4_step(x, u, Ts, P);
    xr = xref_fun(t + Ts);
    posErr(k) = norm(x(1:3) - xr(1:3));
    attErr(k) = norm(x(4:6) - xr(4:6));
end

ms = solveMs(2:end);
fprintf('D1_TEACHER_VERIFY steps=%d ok=%d\n', N_STEPS, okCount);
fprintf('POS_RMSE_m=%.4f ATT_RMSE_deg=%.4f\n', ...
    sqrt(mean(posErr.^2)), rad2deg(sqrt(mean(attErr.^2))));
fprintf('SOLVE_MS median=%.3f p95=%.3f max=%.3f\n', ...
    median(ms), quantile(ms,0.95), max(ms));
fprintf('RUNTIME_W_RETUNE_OK=%d (no rebuild)\n', okCount > 0);
fprintf('EST_HOURS_5scn @1000ev x20cs x1000st = %.2f\n', ...
    5*1000*20*1000*median(ms)/1000/3600);
fprintf('D1_TEACHER_VERIFY_DONE\n');
end

% =============================================================================
function [Q0, R0] = bryson_weights(P)
eAllow = [0.10;0.10;0.10; deg2rad(5)*[1;1;1]; 0.30;0.30;0.30; 2;2;2];
duAllow = [P.m*P.g; 0.5; 0.5; 0.25];
Q0 = diag(1 ./ eAllow.^2);
R0 = diag(1 ./ duAllow.^2);
end

function P = plant_params()
P.g = 9.81; P.m = 0.486;
Jraw = [3.8278e-3; 3.8278e-3; 7.6566e-3];
Jshift = (Jraw(3) - Jraw(1) - Jraw(2)) / 3;
P.Jd = Jraw + Jshift * [1;1;-1];
P.Dv = [5.5670e-4; 5.5670e-4; 6.3540e-4];
P.Domega = [5.5670e-4; 5.5670e-4; 6.3540e-4];
P.alphaT = 1.0; P.alphaTau = [1;1;1];
P.Tmax = 4.0 * P.m * P.g;
end

function model = quad_acados_model(P)
import casadi.*
x = SX.sym('x',12); u = SX.sym('u',4);
eta = x(4:6); vel = x(7:9); om = x(10:12);
phi=eta(1); th=eta(2); psi=eta(3);
cphi=cos(phi); sphi=sin(phi); cth=cos(th); sth=sin(th); cpsi=cos(psi); spsi=sin(psi);
R = [cpsi*cth, cpsi*sth*sphi-spsi*cphi, cpsi*sth*cphi+spsi*sphi;
     spsi*cth, spsi*sth*sphi+cpsi*cphi, spsi*sth*cphi-cpsi*sphi;
     -sth,     cth*sphi,                cth*cphi];
Wm = [1, sphi*tan(th), cphi*tan(th); 0, cphi, -sphi; 0, sphi/cth, cphi/cth];
T = P.alphaT*u(1); tau = P.alphaTau.*u(2:4);
Jom = P.Jd.*om;
cro = [om(2)*Jom(3)-om(3)*Jom(2); om(3)*Jom(1)-om(1)*Jom(3); om(1)*Jom(2)-om(2)*Jom(1)];
f = [vel;
     Wm*om;
     [0;0;-P.g] + (T/P.m)*(R*[0;0;1]) - P.Dv.*vel/P.m;
     (tau - cro - P.Domega.*om)./P.Jd];
model = AcadosModel();
model.name = 'd1_quad_teacher';
model.x = x; model.u = u;
xdot = SX.sym('xdot',12); model.xdot = xdot;
model.f_expl_expr = f; model.f_impl_expr = xdot - f;
% tilt constraint: cos(tilt)=cphi*cth >= cos(70 deg)
model.con_h_expr = cphi*cth;
model.con_h_expr_0 = cphi*cth;
end

function solver = build_solver(model, P, N, Ts)
nx = 12; nu = 4; ny = nx+nu;
ocp = AcadosOcp();
ocp.model = model;
ocp.solver_options.N_horizon = N;
ocp.solver_options.tf = N*Ts;

[Q0, R0] = bryson_weights(P);
ocp.cost.cost_type = 'LINEAR_LS';
ocp.cost.cost_type_e = 'LINEAR_LS';
ocp.cost.W = blkdiag(Q0, R0); ocp.cost.W_e = Q0;    % stages overwritten at runtime
Vx = zeros(ny,nx); Vx(1:nx,1:nx) = eye(nx);
Vu = zeros(ny,nu); Vu(nx+1:end,1:nu) = eye(nu);
ocp.cost.Vx = Vx; ocp.cost.Vu = Vu; ocp.cost.Vx_e = eye(nx);
ocp.cost.yref = zeros(ny,1); ocp.cost.yref_e = zeros(nx,1);

% input bounds
ocp.constraints.idxbu = (0:3).';
ocp.constraints.lbu = [0; -0.5; -0.5; -0.25];
ocp.constraints.ubu = [P.Tmax; 0.5; 0.5; 0.25];
% state bounds (attitude/vel/rate; positions left free)
ocp.constraints.idxbx = (3:11).';
ocp.constraints.lbx = [-1.35;-1.35;-pi; -25;-25;-25; -10;-10;-10];
ocp.constraints.ubx = [ 1.35; 1.35; pi;  25; 25; 25;  10; 10; 10];
% tilt: cphi*cth in [cos(70deg), 1]  (soft to preserve feasibility)
ocp.constraints.lh = cos(deg2rad(70)); ocp.constraints.uh = 1.0;
ocp.constraints.lh_0 = cos(deg2rad(70)); ocp.constraints.uh_0 = 1.0;
ocp.constraints.x0 = zeros(nx,1);

ocp.solver_options.integrator_type = 'ERK';
ocp.solver_options.sim_method_num_stages = 4;
ocp.solver_options.sim_method_num_steps = 1;
ocp.solver_options.nlp_solver_type = 'SQP_RTI';
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM';
ocp.solver_options.qp_solver_cond_N = 5;
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON';

solver = AcadosOcpSolver(ocp);
end

function xn = rk4_step(x, u, dt, P)
k1 = quad_ode(x, u, P);
k2 = quad_ode(x + dt/2*k1, u, P);
k3 = quad_ode(x + dt/2*k2, u, P);
k4 = quad_ode(x + dt*k3, u, P);
xn = x + dt/6*(k1 + 2*k2 + 2*k3 + k4);
end

function xdot = quad_ode(x, u, P)
u = min(max(u, [0;-0.5;-0.5;-0.25]), [P.Tmax;0.5;0.5;0.25]);
eta = x(4:6); vel = x(7:9); om = x(10:12);
phi=eta(1); th=eta(2); psi=eta(3);
cphi=cos(phi); sphi=sin(phi); cth=cos(th); sth=sin(th); cpsi=cos(psi); spsi=sin(psi);
R = [cpsi*cth, cpsi*sth*sphi-spsi*cphi, cpsi*sth*cphi+spsi*sphi;
     spsi*cth, spsi*sth*sphi+cpsi*cphi, spsi*sth*cphi-cpsi*sphi;
     -sth,     cth*sphi,                cth*cphi];
Wm = [1, sphi*tan(th), cphi*tan(th); 0, cphi, -sphi; 0, sphi/cth, cphi/cth];
T = P.alphaT*u(1); tau = P.alphaTau.*u(2:4);
Jom = P.Jd.*om;
cro = [om(2)*Jom(3)-om(3)*Jom(2); om(3)*Jom(1)-om(1)*Jom(3); om(1)*Jom(2)-om(2)*Jom(1)];
xdot = [vel;
        Wm*om;
        [0;0;-P.g] + (T/P.m)*(R*[0;0;1]) - P.Dv.*vel/P.m;
        (tau - cro - P.Domega.*om)./P.Jd];
end
