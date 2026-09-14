function acados_quad_nmpc_benchmark()
%ACADOS_QUAD_NMPC_BENCHMARK Real per-solve timing of the 12-state quadrotor NMPC
%under acados (SQP_RTI + HPIPM). Single nominal scenario, N=20, Ts=0.05.
%
% Purpose: convert the speculative "~3-14 h/seed" estimate into a measured
% per-solve number so the 1000x20 SAC budget can be judged. Plant parameters are
% hardcoded to match configs/step1_plant_config.m so this script is standalone.
% NOT a science run -- an infrastructure benchmark.

check_acados_requirements();
import casadi.*

% ---- plant params (mirror of step1_plant_config nominal) -------------------
g = 9.81; m = 0.486;
Jraw = [3.8278e-3; 3.8278e-3; 7.6566e-3];
Jshift = (Jraw(3) - Jraw(1) - Jraw(2)) / 3;
Jd = Jraw + Jshift * [1; 1; -1];
Dv = [5.5670e-4; 5.5670e-4; 6.3540e-4];
Domega = [5.5670e-4; 5.5670e-4; 6.3540e-4];
alphaT = 1.0; alphaTau = [1; 1; 1];
Tmax = 4.0 * m * g;

% ---- symbolic dynamics (mirror of quad_dynamics) ---------------------------
x = SX.sym('x', 12); u = SX.sym('u', 4);
eta = x(4:6); vel = x(7:9); om = x(10:12);
phi = eta(1); th = eta(2); psi = eta(3);
cphi = cos(phi); sphi = sin(phi);
cth = cos(th); sth = sin(th);
cpsi = cos(psi); spsi = sin(psi);
R = [cpsi*cth, cpsi*sth*sphi - spsi*cphi, cpsi*sth*cphi + spsi*sphi;
     spsi*cth, spsi*sth*sphi + cpsi*cphi, spsi*sth*cphi - cpsi*sphi;
     -sth,     cth*sphi,                   cth*cphi];
Wmat = [1, sphi*tan(th),  cphi*tan(th);
        0, cphi,         -sphi;
        0, sphi/cth,      cphi/cth];
T = alphaT * u(1);
tau = alphaTau .* u(2:4);
e3 = [0; 0; 1];
Jom = Jd .* om;                                  % J diagonal
crossOmJom = [om(2)*Jom(3) - om(3)*Jom(2);
              om(3)*Jom(1) - om(1)*Jom(3);
              om(1)*Jom(2) - om(2)*Jom(1)];
posDot = vel;
etaDot = Wmat * om;
velDot = [0; 0; -g] + (T/m) * (R * e3) - Dv .* vel / m;
omDot = (tau - crossOmJom - Domega .* om) ./ Jd;
f_expl = [posDot; etaDot; velDot; omDot];

model = AcadosModel();
model.name = 'quad12_nmpc';
model.x = x; model.u = u;
xdot = SX.sym('xdot', 12);
model.xdot = xdot;
model.f_expl_expr = f_expl;
model.f_impl_expr = xdot - f_expl;

% ---- OCP -------------------------------------------------------------------
nx = 12; nu = 4; ny = nx + nu;
N = 20; Ts = 0.05;
ocp = AcadosOcp();
ocp.model = model;
ocp.solver_options.N_horizon = N;
ocp.solver_options.tf = N * Ts;

eAllow = [0.10; 0.10; 0.10; deg2rad(5)*[1;1;1]; 0.30; 0.30; 0.30; 2; 2; 2];
duAllow = [m*g; 0.5; 0.5; 0.25];
Q = diag(1 ./ eAllow.^2);
Rw = diag(1 ./ duAllow.^2);
ocp.cost.cost_type = 'LINEAR_LS';
ocp.cost.cost_type_e = 'LINEAR_LS';
ocp.cost.W = blkdiag(Q, Rw);
ocp.cost.W_e = Q;
Vx = zeros(ny, nx); Vx(1:nx, 1:nx) = eye(nx);
Vu = zeros(ny, nu); Vu(nx+1:nx+nu, 1:nu) = eye(nu);
ocp.cost.Vx = Vx; ocp.cost.Vu = Vu;
ocp.cost.Vx_e = eye(nx);
uhover = [m*g; 0; 0; 0];
ocp.cost.yref = [zeros(nx, 1); uhover];
ocp.cost.yref_e = zeros(nx, 1);

ocp.constraints.idxbu = (0:3).';
ocp.constraints.lbu = [0; -0.5; -0.5; -0.25];
ocp.constraints.ubu = [Tmax; 0.5; 0.5; 0.25];
ocp.constraints.x0 = zeros(nx, 1);

ocp.solver_options.integrator_type = 'ERK';
ocp.solver_options.sim_method_num_stages = 4;
ocp.solver_options.sim_method_num_steps = 1;
ocp.solver_options.nlp_solver_type = 'SQP_RTI';
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM';
ocp.solver_options.qp_solver_cond_N = 5;
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON';

tbuild = tic;
solver = AcadosOcpSolver(ocp);
fprintf('BUILD_SECONDS=%.2f\n', toc(tbuild));

% ---- benchmark: RTI over a pseudo-rollout ----------------------------------
Nsolve = 300;
xcur = zeros(nx, 1);
xcur(1:3) = [1.0; 0.5; -0.3];      % initial tracking error
xcur(4:5) = deg2rad([5; -5]);
times = zeros(Nsolve, 1);
okStatus = 0;
for k = 1:Nsolve
    solver.set('constr_x0', xcur);
    ts = tic;
    solver.solve();
    times(k) = toc(ts);
    if solver.get('status') == 0
        okStatus = okStatus + 1;
    end
    xcur = solver.get('x', 1);     % model prediction = next "plant" state
end
ms = 1000 * times(2:end);          % drop first (includes lazy init)
fprintf('ACADOS_NMPC_BENCH single_scenario N=%d solves=%d ok=%d\n', ...
    N, Nsolve, okStatus);
fprintf('SOLVE_MS median=%.3f mean=%.3f p95=%.3f min=%.3f max=%.3f\n', ...
    median(ms), mean(ms), quantile(ms, 0.95), min(ms), max(ms));
fprintf('EST_PER_SEED_HOURS_1scn @1000x20x200 = %.2f\n', ...
    1000*20*200*median(ms)/1000/3600);
fprintf('EST_PER_SEED_HOURS_5scn @1000x20x200 = %.2f (rough x5)\n', ...
    5*1000*20*200*median(ms)/1000/3600);
fprintf('ACADOS_NMPC_BENCH_DONE\n');
end
