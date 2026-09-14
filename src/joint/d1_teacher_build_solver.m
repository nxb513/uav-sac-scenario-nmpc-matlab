function solver = d1_teacher_build_solver(cfg)
%D1_TEACHER_BUILD_SOLVER Build the acados quad NMPC teacher solver ONCE.
%
% Returns an AcadosOcpSolver. The SAC loop sets Q,R per rollout via
%   for s=0:N-1, solver.set('cost_W', blkdiag(Q,R), s); end
% and the per-stage reference via solver.set('cost_y_ref', yref, s), with no
% rebuild (verified: runtime W retune works). Single nominal scenario; M=5 robust
% and Nc=5 blocking are teacher-internal refinements tracked in the audit doc.
%
% cfg fields used: N, Ts, plant params via d1_joint_plant_params.

check_acados_requirements();
import casadi.*

P = cfg.plant;
nx = 12; nu = 4; ny = nx + nu;
N = cfg.N; Ts = cfg.Ts;

% ---- symbolic model (mirror of quad_dynamics, nominal) ----------------------
x = SX.sym('x', nx); u = SX.sym('u', nu);
eta = x(4:6); vel = x(7:9); om = x(10:12);
phi = eta(1); th = eta(2); psi = eta(3);
cphi = cos(phi); sphi = sin(phi); cth = cos(th); sth = sin(th);
cpsi = cos(psi); spsi = sin(psi);
R = [cpsi*cth, cpsi*sth*sphi - spsi*cphi, cpsi*sth*cphi + spsi*sphi;
     spsi*cth, spsi*sth*sphi + cpsi*cphi, spsi*sth*cphi - cpsi*sphi;
     -sth,     cth*sphi,                   cth*cphi];
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
xdot = SX.sym('xdot', nx); model.xdot = xdot;
model.f_expl_expr = f; model.f_impl_expr = xdot - f;
model.con_h_expr = cphi*cth;                         % tilt
model.con_h_expr_0 = cphi*cth;

ocp = AcadosOcp();
ocp.model = model;
ocp.solver_options.N_horizon = N;
ocp.solver_options.tf = N*Ts;

[Q0, R0] = d1_bryson_weights(P);
ocp.cost.cost_type = 'LINEAR_LS';
ocp.cost.cost_type_e = 'LINEAR_LS';
ocp.cost.W = blkdiag(Q0, R0); ocp.cost.W_e = Q0;
Vx = zeros(ny, nx); Vx(1:nx, 1:nx) = eye(nx);
Vu = zeros(ny, nu); Vu(nx+1:end, 1:nu) = eye(nu);
ocp.cost.Vx = Vx; ocp.cost.Vu = Vu; ocp.cost.Vx_e = eye(nx);
ocp.cost.yref = zeros(ny, 1); ocp.cost.yref_e = zeros(nx, 1);

ocp.constraints.idxbu = (0:3).';
ocp.constraints.lbu = [0; -0.5; -0.5; -0.25];
ocp.constraints.ubu = [P.Tmax; 0.5; 0.5; 0.25];
ocp.constraints.idxbx = (3:11).';
ocp.constraints.lbx = [-1.35;-1.35;-pi; -25;-25;-25; -10;-10;-10];
ocp.constraints.ubx = [ 1.35; 1.35; pi;  25; 25; 25;  10; 10; 10];
ocp.constraints.lh = cos(deg2rad(70)); ocp.constraints.uh = 1.0;
ocp.constraints.lh_0 = cos(deg2rad(70)); ocp.constraints.uh_0 = 1.0;
ocp.constraints.x0 = zeros(nx, 1);

ocp.solver_options.integrator_type = 'ERK';
ocp.solver_options.sim_method_num_stages = 4;
ocp.solver_options.sim_method_num_steps = 1;
ocp.solver_options.nlp_solver_type = 'SQP_RTI';
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM';
ocp.solver_options.qp_solver_cond_N = 5;
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON';

solver = AcadosOcpSolver(ocp);
end
