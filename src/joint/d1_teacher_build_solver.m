function solver = d1_teacher_build_solver(cfg, thetaScenarios)
%D1_TEACHER_BUILD_SOLVER Robust scenario NMPC teacher (M=5, Nc=5) via acados.
%
% FROZEN spec realized here:
%   * M = 5 scenarios: augmented state stacks all scenarios' 12-state; the shared
%     control is applied to every scenario; cost = (1/M) * sum_i tracking_i.
%   * Nc = 5 control horizon: delta-u formulation. Augmented state carries u_prev;
%     input is du; u = u_prev + du. du is FREE for stages 0..Nc-1 and CLAMPED to 0
%     for stages Nc..N-1 (move blocking).
%   * DISCRETE dynamics (symbolic RK4 of quad_dynamics per scenario over Ts).
%   * Actuator limits enforced by bounding the u_prev states to [umin,umax]
%     (u_prev_{k+1}=u, so this bounds every applied control).
%   * Runtime Q,R retune via set('cost_W',...) with no rebuild.
%   * PRIVILEGED WIND: the known external force F (world frame, N) is an acados
%     parameter (np = 3) shared by all scenarios, v_dot^i += F/m_i; set every step by
%     d1_teacher_step (current force, held constant over the horizon).
%
% thetaScenarios: 1xM struct array, each field: m, Jd(3), Dv(3), Domega(3),
%   alphaT, alphaTau(3). Baked into the model at build (fixed per seed).

check_acados_requirements();
import casadi.*

P = cfg.plant; Ts = cfg.Ts; N = cfg.N; Nc = cfg.Nc;
M = numel(thetaScenarios);
n1 = 12; nu = 4; nxa = M*n1 + nu;    % [x^1..x^M ; u_prev]
ny = M*n1 + nu; ny_e = M*n1;

% ---- continuous single-scenario dynamics as a parametric CasADi Function ----
xs = SX.sym('x', n1); us = SX.sym('u', nu);
mp = SX.sym('m'); Jp = SX.sym('J',3); Dvp = SX.sym('Dv',3);
Domp = SX.sym('Dom',3); aTp = SX.sym('aT'); aTaup = SX.sym('aTau',3);
Fs = SX.sym('Fext', 3);              % external force (world frame)
eta = xs(4:6); vel = xs(7:9); om = xs(10:12);
phi=eta(1); th=eta(2); psi=eta(3);
cphi=cos(phi); sphi=sin(phi); cth=cos(th); sth=sin(th); cpsi=cos(psi); spsi=sin(psi);
Rm = [cpsi*cth, cpsi*sth*sphi-spsi*cphi, cpsi*sth*cphi+spsi*sphi;
      spsi*cth, spsi*sth*sphi+cpsi*cphi, spsi*sth*cphi-cpsi*sphi;
      -sth,     cth*sphi,                cth*cphi];
Wm = [1, sphi*tan(th), cphi*tan(th); 0, cphi, -sphi; 0, sphi/cth, cphi/cth];
Tt = aTp*us(1); tau = aTaup.*us(2:4);
Jom = Jp.*om;
cro = [om(2)*Jom(3)-om(3)*Jom(2); om(3)*Jom(1)-om(1)*Jom(3); om(1)*Jom(2)-om(2)*Jom(1)];
xdot = [vel;
        Wm*om;
        [0;0;-P.g] + (Tt/mp)*(Rm*[0;0;1]) - Dvp.*vel/mp + Fs/mp;
        (tau - cro - Domp.*om)./Jp];
fc = Function('fc', {xs, us, mp, Jp, Dvp, Domp, aTp, aTaup, Fs}, {xdot});

% ---- augmented DISCRETE dynamics --------------------------------------------
X = SX.sym('X', nxa); du = SX.sym('du', nu);
Fp = SX.sym('F', 3);                 % acados parameter: known external force
uprev = X(M*n1+1 : M*n1+nu);
u = uprev + du;
Xnext = SX.zeros(nxa, 1);
for i = 1:M
    th = thetaScenarios(i);
    xi = X((i-1)*n1+1 : i*n1);
    xi = rk4_step_sym(fc, xi, u, th, Ts, Fp);
    Xnext((i-1)*n1+1 : i*n1) = xi;
end
Xnext(M*n1+1 : M*n1+nu) = u;         % u_prev_{k+1} = u

model = AcadosModel();
model.name = 'd1_quad_teacher_m5w';
model.x = X; model.u = du; model.p = Fp;
model.disc_dyn_expr = Xnext;

ocp = AcadosOcp();
ocp.model = model;
ocp.parameter_values = zeros(3, 1);
ocp.solver_options.N_horizon = N;
ocp.solver_options.tf = N*Ts;

% ---- cost (LINEAR_LS): y=[x^1..x^M ; u], u=u_prev+du ------------------------
[Q0, R0] = d1_bryson_weights(P);
Wblk = repmat({Q0/M}, 1, M); Wblk{end+1} = R0;
Weblk = repmat({Q0/M}, 1, M);
ocp.cost.cost_type = 'LINEAR_LS';
ocp.cost.cost_type_e = 'LINEAR_LS';
ocp.cost.W = blkdiag(Wblk{:});
ocp.cost.W_e = blkdiag(Weblk{:});
Vx = zeros(ny, nxa); Vx(1:M*n1, 1:M*n1) = eye(M*n1);
Vx(M*n1+1:end, M*n1+1:end) = eye(nu);          % pick u_prev
Vu = zeros(ny, nu); Vu(M*n1+1:end, :) = eye(nu); % + du => u
ocp.cost.Vx = Vx; ocp.cost.Vu = Vu;
ocp.cost.Vx_e = [eye(M*n1), zeros(M*n1, nu)];
ocp.cost.yref = zeros(ny, 1); ocp.cost.yref_e = zeros(ny_e, 1);

% ---- constraints ------------------------------------------------------------
% du bounds (full-jump range); stages Nc..N-1 clamped to 0 below
ocp.constraints.idxbu = (0:3).';
ocp.constraints.lbu = [-P.Tmax; -1; -1; -0.5];
ocp.constraints.ubu = [ P.Tmax;  1;  1;  0.5];
% ONLY actuator limits via the u_prev states. NO hard attitude/rate/tilt bounds:
% those made the OCP infeasible on aggressive refs (solve fail -> divergence),
% while the unconstrained LQR tracks the same refs fine. Physical feasibility of
% the refs is guaranteed by S0 (tier-A); teacher need not re-enforce state boxes.
ocp.constraints.idxbx = (M*n1:M*n1+3).';
ocp.constraints.lbx = [0; -0.5; -0.5; -0.25];
ocp.constraints.ubx = [P.Tmax; 0.5; 0.5; 0.25];
ocp.constraints.x0 = zeros(nxa, 1);

ocp.solver_options.integrator_type = 'DISCRETE';
% Solver selectable via cfg.solverType (env D1_SOLVER): 'SQP_RTI' (1 iter, ~10x
% faster, more surrogate samples) vs 'SQP' (up to 30 iters, tighter labels).
ocp.solver_options.nlp_solver_type = cfg.solverType;
if strcmp(cfg.solverType, 'SQP')
    ocp.solver_options.nlp_solver_max_iter = 50;
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING';   % line search -> higher convergence rate
    % time limit per solve (checked after every SQP iteration): a pathologically slow solve
    % (degenerate QP data inside HPIPM, seen as multi-hour cases) ends with status 7 =
    % ACADOS_TIMEOUT, which d1_teacher_step treats as unusable.
    ocp.solver_options.timeout_max_time = cfg.solveTimeout;
    ocp.solver_options.timeout_heuristic = 'ZERO';
end
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM';
ocp.solver_options.qp_solver_cond_N = 5;
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON';
ocp.solver_options.levenberg_marquardt = 1e-3;   % regularize -> avoid QP NaN
ocp.solver_options.qp_solver_iter_max = 100;

solver = AcadosOcpSolver(ocp);

% Nc move-blocking: du = 0 for stages Nc..N-1
for s = Nc:N-1
    solver.set('constr_lbu', zeros(nu,1), s);
    solver.set('constr_ubu', zeros(nu,1), s);
end
end

% ============================================================================
function xn = rk4_step_sym(fc, x, u, th, Ts, F)
k1 = fc(x,          u, th.m, th.Jd, th.Dv, th.Domega, th.alphaT, th.alphaTau, F);
k2 = fc(x+Ts/2*k1,  u, th.m, th.Jd, th.Dv, th.Domega, th.alphaT, th.alphaTau, F);
k3 = fc(x+Ts/2*k2,  u, th.m, th.Jd, th.Dv, th.Domega, th.alphaT, th.alphaTau, F);
k4 = fc(x+Ts*k3,    u, th.m, th.Jd, th.Dv, th.Domega, th.alphaT, th.alphaTau, F);
xn = x + Ts/6*(k1 + 2*k2 + 2*k3 + k4);
end
