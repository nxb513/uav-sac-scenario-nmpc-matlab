function sol = fixed_nmpc_solve(x0, reference, theta, cfg, warmStart)
%FIXED_NMPC_SOLVE Nominal nonlinear MPC baseline over the step-1 plant.

if nargin < 3 || isempty(theta)
    plantCfg = step1_plant_config();
    theta = plantCfg.nominal;
end
if nargin < 4 || isempty(cfg)
    cfg = step2_nmpc_config();
end

horizon = cfg.predictionHorizon;
controlHorizon = nmpc_control_horizon(cfg);
Xref = nmpc_prepare_reference(reference, horizon);
Ucontrol0 = prepare_warm_start(theta, horizon, controlHorizon, warmStart);
[lb, ub] = nmpc_input_bounds(theta, controlHorizon);
z0 = min(max(Ucontrol0(:), lb), ub);

objective = @(z) fixed_objective(z, x0, Xref, theta, cfg);
nonlcon = @(z) fixed_constraints(z, x0, theta, cfg);
if ~cfg.constraints.enableStateBounds
    nonlcon = [];
end

options = nmpc_fmincon_options(cfg);
warmStartCost = objective(z0);

tic;
[zOpt, fval, exitflag, output] = fmincon(objective, z0, [], [], [], [], ...
                                         lb, ub, nonlcon, options);
solveTime = toc;

UcontrolOpt = reshape(zOpt, 4, controlHorizon);
Uopt = nmpc_expand_control_sequence(UcontrolOpt, horizon);
Xpred = nmpc_rollout(x0, Uopt, theta, cfg.sampleTime, ...
                     cfg.rollout.disturbance, cfg.rollout.startTime);

sol.u0 = Uopt(:, 1);
sol.U = Uopt;
sol.Ucontrol = UcontrolOpt;
sol.Xpred = Xpred;
sol.cost = fval;
sol.warmStartCost = warmStartCost;
sol.exitflag = exitflag;
sol.output = output;
sol.solveTime = solveTime;
sol.solver = cfg.solver.name;
sol.algorithm = cfg.solver.algorithm;
end

function Ucontrol0 = prepare_warm_start(theta, horizon, controlHorizon, warmStart)
if nargin < 4 || isempty(warmStart)
    U0 = nmpc_default_warm_start(theta, controlHorizon);
elseif isstruct(warmStart) && isfield(warmStart, 'U')
    U0 = warmStart.U;
else
    U0 = warmStart;
end

if isvector(U0)
    U0 = reshape(U0, 4, []);
end
if size(U0, 1) ~= 4 || ~(size(U0, 2) == horizon || size(U0, 2) == controlHorizon)
    error('fixed_nmpc_solve:BadWarmStart', ...
          'warmStart must be 4-by-N or 4-by-Nc.');
end

U0 = U0(:, 1:controlHorizon);
Ucontrol0 = nmpc_saturate_sequence(U0, theta);
end

function cost = fixed_objective(z, x0, Xref, theta, cfg)
controlHorizon = nmpc_control_horizon(cfg);
Ucontrol = reshape(z, 4, controlHorizon);
U = nmpc_expand_control_sequence(Ucontrol, cfg.predictionHorizon);
X = nmpc_rollout(x0, U, theta, cfg.sampleTime, ...
                 cfg.rollout.disturbance, cfg.rollout.startTime);
cost = nmpc_tracking_cost(X, U, Xref, theta, cfg);
end

function [c, ceq] = fixed_constraints(z, x0, theta, cfg)
controlHorizon = nmpc_control_horizon(cfg);
Ucontrol = reshape(z, 4, controlHorizon);
U = nmpc_expand_control_sequence(Ucontrol, cfg.predictionHorizon);
X = nmpc_rollout(x0, U, theta, cfg.sampleTime, ...
                 cfg.rollout.disturbance, cfg.rollout.startTime);
c = nmpc_state_bound_violations(X, cfg);
ceq = [];
end
