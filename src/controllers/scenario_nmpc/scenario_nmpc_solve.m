function sol = scenario_nmpc_solve( ...
        x0, reference, thetaScenarios, cfg, warmStart, previousInput)
%SCENARIO_NMPC_SOLVE Nonlinear scenario MPC teacher with shared input sequence.

if nargin < 6
    previousInput = [];
end

if nargin < 4 || isempty(cfg)
    cfg = step2_nmpc_config();
end
if nargin < 3 || isempty(thetaScenarios)
    thetaScenarios = sample_default_scenarios(cfg);
end

thetaScenarios = thetaScenarios(:).';
nominalTheta = cfg.plant.nominal;
horizon = cfg.predictionHorizon;
controlHorizon = nmpc_control_horizon(cfg);
Xref = nmpc_prepare_reference(reference, horizon);
Ucontrol0 = prepare_warm_start(nominalTheta, horizon, controlHorizon, warmStart);
[lb, ub] = nmpc_input_bounds(nominalTheta, controlHorizon);
z0 = min(max(Ucontrol0(:), lb), ub);

objective = @(z) scenario_objective( ...
    z, x0, Xref, thetaScenarios, cfg, previousInput);
nonlcon = @(z) scenario_constraints(z, x0, thetaScenarios, cfg);
if ~cfg.constraints.enableStateBounds || ~cfg.constraints.enforceScenarioStateBounds
    nonlcon = [];
end

warmStartCost = objective(z0);

solveClock = tic;
stoppedByWallTime = false;
options = nmpc_fmincon_options(cfg);
maxWallSeconds = solver_wall_limit(cfg);
if isfinite(maxWallSeconds)
    options.OutputFcn = @stop_on_wall_time;
end
[zOpt, fval, exitflag, output] = fmincon(objective, z0, [], [], [], [], ...
                                         lb, ub, nonlcon, options);
solveTime = toc(solveClock);

UcontrolOpt = reshape(zOpt, 4, controlHorizon);
Uopt = nmpc_expand_control_sequence(UcontrolOpt, horizon);
XpredScenarios = rollout_all_scenarios(x0, Uopt, thetaScenarios, cfg);
XpredMean = mean(XpredScenarios, 3);
[maxConstraintViolation, maxStateConstraintViolation, ...
    maxInputConstraintViolation] = solution_constraint_violation( ...
    zOpt, lb, ub, XpredScenarios, cfg);
finiteSolution = all(isfinite(zOpt)) && isfinite(fval) && ...
    all(isfinite(XpredScenarios), 'all');
feasible = finiteSolution && maxConstraintViolation <= ...
    cfg.solver.constraintTolerance;

sol.u0 = Uopt(:, 1);
sol.U = Uopt;
sol.Ucontrol = UcontrolOpt;
sol.Xpred = XpredMean;
sol.XpredScenarios = XpredScenarios;
sol.thetaScenarios = thetaScenarios;
sol.cost = fval;
sol.warmStartCost = warmStartCost;
sol.exitflag = exitflag;
sol.output = output;
sol.solveTime = solveTime;
sol.timedOut = stoppedByWallTime;
sol.converged = exitflag > 0;
sol.limitReached = exitflag == 0;
sol.feasible = feasible;
sol.feasibleSuboptimal = feasible && exitflag == 0 && ...
    ~stoppedByWallTime;
sol.maxConstraintViolation = maxConstraintViolation;
sol.maxStateConstraintViolation = maxStateConstraintViolation;
sol.maxInputConstraintViolation = maxInputConstraintViolation;
sol.solver = cfg.solver.name;
sol.algorithm = cfg.solver.algorithm;

    function stop = stop_on_wall_time(~, ~, ~)
        stop = toc(solveClock) >= maxWallSeconds;
        stoppedByWallTime = stoppedByWallTime || stop;
    end
end

function maxWallSeconds = solver_wall_limit(cfg)
maxWallSeconds = Inf;
if isfield(cfg.solver, 'maxWallSeconds') && ...
        ~isempty(cfg.solver.maxWallSeconds)
    candidate = double(cfg.solver.maxWallSeconds);
    if isscalar(candidate) && isfinite(candidate) && candidate > 0
        maxWallSeconds = candidate;
    end
end
end

function thetaScenarios = sample_default_scenarios(cfg)
thetaScenarios = quad_sample_uncertainty(cfg.plant, cfg.scenario.count, ...
                                         cfg.scenario.domain, ...
                                         cfg.scenario.seed, ...
                                         cfg.scenario.method);
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
    error('scenario_nmpc_solve:BadWarmStart', ...
          'warmStart must be 4-by-N or 4-by-Nc.');
end

U0 = U0(:, 1:controlHorizon);
Ucontrol0 = nmpc_saturate_sequence(U0, theta);
end

function cost = scenario_objective( ...
        z, x0, Xref, thetaScenarios, cfg, previousInput)
controlHorizon = nmpc_control_horizon(cfg);
Ucontrol = reshape(z, 4, controlHorizon);
U = nmpc_expand_control_sequence(Ucontrol, cfg.predictionHorizon);
scenarioCount = numel(thetaScenarios);
cost = 0.0;

for i = 1:scenarioCount
    theta = thetaScenarios(i);
    X = nmpc_rollout(x0, U, theta, cfg.sampleTime, ...
                     cfg.rollout.disturbance, cfg.rollout.startTime);
    cost = cost + nmpc_tracking_cost( ...
        X, U, Xref, theta, cfg, previousInput);
end

cost = cost / scenarioCount;
end

function [maximum, stateMaximum, inputMaximum] = ...
        solution_constraint_violation(z, lb, ub, XpredScenarios, cfg)
inputMaximum = max([0; z(:) - ub(:); lb(:) - z(:)]);
stateMaximum = 0;
for index = 1:size(XpredScenarios, 3)
    violation = nmpc_state_bound_violations( ...
        XpredScenarios(:, :, index), cfg);
    if ~isempty(violation)
        stateMaximum = max(stateMaximum, max([0; violation(:)]));
    end
end
maximum = max(inputMaximum, stateMaximum);
if ~isfinite(maximum)
    maximum = Inf;
end
end

function [c, ceq] = scenario_constraints(z, x0, thetaScenarios, cfg)
controlHorizon = nmpc_control_horizon(cfg);
Ucontrol = reshape(z, 4, controlHorizon);
U = nmpc_expand_control_sequence(Ucontrol, cfg.predictionHorizon);
scenarioCount = numel(thetaScenarios);
c = [];

for i = 1:scenarioCount
    X = nmpc_rollout(x0, U, thetaScenarios(i), cfg.sampleTime, ...
                     cfg.rollout.disturbance, cfg.rollout.startTime);
    c = [c; nmpc_state_bound_violations(X, cfg)]; %#ok<AGROW>
end

ceq = [];
end

function XpredScenarios = rollout_all_scenarios(x0, U, thetaScenarios, cfg)
scenarioCount = numel(thetaScenarios);
XpredScenarios = zeros(12, cfg.predictionHorizon + 1, scenarioCount);

for i = 1:scenarioCount
    XpredScenarios(:, :, i) = nmpc_rollout(x0, U, thetaScenarios(i), ...
                                           cfg.sampleTime, ...
                                           cfg.rollout.disturbance, ...
                                           cfg.rollout.startTime);
end
end
