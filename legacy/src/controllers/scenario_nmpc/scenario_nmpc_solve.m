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
% The reference may be a plain 12-by-(N+1) state window (legacy) or a struct with
% fields .X (state window) and .U (4-by-N input feedforward window) so the
% control-deviation penalty can reference the flatness feedforward.
if isstruct(reference)
    uRefWindow = reference.U;
    reference = reference.X;
else
    uRefWindow = [];
end
Xref = nmpc_prepare_reference(reference, horizon);
Ucontrol0 = prepare_warm_start(nominalTheta, horizon, controlHorizon, warmStart);
[lb, ub] = nmpc_input_bounds(nominalTheta, controlHorizon);
z0 = min(max(Ucontrol0(:), lb), ub);

% Start the solve clock BEFORE building the objective so the objective can
% enforce the per-solve wall limit on EVERY function evaluation (not only per
% fmincon iteration via OutputFcn). This makes a single pathological solve
% impossible to run away for many minutes, which is what caused shard timeouts
% at long horizons/hard cases.
maxWallSeconds = solver_wall_limit(cfg);
solveClock = tic;
stoppedByWallTime = false;

objective = @(z) scenario_objective( ...
    z, x0, Xref, uRefWindow, thetaScenarios, cfg, previousInput, ...
    solveClock, maxWallSeconds);
nonlcon = @(z) scenario_constraints(z, x0, thetaScenarios, cfg);
if ~cfg.constraints.enableStateBounds || ~cfg.constraints.enforceScenarioStateBounds
    nonlcon = [];
end

warmStartCost = objective(z0);

options = nmpc_fmincon_options(cfg);
if isfinite(maxWallSeconds)
    options.OutputFcn = @stop_on_wall_time;
end
% fmincon's internal SQP (nlpSQP) can itself error on a pathological/ill-conditioned
% step. Treat that like a failed solve and fall back to the (saturated) warm start
% so one bad step cannot abort the whole episode/shard.
try
    [zOpt, fval, exitflag, output] = fmincon(objective, z0, [], [], [], [], ...
                                             lb, ub, nonlcon, options);
catch solveErr
    zOpt = z0;
    fval = warmStartCost;
    exitflag = -3;
    output = struct('message', ['fmincon errored; warm-start fallback: ' ...
        solveErr.message], 'iterations', 0, 'funcCount', 0);
end
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
        z, x0, Xref, uRefWindow, thetaScenarios, cfg, previousInput, ...
        solveClock, maxWallSeconds)
% Numerical barrier for a candidate input whose predicted rollout leaves the
% valid attitude chart (ZYX Euler singularity at |pitch| = 90 deg) or otherwise
% diverges to a non-finite state. Returning a dominating finite cost makes the
% sqp solver treat the probe point as infeasible and step away, instead of the
% plant model throwing and aborting the whole solve. The optimum is invariant to
% the barrier magnitude provided it exceeds any attainable feasible cost, so this
% is a class-R numerical constant, not a scientific/tunable parameter.
INVALID_ROLLOUT_PENALTY = 1e12;

% Per-evaluation wall guard: once the per-solve wall limit is exceeded, return the
% barrier for EVERY subsequent evaluation so fmincon exhausts quickly and returns.
% This bounds a single solve reliably even when fmincon is stuck making many slow
% evaluations, which OutputFcn (per-iteration) alone does not catch.
if nargin >= 9 && ~isempty(maxWallSeconds) && isfinite(maxWallSeconds) && ...
        toc(solveClock) > maxWallSeconds
    cost = INVALID_ROLLOUT_PENALTY;
    return;
end

controlHorizon = nmpc_control_horizon(cfg);
Ucontrol = reshape(z, 4, controlHorizon);
U = nmpc_expand_control_sequence(Ucontrol, cfg.predictionHorizon);
scenarioCount = numel(thetaScenarios);
cost = 0.0;

for i = 1:scenarioCount
    theta = thetaScenarios(i);
    try
        X = nmpc_rollout(x0, U, theta, cfg.sampleTime, ...
                         cfg.rollout.disturbance, cfg.rollout.startTime);
        stepCost = nmpc_tracking_cost(X, U, Xref, theta, cfg, ...
                                      previousInput, uRefWindow);
    catch
        stepCost = INVALID_ROLLOUT_PENALTY;
    end
    if ~isfinite(stepCost)
        stepCost = INVALID_ROLLOUT_PENALTY;
    end
    cost = cost + stepCost;
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
    try
        X = nmpc_rollout(x0, U, thetaScenarios(i), cfg.sampleTime, ...
                         cfg.rollout.disturbance, cfg.rollout.startTime);
        c = [c; nmpc_state_bound_violations(X, cfg)]; %#ok<AGROW>
    catch
        % Predicted rollout left the valid attitude chart: mark strongly
        % infeasible so the solver rejects this probe point.
        c = [c; 1e6]; %#ok<AGROW>
    end
end

ceq = [];
end

function XpredScenarios = rollout_all_scenarios(x0, U, thetaScenarios, cfg)
scenarioCount = numel(thetaScenarios);
XpredScenarios = zeros(12, cfg.predictionHorizon + 1, scenarioCount);

for i = 1:scenarioCount
    try
        XpredScenarios(:, :, i) = nmpc_rollout(x0, U, thetaScenarios(i), ...
                                               cfg.sampleTime, ...
                                               cfg.rollout.disturbance, ...
                                               cfg.rollout.startTime);
    catch
        % A selected input that still leaves the valid chart is flagged by
        % NaN, so finiteSolution below marks the solution infeasible.
        XpredScenarios(:, :, i) = NaN;
    end
end
end
