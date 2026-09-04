function run_targeted_solver_acceptance_tests()
%RUN_TARGETED_SOLVER_ACCEPTANCE_TESTS Contract tests for solve acceptance.

projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(projectRoot, 'src', 'rl'));

cfg.environment.solverAcceptancePolicy = 'strict_converged_feasible';
solution = nominal_solution();
[accepted, reason] = targeted_nmpc_accept_solution(solution, cfg);
assert(accepted && reason == "accepted_converged");

solution.exitflag = 0;
solution.converged = false;
solution.feasibleSuboptimal = true;
[accepted, reason] = targeted_nmpc_accept_solution(solution, cfg);
assert(~accepted && reason == "solver_limit_reached");

cfg.environment.solverAcceptancePolicy = ...
    'converged_or_feasible_suboptimal';
[accepted, reason] = targeted_nmpc_accept_solution(solution, cfg);
assert(accepted && reason == "accepted_feasible_suboptimal");

solution.timedOut = true;
[accepted, reason] = targeted_nmpc_accept_solution(solution, cfg);
assert(~accepted && reason == "wall_time_stop");

solution = nominal_solution();
solution.feasible = false;
solution.maxConstraintViolation = 1;
[accepted, reason] = targeted_nmpc_accept_solution(solution, cfg);
assert(~accepted && reason == "infeasible");

solution = nominal_solution();
solution.solveTime = NaN;
[accepted, reason] = targeted_nmpc_accept_solution(solution, cfg);
assert(~accepted && reason == "nonfinite_telemetry");

fprintf('Targeted NMPC solver-acceptance contracts passed.\n');
end

function solution = nominal_solution()
solution.exitflag = 1;
solution.solveTime = 1;
solution.timedOut = false;
solution.converged = true;
solution.feasible = true;
solution.feasibleSuboptimal = false;
solution.maxConstraintViolation = 0;
end
