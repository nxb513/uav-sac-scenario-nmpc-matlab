function [accepted, reason] = targeted_nmpc_accept_solution(solution, cfg)
%TARGETED_NMPC_ACCEPT_SOLUTION Apply the configured teacher-solve policy.

policy = "strict_converged_feasible";
if isfield(cfg.environment, 'solverAcceptancePolicy')
    policy = string(cfg.environment.solverAcceptancePolicy);
end

required = {'exitflag', 'solveTime', 'timedOut', 'converged', ...
    'feasible', 'feasibleSuboptimal', 'maxConstraintViolation'};
for index = 1:numel(required)
    if ~isfield(solution, required{index})
        error('targeted_nmpc_accept_solution:MissingTelemetry', ...
            'NMPC solution is missing telemetry field %s.', required{index});
    end
end

finiteTelemetry = isfinite(solution.exitflag) && ...
    isfinite(solution.solveTime) && ...
    isfinite(solution.maxConstraintViolation);
if ~finiteTelemetry
    accepted = false;
    reason = "nonfinite_telemetry";
    return;
end
if solution.timedOut
    accepted = false;
    reason = "wall_time_stop";
    return;
end
if ~solution.feasible
    accepted = false;
    reason = "infeasible";
    return;
end

switch policy
    case "strict_converged_feasible"
        accepted = solution.converged;
        if accepted
            reason = "accepted_converged";
        else
            reason = "solver_limit_reached";
        end
    case "converged_or_feasible_suboptimal"
        accepted = solution.converged || solution.feasibleSuboptimal;
        if solution.converged
            reason = "accepted_converged";
        elseif solution.feasibleSuboptimal
            reason = "accepted_feasible_suboptimal";
        else
            reason = "unsupported_exitflag";
        end
    otherwise
        error('targeted_nmpc_accept_solution:UnknownPolicy', ...
            'Unknown solver acceptance policy: %s.', policy);
end
end
