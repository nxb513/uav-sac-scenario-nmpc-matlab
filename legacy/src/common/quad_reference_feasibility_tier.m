function [tier, reasons] = quad_reference_feasibility_tier(metric, caps)
%QUAD_REFERENCE_FEASIBILITY_TIER Pre-registered 3-tier (+unresolved) membership.
%
% Classifies one reference case from its feasibility metric (see
% quad_reference_feasibility_metrics) using TIMEWISE MAXIMA, not percentiles, so
% a single-sample excursion cannot be hidden. Membership is controller-
% independent: it depends only on the reference and the declared caps, never on
% any controller outcome. Tiers (frozen in the preregistration):
%
%   'A_primary'    : feasible within the robust design margins -> the evaluation
%                    bank. INCLUDES every feasible-but-hard-for-LQR case.
%   'B_boundary'   : physically feasible but outside the robust margins
%                    (boundary-stress bank, reported separately).
%   'C_infeasible' : physically infeasible exact reference (workspace/state-bound
%                    exit, tilt or input beyond the physical caps, non-finite).
%   'U_unresolved' : otherwise-feasible but the numerical feedforward defect
%                    (max dynamic residual) exceeds the declared ceiling.
%
% caps fields (all required): tiltRobustDeg, tiltPhysicalDeg,
% inputRobustFraction, inputPhysicalFraction, residualMaxLimit.
% See docs/notes/d1_execution_plan_20260912.md and Mueller-Hehn-D'Andrea,
% IEEE T-RO 2015 (reference-feasibility verification).

reasons = strings(0, 1);

physicallyInfeasible = false;
if ~metric.finite
    physicallyInfeasible = true; reasons(end+1) = "nonfinite"; %#ok<AGROW>
end
if metric.stateBoundViolation > 0
    physicallyInfeasible = true; reasons(end+1) = "workspace/state-bound"; %#ok<AGROW>
end
if metric.peakTiltDeg > caps.tiltPhysicalDeg
    physicallyInfeasible = true; reasons(end+1) = "tilt>physical"; %#ok<AGROW>
end
if metric.peakInputFraction > caps.inputPhysicalFraction
    physicallyInfeasible = true; reasons(end+1) = "input>physical"; %#ok<AGROW>
end

if physicallyInfeasible
    tier = 'C_infeasible';
    return;
end

if metric.dynamicResidualMax > caps.residualMaxLimit
    tier = 'U_unresolved';
    reasons(end+1) = "residualMax>ceiling"; %#ok<AGROW>
    return;
end

withinRobust = metric.peakTiltDeg <= caps.tiltRobustDeg && ...
    metric.peakInputFraction <= caps.inputRobustFraction && ...
    metric.stateBoundViolation <= 0 && metric.finite;

if withinRobust
    tier = 'A_primary';
else
    tier = 'B_boundary';
    if metric.peakTiltDeg > caps.tiltRobustDeg
        reasons(end+1) = "tilt>robust"; %#ok<AGROW>
    end
    if metric.peakInputFraction > caps.inputRobustFraction
        reasons(end+1) = "input>robust"; %#ok<AGROW>
    end
end
end
