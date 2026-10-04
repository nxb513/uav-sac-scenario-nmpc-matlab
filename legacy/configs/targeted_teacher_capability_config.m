function cfg = targeted_teacher_capability_config()
%TARGETED_TEACHER_CAPABILITY_CONFIG Bounded paired closed-loop teacher gate.

cfg.version = 'targeted_teacher_capability_h20_v1';
cfg.status = 'development_gate_not_final_solver_or_acceptance_selection';
cfg.targeted = targeted_lqr_weak_config();
cfg.sac = targeted_specialist_sac_config();
cfg.divergence = targeted_lqr_divergence_config();
cfg.outputRoot = fullfile(cfg.targeted.resultRoot, ...
    'specialist_teacher_closed_loop_capability_h20_v1');
cfg.rolloutSteps = cfg.divergence.horizonSteps;
cfg.context.stratumNames = {'short_rank', 'middle_rank', 'long_rank'};
cfg.context.countPerFamily = numel(cfg.context.stratumNames);
cfg.algorithms = {'sqp', 'interior-point'};
cfg.protocols = [ ...
    protocol('short_low', 5, 2, -0.5); ...
    protocol('medium_mid', 10, 5, 0.0); ...
    protocol('long_high', 20, 5, 0.5)];
cfg.acceptancePolicy = 'converged_or_feasible_suboptimal';
cfg.defaultWallLimitSeconds = 60;
cfg.defaultWorkerCount = 3;
cfg.finalSolverLocked = false;
cfg.finalAcceptancePolicyLocked = false;
cfg.fullSacPairGridRetained = true;
cfg.figureGenerationEnabled = false;
end

function value = protocol(name, predictionHorizon, controlHorizon, weightAction)
value = struct('Name', name, 'N', predictionHorizon, ...
    'Nc', controlHorizon, 'WeightAction', weightAction);
end
