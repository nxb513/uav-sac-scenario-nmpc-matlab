function run_targeted_sac_parameterization_tests()
%RUN_TARGETED_SAC_PARAMETERIZATION_TESTS Check joint SAC-NMPC mapping.

projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(projectRoot, 'configs'));
addpath(fullfile(projectRoot, 'src', 'common'));
addpath(fullfile(projectRoot, 'src', 'plant'));
addpath(fullfile(projectRoot, 'src', 'controllers', 'common'));
addpath(fullfile(projectRoot, 'src', 'learning', 'surrogate'));
addpath(fullfile(projectRoot, 'src', 'rl'));

cfg = targeted_specialist_sac_config();
assert(strcmp(cfg.training.budgetUnit, 'episode'));
assert(cfg.training.candidateEpisodeCeiling == 1500);
assert(cfg.training.checkpointFrequencyEpisodes == 50);
assert(~isfield(cfg.training, 'transitionBudget'));

[lowCfg, low] = rl_nmpc_action_to_config(-ones(8, 1), cfg);
assert(lowCfg.predictionHorizon == 5);
assert(lowCfg.controlHorizon == 2);
assert(lowCfg.scenario.count == 5);
assert(low.horizonIndex == 1);

[middleCfg, middle] = rl_nmpc_action_to_config(zeros(8, 1), cfg);
assert(middleCfg.predictionHorizon == 15);
assert(middleCfg.controlHorizon == 10);
assert(middleCfg.scenario.count == 5);
assert(all(middle.groupWeights > 0));

[highCfg, ~] = rl_nmpc_action_to_config(ones(8, 1), cfg);
assert(highCfg.predictionHorizon == 20);
assert(highCfg.controlHorizon == 20);
assert(highCfg.scenario.count == 5);
assert(all(diag(highCfg.weights.Q) > 0));
assert(all(diag(highCfg.weights.R) > 0));
assert(highCfg.controlHorizon <= highCfg.predictionHorizon);
assert(cfg.reward.weights(7) == 0.0);

stateHistory = zeros(12, 4);
inputHistory = repmat(quad_hover_input(cfg.nmpc.plant.nominal), 1, 4);
referenceLookahead = zeros(12, 21);
[observation, rawFeature] = targeted_sac_make_causal_observation( ...
    stateHistory, inputHistory, referenceLookahead, zeros(12, 1), cfg);
assert(isequal(size(observation), [328, 1]));
assert(isequal(size(rawFeature), [328, 1]));
assert(all(isfinite(observation)));
assert(strcmp(cfg.observation.featureVersion, 'causal_feature_v2'));
assert(~cfg.observation.includeSolveTime);
assert(~cfg.observation.includeTeacherPrediction);
assert(strcmp(cfg.environment.positionTerminationPolicy, ...
    'nmpc_state_bounds_only'));
assert(isinf(cfg.environment.maxAbsolutePosition));

targeted_sac_validate_resume_config(cfg, cfg);
legacy = cfg;
legacy.observation.featureVersion = 'legacy_31d_v1';
legacy.observation.dimension = 31;
assert_throws(@() targeted_sac_validate_resume_config(legacy, cfg), ...
    'targeted_sac_validate_resume_config:IncompatibleObservation');
missing = rmfield(cfg, 'observation');
assert_throws(@() targeted_sac_validate_resume_config(missing, cfg), ...
    'targeted_sac_validate_resume_config:MissingMetadata');

fprintf('Targeted SAC parameterization tests passed.\n');
end

function assert_throws(operation, expectedIdentifier)
threw = false;
try
    operation();
catch exception
    threw = true;
    assert(strcmp(exception.identifier, expectedIdentifier), ...
        'Unexpected error identifier: %s.', exception.identifier);
end
assert(threw, 'Expected operation to throw an error.');
end
