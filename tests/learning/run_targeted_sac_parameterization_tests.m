function run_targeted_sac_parameterization_tests()
%RUN_TARGETED_SAC_PARAMETERIZATION_TESTS Check joint SAC-NMPC mapping.

projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(projectRoot, 'configs'));
addpath(fullfile(projectRoot, 'src', 'common'));
addpath(fullfile(projectRoot, 'src', 'plant'));
addpath(fullfile(projectRoot, 'src', 'controllers', 'common'));
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

state = zeros(12, 1);
reference = zeros(12, 1);
previousInput = quad_hover_input(cfg.nmpc.plant.nominal);
observation = rl_nmpc_make_observation(state, reference, ...
    previousInput, zeros(12, 1), 0.1, middle.horizon, cfg, ...
    middle.controlHorizon);
assert(isequal(size(observation), [31, 1]));
assert(all(isfinite(observation)));

fprintf('Targeted SAC parameterization tests passed.\n');
end
