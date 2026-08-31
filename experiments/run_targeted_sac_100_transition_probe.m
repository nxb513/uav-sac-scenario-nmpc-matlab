function run_targeted_sac_100_transition_probe()
%RUN_TARGETED_SAC_100_TRANSITION_PROBE Run one continuous sync-3 SAC probe.

projectRoot = fileparts(fileparts(mfilename('fullpath')));
add_project_paths(projectRoot);
cfg = targeted_specialist_sac_config();
cfg = apply_cloud_pilot_overrides(cfg);
assert(cfg.probe.transitionCount == 100, ...
    'This probe is approved for a 100-transition global target.');
assert(~cfg.training.finalBudgetLocked, ...
    'The final training budget must remain unlocked during this probe.');
assert(cfg.training.workerCount == 3, ...
    'The approved probe requires three synchronous workers.');

bankPath = fullfile(projectRoot, cfg.trainingBankPath);
manifestArtifact = load(bankPath, 'trainContextManifest');
[cellIds, firstCellIndices] = unique( ...
    string(manifestArtifact.trainContextManifest.CellId), 'stable');
bankFile = matfile(bankPath);
bankInfo = whos(bankFile, 'trainContextBank');
probeContextBank = bankFile.trainContextBank(firstCellIndices, 1);
assert(cfg.probe.contextsPerFactorialCell == 1, ...
    'The memory-bounded probe expects one context per factorial cell.');
assert(numel(probeContextBank) == numel(cellIds), ...
    'The compact probe bank must contain one context per cell.');

cfg.probe.sourceContextCount = prod(bankInfo.size);
cfg.probe.activeContextCount = numel(probeContextBank);
cfg.probe.activeFactorialCellCount = numel(cellIds);
cfg.training.activeMode = 'continuous_sync3';
cfg.training.activeWorkerCount = cfg.training.workerCount;
cfg.environment.forceCurriculumStage = cfg.probe.forceCurriculumStage;
cfg.logging.enabled = true;
clear manifestArtifact bankFile bankInfo cellIds firstCellIndices;

runDir = fullfile(projectRoot, cfg.outputRoot, cfg.probe.parallelRunId);
if isfolder(runDir)
    error('run_targeted_sac_100_transition_probe:OutputExists', ...
        'Refusing to overwrite %s.', runDir);
end
mkdir(runDir);
save(fullfile(runDir, 'config.mat'), 'cfg', '-v7.3');
diary(fullfile(runDir, 'training_diary.log'));
diaryCleanup = onCleanup(@() diary('off'));

pool = gcp('nocreate');
if ~isempty(pool)
    delete(pool);
end
cluster = parcluster('Processes');
if cluster.NumWorkers < cfg.training.workerCount
    cluster.NumWorkers = cfg.training.workerCount;
end
parpool(cluster, cfg.training.workerCount);

rng(cfg.environment.baseEpisodeSeed, 'twister');
[env, observationInfo, actionInfo] = ...
    create_targeted_sac_nmpc_environment(cfg, runDir, probeContextBank);
clear probeContextBank;
agent = build_agent(observationInfo, actionInfo, cfg);
clear observationInfo actionInfo;
save(fullfile(runDir, 'initial_agent.mat'), 'agent', '-v7.3');

options = rlTrainingOptions( ...
    'MaxEpisodes', cfg.training.maxEpisodes, ...
    'MaxStepsPerEpisode', cfg.training.maxStepsPerEpisode, ...
    'ScoreAveragingWindowLength', ...
    cfg.training.scoreAveragingWindowLength, ...
    'StopTrainingCriteria', 'GlobalStepCount', ...
    'StopTrainingValue', cfg.probe.transitionCount, ...
    'SaveAgentCriteria', 'None', ...
    'SimulationStorageType', 'none', ...
    'StopOnError', 'on', 'Verbose', true, 'Plots', 'none');
options.UseParallel = true;
options.ParallelizationOptions.Mode = 'sync';
options.ParallelizationOptions.WorkerRandomSeeds = ...
    cfg.training.workerRandomSeeds(1:cfg.training.workerCount);

startedAt = datetime('now', 'TimeZone', 'local');
write_text(fullfile(runDir, 'RUNNING.txt'), sprintf( ...
    ['started_at=%s\nrequested_global_transitions=%d\n' ...
    'mode=continuous_sync3\nworker_count=3\n' ...
    'single_central_agent=true\nsingle_replay_buffer=true\n' ...
    'final_budget_locked=false\n'], char(startedAt), ...
    cfg.probe.transitionCount));
clock = tic;
try
    trainingStats = train(agent, env, options);
    trainingSeconds = toc(clock);
    actualTransitions = double(trainingStats.TotalAgentSteps(end));
    episodeCount = double(trainingStats.EpisodeIndex(end));
    assert(actualTransitions >= cfg.probe.transitionCount, ...
        'Probe stopped before the requested global transition target.');
    finishedAt = datetime('now', 'TimeZone', 'local');
    transitionsPerSecond = actualTransitions / trainingSeconds;
    secondsPerTransition = trainingSeconds / actualTransitions;
    transitionsPerHour = 3600 * transitionsPerSecond;
    save(fullfile(runDir, 'training_stats.mat'), ...
        'trainingStats', '-v7.3');
    save(fullfile(runDir, 'final_agent.mat'), 'agent', '-v7.3');
    runtime = table(cfg.probe.transitionCount, actualTransitions, ...
        episodeCount, trainingSeconds, secondsPerTransition, ...
        transitionsPerSecond, transitionsPerHour, ...
        string(startedAt, 'yyyy-MM-dd HH:mm:ss.SSSXXX'), ...
        string(finishedAt, 'yyyy-MM-dd HH:mm:ss.SSSXXX'), ...
        cfg.agent.numWarmStartSteps, cfg.agent.miniBatchSize, ...
        cfg.nmpc.solver.maxWallSeconds, ...
        'VariableNames', {'RequestedTransitions', 'ActualTransitions', ...
        'EpisodeCount', 'TrainingSeconds', 'SecondsPerTransition', ...
        'TransitionsPerSecond', 'TransitionsPerHour', 'StartedAt', ...
        'FinishedAt', 'NumWarmStartSteps', 'MiniBatchSize', ...
        'SolverMaxWallSeconds'});
    writetable(runtime, fullfile(runDir, 'runtime.csv'));
    write_report(runDir, cfg, runtime);
    write_text(fullfile(runDir, 'COMPLETED.txt'), sprintf( ...
        ['status=completed\nrequested_global_transitions=%d\n' ...
        'actual_global_transitions=%d\nepisodes=%d\n' ...
        'training_seconds=%.6f\nseconds_per_transition=%.9f\n' ...
        'transitions_per_hour=%.6f\nnetwork_updates_expected=false\n'], ...
        cfg.probe.transitionCount, actualTransitions, episodeCount, ...
        trainingSeconds, secondsPerTransition, transitionsPerHour));
    delete(fullfile(runDir, 'RUNNING.txt'));
catch exception
    write_text(fullfile(runDir, 'FAILED.txt'), sprintf( ...
        'identifier=%s\nmessage=%s\nelapsed_seconds=%.6f\n', ...
        exception.identifier, strrep(exception.message, newline, ' '), ...
        toc(clock)));
    if isfile(fullfile(runDir, 'RUNNING.txt'))
        delete(fullfile(runDir, 'RUNNING.txt'));
    end
    rethrow(exception);
end
clear diaryCleanup;
end

function agent = build_agent(observationInfo, actionInfo, cfg)
options = rlSACAgentOptions;
options.SampleTime = cfg.environment.sampleTime;
options.DiscountFactor = cfg.agent.discountFactor;
options.ExperienceBufferLength = cfg.agent.experienceBufferLength;
options.MiniBatchSize = cfg.agent.miniBatchSize;
options.NumWarmStartSteps = cfg.agent.numWarmStartSteps;
options.TargetSmoothFactor = cfg.agent.targetSmoothFactor;
options.LearningFrequency = 1;
options.NumEpoch = 1;
options.MaxMiniBatchPerEpoch = 1;
options.ActorOptimizerOptions.LearnRate = cfg.agent.actorLearnRate;
for index = 1:numel(options.CriticOptimizerOptions)
    options.CriticOptimizerOptions(index).LearnRate = ...
        cfg.agent.criticLearnRate;
end
options.EntropyWeightOptions.LearnRate = cfg.agent.entropyLearnRate;
agent = rlSACAgent(observationInfo, actionInfo, options);
agent.AgentOptions.InfoToSave.ExperienceBuffer = true;
agent.AgentOptions.InfoToSave.Optimizer = true;
agent.AgentOptions.InfoToSave.PolicyState = true;
agent.AgentOptions.InfoToSave.Target = true;
end

function write_report(runDir, cfg, runtime)
fileId = fopen(fullfile(runDir, 'summary.md'), 'w');
assert(fileId >= 0, 'Cannot open probe summary.');
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, '# Continuous SAC 100-transition sync-3 probe\n\n');
fprintf(fileId, '- Requested global transitions: `%d`.\n', ...
    runtime.RequestedTransitions);
fprintf(fileId, ['- Actual global transitions: `%d` (sync workers may ' ...
    'finish a small batch together).\n'], runtime.ActualTransitions);
fprintf(fileId, '- Episodes: `%d`.\n', runtime.EpisodeCount);
fprintf(fileId, '- Training wall time: `%.6f s`.\n', ...
    runtime.TrainingSeconds);
fprintf(fileId, '- Seconds/transition: `%.9f`.\n', ...
    runtime.SecondsPerTransition);
fprintf(fileId, '- Transitions/hour: `%.6f`.\n', ...
    runtime.TransitionsPerHour);
fprintf(fileId, '- Mode/workers: `continuous synchronous/3`.\n');
fprintf(fileId, '- Central agent/replay buffers: `1/1`.\n');
fprintf(fileId, '- Grid: 14 `(N,Nc)` pairs; fixed `M=5`.\n');
fprintf(fileId, ['- Probe context bank: `%d/%d` contexts, one selected ' ...
    'context from each of `%d` balanced factorial cells.\n'], ...
    cfg.probe.activeContextCount, cfg.probe.sourceContextCount, ...
    cfg.probe.activeFactorialCellCount);
fprintf(fileId, '- Final transition budget locked: `false`.\n');
fprintf(fileId, ['- Per-solve wall-time guard: `%.3f s` ' ...
    '(GitHub pilot safety setting, not a final research hyperparameter).\n'], ...
    runtime.SolverMaxWallSeconds);
fprintf(fileId, ['- Replay warm-start/minibatch: `%d/%d`; therefore this ' ...
    '100-transition probe measures collection before gradient updates.\n'], ...
    cfg.agent.numWarmStartSteps, cfg.agent.miniBatchSize);
fprintf(fileId, ['\nEvery counted transition executes an action from the same ' ...
    'central SAC policy, one scenario-NMPC solve, one plant step, reward ' ...
    'calculation and insertion into the same central replay stream.\n']);
clear cleanup;
end

function cfg = apply_cloud_pilot_overrides(cfg)
rawLimit = strtrim(getenv('NMPC_MAX_WALL_SECONDS'));
if isempty(rawLimit)
    return;
end
maxWallSeconds = str2double(rawLimit);
assert(isfinite(maxWallSeconds) && maxWallSeconds > 0, ...
    'NMPC_MAX_WALL_SECONDS must be a positive finite number.');
cfg.nmpc.solver.maxWallSeconds = maxWallSeconds;
cfg.probe.solverWallLimitSource = 'environment_candidate_not_final';
end

function write_text(path, content)
fileId = fopen(path, 'w');
assert(fileId >= 0, 'Cannot open %s.', path);
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, '%s', content);
clear cleanup;
end

function add_project_paths(projectRoot)
addpath(fullfile(projectRoot, 'configs'));
addpath(fullfile(projectRoot, 'src', 'plant'));
addpath(fullfile(projectRoot, 'src', 'controllers', 'common'));
addpath(fullfile(projectRoot, 'src', 'controllers', 'scenario_nmpc'));
addpath(fullfile(projectRoot, 'src', 'rl'));
end
