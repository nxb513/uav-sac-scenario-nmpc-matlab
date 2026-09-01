function run_targeted_sac_checkpoint_stream()
%RUN_TARGETED_SAC_CHECKPOINT_STREAM Train until the external job limit.

projectRoot = fileparts(fileparts(mfilename('fullpath')));
add_project_paths(projectRoot);
cfg = targeted_specialist_sac_config();
cfg = apply_cloud_overrides(cfg);
cfg.status = 'checkpoint_stream_final_budget_not_locked';
cfg.training.activeMode = 'continuous_sync3_checkpoint_stream';
cfg.training.activeWorkerCount = cfg.training.workerCount;
cfg.environment.forceCurriculumStage = 0;
cfg.logging.enabled = true;

assert(~cfg.training.finalBudgetLocked, ...
    'The final research training budget must remain unlocked.');
assert(cfg.divergence.finalThresholdsLocked, ...
    ['The 20-step LQR-divergence thresholds must be reviewed and approved ' ...
    'before SAC can consume the candidate context bank.']);
assert(cfg.training.workerCount == 3, ...
    'The approved checkpoint stream requires three synchronous workers.');

freshStart = strcmpi(strtrim(getenv('SAC_FRESH_START')), 'true');
if freshStart
    agent = [];
    savedAgentResult = [];
    resume = struct('Type', 'fresh_200_step_agent_no_replay', ...
        'TrainingEpisode', 0, 'GlobalEpisodeOffset', 0, ...
        'SourceTrainingEpisode', 0, ...
        'SourceGlobalEpisodeOffset', 0, ...
        'SkippedNewerCheckpointCount', 0, 'SourcePath', '');
else
    resumeRoot = strtrim(getenv('SAC_RESUME_ROOT'));
    assert(~isempty(resumeRoot) && isfolder(resumeRoot), ...
        'SAC_RESUME_ROOT must point to an extracted previous artifact.');
    [agent, savedAgentResult, resume] = load_latest_resume(resumeRoot);
end
cfg.training.executionEpisodeCeiling = ...
    cfg.training.candidateEpisodeCeiling;
assert(resume.TrainingEpisode < cfg.training.executionEpisodeCeiling, ...
    'The resume checkpoint has already reached the candidate episode ceiling.');
cfg.environment.episodeIndexOffset = resume.GlobalEpisodeOffset;
cfg.resume = rmfield(resume, {'SourcePath'});

[contextBank, bankSummary] = load_compact_context_bank(projectRoot, cfg);
cfg.probe.sourceContextCount = bankSummary.SourceCount;
cfg.probe.activeContextCount = bankSummary.ActiveCount;
cfg.probe.activeFactorialCellCount = bankSummary.CellCount;

attemptId = strtrim(getenv('SAC_ATTEMPT_ID'));
if isempty(attemptId)
    attemptId = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
end
runDir = fullfile(projectRoot, cfg.outputRoot, 'checkpoint_stream_runs', ...
    sprintf('run_%s', sanitize_identifier(attemptId)));
if isfolder(runDir)
    error('run_targeted_sac_checkpoint_stream:OutputExists', ...
        'Refusing to overwrite %s.', runDir);
end
checkpointDir = fullfile(runDir, 'checkpoints');
mkdir(checkpointDir);
save(fullfile(runDir, 'config.mat'), 'cfg', 'resume', '-v7.3');
write_resume_report(runDir, resume, cfg);
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

rng(cfg.environment.baseEpisodeSeed + resume.GlobalEpisodeOffset, 'twister');
[env, observationInfo, actionInfo] = ...
    create_targeted_sac_nmpc_environment(cfg, runDir, contextBank);
clear contextBank;
if freshStart
    agent = build_agent(observationInfo, actionInfo, cfg);
    save(fullfile(runDir, 'initial_agent.mat'), 'agent', '-v7.3');
else
    validate_agent_specs(agent, observationInfo, actionInfo, cfg);
end
clear observationInfo actionInfo;

if isempty(savedAgentResult)
    options = make_training_options(cfg, checkpointDir);
else
    savedAgentResult = update_resume_options(savedAgentResult, cfg, ...
        checkpointDir);
end

startedAt = datetime('now', 'TimeZone', 'local');
write_text(fullfile(runDir, 'RUNNING.txt'), sprintf( ...
    ['started_at=%s\nmode=continuous_sync3_checkpoint_stream\n' ...
    'worker_count=3\ncheckpoint_frequency_episodes=%d\n' ...
    'budget_unit=episode\ncandidate_episode_ceiling=%d\n' ...
    'research_budget_locked=false\nexecution_stop=330_minute_train_step\n' ...
    'episode_steps=%d\nfresh_start=%d\n' ...
    'resume_type=%s\nresume_training_episode=%d\n' ...
    'global_episode_offset=%d\nresume_source=%s\n'], ...
    char(startedAt), cfg.training.checkpointFrequencyEpisodes, ...
    cfg.training.candidateEpisodeCeiling, ...
    cfg.environment.stepsPerEpisode, freshStart, ...
    resume.Type, resume.TrainingEpisode, ...
    resume.GlobalEpisodeOffset, resume.SourcePath));

clock = tic;
try
    if isempty(savedAgentResult)
        trainingStats = train(agent, env, options);
    else
        trainingStats = train(agent, env, savedAgentResult);
    end
    trainingSeconds = toc(clock);
    save(fullfile(runDir, 'training_stats.mat'), 'trainingStats', '-v7.3');
    save(fullfile(runDir, 'final_agent.mat'), 'agent', '-v7.3');
    write_text(fullfile(runDir, 'COMPLETED.txt'), sprintf( ...
        ['status=completed_naturally\ntraining_seconds=%.6f\n' ...
        'last_training_episode=%d\n'], trainingSeconds, ...
        double(trainingStats.EpisodeIndex(end))));
    delete_if_present(fullfile(runDir, 'RUNNING.txt'));
catch exception
    write_text(fullfile(runDir, 'FAILED.txt'), sprintf( ...
        'identifier=%s\nmessage=%s\nelapsed_seconds=%.6f\n', ...
        exception.identifier, strrep(exception.message, newline, ' '), ...
        toc(clock)));
    delete_if_present(fullfile(runDir, 'RUNNING.txt'));
    rethrow(exception);
end
clear diaryCleanup;
end

function [bank, summary] = load_compact_context_bank(projectRoot, cfg)
bankPath = fullfile(projectRoot, cfg.trainingBankPath);
artifact = load(bankPath, 'trainContextBank', ...
    'trainContextManifest', 'buildMetadata');
bank = artifact.trainContextBank;
cellIds = unique(string(artifact.trainContextManifest.CellId), 'stable');
assert(cfg.probe.contextsPerFactorialCell == 1, ...
    'The memory-bounded cloud run expects one context per factorial cell.');
assert(numel(bank) == numel(cellIds), ...
    'The compact bank must contain one context per factorial cell.');
assert(all([bank.PreDivergenceEligible]), ...
    'SAC bank contains a context without a valid pre-divergence LQR event.');
sourceCount = numel(bank);
if isfield(artifact, 'buildMetadata') && ...
        isfield(artifact.buildMetadata, 'sourceContextCount')
    sourceCount = artifact.buildMetadata.sourceContextCount;
end
summary = struct('SourceCount', sourceCount, ...
    'ActiveCount', numel(bank), 'CellCount', numel(cellIds));
end

function [agent, savedResult, resume] = load_latest_resume(root)
checkpointFiles = dir(fullfile(root, '**', 'Agent*.mat'));
episodes = -inf(numel(checkpointFiles), 1);
for index = 1:numel(checkpointFiles)
    token = regexp(checkpointFiles(index).name, ...
        '^Agent([0-9]+)\.mat$', 'tokens', 'once');
    if ~isempty(token)
        episodes(index) = str2double(token{1});
    end
end
[sortedEpisodes, order] = sort(episodes, 'descend');
for candidate = 1:numel(order)
    latestEpisode = sortedEpisodes(candidate);
    if ~isfinite(latestEpisode)
        continue;
    end
    latestIndex = order(candidate);
    sourcePath = fullfile(checkpointFiles(latestIndex).folder, ...
        checkpointFiles(latestIndex).name);
    try
        artifact = load(sourcePath, 'saved_agent', 'savedAgentResult');
    catch
        continue;
    end
    if ~isfield(artifact, 'saved_agent') || ...
            ~isfield(artifact, 'savedAgentResult')
        continue;
    end
    agent = artifact.saved_agent;
    savedResult = artifact.savedAgentResult;
    [sourceOffset, sourceTrainingEpisode] = source_offsets(sourcePath);
    globalOffset = sourceOffset + max(0, ...
        latestEpisode - sourceTrainingEpisode);
    resume = struct('Type', 'periodic_exact_resume', ...
        'TrainingEpisode', latestEpisode, ...
        'GlobalEpisodeOffset', globalOffset, ...
        'SourceTrainingEpisode', sourceTrainingEpisode, ...
        'SourceGlobalEpisodeOffset', sourceOffset, ...
        'SkippedNewerCheckpointCount', candidate - 1, ...
        'SourcePath', sourcePath);
    return;
end

finalFiles = dir(fullfile(root, '**', 'final_agent.mat'));
assert(~isempty(finalFiles), ...
    'No AgentK.mat or final_agent.mat was found under %s.', root);
[~, latestIndex] = max([finalFiles.datenum]);
sourcePath = fullfile(finalFiles(latestIndex).folder, ...
    finalFiles(latestIndex).name);
artifact = load(sourcePath, 'agent');
assert(isfield(artifact, 'agent'), ...
    'Final checkpoint %s does not contain variable agent.', sourcePath);
agent = artifact.agent;
savedResult = [];
globalOffset = completed_episode_count(fileparts(sourcePath));
resume = struct('Type', 'final_agent_new_training_result', ...
    'TrainingEpisode', 0, 'GlobalEpisodeOffset', globalOffset, ...
    'SourceTrainingEpisode', 0, ...
    'SourceGlobalEpisodeOffset', globalOffset, ...
    'SkippedNewerCheckpointCount', 0, ...
    'SourcePath', sourcePath);
end

function [globalOffset, trainingEpisode] = source_offsets(checkpointPath)
runDir = fileparts(fileparts(checkpointPath));
configPath = fullfile(runDir, 'config.mat');
globalOffset = 0;
trainingEpisode = 0;
if ~isfile(configPath)
    return;
end
artifact = load(configPath, 'cfg');
if ~isfield(artifact, 'cfg')
    return;
end
if isfield(artifact.cfg.environment, 'episodeIndexOffset')
    globalOffset = double(artifact.cfg.environment.episodeIndexOffset);
end
if isfield(artifact.cfg, 'resume') && ...
        isfield(artifact.cfg.resume, 'TrainingEpisode')
    trainingEpisode = double(artifact.cfg.resume.TrainingEpisode);
end
end

function count = completed_episode_count(runDir)
count = 0;
statsPath = fullfile(runDir, 'training_stats.mat');
if ~isfile(statsPath)
    return;
end
artifact = load(statsPath, 'trainingStats');
if isfield(artifact, 'trainingStats') && ...
        ~isempty(artifact.trainingStats.EpisodeIndex)
    count = double(artifact.trainingStats.EpisodeIndex(end));
end
end

function options = make_training_options(cfg, checkpointDir)
options = rlTrainingOptions( ...
    'MaxEpisodes', cfg.training.executionEpisodeCeiling, ...
    'MaxStepsPerEpisode', cfg.training.maxStepsPerEpisode, ...
    'ScoreAveragingWindowLength', ...
    cfg.training.scoreAveragingWindowLength, ...
    'StopTrainingCriteria', 'None', ...
    'SaveAgentCriteria', 'EpisodeFrequency', ...
    'SaveAgentValue', cfg.training.checkpointFrequencyEpisodes, ...
    'SaveAgentDirectory', checkpointDir, ...
    'SimulationStorageType', 'none', ...
    'StopOnError', 'on', 'Verbose', true, 'Plots', 'none');
options.UseParallel = true;
options.ParallelizationOptions.Mode = 'sync';
options.ParallelizationOptions.WorkerRandomSeeds = ...
    cfg.training.workerRandomSeeds(1:cfg.training.workerCount);
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

function savedResult = update_resume_options(savedResult, cfg, checkpointDir)
savedResult.TrainingOptions.MaxEpisodes = ...
    cfg.training.executionEpisodeCeiling;
savedResult.TrainingOptions.MaxStepsPerEpisode = ...
    cfg.training.maxStepsPerEpisode;
savedResult.TrainingOptions.ScoreAveragingWindowLength = ...
    cfg.training.scoreAveragingWindowLength;
savedResult.TrainingOptions.StopTrainingCriteria = 'None';
savedResult.TrainingOptions.SaveAgentCriteria = 'EpisodeFrequency';
savedResult.TrainingOptions.SaveAgentValue = ...
    cfg.training.checkpointFrequencyEpisodes;
savedResult.TrainingOptions.SaveAgentDirectory = checkpointDir;
savedResult.TrainingOptions.SimulationStorageType = 'none';
savedResult.TrainingOptions.StopOnError = 'on';
savedResult.TrainingOptions.Verbose = true;
savedResult.TrainingOptions.Plots = 'none';
savedResult.TrainingOptions.UseParallel = true;
savedResult.TrainingOptions.ParallelizationOptions.Mode = 'sync';
savedResult.TrainingOptions.ParallelizationOptions.WorkerRandomSeeds = ...
    cfg.training.workerRandomSeeds(1:cfg.training.workerCount);
end

function validate_agent_specs(agent, observationInfo, actionInfo, cfg)
assert(prod(observationInfo.Dimension) == cfg.observation.dimension, ...
    'Observation dimension changed from the saved agent configuration.');
assert(prod(actionInfo.Dimension) == cfg.action.dimension, ...
    'Action dimension changed from the saved agent configuration.');
assert(isa(agent, 'rl.agent.rlSACAgent') || contains(class(agent), 'SAC'), ...
    'The resume checkpoint is not a SAC agent.');
end

function cfg = apply_cloud_overrides(cfg)
rawLimit = strtrim(getenv('NMPC_MAX_WALL_SECONDS'));
if ~isempty(rawLimit)
    maxWallSeconds = str2double(rawLimit);
    assert(isfinite(maxWallSeconds) && maxWallSeconds > 0, ...
        'NMPC_MAX_WALL_SECONDS must be a positive finite number.');
    cfg.nmpc.solver.maxWallSeconds = maxWallSeconds;
end
cfg.probe.solverWallLimitSource = ...
    'github_execution_guard_candidate_not_final_hyperparameter';
end

function write_resume_report(runDir, resume, cfg)
fileId = fopen(fullfile(runDir, 'resume_provenance.md'), 'w');
assert(fileId >= 0, 'Cannot open resume provenance report.');
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, '# SAC checkpoint stream provenance\n\n');
fprintf(fileId, '- Resume type: `%s`.\n', resume.Type);
fprintf(fileId, '- Source checkpoint: `%s`.\n', resume.SourcePath);
fprintf(fileId, '- Saved training episode: `%d`.\n', ...
    resume.TrainingEpisode);
fprintf(fileId, '- Unreadable newer checkpoints skipped: `%d`.\n', ...
    resume.SkippedNewerCheckpointCount);
fprintf(fileId, '- Environment episode/seed offset: `%d`.\n', ...
    resume.GlobalEpisodeOffset);
fprintf(fileId, '- Checkpoint frequency: every `%d` episodes.\n', ...
    cfg.training.checkpointFrequencyEpisodes);
fprintf(fileId, ['- Execution stopping rule: external GitHub step limit; ' ...
    'there is no approved final SAC budget in this run.\n']);
fprintf(fileId, ['- Recovery rule: load the numerically largest valid ' ...
    '`AgentK.mat`; episodes after `K` are intentionally discarded.\n']);
clear cleanup;
end

function value = sanitize_identifier(value)
value = regexprep(char(value), '[^A-Za-z0-9_-]', '_');
end

function write_text(path, content)
fileId = fopen(path, 'w');
assert(fileId >= 0, 'Cannot open %s.', path);
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, '%s', content);
clear cleanup;
end

function delete_if_present(path)
if isfile(path)
    delete(path);
end
end

function add_project_paths(projectRoot)
addpath(fullfile(projectRoot, 'configs'));
addpath(fullfile(projectRoot, 'src', 'common'));
addpath(fullfile(projectRoot, 'src', 'plant'));
addpath(fullfile(projectRoot, 'src', 'controllers', 'common'));
addpath(fullfile(projectRoot, 'src', 'controllers', 'scenario_nmpc'));
addpath(fullfile(projectRoot, 'src', 'rl'));
end
