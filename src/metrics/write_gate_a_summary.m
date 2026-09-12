function write_gate_a_summary(filePath, run, estimate, cfg)
%WRITE_GATE_A_SUMMARY Write a compact Markdown runtime and dataset report.

[fileId, message] = fopen(filePath, 'w');
if fileId < 0
    error('write_gate_a_summary:OpenFailed', 'Cannot open summary: %s', message);
end
cleanup = onCleanup(@() fclose(fileId)); %#ok<NASGU>

fprintf(fileId, '# Gate A runtime and dataset estimate\n\n');
fprintf(fileId, 'Run ID: `%s`.\n\n', run.runId);
fprintf(fileId, 'Generated: `%s`.\n\n', run.generatedAt);
fprintf(fileId, 'MATLAB: `%s`.\n\n', run.matlabVersion);
fprintf(fileId, 'Measured wall time: `%.2f s`.\n\n', run.wallTimeSeconds);

fprintf(fileId, '## Scenario NMPC measured runtime\n\n');
fprintf(fileId, '| N | measured solves | median s/step | P95 s/step |\n');
fprintf(fileId, '| ---: | ---: | ---: | ---: |\n');
for i = 1:height(estimate.runtimeByHorizon)
    row = estimate.runtimeByHorizon(i, :);
    fprintf(fileId, '| %d | %d | %.4f | %.4f |\n', ...
            row.Horizon, row.MeasuredSolves, row.MedianSeconds, row.P95Seconds);
end

fprintf(fileId, '\n## SAC online experience estimate\n\n');
fprintf(fileId, 'SAC generates these transitions online; this is not a static dataset.\n\n');
fprintf(fileId, '| stage | episodes | steps/episode | transitions | low h | base h | high h |\n');
fprintf(fileId, '| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n');
for i = 1:height(estimate.rl)
    row = estimate.rl(i, :);
    fprintf(fileId, '| %s | %d | %d | %d | %.2f | %.2f | %.2f |\n', ...
            row.Stage{1}, row.Episodes, row.StepsPerEpisode, row.Transitions, ...
            row.LowHours, row.BaseHours, row.HighHours);
end
fprintf(fileId, '\nRaw single-precision replay payload for the full candidate: approximately `%.1f MB`.\n', ...
        estimate.rlReplaySingleMegabytes);

fprintf(fileId, '\n## Provisional teacher dataset for step 4\n\n');
fprintf(fileId, '| split | episodes | NMPC solves | usable samples | low h | base h | high h |\n');
fprintf(fileId, '| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n');
for i = 1:height(estimate.dataset)
    row = estimate.dataset(i, :);
    fprintf(fileId, '| %s | %d | %d | %d | %.2f | %.2f | %.2f |\n', ...
            row.Split{1}, row.Episodes, row.GeneratedSteps, row.Samples, ...
            row.LowHours, row.BaseHours, row.HighHours);
end
fprintf(fileId, '\nProvisional feature dimension: `%d`; target dimension: `%d`.\n\n', ...
        estimate.datasetFeatureDimension, estimate.datasetTargetDimension);
fprintf(fileId, 'Estimated dataset storage with 15%% overhead: `%.1f MB` single or `%.1f MB` double.\n\n', ...
        estimate.datasetSingleMegabytes, estimate.datasetDoubleMegabytes);

fprintf(fileId, '## Assumptions and limits\n\n');
fprintf(fileId, '1. Low uses the fastest measured scenario median.\n');
fprintf(fileId, '2. Base uses the measured median at `N=%d`.\n', estimate.baseHorizon);
fprintf(fileId, '3. High uses the largest measured scenario P95.\n');
fprintf(fileId, '4. Times exclude SAC network updates, checkpoint I/O and later evaluation.\n');
fprintf(fileId, '5. The step-4 dataset budget is provisional until papers 4_1 to 4_3 are read.\n');
fprintf(fileId, '6. Dataset split is by episode: `%d/%d/%d`; no sample-level mixing.\n', ...
        cfg.workload.dataset.episodeCounts(1), cfg.workload.dataset.episodeCounts(2), ...
        cfg.workload.dataset.episodeCounts(3));
end
