function V = d1_val_set(cfg, cases)
%D1_VAL_SET Validation flights for choosing among the DAgger students: one reference per
% (family x acceleration level) cell of the training bank (the case with the median speed
% of the cell) = 15 flights, each with a training-law wind drawn from a DEDICATED stream
% RandStream(seed + valSeedOffset), so these wind realizations are not DAgger data.
rs = RandStream('twister', 'Seed', cfg.seed + cfg.valSeedOffset);
cells = struct('key', {}, 'idx', {}, 'v', {});
for i = 1:numel(cases)
    tk = regexp(cases(i).groupId, '^([^|]+)\|v([\d.]+)\|a([\d.]+)', 'tokens', 'once');
    key = [tk{1} '|' tk{3}];
    j = find(strcmp({cells.key}, key), 1);
    if isempty(j), cells(end+1) = struct('key', key, 'idx', i, 'v', str2double(tk{2})); %#ok<AGROW>
    else, cells(j).idx(end+1) = i; cells(j).v(end+1) = str2double(tk{2}); end
end
V = struct('idx', {}, 'ds', {}, 'id', {});
for j = 1:numel(cells)
    [~, o] = sort(cells(j).v); pick = cells(j).idx(o(ceil(numel(o)/2)));
    T = d1_case_len(cases(pick).Xref, cfg);
    V(end+1) = struct('idx', pick, 'ds', d1_sample_wind(cfg, T, rs), 'id', cases(pick).groupId); %#ok<AGROW>
end
end
