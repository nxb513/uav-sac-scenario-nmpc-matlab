function d1_dagger_run(cfg, teacher, lqr, cases, S, stateFile, outFile)
%D1_DAGGER_RUN DAgger (Ross, Gordon, Bagnell, AISTATS 2011, Alg. 3.1) for the linear student,
% with the expert = the SAC-tuned teacher of checkpoint S, FROZEN (a = tanh(mu)).
%   iteration 1        : teacher flies (beta_1 = 1); labels at its states
%   iteration i = 2..N : student W_i flies at alpha = 1; teacher queried at the visited states
%   after every iteration: W_{i+1} = ridge fit on ALL data so far (d1_ridge_fit), linearized
%                          stability check (d1_student_stability), validation on d1_val_set
% Then: W* = the stable candidate with the lowest validation position RMSE, confidences
% (d1_consolidate) and the deployed file outFile (stu + conf). Resumable after every flight
% through stateFile (data statistics, candidates, pending iteration, random stream).
aMean = tanh(extractdata(S.sac.mu));
[Q, R] = d1_action_to_QR(aMean, cfg); d1_set_teacher_weights(teacher, Q, R, cfg);
fprintf('DAGGER teacher frozen at SAC iter %d (a = tanh(mu)); diag(Q)=[%s]\n', S.st.iter, ...
    strtrim(sprintf('%.3g ', diag(Q))));
V = d1_val_set(cfg, cases);
if isfile(stateFile)
    L = load(stateFile); D = L.D; rng(D.rngState);
    fprintf('DAGGER resumed: iteration %d, flight %d\n', D.iter, D.c);
else
    rng(cfg.seed + cfg.daggerSeedOffset, 'twister');
    D = struct('iter', 1, 'c', 1, 'idx', [], 'stats', [], 'cand', [], 'ckptIter', S.st.iter);
end
tStart = tic; lastSave = tic; dur = []; reserve = 600;      % s kept for selection + consolidation
while D.iter <= cfg.daggerIters
    if isempty(D.idx), D.idx = randi(numel(cases), 1, cfg.daggerCases); D.c = 1; end
    if D.iter == 1, W = []; else, W = D.cand(D.iter-1).W; end
    while D.c <= cfg.daggerCases
        if ~isempty(dur) && toc(tStart) + max(dur(max(1,end-4):end)) + reserve > cfg.wallSeconds
            save_state(stateFile, D);
            fprintf('DAGGER_WALL_STOP iteration %d flight %d (resume to continue)\n', D.iter, D.c);
            return;
        end
        kase = cases(D.idx(D.c)); T = d1_case_len(kase.Xref, cfg);
        if cfg.windOn, ds = d1_sample_wind(cfg, T); else, ds = []; end
        [st, info] = d1_dagger_case(teacher, W, kase, ds, lqr, cfg);
        st.fold = mod(numel(D.stats), cfg.cvFolds) + 1;
        if isempty(D.stats), D.stats = st; else, D.stats(end+1) = st; end
        dur(end+1) = info.time; D.c = D.c + 1; %#ok<AGROW>
        fprintf(['  DAGGER_CASE it=%d %-26s %s usable=%.2f restarts=%d ' ...
            'pos=%.3f tsolve max=%.0fms p99=%.0fms t=%.0fs\n'], D.iter, kase.groupId, ...
            ternary(isempty(W), 'teacher', 'student'), info.usable, info.restarts, ...
            info.posRmse, 1e3*info.tmax, 1e3*info.tp99, info.time);
        if toc(lastSave) > cfg.checkpointEverySec, save_state(stateFile, D); lastSave = tic; end
    end
    % ---- iteration complete: fit on all data, stability, validation -------------------
    stu = d1_ridge_fit(D.stats, cfg);
    [rhoMax, ok] = d1_student_stability(stu.W, lqr, cfg);
    val = validate(stu, V, cases, lqr, cfg);
    c = struct('W', stu.W, 'lambda', stu.lambda, 'cv', stu.cv, 'rms', stu.rms, 'n', stu.n, ...
        'rho', rhoMax, 'stable', ok, 'val', val, 'iterData', D.iter);
    if isempty(D.cand), D.cand = c; else, D.cand(end+1) = c; end
    fprintf('DAGGER_ITER %d labels=%d lambda=%.0e cvMSE=%.4g rho_max=%.4f stable=%d val_pos=%.4f\n', ...
        D.iter, stu.n, stu.lambda, min(stu.cv), rhoMax, ok, val);
    D.iter = D.iter + 1; D.idx = []; D.c = 1;
    save_state(stateFile, D); lastSave = tic;
end
% ---- selection + confidences + deployed file -------------------------------------------
okc = find([D.cand.stable]);
assert(~isempty(okc), 'd1_dagger_run:nostable', 'no DAgger candidate passed the stability check');
[~, j] = min([D.cand(okc).val]); best = okc(j);
stu = D.cand(best); stu.selected = best;
stu.valAll = [D.cand.val]; stu.rhoAll = [D.cand.rho]; stu.stableAll = [D.cand.stable];
stu.ckptIter = D.ckptIter;
fprintf('DAGGER_SELECT candidate %d (fit after iteration %d) val_pos=%.4f rho_max=%.4f lambda=%.0e\n', ...
    best, D.cand(best).iterData, D.cand(best).val, D.cand(best).rho, D.cand(best).lambda);
conf = d1_consolidate(cfg, lqr, cases, stu);
conf.iter = D.ckptIter;
save(outFile, 'stu', 'conf', '-v7.3');
fprintf('DAGGER_DONE -> %s\n', outFile);
end

function val = validate(stu, V, cases, lqr, cfg)
% mean capped position RMSE of the student at alpha = 1 on the validation flights
r = zeros(1, numel(V));
for j = 1:numel(V)
    kase = cases(V(j).idx);
    R = d1_fly_student(stu, [], kase.Xref, kase.Uref, V(j).ds, cfg.plant.nominal, lqr, cfg, 'alpha1');
    r(j) = sqrt(mean(d1_track_err(R.X, R.Xr).^2));
end
val = mean(r);
end

function save_state(f, D)
D.rngState = rng;
save(f, 'D', '-v7.3');
end

function s = ternary(c, a, b); if c, s = a; else, s = b; end; end
