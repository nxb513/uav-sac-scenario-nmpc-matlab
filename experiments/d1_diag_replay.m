function d1_diag_replay()
%D1_DIAG_REPLAY DIAGNOSTIC only (not part of the method). Replays the pending SAC iteration
% of checkpoint_seed<s>.mat exactly as paired_rollout flies the teacher (same Q,R of the
% pending action, same cases in the same order, same winds from the restored RNG, same
% teacher_step / reset / restart rules) and reports every slow solve, so a multi-hour case
% can be located and characterized:
%   SLOW lines : solves slower than 0.5 s (status, SQP iterations, QP iterations per SQP
%                iteration from acados 'stat', state norms)
%   HB lines   : heartbeat every 100 steps
% D1_FTZ=1 first sets flush-to-zero / denormals-are-zero in the MATLAB thread (MEX
% tools/ci/d1_set_ftz) to test whether subnormal arithmetic causes the slowness.
cfg = d1_config(); rng(cfg.seed, 'twister');
scen = d1_sample_scenarios(cfg); cases = d1_train_cases(cfg);
S = load(fullfile(cfg.runDir, sprintf('checkpoint_seed%d.mat', cfg.seed))); st = S.st;
assert(isfield(st, 'pend') && ~isempty(st.pend), 'checkpoint has no pending iteration');
ftz = strcmp(d1_getenv_str('D1_FTZ', '0'), '1');
if ftz, before = d1_set_ftz(1); else, before = d1_set_ftz(-1); end
fprintf('REPLAY seed=%d iter=%d pending case %d of %d | FTZ=%d MXCSR before=%s now=%s\n', ...
    cfg.seed, st.iter, st.pend.c, cfg.casesPerEval, ftz, dec2hex(before), dec2hex(d1_set_ftz(-1)));
teacher = d1_teacher_build_solver(cfg, scen);
[Q, R] = d1_action_to_QR(st.pend.a, cfg); d1_set_teacher_weights(teacher, Q, R, cfg);
rng(S.rngState);
Ts = cfg.Ts; theta = cfg.plant.nominal; uh = cfg.uh;
for c = st.pend.c:cfg.casesPerEval
    kase = cases(st.pend.idx(c)); Xref = kase.Xref; T = d1_case_len(Xref, cfg);
    if cfg.windOn, ds = d1_sample_wind(cfg, T); else, ds = []; end   % same draws as paired_rollout
    xN = Xref(:,1); uprev = uh; d1_teacher_reset(teacher, Xref, 1, cfg);
    tCase = tic; tsol = zeros(1,T); nSlow = 0; nUnu = 0; nDiv = 0;
    fprintf('CASE_START %d %s MXCSR=%s\n', c, kase.groupId, dec2hex(d1_set_ftz(-1)));
    for k = 1:T
        t = (k-1)*Ts;
        F = d1_wind_now(ds, t, xN, uprev, theta);
        [uN, status, usable, tsol(k)] = d1_teacher_step(teacher, xN, uprev, Xref, k, F, cfg);
        nUnu = nUnu + ~usable;
        if tsol(k) > 0.5
            nSlow = nSlow + 1;
            try it = teacher.get('sqp_iter'); catch, it = NaN; end
            try sm = teacher.get('stat'); qpit = sm(:, min(7, size(sm,2))).'; catch, qpit = NaN; end
            fprintf(['SLOW case %d k=%d tsolve=%.2fs status=%d usable=%d sqp_iter=%d |e_pos|=%.3g ' ...
                '|eta|=%.3g |v|=%.3g |omega|=%.3g |F|=%.3g qp_iter/sqp=[%s]\n'], c, k, tsol(k), status, ...
                usable, it, norm(xN(1:3)-Xref(1:3,k)), norm(xN(4:6)), norm(xN(7:9)), norm(xN(10:12)), ...
                norm(F), strtrim(sprintf('%g ', qpit)));
        end
        if mod(k, 100) == 0
            fprintf('  HB case %d k=%d elapsed=%.0fs max_tsolve=%.2fs slow=%d unusable=%d\n', ...
                c, k, toc(tCase), max(tsol(1:k)), nSlow, nUnu);
        end
        uN = d1_sat(uN, cfg); uprev = uN;
        [xNnext, divN] = d1_plant_step(t, xN, uN, Ts, theta, ds);
        if divN
            nDiv = nDiv + 1; xN = Xref(:,k+1); uprev = uh; d1_teacher_reset(teacher, Xref, k+1, cfg);
        else
            xN = xNnext;
        end
    end
    fprintf('CASE_END %d %s t=%.0fs max_tsolve=%.2fs p99=%.3fs slow=%d unusable=%d restarts=%d\n', ...
        c, kase.groupId, toc(tCase), max(tsol), prctile(tsol, 99), nSlow, nUnu, nDiv);
end
fprintf('REPLAY_DONE\n');
end
