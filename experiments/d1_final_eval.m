function d1_final_eval()
%D1_FINAL_EVAL Final comparison under REAL wind, train / OOD conditions, 5 families.
%
% Controllers (D1_CTRLS, comma list). Every controller knows only the NOMINAL model;
% the true plant is a repo LHS sample it does not know.
%   LQR      u = sat(uh - K e)                                  (Bryson, nominal)
%   LQI      u = sat(uh - Kx e - KI z), z = int(p - p_ref) dt   (joint dlqr, nominal,
%            clamping anti-windup: no integration on saturated steps)
%   MPC      standard linear MPC, N = D1_MPC_N (5): hover-linear model, Bryson Q,R,
%            terminal DARE P, input box constraints, KWIK active-set QP (warm start)
%   Teacher  SAC-NMPC teacher of the chain (ORACLE): scenario NMPC (acados, M=5 scenarios
%            from the chain seed, N=20, Nc=5, D1_SOLVER) with Q,R = d1_action_to_QR(mu),
%            told the exact current wind force (privileged, not deployable) -- d1_teacher_step
%   P        proposed controller (deployed): d1_blend_control on the LQR base with the
%            chain's DAgger student Du = W*phi and its confidences (student file); it does
%            NOT know the wind (it uses the force estimate F_hat of d1_student_feature)
% Every flight uses the common flight rules of the training pipeline (src/joint/d1_*):
% real time, d1_case_len steps, a divergence restarts the plant on the reference (controller
% state reset) and is counted; per-step position error capped at 5 m (d1_track_err), a
% diverged step counts as 5 m; completed = no restart and never at the cap.
%
% Conditions (D1_COND):
%   train : train plants (LHS 5) x 15 ID references (5 families x {v4 a2, v8 a5, v12 a9})
%   ood   : OOD plants (LHS 5) x 10 OOD references (5 families x {14, 16 m/s}, a = 9)
% Each (reference, plant) pair is flown with 2 real-wind series, cycled over all series
% in D1_WIND_DIR (tools/wind/prepare_wind_series.py): train 150, ood 100 flights.
% The chain (seed, random QR base) comes from D1_SEED / D1_RANDOM_QR exactly as in
% training; D1_CKPT is its SAC checkpoint (teacher) and
% D1_STUDENT its student file student_seed<s><suffix>.mat (required for P).
%
% Outputs (D1_OUT): final_eval_<cond>_<tag>_<k>of<K>.csv (one row per flight x ctrl)
% and final_traj_<cond>_<tag>_<k>of<K>.mat (full trajectories of one representative
% flight per family: hardest level, plant 1, first wind).
cfg = d1_config(); rng(cfg.seed, 'twister');
scen = d1_sample_scenarios(cfg);              % teacher scenarios: first draw after rng(seed)
lqr = d1_build_lqr(cfg);
[KI, KxLQI] = lqi_setup(cfg, lqr);
ctrls = strtrim(strsplit(d1_getenv_str('D1_CTRLS', 'LQR,LQI,MPC'), ','));
kindOf = containers.Map({'LQR','LQI','MPC','Teacher','P'}, {'L','Q','M','T','P'});
assert(all(isKey(kindOf, ctrls)), 'd1_final_eval:ctrl', 'unknown controller in D1_CTRLS');
tag = d1_getenv_str('D1_TAG', 'base'); cond = d1_getenv_str('D1_COND', 'train');
stu = []; conf = []; teacher = []; mp = []; ckIter = NaN;
if any(ismember(ctrls, {'Teacher', 'P'}))
    S = load(getenv('D1_CKPT')); ckIter = S.st.iter;
    fprintf('CKPT seed=%d iter=%d\n', cfg.seed, S.st.iter);
end
if any(strcmp(ctrls, 'P'))
    [stu, conf] = d1_load_deployed(d1_getenv_str('D1_STUDENT', ''), true);
    assert(stu.ckptIter == ckIter, 'student (teacher iter %d) does not belong to checkpoint iter %d', ...
        stu.ckptIter, ckIter);
    fprintf('STUDENT candidate %d lambda=%.0e rho_max=%.4f val_pos=%.4f | finite W: %d\n', ...
        stu.selected, stu.lambda, stu.rho, stu.val, all(isfinite(stu.W(:))));
end
if any(strcmp(ctrls, 'Teacher'))
    teacher = d1_teacher_build_solver(cfg, scen);
    aMean = extractdata(S.sac.mu);                 % deterministic SAC action (mean)
    [Qt, Rt] = d1_action_to_QR(aMean, cfg); d1_set_teacher_weights(teacher, Qt, Rt, cfg);
    fprintf('TEACHER solver=%s diag(Q)=[%s] diag(R)=[%s]\n', cfg.solverType, ...
        num2str(diag(Qt).', '%.3g '), num2str(diag(Rt).', '%.3g '));
end
if any(strcmp(ctrls, 'MPC')), mp = lmpc_setup(cfg, lqr, d1_getenv_num('D1_MPC_N', 5)); end
F = build_final_flights(cfg, cond);
sh = sscanf(d1_getenv_str('D1_SHARD', '1/1'), '%d/%d');
sel = find(mod((1:numel(F)) - 1, sh(2)) == sh(1) - 1);
fprintf('FINAL_EVAL cond=%s tag=%s flights=%d shard=%d/%d selected=%d ctrls=%s\n', ...
    cond, tag, numel(F), sh(1), sh(2), numel(sel), strjoin(ctrls, ','));
rows = {}; TR = struct('flight',{},'family',{},'id',{},'ctrl',{},'X',{},'U',{},'Xr',{},'alpha',{});
for i = sel
    f = F(i); ln = sprintf('FL %4d %-5s %-50s', i, cond, f.id);
    for c = 1:numel(ctrls)
        [X, U, tm, aux, conv, nDiv] = fly_one(kindOf(ctrls{c}), f, cfg, lqr, stu, conf, KI, KxLQI, mp, teacher);
        Xr = f.Xref(:, 2:size(X,2)+1); m = metr(X, U, Xr, tm, cfg, nDiv);
        a = mean(aux(isfinite(aux))); cv = mean(conv(isfinite(conv)));
        ln = [ln sprintf(' | %s %s pos=%.4f vel=%.4f pmax=%.3f restarts=%d tmed=%.1f aux=%.2f', ...
            ctrls{c}, okc(m.ok), m.pos, m.vel, m.pmax, nDiv, m.tmed, a)]; %#ok<AGROW>
        rows(end+1, :) = {cond, tag, cfg.seed, ckIter, i, f.family, f.level, f.ref, f.plant, f.wind, ...
            ctrls{c}, double(m.ok), nDiv, m.pos, m.pmax, m.vel, m.spd, m.duRms, m.tmed, m.tp99, m.tmax, a, cv}; %#ok<AGROW>
        if f.rep
            TR(end+1) = struct('flight', i, 'family', f.family, 'id', f.id, 'ctrl', ctrls{c}, ...
                'X', single(X), 'U', single(U), 'Xr', single(Xr), 'alpha', single(aux)); %#ok<AGROW>
        end
    end
    fprintf('%s\n', ln);
end
out = d1_getenv_str('D1_OUT', fullfile('results', 'final_eval'));
if ~isfolder(out), mkdir(out); end
T = cell2table(rows, 'VariableNames', {'cond','tag','seed','ckpt_iter','flight','family','level','ref', ...
    'plant','wind','ctrl','ok','restarts','pos_rmse','pos_max','vel_rmse','speed_ratio','du_rms','t_med_us', ...
    't_p99_us','t_max_us','alpha_mean','teacher_conv'});
base = sprintf('%s_%s_%dof%d', cond, tag, sh(1), sh(2));
writetable(T, fullfile(out, ['final_eval_' base '.csv']));
save(fullfile(out, ['final_traj_' base '.mat']), 'TR', '-v7');
fprintf('DONE %d flights\n', numel(sel));
end

% ---- flights ------------------------------------------------------------------
function F = build_final_flights(cfg, cond)
D = getenv('D1_WIND_DIR');
fl = [dir(fullfile(D, 'nf_*.csv')); dir(fullfile(D, 'swuf_*.csv'))];
assert(~isempty(fl), 'd1_final_eval:nowind', 'no wind series in %s', D);
W = cell(1, numel(fl)); wn = cell(1, numel(fl));
for j = 1:numel(fl)
    Tb = readtable(fullfile(D, fl(j).name)); W{j} = make_wind_ds(Tb.t, [Tb.Fx, Tb.Fy, Tb.Fz]);
    [~, wn{j}] = fileparts(fl(j).name);
end
pc = step1_plant_config(); nom = cfg.plant.nominal;
refs = struct('Xref',{},'Uref',{},'family',{},'level',{},'id',{});
if strcmp(cond, 'train')
    cases = d1_train_cases(cfg); lv = {[4 2], [8 5], [12 9]};
    for i = 1:numel(cases)
        tk = regexp(cases(i).groupId, '^([^|]+)\|v([\d.]+)\|a([\d.]+)', 'tokens', 'once');
        va = [str2double(tk{2}), str2double(tk{3})];
        if any(cellfun(@(q) isequal(q, va), lv))
            refs(end+1) = struct('Xref', cases(i).Xref, 'Uref', cases(i).Uref, 'family', tk{1}, ...
                'level', sprintf('v%g/a%g', va), 'id', cases(i).groupId); %#ok<AGROW>
        end
    end
    plants = quad_sample_uncertainty(pc, 5, 'train', pc.uncertainty.defaultSeed, 'lhs'); hardest = 'v12/a9';
elseif strcmp(cond, 'ood')
    ref = targeted_lqr_weak_config().reference;
    o = make_ood_refs(cfg, ref, nom, ref.candidateOodSpeedAnchors, 9.0);
    for i = 1:numel(o)
        tk = regexp(o(i).groupId, '^([^|]+)\|v([\d.]+)\|a([\d.]+)', 'tokens', 'once');
        refs(end+1) = struct('Xref', o(i).Xref, 'Uref', o(i).Uref, 'family', tk{1}, ...
            'level', sprintf('v%s/a%s', tk{2}, tk{3}), 'id', o(i).groupId); %#ok<AGROW>
    end
    plants = quad_sample_uncertainty(pc, 5, 'ood', pc.uncertainty.defaultSeed, 'lhs'); hardest = 'v16/a9';
else
    error('d1_final_eval:cond', 'D1_COND must be train or ood');
end
F = struct('id',{},'family',{},'level',{},'ref',{},'plant',{},'wind',{},'rep',{},'Xref',{},'Uref',{},'theta',{},'ds',{});
q = 0;
for r = 1:numel(refs)
    for p = 1:numel(plants)
        q = q + 1;
        for j = 1:2
            w = mod(2*(q-1) + j - 1, numel(W)) + 1;
            F(end+1) = struct('id', sprintf('%s p%d %s', refs(r).id, p, wn{w}), 'family', refs(r).family, ...
                'level', refs(r).level, 'ref', refs(r).id, 'plant', sprintf('p%d', p), 'wind', wn{w}, ...
                'rep', strcmp(refs(r).level, hardest) && p == 1 && j == 1, ...
                'Xref', refs(r).Xref, 'Uref', refs(r).Uref, 'theta', plants(p), 'ds', W{w}); %#ok<AGROW>
        end
    end
end
end

function out = make_ood_refs(cfg, ref, theta, speeds, accel)
out = struct('Xref',{},'Uref',{},'groupId',{}); idx = 0;
for fi = 1:numel(ref.families)
    for si = 1:numel(speeds)
        idx = idx + 1; rng(770000 + idx, 'twister');
        gid = sprintf('%s|v%g|a%g|OOD', ref.families{fi}, speeds(si), accel);
        try
            opt = quad_sample_targeted_reference_options(ref.families{fi}, ref, speeds(si), accel);
            [Xref, ~, Uref] = quad_targeted_reference_trajectory(ref.families{fi}, cfg.Ts, cfg.stepsPerCase, opt, theta);
            if all(isfinite(Xref(:)))
                out(end+1) = struct('Xref', Xref, 'Uref', Uref, 'groupId', gid); %#ok<AGROW>
            else
                fprintf('OODREF_FAIL %s non-finite\n', gid);
            end
        catch ME
            fprintf('OODREF_FAIL %s %s\n', gid, ME.message);
        end
    end
end
end

function ds = make_wind_ds(tt, FF)
% world-frame external force, linear interpolation in time (held constant past the ends)
Gx = griddedInterpolant(tt, FF(:,1), 'linear', 'nearest');
Gy = griddedInterpolant(tt, FF(:,2), 'linear', 'nearest');
Gz = griddedInterpolant(tt, FF(:,3), 'linear', 'nearest');
ds = @(t, x, u, th) struct('force', [Gx(t); Gy(t); Gz(t)], 'torque', zeros(3,1));
end

% ---- closed-loop flight ---------------------------------------------------------
function [X, U, tm, aux, conv, nDiv] = fly_one(kind, f, cfg, lqr, stu, conf, KI, KxLQI, mp, teacher)
% One flight with the COMMON flight rules of the training pipeline: real time (k-1)*Ts,
% d1_plant_step, d1_case_len steps; after a divergence the plant restarts on the
% reference and the controller's internal state (student memory, u_prev, integrator z, solver /
% active set) is reset. Diverged steps stay NaN in X; nDiv counts restarts. tm = time of
% the control computation only (for the teacher incl. its wind-consistent target).
Ts = cfg.Ts; uh = cfg.uh; Xref = f.Xref; T = d1_case_len(Xref, cfg);
X = nan(12,T); U = nan(4,T); tm = nan(1,T); aux = nan(1,T); conv = nan(1,T); nDiv = 0;
x = Xref(:,1); s = d1_student_init(x, cfg); zI = zeros(3,1); uprev = uh;
if kind == 'M', ws.iA = false(size(mp.bin)); end
if kind == 'T', d1_teacher_reset(teacher, Xref, 1, cfg); end     % clean solver per flight
for k = 1:T
    t = (k-1)*Ts;
    if kind == 'T', Fk = d1_wind_now(f.ds, t, x, uprev, f.theta); end   % oracle information
    t0 = tic;
    e = x - Xref(:,k);
    switch kind
        case 'L'
            u = d1_sat(uh - lqr.K*e, cfg);
        case 'Q'
            uu = uh - KxLQI*e - KI*zI; u = d1_sat(uu, cfg);
            if all(u == uu), zI = zI + Ts*e(1:3); end            % clamping anti-windup
        case 'M'
            [u, ws, nit] = lmpc_solve(mp, x, Xref(:, k+1:k+mp.N), ws); aux(k) = nit;
        case 'T'   % identical to training: d1_teacher_step (reset after an unusable solve)
            [u, status] = d1_teacher_step(teacher, x, uprev, Xref, k, Fk, cfg); conv(k) = (status == 0);
            u = d1_sat(u, cfg); uprev = u;
        case 'P'   % proposed: d1_blend_control on the LQR base (linear student)
            phi = d1_student_feature(s, x, k, Xref, f.Uref, cfg);
            [u, aux(k)] = d1_blend_control(uh - lqr.K*e, e, phi, stu, conf, lqr, cfg);
    end
    tm(k) = toc(t0); U(:,k) = u;
    [xNew, div] = d1_plant_step(t, x, u, Ts, f.theta, f.ds);
    if div
        nDiv = nDiv + 1; x = Xref(:,k+1);
        s = d1_student_init(x, cfg); zI = zeros(3,1); uprev = uh;
        if kind == 'M', ws.iA = false(size(mp.bin)); end
        if kind == 'T', d1_teacher_reset(teacher, Xref, k+1, cfg); end
    else
        if kind == 'P', s = d1_student_push(s, x, u); end
        x = xNew; X(:,k) = x;
    end
end
end

function [KI, KxLQI] = lqi_setup(cfg, lqr)
% LQI on the nominal hover model: z_{k+1} = z_k + Ts*C*e_k (C = position rows),
% Q_aug = blkdiag(Q0, I/z_allow^2), z_allow = 0.10 m (Bryson e_allow_pos) x 1 s.
Ts = cfg.Ts; C = [eye(3), zeros(3,9)];
Aa = [lqr.Ad, zeros(12,3); Ts*C, eye(3)]; Ba = [lqr.Bd; zeros(3,4)];
zAllow = 0.10 * 1.0; Qa = blkdiag(lqr.Q, eye(3)/zAllow^2);
Ka = dlqr(Aa, Ba, Qa, lqr.R); KI = Ka(:, 13:15); KxLQI = Ka(:, 1:12);
end

% ---- standard linear MPC (condensed, box-constrained QP) ------------------------
function mp = lmpc_setup(cfg, lqr, N)
% x_{j+1} = Ad x_j + Bd (u_j - uh)  (hover linearization, identical to the LQR model)
% J = sum_{j=1}^{N-1} |x_j - r_j|_Q^2 + |x_N - r_N|_P^2 + sum_{j=0}^{N-1} |u_j - uh|_R^2
A = lqr.Ad; B = lqr.Bd; nx = 12; nu = 4;
Phi = zeros(nx*N, nx); Gam = zeros(nx*N, nu*N); Aj = eye(nx);
for j = 1:N
    Aj = A*Aj; Phi((j-1)*nx+(1:nx), :) = Aj;
    for i = 1:j
        Gam((j-1)*nx+(1:nx), (i-1)*nu+(1:nu)) = A^(j-i) * B;
    end
end
W = blkdiag(kron(eye(N-1), lqr.Q), lqr.P); Rb = kron(eye(N), lqr.R);
H = 2*(Gam.'*W*Gam + Rb); H = (H + H.')/2;
mp.N = N; mp.Phi = Phi; mp.GtW = 2*Gam.'*W; mp.Linv = inv(chol(H, 'lower'));
mp.Ain = [eye(nu*N); -eye(nu*N)]; mp.bin = [repmat(cfg.uHi - cfg.uh, N, 1); -repmat(cfg.uLo - cfg.uh, N, 1)];
mp.Aeq = zeros(0, nu*N); mp.beq = zeros(0, 1);
mp.opt = mpcActiveSetOptions; mp.opt.UseHessianAsInput = false;
mp.uh = cfg.uh; mp.cfg = cfg;
end

function [u, ws, nit] = lmpc_solve(mp, x0, Rf, ws)
f = mp.GtW * (mp.Phi*x0 - Rf(:));
[dU, flag, iA] = mpcActiveSetSolver(mp.Linv, f, mp.Ain, mp.bin, mp.Aeq, mp.beq, ws.iA, mp.opt);
if flag > 0, ws.iA = iA; nit = flag; else, ws.iA = false(size(mp.bin)); nit = NaN; end
u = d1_sat(mp.uh + dU(1:4), mp.cfg);
end

% ---- metrics / misc -------------------------------------------------------------
function m = metr(X, U, Xr, tm, cfg, nDiv)
% Position metrics use the common capped per-step error (d1_track_err: 5 m cap, a diverged
% step counts as 5 m) over the FULL flight. Completed = no restart and never at the cap.
tv = tm(isfinite(tm))*1e6; m.tmed = median(tv); m.tp99 = prctile(tv, 99); m.tmax = max(tv);
ep = d1_track_err(X, Xr);
m.pos = sqrt(mean(ep.^2)); m.pmax = max(ep);
m.ok = (nDiv == 0) && (m.pmax < 5);
fin = all(isfinite(X), 1);
ev = vecnorm(X(7:9,fin) - Xr(7:9,fin)); m.vel = sqrt(mean(ev.^2));
m.spd = mean(vecnorm(X(7:9,fin))) / mean(vecnorm(Xr(7:9,fin)));
dU = diff(U, 1, 2) ./ cfg.resHalf; dU = dU(:, all(isfinite(dU), 1));
m.duRms = sqrt(mean(dU(:).^2));
end

function s = okc(ok), if ok, s = 'OK'; else, s = 'NO'; end, end
