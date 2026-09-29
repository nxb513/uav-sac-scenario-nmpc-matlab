function d1_final_eval()
%D1_FINAL_EVAL Final comparison under REAL wind, train / OOD conditions, 5 families.
%
% Controllers (D1_CTRLS, comma list). Every controller knows only the NOMINAL model;
% the true plant is a repo LHS sample it does not know.
%   LQR      u = sat(uh - K e)                                  (Bryson, nominal)
%   LQI      u = sat(uh - Kx e - KI z), z = int(p - p_ref) dt   (joint dlqr, nominal)
%   MPC      standard linear MPC, N = D1_MPC_N (5): hover-linear model, Bryson Q,R,
%            terminal DARE P, input box constraints, KWIK active-set QP (warm start)
%   Teacher  SAC-NMPC teacher of the chain: scenario NMPC (acados, M=5 scenarios from
%            the chain seed, N=20, Nc=5, D1_SOLVER) with Q,R = action_to_QR(tanh(mu))
%            -- same teacher_step as training (solver reset per flight and after a failed solve)
%   P        proposed blend u = sat(uh - K e + alpha*Du)         (pipeline fly_gate (d))
%   PI       proposed + I on the LQI base u = sat(uh - Kx e - KI z + alpha*Du)
% Integral states use clamping anti-windup (no integration on saturated steps).
%
% Conditions (D1_COND):
%   train : train plants (LHS 5) x 15 ID references (5 families x {v4 a2, v8 a5, v12 a9})
%   ood   : OOD plants (LHS 5) x 10 OOD references (5 families x {14, 16 m/s}, a = 9)
% Each (reference, plant) pair is flown with 2 real-wind series, cycled over all series
% in D1_WIND_DIR (tools/wind/prepare_wind_series.py): train 150, ood 100 flights.
% The chain (seed, random QR base, SAC width) comes from D1_SEED / D1_RANDOM_QR /
% D1_LOGMULT_DEC exactly as in training; D1_CKPT (+ D1_CONF) is its checkpoint.
%
% Outputs (D1_OUT): final_eval_<cond>_<tag>_<k>of<K>.csv (one row per flight x ctrl)
% and final_traj_<cond>_<tag>_<k>of<K>.mat (full trajectories of one representative
% flight per family: hardest level, plant 1, first wind).
cfg = joint_config(); rng(cfg.seed, 'twister');
scen = sample_scenarios(cfg);                 % teacher scenarios: first draw after rng(seed)
lqr = build_lqr(cfg);
[KI, KxLQI] = lqi_setup(cfg, lqr);
ctrls = strtrim(strsplit(getenv_str('D1_CTRLS', 'LQR,LQI,MPC'), ','));
tag = getenv_str('D1_TAG', 'base'); cond = getenv_str('D1_COND', 'train');
sur = []; conf = []; teacher = []; mp = []; ckIter = NaN;
if any(ismember(ctrls, {'Teacher', 'P', 'PI'}))
    S = load(getenv('D1_CKPT')); sur = S.sur; conf = pick_conf(S, getenv_str('D1_CONF', ''));
    ckIter = S.st.iter;
    L = S.sur.net.Learnables; nb = 0; nt = 0;
    for q = 1:height(L), v = extractdata(L.Value{q}); nb = nb + sum(~isfinite(v(:))); nt = nt + numel(v); end
    fprintf('CKPT seed=%d iter=%d samples=%d | surrogate non-finite params %d/%d\n', ...
        cfg.seed, S.st.iter, S.st.totalSamples, nb, nt);
end
if any(strcmp(ctrls, 'Teacher'))
    teacher = d1_teacher_build_solver(cfg, scen);
    aMean = tanh(extractdata(S.sac.mu));
    [Qt, Rt] = action_to_QR(aMean, cfg); set_teacher_weights(teacher, Qt, Rt, cfg);
    fprintf('TEACHER solver=%s diag(Q)=[%s] diag(R)=[%s]\n', cfg.solverType, ...
        num2str(diag(Qt).', '%.3g '), num2str(diag(Rt).', '%.3g '));
end
if any(strcmp(ctrls, 'MPC')), mp = lmpc_setup(cfg, lqr, getenv_num('D1_MPC_N', 5)); end
F = build_final_flights(cfg, cond);
sh = sscanf(getenv_str('D1_SHARD', '1/1'), '%d/%d');
sel = find(mod((1:numel(F)) - 1, sh(2)) == sh(1) - 1);
fprintf('FINAL_EVAL cond=%s tag=%s flights=%d shard=%d/%d selected=%d ctrls=%s\n', ...
    cond, tag, numel(F), sh(1), sh(2), numel(sel), strjoin(ctrls, ','));
kindOf = containers.Map({'LQR','LQI','MPC','Teacher','P','PI'}, {'L','Q','M','T','P','J'});
rows = {}; TR = struct('flight',{},'family',{},'id',{},'ctrl',{},'X',{},'U',{},'Xr',{},'alpha',{});
for i = sel
    f = F(i); ln = sprintf('FL %4d %-5s %-50s', i, cond, f.id);
    for c = 1:numel(ctrls)
        [X, U, tm, aux, conv] = fly_one(kindOf(ctrls{c}), f, cfg, lqr, sur, conf, KI, KxLQI, mp, teacher);
        Xr = f.Xref(:, 2:size(X,2)+1); m = metr(X, U, Xr, tm, cfg);
        a = mean(aux(isfinite(aux))); cv = mean(conv(isfinite(conv)));
        ln = [ln sprintf(' | %s %s pos=%.4f vel=%.4f pmax=%.3f tmed=%.1f aux=%.2f', ...
            ctrls{c}, okc(m.ok), m.pos, m.vel, m.pmax, m.tmed, a)]; %#ok<AGROW>
        rows(end+1, :) = {cond, tag, cfg.seed, ckIter, i, f.family, f.level, f.ref, f.plant, f.wind, ...
            ctrls{c}, double(m.ok), m.pos, m.pmax, m.vel, m.spd, m.duRms, m.tmed, m.tp99, m.tmax, a, cv}; %#ok<AGROW>
        if f.rep
            TR(end+1) = struct('flight', i, 'family', f.family, 'id', f.id, 'ctrl', ctrls{c}, ...
                'X', single(X), 'U', single(U), 'Xr', single(Xr), 'alpha', single(aux)); %#ok<AGROW>
        end
    end
    fprintf('%s\n', ln);
end
out = getenv_str('D1_OUT', fullfile('results', 'final_eval'));
if ~isfolder(out), mkdir(out); end
T = cell2table(rows, 'VariableNames', {'cond','tag','seed','ckpt_iter','flight','family','level','ref', ...
    'plant','wind','ctrl','ok','pos_rmse','pos_max','vel_rmse','speed_ratio','du_rms','t_med_us', ...
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
refs = struct('Xref',{},'family',{},'level',{},'id',{});
if strcmp(cond, 'train')
    cases = generate_teacher_dev_cases(cfg); lv = {[4 2], [8 5], [12 9]};
    for i = 1:numel(cases)
        tk = regexp(cases(i).groupId, '^([^|]+)\|v([\d.]+)\|a([\d.]+)', 'tokens', 'once');
        va = [str2double(tk{2}), str2double(tk{3})];
        if any(cellfun(@(q) isequal(q, va), lv))
            refs(end+1) = struct('Xref', cases(i).Xref, 'family', tk{1}, ...
                'level', sprintf('v%g/a%g', va), 'id', cases(i).groupId); %#ok<AGROW>
        end
    end
    plants = quad_sample_uncertainty(pc, 5, 'train', pc.uncertainty.defaultSeed, 'lhs'); hardest = 'v12/a9';
elseif strcmp(cond, 'ood')
    ref = targeted_lqr_weak_config().reference;
    o = make_ood_refs(cfg, ref, nom, ref.candidateOodSpeedAnchors, 9.0);
    for i = 1:numel(o)
        tk = regexp(o(i).groupId, '^([^|]+)\|v([\d.]+)\|a([\d.]+)', 'tokens', 'once');
        refs(end+1) = struct('Xref', o(i).Xref, 'family', tk{1}, ...
            'level', sprintf('v%s/a%s', tk{2}, tk{3}), 'id', o(i).groupId); %#ok<AGROW>
    end
    plants = quad_sample_uncertainty(pc, 5, 'ood', pc.uncertainty.defaultSeed, 'lhs'); hardest = 'v16/a9';
else
    error('d1_final_eval:cond', 'D1_COND must be train or ood');
end
F = struct('id',{},'family',{},'level',{},'ref',{},'plant',{},'wind',{},'rep',{},'Xref',{},'theta',{},'ds',{});
q = 0;
for r = 1:numel(refs)
    for p = 1:numel(plants)
        q = q + 1;
        for j = 1:2
            w = mod(2*(q-1) + j - 1, numel(W)) + 1;
            F(end+1) = struct('id', sprintf('%s p%d %s', refs(r).id, p, wn{w}), 'family', refs(r).family, ...
                'level', refs(r).level, 'ref', refs(r).id, 'plant', sprintf('p%d', p), 'wind', wn{w}, ...
                'rep', strcmp(refs(r).level, hardest) && p == 1 && j == 1, ...
                'Xref', refs(r).Xref, 'theta', plants(p), 'ds', W{w}); %#ok<AGROW>
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
function [X, U, tm, aux, conv] = fly_one(kind, f, cfg, lqr, sur, conf, KI, KxLQI, mp, teacher)
% Real time (k-1)*Ts is passed to the plant so the time-varying wind acts.
Ts = cfg.Ts; N = cfg.N; Xref = f.Xref; T = min(cfg.stepsPerCase, size(Xref,2)-N-1);
uh = [cfg.plant.m*cfg.plant.g; 0; 0; 0];
lo = [0;-0.5;-0.5;-0.25]; hi = [cfg.plant.Tmax;0.5;0.5;0.25];
X = nan(12,T); U = nan(4,T); tm = nan(1,T); aux = nan(1,T); conv = nan(1,T);
haveConf = ~isempty(conf) && isfield(conf,'LQR') && ~isempty(conf.LQR.w);
x = Xref(:,1); stateHist = repmat(x,1,4); inputHist = repmat(uh,1,4); zI = zeros(3,1);
uprev = uh; if kind == 'M', ws.iA = false(size(mp.bin)); end
if kind == 'T', teacher_reset(teacher, Xref, 1, uh, cfg); end     % clean solver per flight
for k = 1:T
    t0 = tic;
    e = x - Xref(:,k);
    switch kind
        case 'L'
            u = min(max(uh - lqr.K*e, lo), hi);
        case 'Q'
            uu = uh - KxLQI*e - KI*zI; u = min(max(uu, lo), hi);
        case 'M'
            [u, ws, nit] = lmpc_solve(mp, x, Xref(:, k+1:k+mp.N), ws); aux(k) = nit;
        case 'T'   % identical to training: teacher_step (reset after a failed solve)
            [u, status] = teacher_step(teacher, x, uprev, Xref, k, uh, cfg); conv(k) = (status == 0);
            u = min(max(u,lo),hi); uprev = u;
        case {'P', 'J'}
            if kind == 'J'
                uLk = uh - KxLQI*e - KI*zI;                % LQI base [Kx, KI]
            else
                uLk = uh - lqr.K*e;                        % pipeline blend base
            end
            feat = surrogate_build_feature(stateHist, inputHist, Xref(:,k:k+10), zeros(12,1));
            if all(isfinite(feat))
                du = surrogate_predict_du(sur, feat); cS = surrogate_predict_cs(sur, feat);
                if ~all(isfinite([du; cS])), du = zeros(4,1); cS = 0; end   % NaN guard -> pure base
            else
                du = zeros(4,1); cS = 0;
            end
            if haveConf, cLp = predict_logistic(conf.LQR, conf_feature_online(e).'); else, cLp = 0; end
            gL = min(max((cfg.cHigh - cLp)/(cfg.cHigh - cfg.cLow), 0), 1);
            alpha = cS * gL;
            if cfg.alphaSafe
                alpha = min(alpha, (1-cfg.epsSafe)*alpha_bar_est(e, du, lqr));
            end
            uu = uLk + alpha*du; u = min(max(uu, lo), hi); aux(k) = alpha;
    end
    if any(kind == 'QJ') && all(u == uu), zI = zI + Ts*e(1:3); end   % clamping anti-windup
    tm(k) = toc(t0); U(:,k) = u;
    if any(kind == 'PJ'), stateHist = [stateHist(:,2:end), x]; inputHist = [inputHist(:,2:end), u]; end
    x = quad_step_rk4((k-1)*Ts, x, u, Ts, f.theta, f.ds); X(:,k) = x;
    if ~all(isfinite(x)) || norm(x(1:3)) > 1e4, X(:,k:end) = NaN; break; end
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
lo = [0;-0.5;-0.5;-0.25]; hi = [cfg.plant.Tmax;0.5;0.5;0.25];
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
mp.Ain = [eye(nu*N); -eye(nu*N)]; mp.bin = [repmat(hi - lqr.uh, N, 1); -repmat(lo - lqr.uh, N, 1)];
mp.Aeq = zeros(0, nu*N); mp.beq = zeros(0, 1);
mp.opt = mpcActiveSetOptions; mp.opt.UseHessianAsInput = false;
mp.uh = lqr.uh; mp.lo = lo; mp.hi = hi;
end

function [u, ws, nit] = lmpc_solve(mp, x0, Rf, ws)
f = mp.GtW * (mp.Phi*x0 - Rf(:));
[dU, flag, iA] = mpcActiveSetSolver(mp.Linv, f, mp.Ain, mp.bin, mp.Aeq, mp.beq, ws.iA, mp.opt);
if flag > 0, ws.iA = iA; nit = flag; else, ws.iA = false(size(mp.bin)); nit = NaN; end
u = min(max(mp.uh + dU(1:4), mp.lo), mp.hi);
end

% ---- metrics / misc -------------------------------------------------------------
function m = metr(X, U, Xr, tm, cfg)
tv = tm(isfinite(tm))*1e6; m.tmed = median(tv); m.tp99 = prctile(tv, 99); m.tmax = max(tv);
m.ok = all(isfinite(X(:)));
if ~m.ok, m.pos = NaN; m.vel = NaN; m.pmax = NaN; m.spd = NaN; m.duRms = NaN; return; end
ep = vecnorm(X(1:3,:) - Xr(1:3,:)); ev = vecnorm(X(7:9,:) - Xr(7:9,:));
m.pos = sqrt(mean(ep.^2)); m.vel = sqrt(mean(ev.^2)); m.pmax = max(ep);
m.spd = mean(vecnorm(X(7:9,:))) / mean(vecnorm(Xr(7:9,:)));
dU = diff(U, 1, 2) ./ cfg.resHalf;                    % input increments, surrogate normalization
m.duRms = sqrt(mean(dU(:).^2));
end

function s = okc(ok), if ok, s = 'OK'; else, s = 'NO'; end, end

function c = pick_conf(S, confFile)
if isfield(S,'conf'), c = S.conf; elseif ~isempty(confFile) && isfile(confFile), C = load(confFile); c = C.conf; else, c = []; end
end

% ============================================================================
% Helpers below are copied VERBATIM from experiments/run_d1_joint_pipeline.m
% (local functions there), so the controllers match the training pipeline exactly.
% ============================================================================
function cfg = joint_config()
cfg.seed = getenv_num('D1_SEED', 260914001);
cfg.runDir = getenv_str('D1_RUN_DIR', fullfile('results','d1_joint', ...
    sprintf('seed%d', getenv_num('D1_SEED',260914001))));
cfg.wallSeconds = getenv_num('D1_WALL_SECONDS', 300);
cfg.stopIter = getenv_num('D1_STOP_ITER', 0);        % stop exactly at this SAC iter (0 = off)
cfg.resume = strcmp(getenv_str('D1_RESUME','0'),'1');
cfg.Ts = 0.05; cfg.N = 20; cfg.Nc = 5; cfg.H = 20; cfg.Qf = 0; cfg.dU = 0;
cfg.M = 5;                                          % robust scenarios (frozen)
cfg.solverType = getenv_str('D1_SOLVER', 'SQP_RTI'); % 'SQP_RTI' (fast) | 'SQP' (accurate)
cfg.stepsPerCase = getenv_num('D1_STEPS', 1000);
cfg.casesPerEval = getenv_num('D1_CASES_PER_EVAL', 20);
cfg.plant = d1_joint_plant_params();
cfg.actionDim = 6;                                  % Q:{pos,att,vel,rate}, R:{T,tau}
% SAC Q,R search half-width in decades around the base (mult in 10^[-dec, +dec]).
% Default 1.5 (0.03x..32x, wide). Smaller = SAC stays CLOSER to Bryson (e.g. 0.5 =
% 0.32x..3.2x) -> teacher can't be pushed into an unsolvable corner.
cfg.logMultDec = getenv_num('D1_LOGMULT_DEC', 1.5);
cfg.logMultBounds = [10^(-cfg.logMultDec), 10^(cfg.logMultDec)];
% Q,R base for the SAC-tuned teacher: 0 = Bryson warm-start (default), 1 = RANDOM
% (no Bryson) log-uniform diag weights, deterministic per seed. Ablation: does the
% Bryson warm-start matter? The LQR baseline stays Bryson in BOTH (fixed yardstick).
cfg.randomQR = strcmp(getenv_str('D1_RANDOM_QR','0'),'1');
cfg.rqrLog   = [-2, 2];                              % random base: 10^[-2,2] per weight
% surrogate (2-head: residual Delta_u + confidence c_S)
cfg.surHidden = 128; cfg.surLR = 1e-3; cfg.surBatch = 256;
cfg.surBufferCap = 1e5; cfg.surRecentFrac = 0.5;
cfg.resHalf = [cfg.plant.Tmax; 1; 1; 0.5];           % residual normalization scale
% blend / confidence design params (fixed, disclosed)
cfg.epsP   = getenv_num('D1_EPS_P',  0.5);           % c_S error scale (m): s=exp(-(RMS/epsP)^2)
cfg.cLow   = getenv_num('D1_C_LOW',  0.3);           % g_L gate low threshold on c_LQR
cfg.cHigh  = getenv_num('D1_C_HIGH', 0.7);           % g_L gate high threshold on c_LQR
cfg.epsSafe= getenv_num('D1_EPS_SAFE', 0.1);         % Lyapunov safeguard margin (alpha_safe mode)
cfg.alphaSafe = strcmp(getenv_str('D1_ALPHA_SAFE','0'),'1');
cfg.lamU = 1; cfg.lamC = 1;                          % loss weights (on normalized targets)
cfg.csCasesPerCall = getenv_num('D1_CS_CASES', 40);  % surrogate closed-loop cases for c_S labels
cfg.csEpochs = getenv_num('D1_CS_EPOCHS', 300);
% sac
cfg.sacLR = 3e-4; cfg.sacBatch = 256; cfg.sacBufferCap = 5e4;
cfg.sacGamma = 0.0;                                  % 1-step bandit (done each ep)
cfg.sacTau = 0.005; cfg.sacTargetEntropy = -cfg.actionDim;
cfg.logEvery = 1; cfg.checkpointEverySec = 120;
% frozen milestone checkpoints (+ their own confidences) every N SAC iterations; 0 = off
cfg.ckptEvery = getenv_num('D1_CKPT_EVERY_ITER', 50);
% training wind (random, synthetic; NO measured wind data is used): see sample_wind
cfg.windOn   = strcmp(getenv_str('D1_WIND','1'),'1');
cfg.windMin  = getenv_num('D1_WIND_MIN', 1);         % mean wind speed range [m/s]
cfg.windMax  = getenv_num('D1_WIND_MAX', 10);
cfg.windDrag = [0.425; 0.256; 0];                    % mass-normalized rotor drag [1/s], Faessler et al. RA-L 2018
end

function v = getenv_num(name, dflt)
s = getenv(name); if isempty(s), v = dflt; else, v = str2double(s); end
end

function v = getenv_str(name, dflt)
s = getenv(name); if isempty(s), v = dflt; else, v = s; end
end

function lqr = build_lqr(cfg)
P = cfg.plant; theta = P.nominal;
xh = zeros(12,1); uh = [P.m*P.g; 0; 0; 0];
[A, B] = num_linearize(@(x,u) quad_dynamics(0, x, u, theta, []), xh, uh);
sysd = c2d(ss(A, B, eye(12), zeros(12,4)), cfg.Ts);
[Q0, R0] = d1_bryson_weights(P);
[K, Sr] = dlqr(sysd.A, sysd.B, Q0, R0);
lqr.K = K; lqr.P = Sr; lqr.uh = uh; lqr.Ad = sysd.A; lqr.Bd = sysd.B;
lqr.Q = Q0; lqr.R = R0;                               % for alpha_bar (Prop 1)
end

function [A, B] = num_linearize(f, x0, u0)
n = numel(x0); m = numel(u0); h = 1e-6;
A = zeros(n); B = zeros(n, m);
for i = 1:n
    dx = zeros(n,1); dx(i) = h;
    A(:,i) = (f(x0+dx,u0) - f(x0-dx,u0)) / (2*h);
end
for j = 1:m
    du = zeros(m,1); du(j) = h;
    B(:,j) = (f(x0,u0+du) - f(x0,u0-du)) / (2*h);
end
end

function cases = generate_teacher_dev_cases(cfg)
ref = targeted_lqr_weak_config().reference;
theta = cfg.plant.nominal;
families = ref.families; speeds = ref.approvedIdSpeedAnchors;
accels = ref.approvedAccelerationTargets;
cases = struct('Xref',{},'Uref',{},'groupId',{});
count = 0; target = 120;
% deterministic stratified sweep family x accel x speed until ~120
for fi = 1:numel(families)
  for ai = 1:numel(accels)
    for si = 1:numel(speeds)
      if count >= target, break; end
      gid = sprintf('%s|v%g|a%g', families{fi}, speeds(si), accels(ai));
      rng(d1_case_seed(gid), 'twister');
      try
        opt = quad_sample_targeted_reference_options(families{fi}, ref, ...
            speeds(si), accels(ai));
        [Xref,~,Uref] = quad_targeted_reference_trajectory(families{fi}, ...
            cfg.Ts, cfg.stepsPerCase, opt, theta);
        if all(isfinite(Xref(:)))
          count = count + 1;
          cases(count).Xref = Xref; cases(count).Uref = Uref;
          cases(count).groupId = gid;
        end
      catch
      end
    end
  end
end
assert(count > 0, 'No teacher-dev cases generated.');
end

function [Q, R] = action_to_QR(a, cfg)
[Q0, R0] = qr_base(cfg);                            % Bryson, or random (no-Bryson)
lo = log(cfg.logMultBounds(1)); hi = log(cfg.logMultBounds(2));
mult = exp(lo + 0.5*(a(:)+1)*(hi-lo));              % 6 log-multipliers
q = diag(Q0);
q(1:3)=q(1:3)*mult(1); q(4:6)=q(4:6)*mult(2);
q(7:9)=q(7:9)*mult(3); q(10:12)=q(10:12)*mult(4);
r = diag(R0); r(1)=r(1)*mult(5); r(2:4)=r(2:4)*mult(6);
Q = diag(q); R = diag(r);
end

function [Q0, R0] = qr_base(cfg)
% Base Q,R for the SAC-tuned teacher. Default = Bryson (1/e_allow^2, 1/du_allow^2).
% D1_RANDOM_QR=1 -> random diagonal weights, log-uniform 10^cfg.rqrLog per element,
% deterministic per seed (ablation vs Bryson warm-start). NOTE: only the TEACHER
% base changes; build_lqr keeps Bryson so the LQR yardstick is identical across both.
if cfg.randomQR
    rs = RandStream('twister', 'Seed', cfg.seed + 90210);   % independent of global rng
    lo = cfg.rqrLog(1); span = cfg.rqrLog(2) - cfg.rqrLog(1);
    Q0 = diag(10.^(lo + span*rand(rs,12,1)));
    R0 = diag(10.^(lo + span*rand(rs,4,1)));
else
    [Q0, R0] = d1_bryson_weights(cfg.plant);
end
end

function scen = sample_scenarios(cfg)
% M uncertain plant realizations; scenario 1 = nominal, rest perturbed by +/-rho
% (step1_plant_config train uncertainty). Deterministic given the seed rng.
nom = cfg.plant.nominal;
rho = step1_plant_config().uncertainty.train.rho;   % 14x1
Jnom = diag(nom.J);
scen = struct('m',{},'Jd',{},'Dv',{},'Domega',{},'alphaT',{},'alphaTau',{});
for i = 1:cfg.M
    if i==1, xi = zeros(14,1); else, xi = 2*rand(14,1)-1; end
    f = 1 + xi.*rho;
    scen(i).m = nom.m*f(1);
    scen(i).Jd = Jnom.*f(2:4);
    scen(i).Dv = nom.Dv(:).*f(5:7);
    scen(i).Domega = nom.Domega(:).*f(8:10);
    scen(i).alphaT = nom.alphaT*f(11);
    scen(i).alphaTau = nom.alphaTau(:).*f(12:14);
end
end

function set_teacher_weights(solver, Q, R, cfg)
Wblk = repmat({Q/cfg.M}, 1, cfg.M); Wblk{end+1} = R;
W = blkdiag(Wblk{:});                                 % (M*12+4) x (M*12+4)
for s = 0:cfg.N-1, solver.set('cost_W', W, s); end
% terminal weight kept at build-time (Qf=0; terminal retune not critical).
end

function warmstart_ref(teacher, Xref, k, uh, cfg)
% seed the SQP initial guess along the reference (helps convergence on fast refs)
M = cfg.M; N = cfg.N; nc = size(Xref,2);
for j = 0:N
    teacher.set('init_x', [repmat(Xref(:,min(k+j,nc)),M,1); uh], j);
end
for j = 0:N-1
    teacher.set('init_u', zeros(4,1), j);
end
end

function teacher_reset(teacher, Xref, k, uh, cfg)
% Clear ALL solver memory (iterates, multipliers, QP warm start) and re-seed the
% initial guess along the reference. The acados solver object is reused for the whole
% process; without this, one failed solve left corrupted multipliers that made every
% later solve fail (the former reward -10 "collapse" that only a restart cured).
teacher.reset();
warmstart_ref(teacher, Xref, k, uh, cfg);
end

function [u, status, usable] = teacher_step(teacher, x, uprev, Xref, k, uh, cfg)
% One teacher NMPC step. The SQP iterate is applied when the solver converged
% (status 0) or hit its iteration cap (status 2) with a finite iterate; otherwise the
% last control is held and the solver is reset before the next step.
N = cfg.N; M = cfg.M;
for s = 0:N-1
    teacher.set('cost_y_ref', [repmat(Xref(:,k+s),M,1); uh], s);
end
teacher.set('cost_y_ref_e', repmat(Xref(:,k+N),M,1));
teacher.set('constr_x0', [repmat(x,M,1); uprev]);
teacher.solve();
status = teacher.get('status'); du0 = teacher.get('u', 0);
usable = any(status == [0 2]) && all(isfinite(du0));
if usable
    u = uprev + du0;
else
    u = uprev;
    teacher_reset(teacher, Xref, min(k+1, size(Xref,2)-N), uh, cfg);
end
end

function du = surrogate_predict_du(sur, feat)
% predicted residual Delta_u (physical units) from a raw feature vector
z = single(feat) ./ sur.featScale;
p = predict(sur.net, dlarray(z,'CB'), 'Outputs', 'du');
du = double(extractdata(p(:))) .* sur.resHalf;
end

function c = surrogate_predict_cs(sur, feat)
z = single(feat) ./ sur.featScale;
p = predict(sur.net, dlarray(z,'CB'), 'Outputs', 'cs');
c = double(extractdata(p(1)));
end

function F = conf_feature(Ew)
% 12 (wrapped) error states + 4 group magnitudes (pos/att/vel/rate)
pos = vecnorm(Ew(1:3,:)); att = vecnorm(Ew(4:6,:));
vel = vecnorm(Ew(7:9,:)); rate = vecnorm(Ew(10:12,:));
F = [Ew; pos; att; vel; rate];                              % 16 x N
end

function f = conf_feature_online(e)
ew = e; ew(4:6) = mod(ew(4:6)+pi, 2*pi) - pi;
f = conf_feature(ew);                                       % 16 x 1
end

function p = predict_logistic(clf, X)
% X: n x d -> p: n x 1 = P(contract)
if isempty(clf) || isempty(clf.w), p = 0.5*ones(size(X,1),1); return; end
Z = (X - clf.mu)./clf.sg; p = 1./(1+exp(-(Z*clf.w + clf.b)));
end

function ab = alpha_bar_est(e, d, lqr)
% linearized one-step contraction budget alpha_bar (Prop 1); d = residual Delta_u
Acl = lqr.Ad - lqr.Bd*lqr.K; S = lqr.P;
m = e.'*(lqr.Q + lqr.K.'*lqr.R*lqr.K)*e;
b = 2 * e.'*Acl.'*S*lqr.Bd*d;
q = d.'*lqr.Bd.'*S*lqr.Bd*d;
if q > 1e-12
    ab = (-b + sqrt(max(b^2 + 4*q*m,0)))/(2*q);
elseif b > 1e-12
    ab = m/b;
else
    ab = inf;
end
ab = max(ab, 0);
end
