function run_d1_joint_pipeline()
%RUN_D1_JOINT_PIPELINE One-seed joint pipeline: SAC tunes the NMPC Q,R -> scenario-NMPC
% teacher (M=5, N=20, Nc=5) with PRIVILEGED wind knowledge + paired LQR -> streaming
% residual surrogate -> c_S / c_LQR consolidation -> checkpoint. Method document:
% docs/D1_method.tex. The shared definitions (configuration, flight rules, teacher step,
% surrogate history/feature, deployed blend) are the src/joint/d1_*.m functions, also used
% by experiments/d1_final_eval.m, so training and evaluation cannot drift apart.
%
% Open-ended: runs SAC iterations until the wall-time quota (or D1_STOP_ITER), saving
% a FULL checkpoint (SAC, surrogate, counters, pending iteration, RNG) that a later job
% resumes, down to the case level. Milestone checkpoints every D1_CKPT_EVERY_ITER.
% All training flights use the common flight rules (d1_plant_step, d1_case_len,
% d1_teacher_step, d1_track_err) and the random training wind (d1_sample_wind).
%
% Env: D1_SEED, D1_RUN_DIR, D1_WALL_SECONDS, D1_RESUME ('1' to resume), D1_STOP_ITER,
% D1_CKPT_EVERY_ITER, D1_WIND, D1_SOLVER, D1_RANDOM_QR, D1_LOGMULT_DEC, ...
% Modes: D1_DIAG, D1_SURR_EVAL, D1_CONSOLIDATE, D1_COMPARE, D1_GATE_GRID.
% FROZEN: Np=20, Nc=5, Ts=0.05, H=20, dU=0 (asserted; cfg.Qf is an unused flag, the
% teacher terminal weight is the fixed Bryson Q0/M).

cfg = d1_config();
fprintf('== D1 joint pipeline seed=%d wall=%ds run_dir=%s resume=%d ==\n', ...
    cfg.seed, cfg.wallSeconds, cfg.runDir, cfg.resume);
assert(cfg.N==20 && cfg.Nc==5 && abs(cfg.Ts-0.05)<1e-12 && cfg.H==20, ...
    'Frozen horizon/Ts/H violated.');
assert(cfg.Qf==0 && cfg.dU==0, 'Frozen Qf/dU violated.');

if ~isfolder(cfg.runDir), mkdir(cfg.runDir); end
rng(cfg.seed, 'twister');

% ---- fixed components (built once) ------------------------------------------
% Teacher-free modes (surrogate eval, consolidate, gate grid) never touch the
% NMPC teacher, so they SKIP the acados build entirely -> those CI jobs need no
% acados/CasADi toolchain.
noTeacher = strcmp(d1_getenv_str('D1_SURR_EVAL','0'),'1') || ...
            strcmp(d1_getenv_str('D1_CONSOLIDATE','0'),'1') || ...
            strcmp(d1_getenv_str('D1_GATE_GRID','0'),'1');
scen = d1_sample_scenarios(cfg);            % M=5 uncertain plant realizations (first rng draw)
lqr = d1_build_lqr(cfg);                    % K, P (Bryson LQR + Riccati)
cases = d1_train_cases(cfg);                % 120 cases, 1000-sample references
if noTeacher
    teacher = [];
    fprintf('teacher SKIPPED (teacher-free mode) + LQR(P minEig=%.4g) + %d cases\n', ...
        min(eig(lqr.P)), numel(cases));
else
    teacher = d1_teacher_build_solver(cfg, scen);
    fprintf('built teacher(acados M=%d Nc=%d, privileged wind) + LQR(P minEig=%.4g) + %d cases\n', ...
        cfg.M, cfg.Nc, min(eig(lqr.P)), numel(cases));
end

% ---- diagnostic mode: fly a few cases, dump trajectories, exit --------------
if strcmp(d1_getenv_str('D1_DIAG','0'), '1')
    diag_flight(cfg, teacher, lqr, cases);
    return;
end

% ---- init or resume learners ------------------------------------------------
ckptPath = fullfile(cfg.runDir, sprintf('checkpoint_seed%d.mat', cfg.seed));
confPath = fullfile(cfg.runDir, sprintf('conf_seed%d.mat', cfg.seed));
if strcmp(d1_getenv_str('D1_SURR_EVAL','0'), '1')
    assert(isfile(ckptPath), 'surrogate eval requires a checkpoint (set resume_run_id).');
    S = load(ckptPath); sur = d1_load_deployed(S, confPath); surrogate_eval(cfg, lqr, cases, sur); return;
end
if strcmp(d1_getenv_str('D1_CONSOLIDATE','0'), '1')
    % Train c_S and c_LQR over LQR / surrogate closed-loop rollouts; writes
    % conf_seed<s>.mat (conf + consolidated surrogate = the deployed pair).
    assert(isfile(ckptPath), 'consolidate requires a checkpoint (set resume_run_id).');
    S = load(ckptPath); consolidate_confidence(cfg, lqr, cases, S.sur, S.st); return;
end
if strcmp(d1_getenv_str('D1_COMPARE','0'), '1')
    % Fly LQR-only / teacher-NMPC / pure surrogate / proposed(LQR+surrogate blend)
    % on one case per family, dump trajectories + tracking error for the figures.
    assert(isfile(ckptPath), 'compare requires a checkpoint (set resume_run_id).');
    S = load(ckptPath); [sur, conf] = d1_load_deployed(S, confPath);
    compare_flight(cfg, teacher, lqr, cases, sur, S.sac, conf); return;
end
if strcmp(d1_getenv_str('D1_GATE_GRID','0'), '1')
    % Evaluate ONE (c_low,c_high) gate (from D1_C_LOW/D1_C_HIGH) on the validation
    % families: fly LQR + proposed blend (teacher-free) and report tracking RMSE.
    % The workflow matrix sweeps the grid -> collect GATE_RESULT lines.
    assert(isfile(ckptPath), 'gate_grid requires a consolidated checkpoint (set resume_run_id).');
    S = load(ckptPath); [sur, conf] = d1_load_deployed(S, confPath);
    gate_grid_flight(cfg, lqr, cases, sur, conf); return;
end
if cfg.resume && isfile(ckptPath)
    S = load(ckptPath); sac = S.sac; sur = S.sur; st = S.st;
    rng(S.rngState);
    fprintf('RESUMED from %s at iter=%d (pending case %d)\n', ckptPath, st.iter, ...
        pending_case(st));
else
    sac = init_sac(cfg); sur = init_surrogate(cfg);
    st = struct('iter', 0, 'totalSamples', 0, 'teacherVersion', 0, 'lastReward', NaN, 'pend', []);
    fprintf('FRESH start\n');
end

% ---- open-ended SAC loop (resumable after every case) -------------------------
% One SAC iteration = casesPerEval paired rollouts with one sampled Q,R. The partly
% done iteration (action, case indices, rewards so far) lives in st.pend and is saved
% with the checkpoint, so an iteration longer than one CI job continues in the next
% job with the identical random stream (RNG state is part of the checkpoint).
tStart = tic; lastCkpt = tic; caseDur = [];
wallReserve = 120;                                   % s kept for final save + consolidate
if ~isfield(st, 'pend'), st.pend = []; end
stopRun = false;
while ~stopRun
    if isempty(st.pend)
        % Optional hard stop at an exact SAC iteration (D1_STOP_ITER); 0 = no stop.
        if cfg.stopIter > 0 && st.iter >= cfg.stopIter
            fprintf('STOP_ITER reached iter=%d (target %d)\n', st.iter, cfg.stopIter);
            break;
        end
        st.iter = st.iter + 1;
        st.teacherVersion = st.iter;                   % Q,R identity per rollout
        a = sac_sample_action(sac, cfg);               % candidate in [-1,1]^6
        idx = randi(numel(cases), 1, cfg.casesPerEval);
        st.pend = struct('a', a, 'idx', idx, 'rewards', zeros(cfg.casesPerEval, 1), ...
            'cL', zeros(0,1), 'c', 1, 'dur', 0);
    end
    [Q, R] = d1_action_to_QR(st.pend.a, cfg);
    d1_set_teacher_weights(teacher, Q, R, cfg);
    % --- remaining rollouts of this iteration (paired NMPC + LQR) ----------------
    while st.pend.c <= cfg.casesPerEval
        % Do not START a case predicted to overrun the wall budget (the CI step timeout
        % would kill the job before the final checkpoint). Estimate = max of last 5 cases.
        if ~isempty(caseDur) && toc(tStart) + max(caseDur(max(1,end-4):end)) + wallReserve > cfg.wallSeconds
            fprintf('WALL_STOP elapsed=%.0fs next-case est=%.0fs (wall=%ds) iter=%d case=%d\n', ...
                toc(tStart), max(caseDur(max(1,end-4):end)), cfg.wallSeconds, st.iter, st.pend.c);
            stopRun = true; break;
        end
        tCase = tic; c = st.pend.c;
        [rew, sur, cLdata, st] = paired_rollout(teacher, lqr, cases(st.pend.idx(c)), sur, st, cfg);
        st.pend.rewards(c) = rew; st.pend.cL = [st.pend.cL; cLdata];
        st.pend.c = c + 1; caseDur(end+1) = toc(tCase); st.pend.dur = st.pend.dur + caseDur(end); %#ok<AGROW>
        if toc(lastCkpt) > cfg.checkpointEverySec
            save_checkpoint(ckptPath, sac, sur, st); lastCkpt = tic;
        end
    end
    if stopRun, break; end
    % --- iteration complete: SAC update -------------------------------------------
    reward = mean(st.pend.rewards);
    sac = sac_update(sac, st.pend.a, reward, cfg);     % 1-step bandit SAC
    st.lastReward = reward; st.lastQ = diag(Q).'; st.lastR = diag(R).';
    st.cL = st.pend.cL; iterTime = st.pend.dur; st.pend = [];
    if mod(st.iter, cfg.logEvery)==0
        fprintf('iter=%d reward=%.4f alpha=%.3g samples=%d elapsed=%.0fs iter_time=%.0fs\n', ...
            st.iter, reward, sac.alpha, st.totalSamples, toc(tStart), iterTime);
    end
    save_checkpoint(ckptPath, sac, sur, st); lastCkpt = tic;
    if cfg.ckptEvery > 0 && mod(st.iter, cfg.ckptEvery) == 0
        save_milestone(cfg, lqr, cases, sac, sur, st);
    end
end
save_checkpoint(ckptPath, sac, sur, st);             % main checkpoint FIRST (safe)
fprintf('D1_JOINT_DONE iters=%d pending_case=%d samples=%d last_reward=%.4f\n', ...
    st.iter, pending_case(st), st.totalSamples, st.lastReward);
% ---- phase B: train c_S (surrogate closed-loop alpha=1) + c_LQR (logistic) ---
% Guarded so a failure never loses the SAC/surrogate/residual checkpoint.
try
    consolidate_confidence(cfg, lqr, cases, sur, st);
catch ME
    fprintf('CONF_FAIL %s (main ckpt intact; re-run with D1_CONSOLIDATE=1)\n', ME.message);
end
end

% ============================================================================
function theta = perturb_plant(nom, scale, seed)
% Off-nominal FLIGHT plant for the diagnostic robustness test: nominal parameters scaled
% by (1 + scale*xi.*rho), xi deterministic from seed, rho = step1 train uncertainty
% (14x1). scale<=0 returns the nominal plant unchanged.
theta = nom;
if scale <= 0, return; end
rho = step1_plant_config().uncertainty.train.rho;   % 14x1
rs  = RandStream('twister', 'Seed', seed);          % independent of the global rng
xi  = 2*rand(rs, 14, 1) - 1;
f   = 1 + scale * xi .* rho;
theta.m        = nom.m * f(1);
theta.J        = diag(diag(nom.J) .* f(2:4));        % scale diagonal inertia
theta.Dv       = nom.Dv(:)       .* f(5:7);
theta.Domega   = nom.Domega(:)   .* f(8:10);
theta.alphaT   = nom.alphaT * f(11);
theta.alphaTau = nom.alphaTau(:) .* f(12:14);
end

% ---- paired NMPC + LQR rollout on one case ---------------------------------
function [reward, sur, cLdata, st] = paired_rollout(teacher, lqr, kase, sur, st, cfg)
% Teacher copy and LQR copy of the nominal plant fly the same reference in the same wind
% realization. The teacher is told the current wind force (privileged); the residual
% label Du* = u_teacher - sat(u_LQR) at the teacher state is streamed with the history
% feature the deployed surrogate sees (no wind information), usable solves only.
Ts = cfg.Ts; theta = cfg.plant.nominal; uh = cfg.uh;
Xref = kase.Xref; T = d1_case_len(Xref, cfg);
xN = Xref(:,1); xL = Xref(:,1); uprev = uh; h = d1_hist_init(xN, cfg);
posErrN = zeros(T,1); duAcc = 0; cViol = 0; prevU = uh;
tCase = tic; nSt = [0 0 0];                          % teacher steps: converged / max-iter / unusable
nDivN = 0; nDivL = 0;                                % divergence restarts (teacher copy / LQR copy)
EL = zeros(12, T+1); EL(:,1) = xL - Xref(:,1);       % LQR error traj for c_L
d1_teacher_reset(teacher, Xref, 1, cfg);             % clean solver memory for EVERY case
if cfg.windOn, ds = d1_sample_wind(cfg, T); else, ds = []; end   % same wind for both copies

for k = 1:T
    t = (k-1)*Ts;
    % ---- NMPC teacher branch (privileged current wind force) ------------------
    F = d1_wind_now(ds, t, xN, uprev, theta);
    [uN, status, usable] = d1_teacher_step(teacher, xN, uprev, Xref, k, F, cfg);
    uN = d1_sat(uN, cfg); uprev = uN;
    nSt = nSt + [status == 0, status == 2, ~usable];
    if usable
        feat = d1_hist_feature(h, Xref(:, k:k+10));
        if all(isfinite(feat))
            uLk = d1_sat(uh - lqr.K*(xN - Xref(:,k)), cfg);
            sur = surrogate_stream_update(sur, feat, uN - uLk, st.teacherVersion, cfg);
            st.totalSamples = st.totalSamples + 1;
        end
    end
    [xNnext, divN] = d1_plant_step(t, xN, uN, Ts, theta, ds);
    duAcc = duAcc + sum((uN-prevU).^2); prevU = uN;
    if divN
        % diverged: charge this step the capped error + an attitude violation, then
        % restart this plant copy ON the reference (history, actuator state and solver
        % reset) so the case always runs its full length
        nDivN = nDivN + 1; posErrN(k) = 5; cViol = cViol + 1;
        xN = Xref(:,k+1); uprev = uh; prevU = uh; h = d1_hist_init(xN, cfg);
        d1_teacher_reset(teacher, Xref, k+1, cfg);
    else
        h = d1_hist_push(h, uN, xNnext, cfg); xN = xNnext;
        posErrN(k) = norm(xN(1:3) - Xref(1:3,k+1));
        cViol = cViol + any(abs(xN(4:5)) > 1.35);
    end
    % ---- LQR paired branch (own plant copy) --------------------------------
    uL = d1_sat(uh - lqr.K*(xL - Xref(:,k)), cfg);
    [xL, divL] = d1_plant_step(t, xL, uL, Ts, theta, ds);
    if divL
        nDivL = nDivL + 1; EL(:,k+1) = NaN;          % NaN breaks every contraction window across it
        xL = Xref(:,k+1);                            % restart the LQR copy on the reference
    else
        EL(:,k+1) = xL - Xref(:,k+1);
    end
end
% reward from the teacher's KPIs over the FULL case (negative cost). posErr capped at
% 5 m/step; a divergence step is charged the cap and the copy restarts on the reference.
% failRate = share of UNUSABLE solves (same definition as the apply / label rule).
pe = min(posErrN, 5);
posRmse = sqrt(mean(pe.^2));
failRate = nSt(3)/T;
reward = -(posRmse + 0.01*sqrt(duAcc/T) + 0.5*(cViol/T) + 5*failRate);
% c_L contraction data from LQR error trajectory
try
    outL = d1_finite_horizon_contraction(EL, lqr.P, cfg.H, struct());
    cLdata = outL.g_H(isfinite(outL.g_H))';
catch
    cLdata = [];
end
if isempty(cLdata), cLdata = zeros(0,1); end
fprintf('  CASE %-26s steps=%4d conv=%.2f maxit=%.2f unusable=%.2f restarts NMPC=%d LQR=%d t=%.0fs\n', ...
    kase.groupId, T, nSt/T, nDivN, nDivL, toc(tCase));
end

% ---- surrogate (2-head: residual Delta_u + confidence c_S) ------------------
function net = build_two_head_net(cfg)
% shared trunk 208-128-128-128, two heads: du (4,tanh) and cs (1,sigmoid).
lg = layerGraph();
trunk = [featureInputLayer(208,'Name','in','Normalization','none')
         fullyConnectedLayer(cfg.surHidden,'Name','t1'); swishLayer('Name','s1')
         fullyConnectedLayer(cfg.surHidden,'Name','t2'); swishLayer('Name','s2')
         fullyConnectedLayer(cfg.surHidden,'Name','t3'); swishLayer('Name','s3')];
lg = addLayers(lg, trunk);
lg = addLayers(lg, [fullyConnectedLayer(4,'Name','du_fc'); tanhLayer('Name','du')]);
lg = addLayers(lg, [fullyConnectedLayer(1,'Name','cs_fc'); sigmoidLayer('Name','cs')]);
lg = connectLayers(lg, 's3', 'du_fc');
lg = connectLayers(lg, 's3', 'cs_fc');
net = dlnetwork(lg);
end

function sur = init_surrogate(cfg)
sur.net = build_two_head_net(cfg);
sur.avgG = []; sur.avgSqG = []; sur.step = 0;      % adam state (du/trunk in phase A)
sur.avgC = []; sur.avgSqC = []; sur.stepC = 0;     % adam state (cs head in phase B)
sur.featScale = d1_feature_scale(cfg);
sur.resHalf = cfg.resHalf;                          % residual normalization
% Delta_u residual buffer (phase A, teacher-driven)
sur.buf.feat = zeros(208, cfg.surBufferCap, 'single');
sur.buf.tgt  = zeros(4,  cfg.surBufferCap, 'single');
sur.buf.ver  = zeros(1,  cfg.surBufferCap);
sur.buf.n = 0; sur.buf.pos = 0; sur.cap = cfg.surBufferCap;
end

function sur = surrogate_stream_update(sur, feat, duResidual, ver, cfg)
% Stream a RESIDUAL label Delta_u* = u_teacher - sat(u_LQR) (same state) to the du head.
sur.buf.pos = mod(sur.buf.pos, sur.cap) + 1;
tgt = duResidual ./ sur.resHalf;                     % normalize residual
sur.buf.feat(:,sur.buf.pos) = single(feat ./ sur.featScale);
sur.buf.tgt(:,sur.buf.pos)  = single(min(max(tgt,-1),1));
sur.buf.ver(sur.buf.pos) = ver;
sur.buf.n = min(sur.buf.n + 1, sur.cap);
if sur.buf.n < cfg.surBatch, return; end
idx = sample_recent(sur.buf, ver, cfg);
Xb = dlarray(sur.buf.feat(:,idx), 'CB');
Yb = dlarray(sur.buf.tgt(:,idx), 'CB');
[grad, loss] = dlfeval(@du_loss, sur.net, Xb, Yb);
if ~grads_finite(grad, loss)                          % never let one bad batch NaN the net
    sur.nSkipped = getfield_or(sur, 'nSkipped', 0) + 1; return;
end
sur.step = sur.step + 1;
[sur.net, sur.avgG, sur.avgSqG] = adamupdate(sur.net, grad, ...
    sur.avgG, sur.avgSqG, sur.step, cfg.surLR);
end

function idx = sample_recent(buf, ver, cfg)
n = buf.n; k = cfg.surBatch;
recentMask = find(buf.ver(1:n) >= ver-2 & buf.ver(1:n) > 0);
nRecent = round(cfg.surRecentFrac*k);
if numel(recentMask) >= nRecent && ~isempty(recentMask)
    idx1 = recentMask(randi(numel(recentMask),1,nRecent));
    idx2 = randi(n, 1, k-nRecent);
    idx = [idx1, idx2];
else
    idx = randi(n, 1, k);
end
end

function [grad, loss] = du_loss(net, X, Y)
Yhat = forward(net, X, 'Outputs', 'du');
loss = mean((Yhat - Y).^2, 'all');
grad = dlgradient(loss, net.Learnables);
end
function [grad, loss] = cs_loss(net, X, S)
csHat = forward(net, X, 'Outputs', 'cs');
loss = mean((csHat - S).^2, 'all');
grad = dlgradient(loss, net.Learnables);
end
function tf = grads_finite(grad, loss)
% true when the loss and every gradient entry are finite
tf = isfinite(double(extractdata(loss)));
for i = 1:height(grad)
    if ~tf, return; end
    tf = all(isfinite(extractdata(grad.Value{i})), 'all');
end
end
function v = getfield_or(s, f, dflt)
if isfield(s, f), v = s.(f); else, v = dflt; end
end
function grad = keep_layers(grad, names)
% zero gradients of every learnable NOT in `names` (freeze those params)
for i = 1:height(grad)
    li = grad.Layer(i); if iscell(li), li = li{1}; end
    if ~ismember(char(string(li)), names)
        grad.Value{i} = 0*grad.Value{i};
    end
end
end

% ---- manual SAC (1-step bandit; per-rollout Q,R tuner) ----------------------
function sac = init_sac(cfg)
d = cfg.actionDim;
sac.mu = dlarray(zeros(d,1)); sac.logStd = dlarray(-0.5*ones(d,1));
sac.q1 = q_net(d); sac.q2 = q_net(d);
sac.logAlpha = dlarray(0);
sac.avgA=[]; sac.avgSqA=[]; sac.aStd=[]; sac.aSqStd=[];
sac.avg1=[]; sac.avgSq1=[]; sac.avg2=[]; sac.avgSq2=[];
sac.avgAl=[]; sac.avgSqAl=[]; sac.step=0;
sac.alpha = 1.0;
sac.buf.a = zeros(d, cfg.sacBufferCap); sac.buf.r = zeros(1, cfg.sacBufferCap);
sac.buf.n = 0; sac.buf.pos = 0; sac.cap = cfg.sacBufferCap;
end

function net = q_net(d)
lg = [featureInputLayer(d,'Normalization','none')
      fullyConnectedLayer(64); reluLayer
      fullyConnectedLayer(64); reluLayer
      fullyConnectedLayer(1)];
net = dlnetwork(lg);
end

function a = sac_sample_action(sac, ~)
mu = extractdata(sac.mu); std = exp(extractdata(sac.logStd));
a = tanh(mu + std.*randn(size(mu)));
end

function sac = sac_update(sac, a, r, cfg)
% store
sac.buf.pos = mod(sac.buf.pos, sac.cap) + 1;
sac.buf.a(:,sac.buf.pos) = a(:); sac.buf.r(sac.buf.pos) = r;
sac.buf.n = min(sac.buf.n+1, sac.cap);
if sac.buf.n < min(64, cfg.sacBatch), return; end
k = min(cfg.sacBatch, sac.buf.n);
idx = randi(sac.buf.n, 1, k);
Ab = dlarray(sac.buf.a(:,idx), 'CB'); Rb = dlarray(sac.buf.r(idx), 'CB');
sac.step = sac.step + 1;
% critic update: target = r (1-step, done)
[g1, g2] = dlfeval(@critic_loss, sac.q1, sac.q2, Ab, Rb);
[sac.q1, sac.avg1, sac.avgSq1] = adamupdate(sac.q1, g1, sac.avg1, sac.avgSq1, sac.step, cfg.sacLR);
[sac.q2, sac.avg2, sac.avgSq2] = adamupdate(sac.q2, g2, sac.avg2, sac.avgSq2, sac.step, cfg.sacLR);
% actor + alpha update (state-independent squashed Gaussian)
[gmu, gstd, gAl] = dlfeval(@actor_loss, sac.mu, sac.logStd, ...
    sac.logAlpha, sac.q1, sac.q2, cfg.actionDim, cfg.sacTargetEntropy);
[sac.mu, sac.avgA, sac.avgSqA] = adamupdate(sac.mu, gmu, sac.avgA, sac.avgSqA, sac.step, cfg.sacLR);
[sac.logStd, sac.aStd, sac.aSqStd] = adamupdate(sac.logStd, gstd, sac.aStd, sac.aSqStd, sac.step, cfg.sacLR);
[sac.logAlpha, sac.avgAl, sac.avgSqAl] = adamupdate(sac.logAlpha, gAl, sac.avgAl, sac.avgSqAl, sac.step, cfg.sacLR);
sac.alpha = exp(extractdata(sac.logAlpha));
end

function [g1, g2] = critic_loss(q1, q2, A, R)
y = R;
q1v = forward(q1, A); q2v = forward(q2, A);
l1 = mean((q1v - y).^2, 'all'); l2 = mean((q2v - y).^2, 'all');
g1 = dlgradient(l1, q1.Learnables, 'RetainData', true);
g2 = dlgradient(l2, q2.Learnables);
end

function [gmu, gstd, gAl, ent] = actor_loss(mu, logStd, logAlpha, q1, q2, d, targetEnt)
eps = randn(d,1);
std = exp(logStd);
preTanh = mu + std.*eps;
a = tanh(preTanh);
% log prob of squashed gaussian
logp = sum(-0.5*((preTanh-mu)./std).^2 - logStd - 0.5*log(2*pi)) ...
       - sum(log(1 - a.^2 + 1e-6));
alpha = exp(logAlpha);
Ab = dlarray(a, 'CB');
qmin = min(forward(q1, Ab), forward(q2, Ab));
actorLoss = alpha*logp - qmin;
[gmu, gstd] = dlgradient(actorLoss, mu, logStd, 'RetainData', true);
alphaLoss = -logAlpha*(logp + targetEnt);
gAl = dlgradient(alphaLoss, logAlpha);
ent = -logp;
end

% ---- checkpoint -------------------------------------------------------------
function save_checkpoint(path, sac, sur, st)
rngState = rng;
save(path, 'sac', 'sur', 'st', 'rngState', '-v7.3');
fprintf('CHECKPOINT saved iter=%d samples=%d -> %s\n', st.iter, st.totalSamples, path);
end

function c = pending_case(st)
% next case index of a partly done SAC iteration (0 = none pending)
if isfield(st, 'pend') && ~isempty(st.pend), c = st.pend.c; else, c = 0; end
end

function save_milestone(cfg, lqr, cases, sac, sur, st)
% Frozen copy at an exact SAC iteration (checkpoint_seed<s>_iter<NNNN>.mat) plus its
% own deployed pair (conf_seed<s>_iter<NNNN>.mat). Consolidation runs on a COPY of the
% surrogate and the training RNG stream is restored, so training is unaffected.
tag = sprintf('seed%d_iter%04d', cfg.seed, st.iter);
p = fullfile(cfg.runDir, ['checkpoint_' tag '.mat']);
save_checkpoint(p, sac, sur, st);
r0 = rng;
try
    consolidate_confidence(cfg, lqr, cases, sur, st, ['conf_' tag '.mat']);
catch ME
    fprintf('MILESTONE_CONF_FAIL iter=%d %s\n', st.iter, ME.message);
end
rng(r0);
fprintf('MILESTONE saved iter=%d -> %s\n', st.iter, p);
end

% ---- diagnostics ------------------------------------------------------------
function diag_flight(cfg, teacher, lqr, cases)
[Q0, R0] = d1_bryson_weights(cfg.plant);
d1_set_teacher_weights(teacher, Q0, R0, cfg);
sel = unique(round(linspace(1, numel(cases), min(4, numel(cases)))));
D = struct('groupId',{},'Xref',{},'xN',{},'xL',{},'ok',{},'restarts',{});
for ci = 1:numel(sel)
    kase = cases(sel(ci));
    [Xr, xN, xL, okv, nDiv] = fly_case(teacher, lqr, kase, cfg);
    D(ci).groupId = kase.groupId; D(ci).Xref = Xr;
    D(ci).xN = xN; D(ci).xL = xL; D(ci).ok = okv; D(ci).restarts = nDiv;
    peN = d1_track_err(xN, Xr); peL = d1_track_err(xL, Xr);
    fprintf(['DIAG %s: NMPC posErr med=%.2f max=%.2f okRate=%.2f restarts=%d | ' ...
        'LQR posErr med=%.2f max=%.2f restarts=%d\n'], kase.groupId, median(peN), max(peN), ...
        mean(okv), nDiv(1), median(peL), max(peL), nDiv(2));
end
if ~isfolder(cfg.runDir), mkdir(cfg.runDir); end
save(fullfile(cfg.runDir, sprintf('diag_seed%d.mat', cfg.seed)), 'D', '-v7.3');
fprintf('DIAG_DONE saved %d cases\n', numel(D));
end

function [Xr, xNt, xLt, okv, nDiv] = fly_case(teacher, lqr, kase, cfg)
% Teacher (Bryson weights set by the caller, privileged wind) and LQR on the nominal
% plant with one training-wind realization, full length, common divergence/restart rule.
Ts = cfg.Ts; theta = cfg.plant.nominal; uh = cfg.uh;
Xref = kase.Xref; T = d1_case_len(Xref, cfg);
Xr = Xref(:, 2:T+1);                                 % state after step k vs ref k+1
xNt = nan(12, T); xLt = nan(12, T); okv = zeros(1, T); nDiv = [0 0];
if cfg.windOn, ds = d1_sample_wind(cfg, T); else, ds = []; end
% NMPC teacher branch
xN = Xref(:,1); uprev = uh;
d1_teacher_reset(teacher, Xref, 1, cfg);
for k = 1:T
    t = (k-1)*Ts; F = d1_wind_now(ds, t, xN, uprev, theta);
    [uN, status] = d1_teacher_step(teacher, xN, uprev, Xref, k, F, cfg);
    uN = d1_sat(uN, cfg); uprev = uN; okv(k) = (status == 0);
    [xN, div] = d1_plant_step(t, xN, uN, Ts, theta, ds);
    if div
        nDiv(1) = nDiv(1) + 1; xN = Xref(:,k+1); uprev = uh;
        d1_teacher_reset(teacher, Xref, k+1, cfg);
    else
        xNt(:,k) = xN;
    end
end
% LQR branch (same wind realization)
xL = Xref(:,1);
for k = 1:T
    uL = d1_sat(uh - lqr.K*(xL - Xref(:,k)), cfg);
    [xL, div] = d1_plant_step((k-1)*Ts, xL, uL, Ts, theta, ds);
    if div, nDiv(2) = nDiv(2) + 1; xL = Xref(:,k+1); else, xLt(:,k) = xL; end
end
end

function surrogate_eval(cfg, lqr, cases, sur)
% Fly the surrogate at alpha=1 (u = sat(sat(u_LQR) + Delta_u_hat)); report tracking +
% mean predicted c_S per case.
Ts = cfg.Ts; theta = cfg.plant.nominal; uh = cfg.uh;
sel = unique(round(linspace(1, numel(cases), min(6, numel(cases)))));
D = struct('groupId',{},'Xref',{},'xS',{},'posErr',{},'meanCS',{},'restarts',{});
for ci = 1:numel(sel)
    kase = cases(sel(ci)); Xref = kase.Xref; T = d1_case_len(Xref, cfg);
    xS = Xref(:,1); h = d1_hist_init(xS, cfg);
    xSt = nan(12,T); csAll = nan(1,T); nDiv = 0;
    if cfg.windOn, ds = d1_sample_wind(cfg, T); else, ds = []; end
    for k = 1:T
        feat = d1_hist_feature(h, Xref(:,k:k+10));
        du = zeros(4,1);
        if all(isfinite(feat))
            du = d1_sur_predict_du(sur, feat); csAll(k) = d1_sur_predict_cs(sur, feat);
            if ~all(isfinite(du)), du = zeros(4,1); end
        end
        u = d1_sat(d1_sat(uh - lqr.K*(xS - Xref(:,k)), cfg) + du, cfg);
        [xNew, div] = d1_plant_step((k-1)*Ts, xS, u, Ts, theta, ds);
        if div
            nDiv = nDiv + 1; xS = Xref(:,k+1); h = d1_hist_init(xS, cfg);
        else
            h = d1_hist_push(h, u, xNew, cfg); xS = xNew; xSt(:,k) = xS;
        end
    end
    pe = d1_track_err(xSt, Xref(:,2:T+1));
    mcs = mean(csAll(isfinite(csAll)));
    D(ci).groupId = kase.groupId; D(ci).Xref = Xref(:,2:T+1); D(ci).xS = xSt;
    D(ci).posErr = pe; D(ci).meanCS = mcs; D(ci).restarts = nDiv;
    fprintf('SURR %s: posErr med=%.3f max=%.3f restarts=%d (T=%d) | meanCS=%.2f\n', ...
        kase.groupId, median(pe), max(pe), nDiv, T, mcs);
end
if ~isfolder(cfg.runDir), mkdir(cfg.runDir); end
save(fullfile(cfg.runDir, sprintf('surr_eval_seed%d.mat', cfg.seed)), 'D', '-v7.3');
fprintf('SURR_EVAL_DONE %d cases\n', numel(D));
end

% ---- 4-controller comparison flight ----------------------------------------
function sel = family_cases(cases)
% One case per trajectory family: the middle case of the family block, or the hardest
% (max speed, then max acceleration) with D1_HARD=1. Shared by compare and gate grid.
fams = cell(1, numel(cases));
for i = 1:numel(cases)
    g = cases(i).groupId; p = find(g=='|', 1); fams{i} = g(1:p-1);
end
uf = unique(fams, 'stable');
hard = strcmp(d1_getenv_str('D1_HARD','0'), '1');
sel = zeros(1, numel(uf));
for f = 1:numel(uf)
    ids = find(strcmp(fams, uf{f}));
    if hard
        sc = zeros(numel(ids),1);
        for t = 1:numel(ids)
            tk = regexp(cases(ids(t)).groupId, 'v([\d.]+)\|a([\d.]+)', 'tokens', 'once');
            sc(t) = str2double(tk{1})*100 + str2double(tk{2});   % speed dominates accel
        end
        [~, jj] = max(sc); sel(f) = ids(jj);
    else
        sel(f) = ids(max(1, round(numel(ids)/2)));
    end
end
end

function compare_flight(cfg, teacher, lqr, cases, sur, sac, conf)
% One representative case per trajectory FAMILY; fly all controllers on the SAME
% reference, plant and wind realization and dump full trajectories + tracking error:
%   (a) LQR-only        u = sat(uh - K e)
%   (b) teacher NMPC    SAC-NMPC (deterministic SAC policy a = tanh(mu)), privileged
%                       wind, d1_teacher_step, no LQR fallback
%   (c) surrogate alpha=1  u = sat(sat(u_LQR) + Delta_u_hat)  (full residual, no gate)
%   (d) proposed blend     d1_blend_control (alpha = c_S * g_L(c_LQR))
sel = family_cases(cases);
aMean = tanh(extractdata(sac.mu));
[Qt, Rt] = d1_action_to_QR(aMean, cfg); d1_set_teacher_weights(teacher, Qt, Rt, cfg);
fprintf('COMPARE teacher weights = SAC-NMPC (a=tanh(mu)); diag(Q)=[%s]\n', ...
    strtrim(sprintf('%.3g ', diag(Qt))));
haveConf = ~isempty(conf) && isfield(conf,'LQR') && ~isempty(conf.LQR.w);
fprintf('COMPARE blend: u=sat(sat(u_LQR)+alpha*Du), alpha=c_S*g_L(c_LQR); c_LQR %s, alphaSafe=%d\n', ...
    ternary(haveConf, 'trained', 'MISSING (g_L=1, alpha=c_S)'), cfg.alphaSafe);
% Flight plant: nominal, or OFF-NOMINAL (same perturbed plant for ALL controllers;
% LQR gain and the teacher model stay nominal-designed -> a fair robustness test).
pscale = d1_getenv_num('D1_PLANT_PERTURB', 0);
testTheta = perturb_plant(cfg.plant.nominal, pscale, cfg.seed);
fprintf('COMPARE flight plant = %s (perturb scale %.2f x train rho)\n', ...
    ternary(pscale > 0, 'OFF-NOMINAL', 'nominal'), pscale);
D = struct('groupId',{},'family',{},'Xref',{},'xL',{},'xN',{},'xS',{},'xB',{}, ...
    'peL',{},'peN',{},'peS',{},'peB',{},'alpha',{},'okN',{},'restarts',{});
for ci = 1:numel(sel)
    kase = cases(sel(ci));
    [Xr, xL, xN, xS, xB, okN, alp, nDiv] = fly_compare(teacher, lqr, sur, kase, cfg, conf, testTheta);
    peL = d1_track_err(xL, Xr); peN = d1_track_err(xN, Xr);
    peS = d1_track_err(xS, Xr); peB = d1_track_err(xB, Xr);
    D(ci).groupId = kase.groupId; D(ci).family = kase.groupId(1:find(kase.groupId=='|',1)-1);
    D(ci).Xref = Xr; D(ci).xL = xL; D(ci).xN = xN; D(ci).xS = xS; D(ci).xB = xB;
    D(ci).peL = peL; D(ci).peN = peN; D(ci).peS = peS; D(ci).peB = peB;
    D(ci).alpha = alp; D(ci).okN = okN; D(ci).restarts = nDiv;
    fprintf(['CMP %-18s | LQR rmse=%.3f max=%.3f | NMPC rmse=%.3f max=%.3f | ' ...
        'SUR rmse=%.3f max=%.3f | BLEND rmse=%.3f max=%.3f | meanA=%.2f okN=%.2f | ' ...
        'restarts L/N/S/B=%d/%d/%d/%d\n'], ...
        kase.groupId, rmse_(peL), max(peL), rmse_(peN), max(peN), ...
        rmse_(peS), max(peS), rmse_(peB), max(peB), mean(alp(isfinite(alp))), mean(okN), nDiv);
end
if ~isfolder(cfg.runDir), mkdir(cfg.runDir); end
save(fullfile(cfg.runDir, sprintf('compare_seed%d.mat', cfg.seed)), 'D', '-v7.3');
fprintf('COMPARE_DONE %d families\n', numel(D));
end

function r = rmse_(pe)
r = sqrt(mean(pe.^2));
end
function s = ternary(c, a, b); if c, s = a; else, s = b; end; end

function [Xr, xL, xN, xS, xB, okN, alphaTraj, nDiv] = fly_compare(teacher, lqr, sur, kase, cfg, conf, theta)
% Four controllers on the SAME reference, plant and training-wind realization; common
% flight rules (full length, restart on the reference after divergence).
Ts = cfg.Ts; uh = cfg.uh;
Xref = kase.Xref; T = d1_case_len(Xref, cfg);
% x?(:,k) is the state AFTER integrating step k -> compared with Xref(:,k+1)
% (same alignment as the training reward). Diverged steps stay NaN.
Xr = Xref(:, 2:T+1);
xL = nan(12,T); xN = nan(12,T); xS = nan(12,T); xB = nan(12,T);
okN = zeros(1,T); alphaTraj = nan(1,T); nDiv = [0 0 0 0];      % restarts LQR/NMPC/SUR/BLEND
if cfg.windOn, ds = d1_sample_wind(cfg, T); else, ds = []; end

% (a) LQR-only ----------------------------------------------------------------
x = Xref(:,1);
for k = 1:T
    u = d1_sat(uh - lqr.K*(x - Xref(:,k)), cfg);
    [x, div] = d1_plant_step((k-1)*Ts, x, u, Ts, theta, ds);
    if div, nDiv(1) = nDiv(1) + 1; x = Xref(:,k+1); else, xL(:,k) = x; end
end

% (b) teacher = SAC-NMPC (weights set by the caller), privileged wind ----------
x = Xref(:,1); uprev = uh; d1_teacher_reset(teacher, Xref, 1, cfg);
for k = 1:T
    t = (k-1)*Ts; F = d1_wind_now(ds, t, x, uprev, theta);
    [u, status] = d1_teacher_step(teacher, x, uprev, Xref, k, F, cfg); okN(k) = (status == 0);
    u = d1_sat(u, cfg); uprev = u;
    [x, div] = d1_plant_step(t, x, u, Ts, theta, ds);
    if div
        nDiv(2) = nDiv(2) + 1; x = Xref(:,k+1); uprev = uh;
        d1_teacher_reset(teacher, Xref, k+1, cfg);
    else
        xN(:,k) = x;
    end
end

% (c) surrogate at alpha=1: u = sat(sat(u_LQR) + Delta_u_hat) -- full residual, no gate
x = Xref(:,1); h = d1_hist_init(x, cfg);
for k = 1:T
    feat = d1_hist_feature(h, Xref(:,k:k+10));
    du = zeros(4,1);
    if all(isfinite(feat))
        du = d1_sur_predict_du(sur, feat);
        if ~all(isfinite(du)), du = zeros(4,1); end
    end
    u = d1_sat(d1_sat(uh - lqr.K*(x - Xref(:,k)), cfg) + du, cfg);
    [xNew, div] = d1_plant_step((k-1)*Ts, x, u, Ts, theta, ds);
    if div
        nDiv(3) = nDiv(3) + 1; x = Xref(:,k+1); h = d1_hist_init(x, cfg);
    else
        h = d1_hist_push(h, u, xNew, cfg); x = xNew; xS(:,k) = x;
    end
end

% (d) proposed blend (d1_blend_control) -----------------------------------------
x = Xref(:,1); h = d1_hist_init(x, cfg);
for k = 1:T
    e = x - Xref(:,k);
    [u, alphaTraj(k)] = d1_blend_control(uh - lqr.K*e, e, h, Xref(:,k:k+10), sur, conf, lqr, cfg);
    [xNew, div] = d1_plant_step((k-1)*Ts, x, u, Ts, theta, ds);
    if div
        nDiv(4) = nDiv(4) + 1; x = Xref(:,k+1); h = d1_hist_init(x, cfg);
    else
        h = d1_hist_push(h, u, xNew, cfg); x = xNew; xB(:,k) = x;
    end
end
end

% ============================================================================
% GATE GRID: evaluate ONE (c_low, c_high) blend-gate setting on the validation
% families. Teacher-free (no acados): flies only LQR-only and the proposed blend.
% The CI matrix sweeps the (c_low, c_high) grid; each job prints one GATE_RESULT line.
function gate_grid_flight(cfg, lqr, cases, sur, conf)
assert(cfg.cHigh > cfg.cLow, 'gate grid needs c_high > c_low (got %.3f, %.3f).', ...
    cfg.cHigh, cfg.cLow);
haveConf = ~isempty(conf) && isfield(conf,'LQR') && ~isempty(conf.LQR.w);
if ~haveConf
    fprintf('GATE_WARN c_LQR MISSING -> g_L=1 -> alpha = c_S\n');
end
sel = family_cases(cases);
hard = strcmp(d1_getenv_str('D1_HARD','0'), '1');
pscale = d1_getenv_num('D1_PLANT_PERTURB', 0);
theta = perturb_plant(cfg.plant.nominal, pscale, cfg.seed);
fprintf('GATE start c_low=%.3f c_high=%.3f hard=%d pscale=%.2f alphaSafe=%d (%d families)\n', ...
    cfg.cLow, cfg.cHigh, hard, pscale, cfg.alphaSafe, numel(sel));
peLall = []; peBall = []; aAll = [];
for ci = 1:numel(sel)
    kase = cases(sel(ci));
    [peL, peB, alp] = fly_gate(lqr, sur, kase, cfg, conf, theta);
    peLall = [peLall, peL]; peBall = [peBall, peB]; aAll = [aAll, alp]; %#ok<AGROW>
    fprintf('GATE_FAM %-18s | LQR rmse=%.3f | BLEND rmse=%.3f | meanA=%.2f\n', ...
        kase.groupId, rmse_(peL), rmse_(peB), mean(alp(isfinite(alp))));
end
rmseB = rmse_(peBall); rmseL = rmse_(peLall);
fprintf('GATE_RESULT c_low=%.3f c_high=%.3f rmse_blend=%.4f rmse_lqr=%.4f delta=%.4f meanA=%.3f n=%d\n', ...
    cfg.cLow, cfg.cHigh, rmseB, rmseL, rmseB-rmseL, mean(aAll(isfinite(aAll))), numel(sel));
if ~isfolder(cfg.runDir), mkdir(cfg.runDir); end
res = struct('cLow',cfg.cLow,'cHigh',cfg.cHigh,'rmseB',rmseB,'rmseL',rmseL, ...
    'meanA',mean(aAll(isfinite(aAll))),'hard',hard,'pscale',pscale,'seed',cfg.seed);
save(fullfile(cfg.runDir, sprintf('gate_seed%d_cl%.2f_ch%.2f.mat', ...
    cfg.seed, cfg.cLow, cfg.cHigh)), 'res');
end

function [peL, peB, alphaTraj] = fly_gate(lqr, sur, kase, cfg, conf, theta)
% Teacher-free flight for gate tuning: (a) LQR-only, (d) proposed blend -- the same laws
% and flight rules as fly_compare branches (a)/(d).
Ts = cfg.Ts; uh = cfg.uh;
Xref = kase.Xref; T = d1_case_len(Xref, cfg);
Xr = Xref(:, 2:T+1);
xL = nan(12,T); xB = nan(12,T); alphaTraj = nan(1,T);
if cfg.windOn, ds = d1_sample_wind(cfg, T); else, ds = []; end
% (a) LQR-only
x = Xref(:,1);
for k = 1:T
    u = d1_sat(uh - lqr.K*(x - Xref(:,k)), cfg);
    [x, div] = d1_plant_step((k-1)*Ts, x, u, Ts, theta, ds);
    if div, x = Xref(:,k+1); else, xL(:,k) = x; end
end
% (d) proposed blend
x = Xref(:,1); h = d1_hist_init(x, cfg);
for k = 1:T
    e = x - Xref(:,k);
    [u, alphaTraj(k)] = d1_blend_control(uh - lqr.K*e, e, h, Xref(:,k:k+10), sur, conf, lqr, cfg);
    [xNew, div] = d1_plant_step((k-1)*Ts, x, u, Ts, theta, ds);
    if div
        x = Xref(:,k+1); h = d1_hist_init(x, cfg);
    else
        h = d1_hist_push(h, u, xNew, cfg); x = xNew; xB(:,k) = x;
    end
end
peL = d1_track_err(xL, Xr);
peB = d1_track_err(xB, Xr);
end

% ============================================================================
% CONFIDENCE: c_S (surrogate head) and c_LQR = P(next H=20 steps contract | error).
% ============================================================================
function consolidate_confidence(cfg, lqr, cases, sur, st, confName)
% Phase B: train the c_S head (surrogate closed-loop alpha=1 tracking quality) on a COPY
% of the surrogate + the c_LQR logistic (LQR V-contraction), both in the training wind.
% The training checkpoint is NEVER modified. The consolidated surrogate and conf are
% saved TOGETHER in <runDir>/<confName>: that pair is the deployed controller used by
% compare, gate, surr_eval and the final evaluation.
if nargin < 6, confName = sprintf('conf_seed%d.mat', cfg.seed); end
sur = train_cS_head(cfg, lqr, cases, sur);
confLQR = train_cLQR(cfg, lqr, cases);
conf.LQR = confLQR; conf.epsP = cfg.epsP; conf.H = cfg.H;
conf.def = ['c_S=exp(-(RMS_pastH_pos/epsP)^2) predicted by surrogate cs head; ' ...
    'c_LQR=logistic P(V_{k+H}<V_k)'];
conf.iter = st.iter;
save(fullfile(cfg.runDir, confName), 'conf', 'sur', '-v7.3');
fprintf('CONF_DONE c_LQR(acc=%.2f base=%.2f n=%d) epsP=%.3g -> %s\n', ...
    confLQR.acc, confLQR.base, confLQR.n, cfg.epsP, confName);
end

function sur = train_cS_head(cfg, lqr, cases, sur)
% Fly the surrogate closed-loop at alpha=1: u = sat(sat(u_LQR) + Delta_u_hat), in the
% training wind. Collect (feature_k, s_k), s_k = exp(-(RMS pos err over the PAST H
% steps / epsP)^2). Train ONLY the cs head (trunk + du head frozen).
Ts=cfg.Ts; theta=cfg.plant.nominal; H=cfg.H; uh=cfg.uh;
nSel = min(cfg.csCasesPerCall, numel(cases));
sel = unique(round(linspace(1, numel(cases), nSel)));
Zall = zeros(208,0,'single'); Sall = zeros(1,0,'single');
for ci = 1:numel(sel)
    kase = cases(sel(ci)); Xref = kase.Xref; T = d1_case_len(Xref, cfg);
    x = Xref(:,1); h = d1_hist_init(x, cfg);
    feats = nan(208, T, 'single'); perr = nan(1,T);
    if cfg.windOn, ds = d1_sample_wind(cfg, T); else, ds = []; end   % same wind law as training
    for k = 1:T
        feat = d1_hist_feature(h, Xref(:,k:k+10));
        du = zeros(4,1);
        if all(isfinite(feat))
            du = d1_sur_predict_du(sur, feat);
            if ~all(isfinite(du)), du = zeros(4,1); end
            feats(:,k) = single(feat) ./ sur.featScale;
        end
        u = d1_sat(d1_sat(uh - lqr.K*(x - Xref(:,k)), cfg) + du, cfg);   % alpha = 1
        [xNew, div] = d1_plant_step((k-1)*Ts, x, u, Ts, theta, ds);
        if div
            % diverged: charge the 5 m cap, restart on the reference, keep flying
            perr(k) = 5;
            x = Xref(:,k+1); h = d1_hist_init(x, cfg);
        else
            h = d1_hist_push(h, u, xNew, cfg); x = xNew;
            perr(k) = min(norm(x(1:3) - Xref(1:3,k+1)), 5);   % same capped error as d1_track_err
        end
    end
    for k = H:T
        if ~all(isfinite(feats(:,k))), continue; end
        E20 = sqrt(mean(perr(k-H+1:k).^2)); sk = exp(-(E20/cfg.epsP)^2);
        Zall(:,end+1) = feats(:,k); Sall(end+1) = single(sk); %#ok<AGROW>
    end
end
nAll = numel(Sall);
if nAll == 0, fprintf('CS_TRAIN: no samples\n'); return; end
for ep = 1:cfg.csEpochs
    idx = randi(nAll, 1, min(cfg.surBatch, nAll));
    Xb = dlarray(Zall(:,idx),'CB'); Sb = dlarray(Sall(idx),'CB');
    [g, lossC] = dlfeval(@cs_loss, sur.net, Xb, Sb);
    if ~grads_finite(g, lossC), continue; end
    g = keep_layers(g, {'cs_fc'});
    sur.stepC = sur.stepC + 1;
    [sur.net, sur.avgC, sur.avgSqC] = adamupdate(sur.net, g, sur.avgC, sur.avgSqC, sur.stepC, cfg.surLR);
end
pcs = extractdata(predict(sur.net, dlarray(Zall,'CB'),'Outputs','cs'));
fprintf('CS_TRAIN n=%d meanS=%.3f meanPredCS=%.3f\n', nAll, mean(Sall), mean(pcs));
end

function confLQR = train_cLQR(cfg, lqr, cases)
P = lqr.P; H = cfg.H; nSel = min(60, numel(cases));
sel = unique(round(linspace(1, numel(cases), nSel)));
XL = []; yL = [];
for ci = 1:numel(sel)
    EL = rollout_error_lqr(cfg, lqr, cases(sel(ci)));
    [xl, yl] = conf_samples(EL, P, H); XL=[XL,xl]; yL=[yL,yl]; %#ok<AGROW>
end
confLQR = fit_logistic(XL.', yL.');
end

function [X, y] = conf_samples(E, P, H)
% feature (16-dim, d1_conf_feature) + binary contract label per step with a full
% finite window
if isempty(E) || size(E,2) <= H, X = zeros(16,0); y = zeros(1,0); return; end
o = d1_finite_horizon_contraction(E, P, H, struct());
m = o.validMask & o.windowFinite;
F = d1_conf_feature(E);
X = F(:, m); y = double(o.isContracting(m));
end

function clf = fit_logistic(X, y)
% L2 logistic regression with class-balanced weights. X: n x d, y: n x 1 in {0,1}.
if isempty(y)
    clf = struct('w',[],'b',0,'mu',[],'sg',[],'acc',NaN,'base',NaN,'n',0); return;
end
y = y(:); mu = mean(X,1); sg = std(X,0,1) + 1e-6; Z = (X - mu)./sg;
[n, d] = size(Z); w = zeros(d,1); b = 0; lr = 0.5; lam = 1e-3;
p1 = mean(y); wpos = 1/max(p1,1e-3); wneg = 1/max(1-p1,1e-3);
sw = y*wpos + (1-y)*wneg; sw = sw/mean(sw);
for it = 1:800
    p = 1./(1+exp(-(Z*w + b)));
    g = Z.'*((p - y).*sw)/n + lam*w; gb = mean((p - y).*sw);
    w = w - lr*g; b = b - lr*gb;
end
p = 1./(1+exp(-(Z*w + b)));
clf = struct('w',w,'b',b,'mu',mu,'sg',sg,'acc',mean((p>0.5)==y),'base',mean(y),'n',n);
end

function E = rollout_error_lqr(cfg, lqr, kase)
% LQR closed loop on the nominal plant in the training wind, full length; a divergence
% restarts on the reference and leaves a NaN column (breaks every window across it).
Ts = cfg.Ts; theta = cfg.plant.nominal; Xref = kase.Xref; uh = cfg.uh;
T = d1_case_len(Xref, cfg);
x = Xref(:,1); X = nan(12,T);
if cfg.windOn, ds = d1_sample_wind(cfg, T); else, ds = []; end   % same wind law as training
for k = 1:T
    u = d1_sat(uh - lqr.K*(x - Xref(:,k)), cfg);
    [x, div] = d1_plant_step((k-1)*Ts, x, u, Ts, theta, ds);
    if div
        X(:,k) = NaN;
        x = Xref(:,k+1);                             % restart on the reference, run the full case
    else
        X(:,k) = x;
    end
end
E = X - Xref(:,2:T+1);                               % align state_k with ref_{k+1}
end
