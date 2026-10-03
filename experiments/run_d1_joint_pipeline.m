function run_d1_joint_pipeline()
%RUN_D1_JOINT_PIPELINE One-seed D1 pipeline.
%   SAC phase (default): SAC tunes the NMPC Q,R of the scenario-NMPC teacher (M=5, N=20,
%   Nc=5, privileged wind) on paired teacher/LQR rollouts; checkpoints and milestones hold
%   the SAC state (= the teacher). No student is trained here.
%   DAgger phase (D1_DAGGER=1): the teacher of checkpoint checkpoint_seed<s><D1_CKPT_SUFFIX>
%   is frozen and the linear student Du = W*phi is learned by DAgger (d1_dagger_run), then
%   its confidences; output student_seed<s><suffix>.mat = the deployed controller.
% Method document: docs/D1_method.tex. Shared definitions: src/joint/d1_*.m (also used by
% experiments/d1_final_eval.m).
%
% Resumable: the SAC checkpoint (saved after every iteration) holds SAC, counters and the
% RNG; the DAgger state file (saved after every DAgger iteration) holds the data
% statistics, candidates and the RNG. Milestone SAC checkpoints every D1_CKPT_EVERY_ITER
% iterations (1 = every iteration). The flights of one iteration run in parallel on
% D1_WORKERS local workers (d1_teacher_pool); their winds are drawn by the client in order.
% All flights use the common flight rules (d1_plant_step, d1_case_len, d1_teacher_step,
% d1_track_err) and the random training wind (d1_sample_wind).
%
% Env: D1_SEED, D1_RUN_DIR, D1_WALL_SECONDS, D1_RESUME ('1' to resume), D1_STOP_ITER,
% D1_CKPT_EVERY_ITER, D1_WIND, D1_SOLVER, D1_RANDOM_QR, D1_WORKERS, D1_CKPT_SUFFIX, ...
% Modes: D1_DAGGER, D1_DIAG, D1_SURR_EVAL, D1_CONSOLIDATE, D1_COMPARE, D1_GATE_GRID.
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
% Teacher-free modes never touch the NMPC teacher and skip the acados build.
noTeacher = strcmp(d1_getenv_str('D1_SURR_EVAL','0'),'1') || ...
            strcmp(d1_getenv_str('D1_CONSOLIDATE','0'),'1') || ...
            strcmp(d1_getenv_str('D1_GATE_GRID','0'),'1');
scen = d1_sample_scenarios(cfg);            % M=5 uncertain plant realizations (first rng draw)
lqr = d1_build_lqr(cfg);                    % K, P (Bryson LQR + Riccati)
cases = d1_train_cases(cfg);                % 120 cases, 1000-sample references (+ Uref)
if noTeacher
    teacher = [];
    fprintf('teacher SKIPPED (teacher-free mode) + LQR(P minEig=%.4g) + %d cases\n', ...
        min(eig(lqr.P)), numel(cases));
else
    teacher = d1_teacher_build_solver(cfg, scen);
    fprintf('built teacher(acados M=%d Nc=%d, privileged wind) + LQR(P minEig=%.4g) + %d cases\n', ...
        cfg.M, cfg.Nc, min(eig(lqr.P)), numel(cases));
end

% ---- files --------------------------------------------------------------------
ckptPath = fullfile(cfg.runDir, sprintf('checkpoint_seed%d.mat', cfg.seed));        % SAC training
sacPath  = fullfile(cfg.runDir, sprintf('checkpoint_seed%d%s.mat', cfg.seed, cfg.ckptSuffix));
stuPath  = fullfile(cfg.runDir, sprintf('student_seed%d%s.mat', cfg.seed, cfg.ckptSuffix));
dagPath  = fullfile(cfg.runDir, sprintf('dagger_seed%d%s.mat', cfg.seed, cfg.ckptSuffix));

% ---- modes ----------------------------------------------------------------------
if strcmp(d1_getenv_str('D1_DIAG','0'), '1')
    diag_flight(cfg, teacher, lqr, cases); return;
end
if strcmp(d1_getenv_str('D1_DAGGER','0'), '1')
    % learn the linear student from the frozen teacher of sacPath (resumable)
    assert(isfile(sacPath), 'DAgger requires the SAC checkpoint %s', sacPath);
    S = load(sacPath);
    d1_dagger_run(cfg, teacher, scen, lqr, cases, S, dagPath, stuPath); return;
end
if strcmp(d1_getenv_str('D1_CONSOLIDATE','0'), '1')
    % recompute the confidences of an existing student (teacher-free)
    [stu, ~] = d1_load_deployed(stuPath, true);
    conf = d1_consolidate(cfg, lqr, cases, stu); conf.iter = stu.ckptIter;
    save(stuPath, 'stu', 'conf', '-v7.3'); return;
end
if strcmp(d1_getenv_str('D1_SURR_EVAL','0'), '1')
    [stu, conf] = d1_load_deployed(stuPath, true);
    student_eval(cfg, lqr, cases, stu, conf); return;
end
if strcmp(d1_getenv_str('D1_COMPARE','0'), '1')
    assert(isfile(sacPath), 'compare requires the SAC checkpoint %s', sacPath);
    S = load(sacPath); [stu, conf] = d1_load_deployed(stuPath, true);
    compare_flight(cfg, teacher, lqr, cases, stu, S.sac, conf); return;
end
if strcmp(d1_getenv_str('D1_GATE_GRID','0'), '1')
    [stu, conf] = d1_load_deployed(stuPath, true);
    gate_grid_flight(cfg, lqr, cases, stu, conf); return;
end

% ---- SAC phase: init or resume ---------------------------------------------------
if cfg.resume && isfile(ckptPath)
    S = load(ckptPath); sac = S.sac; st = S.st;
    rng(S.rngState);
    fprintf('RESUMED from %s at iter=%d\n', ckptPath, st.iter);
else
    sac = init_sac(cfg);
    st = struct('iter', 0, 'lastReward', NaN, 'pend', []);
    fprintf('FRESH start\n');
end

% ---- open-ended SAC loop (checkpoint after every iteration) ---------------------
% One SAC iteration = casesPerEval paired rollouts with one sampled Q,R. The client draws
% the action, the case indices and the wind of every case IN ORDER from the global random
% stream, then the rollouts run in parallel on local workers (d1_teacher_pool); results
% are gathered in case order. The checkpoint (SAC, counters, RNG) is saved after every
% iteration; an iteration predicted to overrun the wall budget is not started.
[nW, teacherC] = d1_teacher_pool(cfg, scen, teacher);
tStart = tic; iterDur = [];
wallReserve = 120;                                   % s kept for the final save
if ~isfield(st, 'pend'), st.pend = []; end
while true
    if isempty(st.pend)
        if cfg.stopIter > 0 && st.iter >= cfg.stopIter
            fprintf('STOP_ITER reached iter=%d (target %d)\n', st.iter, cfg.stopIter);
            break;
        end
        if ~isempty(iterDur) && toc(tStart) + max(iterDur(max(1,end-2):end)) + wallReserve > cfg.wallSeconds
            fprintf('WALL_STOP elapsed=%.0fs next-iteration est=%.0fs (wall=%ds) iter=%d\n', ...
                toc(tStart), max(iterDur(max(1,end-2):end)), cfg.wallSeconds, st.iter);
            break;
        end
        st.iter = st.iter + 1;
        a = sac_sample_action(sac, cfg);               % candidate in R^6 (log10 multipliers)
        idx = randi(numel(cases), 1, cfg.casesPerEval);
        ds = cell(cfg.casesPerEval, 1);                % wind of every case, drawn in order
        for cc = 1:cfg.casesPerEval
            if cfg.windOn, ds{cc} = d1_sample_wind(cfg, d1_case_len(cases(idx(cc)).Xref, cfg)); end
        end
        st.pend = struct('a', a, 'idx', idx, 'ds', {ds});
    end
    [Q, R] = d1_action_to_QR(st.pend.a, cfg);
    % --- the casesPerEval paired rollouts of this iteration, in parallel ----------
    tIter = tic; n = cfg.casesPerEval;
    rewards = zeros(n, 1); usable = zeros(n, 1); cL = cell(n, 1);
    kases = cases(st.pend.idx); ds = st.pend.ds;
    parfor (c = 1:n, nW)
        tch = teacherC.Value;                          %#ok<PFBNS> pool Constant: one solver per worker
        d1_set_teacher_weights(tch, Q, R, cfg);
        [rewards(c), usable(c), cL{c}] = paired_rollout(tch, lqr, kases(c), ds{c}, cfg);
    end
    iterDur(end+1) = toc(tIter); %#ok<AGROW>
    % --- iteration complete: SAC update -------------------------------------------
    reward = mean(rewards);
    sac = sac_update(sac, st.pend.a, reward, cfg);     % 1-step bandit SAC
    st.lastReward = reward; st.lastQ = diag(Q).'; st.lastR = diag(R).';
    st.cL = vertcat(cL{:}); st.pend = [];
    if mod(st.iter, cfg.logEvery)==0
        fprintf('iter=%d reward=%.4f alpha=%.3g usable=%.3f elapsed=%.0fs iter_time=%.0fs\n', ...
            st.iter, reward, sac.alpha, mean(usable), toc(tStart), iterDur(end));
    end
    save_checkpoint(ckptPath, sac, st);
    if cfg.ckptEvery > 0 && mod(st.iter, cfg.ckptEvery) == 0
        p = fullfile(cfg.runDir, sprintf('checkpoint_seed%d_iter%04d.mat', cfg.seed, st.iter));
        save_checkpoint(p, sac, st);
        fprintf('MILESTONE saved iter=%d -> %s\n', st.iter, p);
    end
end
save_checkpoint(ckptPath, sac, st);
fprintf('D1_JOINT_DONE iters=%d last_reward=%.4f\n', st.iter, st.lastReward);
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

% ---- paired NMPC + LQR rollout on one case (SAC reward) -----------------------
function [reward, useShare, cLdata] = paired_rollout(teacher, lqr, kase, ds, cfg)
% Teacher copy and LQR copy of the nominal plant fly the same reference in the same wind
% realization ds (drawn by the client in case order; [] = no wind). The teacher is told the current wind force (privileged). The reward is
% computed from the teacher's KPIs over the full case; the LQR copy only provides the
% contraction diagnostics g_H.
Ts = cfg.Ts; theta = cfg.plant.nominal; uh = cfg.uh;
Xref = kase.Xref; T = d1_case_len(Xref, cfg);
xN = Xref(:,1); xL = Xref(:,1); uprev = uh;
posErrN = zeros(T,1); duAcc = 0; cViol = 0; prevU = uh;
tCase = tic; nSt = [0 0 0];                          % converged / max-iter / unusable
tsol = zeros(1,T);
nDivN = 0; nDivL = 0;                                % divergence restarts (teacher copy / LQR copy)
EL = zeros(12, T+1); EL(:,1) = xL - Xref(:,1);       % LQR error traj for g_H
d1_teacher_reset(teacher, Xref, 1, cfg);             % clean solver memory for EVERY case

for k = 1:T
    t = (k-1)*Ts;
    % ---- NMPC teacher branch (privileged current wind force) ------------------
    F = d1_wind_now(ds, t, xN, uprev, theta);
    [uN, status, usable, tsol(k)] = d1_teacher_step(teacher, xN, uprev, Xref, k, F, cfg);
    uN = d1_sat(uN, cfg); uprev = uN;
    nSt = nSt + [status == 0, status == 2, ~usable];
    [xNnext, divN] = d1_plant_step(t, xN, uN, Ts, theta, ds);
    duAcc = duAcc + sum((uN-prevU).^2); prevU = uN;
    if divN
        % diverged: charge this step the capped error + an attitude violation, then
        % restart this plant copy ON the reference (actuator state and solver reset)
        % so the case always runs its full length
        nDivN = nDivN + 1; posErrN(k) = 5; cViol = cViol + 1;
        xN = Xref(:,k+1); uprev = uh; prevU = uh;
        d1_teacher_reset(teacher, Xref, k+1, cfg);
    else
        xN = xNnext;
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
failRate = nSt(3)/T; useShare = 1 - failRate;
reward = -(posRmse + 0.01*sqrt(duAcc/T) + 0.5*(cViol/T) + 5*failRate);
try
    outL = d1_finite_horizon_contraction(EL, lqr.P, cfg.H, struct());
    cLdata = outL.g_H(isfinite(outL.g_H))';
catch
    cLdata = [];
end
if isempty(cLdata), cLdata = zeros(0,1); end
fprintf(['  CASE %-26s steps=%4d conv=%.2f maxit=%.2f unusable=%.2f ' ...
    'tsolve max=%.0fms p99=%.0fms restarts NMPC=%d LQR=%d t=%.0fs\n'], ...
    kase.groupId, T, nSt/T, 1e3*max(tsol), 1e3*prctile(tsol,99), nDivN, nDivL, toc(tCase));
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
a = mu + std.*randn(size(mu));                     % unbounded (log10 multipliers)
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
% actor + alpha update (state-independent Gaussian, unbounded action)
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
a = mu + std.*eps;                                   % unbounded action (no tanh)
% log prob of the diagonal Gaussian
logp = sum(-0.5*((a-mu)./std).^2 - logStd - 0.5*log(2*pi));
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
function save_checkpoint(path, sac, st)
rngState = rng;
save(path, 'sac', 'st', 'rngState', '-v7.3');
fprintf('CHECKPOINT saved iter=%d -> %s\n', st.iter, path);
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
xNt = nan(12, T); okv = zeros(1, T); nDiv = [0 0];
if cfg.windOn, ds = d1_sample_wind(cfg, T); else, ds = []; end
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
R = d1_fly_student([], [], Xref, kase.Uref, ds, theta, lqr, cfg, 'lqr');   % same wind
xLt = R.X; nDiv(2) = R.nDiv;
end

function student_eval(cfg, lqr, cases, stu, conf)
% Fly the student at alpha = 1 and the deployed blend on 6 bank cases (training wind);
% report tracking and the mean predicted c_S / alpha.
theta = cfg.plant.nominal;
sel = unique(round(linspace(1, numel(cases), min(6, numel(cases)))));
D = struct('groupId',{},'Xref',{},'xS',{},'xB',{},'peS',{},'peB',{},'alpha',{},'restarts',{});
for ci = 1:numel(sel)
    kase = cases(sel(ci)); T = d1_case_len(kase.Xref, cfg);
    if cfg.windOn, ds = d1_sample_wind(cfg, T); else, ds = []; end
    RS = d1_fly_student(stu, conf, kase.Xref, kase.Uref, ds, theta, lqr, cfg, 'alpha1');
    RB = d1_fly_student(stu, conf, kase.Xref, kase.Uref, ds, theta, lqr, cfg, 'blend');
    peS = d1_track_err(RS.X, RS.Xr); peB = d1_track_err(RB.X, RB.Xr);
    D(ci) = struct('groupId', kase.groupId, 'Xref', RS.Xr, 'xS', RS.X, 'xB', RB.X, 'peS', peS, ...
        'peB', peB, 'alpha', RB.alpha, 'restarts', [RS.nDiv RB.nDiv]);
    fprintf('STUDENT %s: alpha1 pos rmse=%.3f max=%.3f | blend rmse=%.3f meanA=%.2f | restarts %d/%d\n', ...
        kase.groupId, rmse_(peS), max(peS), rmse_(peB), mean(RB.alpha), RS.nDiv, RB.nDiv);
end
if ~isfolder(cfg.runDir), mkdir(cfg.runDir); end
save(fullfile(cfg.runDir, sprintf('student_eval_seed%d.mat', cfg.seed)), 'D', '-v7.3');
fprintf('STUDENT_EVAL_DONE %d cases\n', numel(D));
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

function compare_flight(cfg, teacher, lqr, cases, stu, sac, conf)
% One representative case per trajectory FAMILY; fly all controllers on the SAME
% reference, plant and wind realization and dump full trajectories + tracking error:
%   (a) LQR-only        u = sat(uh - K e)
%   (b) teacher NMPC    SAC-NMPC (deterministic SAC policy a = mu), privileged wind
%   (c) student alpha=1 u = sat(sat(u_LQR) + W*phi)
%   (d) proposed blend  d1_blend_control (alpha = c_S * g_L(c_LQR))
sel = family_cases(cases);
aMean = extractdata(sac.mu);
[Qt, Rt] = d1_action_to_QR(aMean, cfg); d1_set_teacher_weights(teacher, Qt, Rt, cfg);
fprintf('COMPARE teacher weights = SAC-NMPC (a=mu); diag(Q)=[%s]\n', ...
    strtrim(sprintf('%.3g ', diag(Qt))));
pscale = d1_getenv_num('D1_PLANT_PERTURB', 0);
theta = perturb_plant(cfg.plant.nominal, pscale, cfg.seed);
fprintf('COMPARE flight plant = %s (perturb scale %.2f x train rho), alphaSafe=%d\n', ...
    ternary(pscale > 0, 'OFF-NOMINAL', 'nominal'), pscale, cfg.alphaSafe);
D = struct('groupId',{},'family',{},'Xref',{},'xL',{},'xN',{},'xS',{},'xB',{}, ...
    'peL',{},'peN',{},'peS',{},'peB',{},'alpha',{},'okN',{},'restarts',{});
for ci = 1:numel(sel)
    kase = cases(sel(ci)); Xref = kase.Xref; T = d1_case_len(Xref, cfg);
    if cfg.windOn, ds = d1_sample_wind(cfg, T); else, ds = []; end
    RL = d1_fly_student([], [], Xref, kase.Uref, ds, theta, lqr, cfg, 'lqr');
    [xN, okN, nDivN] = fly_teacher(teacher, Xref, ds, theta, cfg);
    RS = d1_fly_student(stu, conf, Xref, kase.Uref, ds, theta, lqr, cfg, 'alpha1');
    RB = d1_fly_student(stu, conf, Xref, kase.Uref, ds, theta, lqr, cfg, 'blend');
    Xr = RL.Xr;
    peL = d1_track_err(RL.X, Xr); peN = d1_track_err(xN, Xr);
    peS = d1_track_err(RS.X, Xr); peB = d1_track_err(RB.X, Xr);
    D(ci) = struct('groupId', kase.groupId, 'family', kase.groupId(1:find(kase.groupId=='|',1)-1), ...
        'Xref', Xr, 'xL', RL.X, 'xN', xN, 'xS', RS.X, 'xB', RB.X, 'peL', peL, 'peN', peN, ...
        'peS', peS, 'peB', peB, 'alpha', RB.alpha, 'okN', okN, ...
        'restarts', [RL.nDiv nDivN RS.nDiv RB.nDiv]);
    fprintf(['CMP %-18s | LQR rmse=%.3f max=%.3f | NMPC rmse=%.3f max=%.3f | ' ...
        'STU rmse=%.3f max=%.3f | BLEND rmse=%.3f max=%.3f | meanA=%.2f okN=%.2f | ' ...
        'restarts L/N/S/B=%d/%d/%d/%d\n'], ...
        kase.groupId, rmse_(peL), max(peL), rmse_(peN), max(peN), rmse_(peS), max(peS), ...
        rmse_(peB), max(peB), mean(RB.alpha), mean(okN), RL.nDiv, nDivN, RS.nDiv, RB.nDiv);
end
if ~isfolder(cfg.runDir), mkdir(cfg.runDir); end
save(fullfile(cfg.runDir, sprintf('compare_seed%d.mat', cfg.seed)), 'D', '-v7.3');
fprintf('COMPARE_DONE %d families\n', numel(D));
end

function [X, okN, nDiv] = fly_teacher(teacher, Xref, ds, theta, cfg)
% teacher = SAC-NMPC (weights set by the caller), privileged wind, d1_teacher_step rule
T = d1_case_len(Xref, cfg); Ts = cfg.Ts; uh = cfg.uh;
X = nan(12,T); okN = zeros(1,T); nDiv = 0;
x = Xref(:,1); uprev = uh; d1_teacher_reset(teacher, Xref, 1, cfg);
for k = 1:T
    t = (k-1)*Ts; F = d1_wind_now(ds, t, x, uprev, theta);
    [u, status] = d1_teacher_step(teacher, x, uprev, Xref, k, F, cfg); okN(k) = (status == 0);
    u = d1_sat(u, cfg); uprev = u;
    [x, div] = d1_plant_step(t, x, u, Ts, theta, ds);
    if div
        nDiv = nDiv + 1; x = Xref(:,k+1); uprev = uh;
        d1_teacher_reset(teacher, Xref, k+1, cfg);
    else
        X(:,k) = x;
    end
end
end

function r = rmse_(pe)
r = sqrt(mean(pe.^2));
end
function s = ternary(c, a, b); if c, s = a; else, s = b; end; end

% ============================================================================
% GATE GRID: evaluate ONE (c_low, c_high) blend-gate setting on the validation
% families. Teacher-free: flies only LQR-only and the proposed blend.
function gate_grid_flight(cfg, lqr, cases, stu, conf)
assert(cfg.cHigh > cfg.cLow, 'gate grid needs c_high > c_low (got %.3f, %.3f).', ...
    cfg.cHigh, cfg.cLow);
sel = family_cases(cases);
hard = strcmp(d1_getenv_str('D1_HARD','0'), '1');
pscale = d1_getenv_num('D1_PLANT_PERTURB', 0);
theta = perturb_plant(cfg.plant.nominal, pscale, cfg.seed);
fprintf('GATE start c_low=%.3f c_high=%.3f hard=%d pscale=%.2f alphaSafe=%d (%d families)\n', ...
    cfg.cLow, cfg.cHigh, hard, pscale, cfg.alphaSafe, numel(sel));
peLall = []; peBall = []; aAll = [];
for ci = 1:numel(sel)
    kase = cases(sel(ci)); T = d1_case_len(kase.Xref, cfg);
    if cfg.windOn, ds = d1_sample_wind(cfg, T); else, ds = []; end
    RL = d1_fly_student([], [], kase.Xref, kase.Uref, ds, theta, lqr, cfg, 'lqr');
    RB = d1_fly_student(stu, conf, kase.Xref, kase.Uref, ds, theta, lqr, cfg, 'blend');
    peL = d1_track_err(RL.X, RL.Xr); peB = d1_track_err(RB.X, RB.Xr);
    peLall = [peLall, peL]; peBall = [peBall, peB]; aAll = [aAll, RB.alpha]; %#ok<AGROW>
    fprintf('GATE_FAM %-18s | LQR rmse=%.3f | BLEND rmse=%.3f | meanA=%.2f\n', ...
        kase.groupId, rmse_(peL), rmse_(peB), mean(RB.alpha));
end
rmseB = rmse_(peBall); rmseL = rmse_(peLall);
fprintf('GATE_RESULT c_low=%.3f c_high=%.3f rmse_blend=%.4f rmse_lqr=%.4f delta=%.4f meanA=%.3f n=%d\n', ...
    cfg.cLow, cfg.cHigh, rmseB, rmseL, rmseB-rmseL, mean(aAll), numel(sel));
if ~isfolder(cfg.runDir), mkdir(cfg.runDir); end
res = struct('cLow',cfg.cLow,'cHigh',cfg.cHigh,'rmseB',rmseB,'rmseL',rmseL, ...
    'meanA',mean(aAll),'hard',hard,'pscale',pscale,'seed',cfg.seed);
save(fullfile(cfg.runDir, sprintf('gate_seed%d_cl%.2f_ch%.2f.mat', ...
    cfg.seed, cfg.cLow, cfg.cHigh)), 'res');
end
