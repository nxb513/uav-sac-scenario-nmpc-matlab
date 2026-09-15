function run_d1_joint_pipeline()
%RUN_D1_JOINT_PIPELINE One-seed joint pipeline (SAC tunes NMPC Q,R -> teacher
% NMPC + paired LQR -> streaming surrogate -> contraction -> checkpoint).
%
% Open-ended: runs SAC iterations until the wall-time quota, then saves a FULL
% checkpoint so a later job resumes and keeps learning. All state (SAC actor/
% critics/targets/alpha/optimizers/replay + surrogate net/optimizer/buffer/norm +
% RNG + counters + current Q,R) is checkpointed.
%
% Env: D1_SEED, D1_RUN_DIR, D1_WALL_SECONDS, D1_RESUME ('1' to resume).
% FROZEN: Qf=0, Np=20, Nc=5, Ts=0.05, H=20, dU=0 (asserted). Teacher is single
% nominal scenario + full control horizon for now (M=5 robust + Nc=5 blocking are
% teacher-internal refinements tracked in docs/notes/d1_joint_pipeline_audit_gate).

cfg = joint_config();
fprintf('== D1 joint pipeline seed=%d wall=%ds run_dir=%s resume=%d ==\n', ...
    cfg.seed, cfg.wallSeconds, cfg.runDir, cfg.resume);
assert(cfg.N==20 && cfg.Nc==5 && abs(cfg.Ts-0.05)<1e-12 && cfg.H==20, ...
    'Frozen horizon/Ts/H violated.');
assert(cfg.Qf==0 && cfg.dU==0, 'Frozen Qf/dU violated.');

if ~isfolder(cfg.runDir), mkdir(cfg.runDir); end
rng(cfg.seed, 'twister');

% ---- fixed components (built once) ------------------------------------------
scen = sample_scenarios(cfg);               % M=5 uncertain plant realizations
teacher = d1_teacher_build_solver(cfg, scen);
lqr = build_lqr(cfg);                       % K, P (Bryson LQR + Riccati)
cases = generate_teacher_dev_cases(cfg);    % ~120 cases, 1000-step references
fprintf('built teacher(acados M=%d Nc=%d) + LQR(P minEig=%.4g) + %d cases\n', ...
    cfg.M, cfg.Nc, min(eig(lqr.P)), numel(cases));

% ---- diagnostic mode: fly a few cases, dump trajectories, exit --------------
if strcmp(getenv_str('D1_DIAG','0'), '1')
    diag_flight(cfg, teacher, lqr, cases);
    return;
end

% ---- init or resume learners ------------------------------------------------
ckptPath = fullfile(cfg.runDir, sprintf('checkpoint_seed%d.mat', cfg.seed));
if cfg.resume && isfile(ckptPath)
    S = load(ckptPath); sac = S.sac; sur = S.sur; st = S.st;
    rng(S.rngState);
    fprintf('RESUMED from %s at iter=%d\n', ckptPath, st.iter);
else
    sac = init_sac(cfg); sur = init_surrogate(cfg);
    st = struct('iter', 0, 'totalSamples', 0, 'teacherVersion', 0);
    fprintf('FRESH start\n');
end

% ---- open-ended SAC loop ----------------------------------------------------
tStart = tic; lastCkpt = tic;
while toc(tStart) < cfg.wallSeconds
    st.iter = st.iter + 1;
    st.teacherVersion = st.iter;                       % Q,R identity per rollout
    a = sac_sample_action(sac, cfg);                   % candidate in [-1,1]^6
    [Q, R] = action_to_QR(a, cfg);
    set_teacher_weights(teacher, Q, R, cfg);
    % --- rollout over a batch of cases (paired NMPC + LQR) -------------------
    idx = randi(numel(cases), 1, cfg.casesPerEval);
    rewards = zeros(cfg.casesPerEval, 1); cLall = [];
    for c = 1:cfg.casesPerEval
        [rew, sur, cLdata, st] = paired_rollout(teacher, lqr, cases(idx(c)), ...
            sur, st, cfg);
        rewards(c) = rew; cLall = [cLall; cLdata]; %#ok<AGROW>
    end
    reward = mean(rewards);
    sac = sac_update(sac, a, reward, cfg);             % 1-step bandit SAC
    st.lastReward = reward; st.lastQ = diag(Q).'; st.lastR = diag(R).';
    st.cL = cLall;
    if mod(st.iter, cfg.logEvery)==0
        fprintf('iter=%d reward=%.4f alpha=%.3g samples=%d elapsed=%.0fs\n', ...
            st.iter, reward, sac.alpha, st.totalSamples, toc(tStart));
    end
    if toc(lastCkpt) > cfg.checkpointEverySec
        save_checkpoint(ckptPath, sac, sur, st, cfg); lastCkpt = tic;
    end
end
save_checkpoint(ckptPath, sac, sur, st, cfg);
fprintf('D1_JOINT_DONE iters=%d samples=%d final_reward=%.4f\n', ...
    st.iter, st.totalSamples, st.lastReward);
end

% ============================================================================
function cfg = joint_config()
cfg.seed = getenv_num('D1_SEED', 260914001);
cfg.runDir = getenv_str('D1_RUN_DIR', fullfile('results','d1_joint', ...
    sprintf('seed%d', getenv_num('D1_SEED',260914001))));
cfg.wallSeconds = getenv_num('D1_WALL_SECONDS', 300);
cfg.resume = strcmp(getenv_str('D1_RESUME','0'),'1');
cfg.Ts = 0.05; cfg.N = 20; cfg.Nc = 5; cfg.H = 20; cfg.Qf = 0; cfg.dU = 0;
cfg.M = 5;                                          % robust scenarios (frozen)
cfg.stepsPerCase = getenv_num('D1_STEPS', 1000);
cfg.casesPerEval = getenv_num('D1_CASES_PER_EVAL', 20);
cfg.plant = d1_joint_plant_params();
cfg.actionDim = 6;                                  % Q:{pos,att,vel,rate}, R:{T,tau}
cfg.logMultBounds = [10^-1.5, 10^1.5];              % search window around Bryson
% surrogate
cfg.surHidden = 128; cfg.surLR = 1e-3; cfg.surBatch = 256;
cfg.surBufferCap = 1e5; cfg.surRecentFrac = 0.5;
% sac
cfg.sacLR = 3e-4; cfg.sacBatch = 256; cfg.sacBufferCap = 5e4;
cfg.sacGamma = 0.0;                                  % 1-step bandit (done each ep)
cfg.sacTau = 0.005; cfg.sacTargetEntropy = -cfg.actionDim;
cfg.logEvery = 1; cfg.checkpointEverySec = 120;
end

function v = getenv_num(name, dflt)
s = getenv(name); if isempty(s), v = dflt; else, v = str2double(s); end
end
function v = getenv_str(name, dflt)
s = getenv(name); if isempty(s), v = dflt; else, v = s; end
end

% ---- LQR (Bryson) -----------------------------------------------------------
function lqr = build_lqr(cfg)
P = cfg.plant; theta = P.nominal;
xh = zeros(12,1); uh = [P.m*P.g; 0; 0; 0];
[A, B] = num_linearize(@(x,u) quad_dynamics(0, x, u, theta, []), xh, uh);
sysd = c2d(ss(A, B, eye(12), zeros(12,4)), cfg.Ts);
[Q0, R0] = d1_bryson_weights(P);
[K, Sr] = dlqr(sysd.A, sysd.B, Q0, R0);
lqr.K = K; lqr.P = Sr; lqr.uh = uh; lqr.Ad = sysd.A; lqr.Bd = sysd.B;
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

% ---- cases ------------------------------------------------------------------
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

% ---- action -> Q,R ----------------------------------------------------------
function [Q, R] = action_to_QR(a, cfg)
[Q0, R0] = d1_bryson_weights(cfg.plant);
lo = log(cfg.logMultBounds(1)); hi = log(cfg.logMultBounds(2));
mult = exp(lo + 0.5*(a(:)+1)*(hi-lo));              % 6 log-multipliers
q = diag(Q0);
q(1:3)=q(1:3)*mult(1); q(4:6)=q(4:6)*mult(2);
q(7:9)=q(7:9)*mult(3); q(10:12)=q(10:12)*mult(4);
r = diag(R0); r(1)=r(1)*mult(5); r(2:4)=r(2:4)*mult(6);
Q = diag(q); R = diag(r);
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

% ---- paired NMPC + LQR rollout on one case ---------------------------------
function [reward, sur, cLdata, st] = paired_rollout(teacher, lqr, kase, sur, st, cfg)
Ts = cfg.Ts; N = cfg.N; theta = cfg.plant.nominal;
Xref = kase.Xref; T = min(cfg.stepsPerCase, size(Xref,2)-N-1);
uh = [cfg.plant.m*cfg.plant.g; 0; 0; 0];

% independent plant copies, same initial condition
M = cfg.M;
xN = Xref(:,1); xL = Xref(:,1); uprev = uh;
stateHist = repmat(xN,1,4); inputHist = repmat(uh,1,4);
posErrN = zeros(T,1); duAcc = 0; okN = 0; cViol = 0; prevU = uh;
diverged = false; kdone = T;
EL = zeros(12, T+1); EL(:,1) = xL - Xref(:,1);       % LQR error traj for c_L
usat_lo = [0;-0.5;-0.5;-0.25]; usat_hi = [cfg.plant.Tmax;0.5;0.5;0.25];
warmstart_ref(teacher, Xref, 1, uh, cfg);

for k = 1:T
    % ---- NMPC teacher branch (augmented M-scenario state, delta-u) ----------
    for s = 0:N-1
        teacher.set('cost_y_ref', [repmat(Xref(:,k+s),M,1); uh], s);
    end
    teacher.set('cost_y_ref_e', repmat(Xref(:,k+N),M,1));
    teacher.set('constr_x0', [repmat(xN,M,1); uprev]);
    teacher.solve();
    okStatus = (teacher.get('status')==0);
    du0 = teacher.get('u', 0);
    if okStatus && all(isfinite(du0))
        uN = uprev + du0;                            % u = u_prev + du0
    else
        uN = uh - lqr.K*(xN - Xref(:,k));            % LQR fallback keeps plant stable
    end
    uN = min(max(uN, usat_lo), usat_hi);
    okN = okN + okStatus; uprev = uN;
    % streaming surrogate sample (skip if any non-finite) ---------------------
    if all(isfinite(stateHist(:))) && all(isfinite(inputHist(:)))
        refLook = Xref(:, k:k+10);
        feat = surrogate_build_feature(stateHist, inputHist, refLook, zeros(12,1));
        if all(isfinite(feat))
            sur = surrogate_stream_update(sur, feat, uN, st.teacherVersion, cfg);
            st.totalSamples = st.totalSamples + 1;
        end
    end
    % advance NMPC plant + histories
    xNnext = quad_step_rk4(0, xN, uN, Ts, theta, []);
    stateHist = [stateHist(:,2:end), xN];
    inputHist = [inputHist(:,2:end), uN];
    duAcc = duAcc + sum((uN-prevU).^2); prevU = uN; xN = xNnext;
    if ~all(isfinite(xN)) || norm(xN(1:3)) > 1e4
        diverged = true; kdone = k; break;           % diverged -> penalize case
    end
    posErrN(k) = norm(xN(1:3) - Xref(1:3,k+1));
    cViol = cViol + any(abs(xN(4:5)) > 1.35);
    % ---- LQR paired branch (own plant copy) --------------------------------
    uL = uh - lqr.K*(xL - Xref(:,k));
    uL = min(max(uL,[0;-0.5;-0.5;-0.25]),[cfg.plant.Tmax;0.5;0.5;0.25]);
    xL = quad_step_rk4(0, xL, uL, Ts, theta, []);
    EL(:,k+1) = xL - Xref(:,k+1);
end
% reward from NMPC KPIs (negative cost; lower error/failure = higher reward).
% posErr capped at 5 m/step so a rare hard case cannot swamp the mean; LQR
% fallback keeps the plant bounded, so no separate divergence term.
nOk = max(kdone,1);
pe = min(posErrN(1:nOk), 5);
posRmse = sqrt(mean(pe.^2));
failRate = 1 - okN/nOk;
reward = -(posRmse + 0.01*sqrt(duAcc/nOk) + 0.5*(cViol/nOk) + 5*failRate);
% c_L contraction data from LQR error trajectory
try
    outL = d1_finite_horizon_contraction(EL, lqr.P, cfg.H, struct());
    cLdata = outL.g_H(isfinite(outL.g_H))';
catch
    cLdata = [];
end
if isempty(cLdata), cLdata = zeros(0,1); end
end

% ---- surrogate --------------------------------------------------------------
function sur = init_surrogate(cfg)
lg = [featureInputLayer(208,'Name','in','Normalization','none')
      fullyConnectedLayer(cfg.surHidden); swishLayer
      fullyConnectedLayer(cfg.surHidden); swishLayer
      fullyConnectedLayer(cfg.surHidden); swishLayer
      fullyConnectedLayer(4); tanhLayer];
sur.net = dlnetwork(lg);
sur.avgG = []; sur.avgSqG = []; sur.step = 0;
sur.featScale = feature_scale();
sur.tgtMid = [cfg.plant.m*cfg.plant.g;0;0;0];
sur.tgtHalf = [cfg.plant.Tmax/2; 0.5; 0.5; 0.25];
sur.buf.feat = zeros(208, cfg.surBufferCap, 'single');
sur.buf.tgt = zeros(4, cfg.surBufferCap, 'single');
sur.buf.ver = zeros(1, cfg.surBufferCap);
sur.buf.n = 0; sur.buf.pos = 0; sur.cap = cfg.surBufferCap;
end

function s = feature_scale()
ss = max(abs([-100;-100;-10;-1.35;-1.35;-pi;-25;-25;-25;-10;-10;-10]), ...
         abs([100;100;100;1.35;1.35;pi;25;25;25;10;10;10]));
us = [40;1;1;0.5];
s = [repmat(ss,4,1); repmat(us,4,1); repmat(ss,11,1); ss];  % 48+16+132+12=208
end

function sur = surrogate_stream_update(sur, feat, uTeacher, ver, cfg)
% push to ring buffer
sur.buf.pos = mod(sur.buf.pos, sur.cap) + 1;
tgt = (uTeacher - sur.tgtMid) ./ sur.tgtHalf;
sur.buf.feat(:,sur.buf.pos) = single(feat ./ sur.featScale);
sur.buf.tgt(:,sur.buf.pos) = single(min(max(tgt,-1),1));
sur.buf.ver(sur.buf.pos) = ver;
sur.buf.n = min(sur.buf.n + 1, sur.cap);
if sur.buf.n < cfg.surBatch, return; end
% recent-teacher-prioritized minibatch
idx = sample_recent(sur.buf, ver, cfg);
Xb = dlarray(sur.buf.feat(:,idx), 'CB');
Yb = dlarray(sur.buf.tgt(:,idx), 'CB');
[grad, ~] = dlfeval(@sur_loss, sur.net, Xb, Yb);
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

function [grad, loss] = sur_loss(net, X, Y)
Yhat = forward(net, X);
loss = mean((Yhat - Y).^2, 'all');
grad = dlgradient(loss, net.Learnables);
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
[gmu, gstd, gAl, entropy] = dlfeval(@actor_loss, sac.mu, sac.logStd, ...
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

function tnet = soft_update(tnet, net, tau)
tl = tnet.Learnables; nl = net.Learnables;
for i = 1:height(tl)
    tl.Value{i} = (1-tau)*tl.Value{i} + tau*nl.Value{i};
end
tnet.Learnables = tl;
end

% ---- checkpoint -------------------------------------------------------------
function save_checkpoint(path, sac, sur, st, cfg) %#ok<INUSD>
rngState = rng; %#ok<NASGU>
save(path, 'sac', 'sur', 'st', 'rngState', '-v7.3');
fprintf('CHECKPOINT saved iter=%d samples=%d -> %s\n', st.iter, st.totalSamples, path);
end

% ---- diagnostics ------------------------------------------------------------
function diag_flight(cfg, teacher, lqr, cases)
[Q0, R0] = d1_bryson_weights(cfg.plant);
set_teacher_weights(teacher, Q0, R0, cfg);
sel = unique(round(linspace(1, numel(cases), min(4, numel(cases)))));
D = struct('groupId',{},'Xref',{},'xN',{},'xL',{},'ok',{},'kdiv',{});
for ci = 1:numel(sel)
    kase = cases(sel(ci));
    [Xref, xN, xL, okv, kdiv] = fly_case(teacher, lqr, kase, cfg);
    D(ci).groupId = kase.groupId; D(ci).Xref = Xref;
    D(ci).xN = xN; D(ci).xL = xL; D(ci).ok = okv; D(ci).kdiv = kdiv;
    peN = vecnorm(xN(1:3,:) - Xref(1:3,:));
    peL = vecnorm(xL(1:3,:) - Xref(1:3,:));
    fprintf(['DIAG %s: NMPC posErr med=%.2f max=%.2f okRate=%.2f kdiv=%d | ' ...
        'LQR posErr med=%.2f max=%.2f\n'], kase.groupId, median(peN), max(peN), ...
        mean(okv), kdiv, median(peL), max(peL));
end
if ~isfolder(cfg.runDir), mkdir(cfg.runDir); end
save(fullfile(cfg.runDir, sprintf('diag_seed%d.mat', cfg.seed)), 'D', '-v7.3');
fprintf('DIAG_DONE saved %d cases\n', numel(D));
end

function [Xr, xNt, xLt, okv, kdiv] = fly_case(teacher, lqr, kase, cfg)
Ts = cfg.Ts; N = cfg.N; M = cfg.M; theta = cfg.plant.nominal;
Xref = kase.Xref; T = min(cfg.stepsPerCase, size(Xref,2)-N-1);
uh = [cfg.plant.m*cfg.plant.g; 0; 0; 0];
lo = [0;-0.5;-0.5;-0.25]; hi = [cfg.plant.Tmax;0.5;0.5;0.25];
Xr = Xref(:, 1:T);
xNt = nan(12, T); xLt = nan(12, T); okv = zeros(1, T); kdiv = 0;
% NMPC teacher branch
xN = Xref(:,1); uprev = uh;
warmstart_ref(teacher, Xref, 1, uh, cfg);
for k = 1:T
    for s = 0:N-1, teacher.set('cost_y_ref', [repmat(Xref(:,k+s),M,1); uh], s); end
    teacher.set('cost_y_ref_e', repmat(Xref(:,k+N),M,1));
    teacher.set('constr_x0', [repmat(xN,M,1); uprev]);
    teacher.solve();
    ok = (teacher.get('status')==0); du0 = teacher.get('u',0);
    if ok && all(isfinite(du0)), uN = uprev + du0; else, uN = uprev; end
    uN = min(max(uN, lo), hi); uprev = uN; okv(k) = ok;
    xN = quad_step_rk4(0, xN, uN, Ts, theta, []); xNt(:,k) = xN;
    if ~all(isfinite(xN)) || norm(xN(1:3)) > 1e4, kdiv = k; break; end
end
% LQR paired branch (independent, full length)
xL = Xref(:,1);
for k = 1:T
    uL = uh - lqr.K*(xL - Xref(:,k)); uL = min(max(uL, lo), hi);
    xL = quad_step_rk4(0, xL, uL, Ts, theta, []); xLt(:,k) = xL;
    if ~all(isfinite(xL)) || norm(xL(1:3)) > 1e4, break; end
end
end
