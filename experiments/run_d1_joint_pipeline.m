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
if strcmp(getenv_str('D1_SURR_EVAL','0'), '1')
    assert(isfile(ckptPath), 'surrogate eval requires a checkpoint (set resume_run_id).');
    S = load(ckptPath); surrogate_eval(cfg, lqr, cases, S.sur); return;
end
if strcmp(getenv_str('D1_CONSOLIDATE','0'), '1')
    % Train c_S and c_LQR = P(next H steps contract | error state) as logistic
    % classifiers over LQR + surrogate closed-loop rollouts. Append to checkpoint.
    assert(isfile(ckptPath), 'consolidate requires a checkpoint (set resume_run_id).');
    S = load(ckptPath); consolidate_confidence(cfg, lqr, cases, S.sur, S.st, ckptPath); return;
end
if strcmp(getenv_str('D1_COMPARE','0'), '1')
    % Fly LQR-only / teacher-NMPC / pure surrogate / proposed(LQR+surrogate blend)
    % on one case per family, dump trajectories + tracking error for the figures.
    assert(isfile(ckptPath), 'compare requires a checkpoint (set resume_run_id).');
    S = load(ckptPath);
    if isfield(S,'conf'), conf = S.conf; else, conf = []; end
    compare_flight(cfg, teacher, lqr, cases, S.sur, S.sac, conf); return;
end
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
save_checkpoint(ckptPath, sac, sur, st, cfg);        % main checkpoint FIRST (safe)
fprintf('D1_JOINT_DONE iters=%d samples=%d final_reward=%.4f\n', ...
    st.iter, st.totalSamples, st.lastReward);
% ---- phase B: train c_S (surrogate closed-loop alpha=1) + c_LQR (logistic) ---
% Guarded so a failure never loses the SAC/surrogate/residual checkpoint.
try
    consolidate_confidence(cfg, lqr, cases, sur, st, ckptPath);
catch ME
    fprintf('CONF_FAIL %s (main ckpt intact; re-run with D1_CONSOLIDATE=1)\n', ME.message);
end
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
cfg.solverType = getenv_str('D1_SOLVER', 'SQP_RTI'); % 'SQP_RTI' (fast) | 'SQP' (accurate)
cfg.stepsPerCase = getenv_num('D1_STEPS', 1000);
cfg.casesPerEval = getenv_num('D1_CASES_PER_EVAL', 20);
cfg.plant = d1_joint_plant_params();
cfg.actionDim = 6;                                  % Q:{pos,att,vel,rate}, R:{T,tau}
cfg.logMultBounds = [10^-1.5, 10^1.5];              % search window around Bryson
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
    okStatus = (teacher.get('status')==0);           % true SQP convergence (diagnostic)
    du0 = teacher.get('u', 0);
    solved = all(isfinite(du0));                      % NMPC produced a finite SQP iterate
    if solved
        uN = uprev + du0;                            % apply the NMPC control (teacher = SAC-NMPC).
    else                                             % converged or max-iter iterate; NEVER LQR.
        uN = uprev;                                  % rare numerical failure -> hold last NMPC control
    end
    uN = min(max(uN, usat_lo), usat_hi);
    okN = okN + okStatus; uprev = uN;
    % stream RESIDUAL label Delta_u* = u_teacher - u_LQR(same state xN); genuine NMPC only
    if solved && all(isfinite(stateHist(:))) && all(isfinite(inputHist(:)))
        refLook = Xref(:, k:k+10);
        feat = surrogate_build_feature(stateHist, inputHist, refLook, zeros(12,1));
        if all(isfinite(feat))
            uLk = uh - lqr.K*(xN - Xref(:,k));
            uLk = min(max(uLk, usat_lo), usat_hi);
            sur = surrogate_stream_update(sur, feat, uN - uLk, st.teacherVersion, cfg);
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
sur.featScale = feature_scale();
sur.resHalf = cfg.resHalf;                          % residual normalization
% Delta_u residual buffer (phase A, teacher-driven)
sur.buf.feat = zeros(208, cfg.surBufferCap, 'single');
sur.buf.tgt  = zeros(4,  cfg.surBufferCap, 'single');
sur.buf.ver  = zeros(1,  cfg.surBufferCap);
sur.buf.n = 0; sur.buf.pos = 0; sur.cap = cfg.surBufferCap;
end

function s = feature_scale()
ss = max(abs([-100;-100;-10;-1.35;-1.35;-pi;-25;-25;-25;-10;-10;-10]), ...
         abs([100;100;100;1.35;1.35;pi;25;25;25;10;10;10]));
us = [40;1;1;0.5];
s = [repmat(ss,4,1); repmat(us,4,1); repmat(ss,11,1); ss];  % 48+16+132+12=208
end

function sur = surrogate_stream_update(sur, feat, duResidual, ver, cfg)
% Stream a RESIDUAL label Delta_u* = u_teacher - u_LQR(same state) to the du head.
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
[grad, ~] = dlfeval(@du_loss, sur.net, Xb, Yb);
sur.step = sur.step + 1;
[sur.net, sur.avgG, sur.avgSqG] = adamupdate(sur.net, grad, ...
    sur.avgG, sur.avgSqG, sur.step, cfg.surLR);
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

function surrogate_eval(cfg, lqr, cases, sur)
% Fly the surrogate at alpha=1 (u = sat(u_LQR + Delta_u_hat)); report tracking +
% mean predicted c_S per case.
Ts = cfg.Ts; theta = cfg.plant.nominal;
uh = [cfg.plant.m*cfg.plant.g; 0; 0; 0];
lo = [0;-0.5;-0.5;-0.25]; hi = [cfg.plant.Tmax;0.5;0.5;0.25];
sel = unique(round(linspace(1, numel(cases), min(6, numel(cases)))));
D = struct('groupId',{},'Xref',{},'xS',{},'posErr',{},'meanCS',{});
for ci = 1:numel(sel)
    kase = cases(sel(ci)); Xref = kase.Xref; T = min(cfg.stepsPerCase, size(Xref,2)-11);
    xS = Xref(:,1); stateHist = repmat(xS,1,4); inputHist = repmat(uh,1,4);
    xSt = nan(12,T); csAll = nan(1,T);
    for k = 1:T
        feat = surrogate_build_feature(stateHist, inputHist, Xref(:,k:k+10), zeros(12,1));
        uLk = uh - lqr.K*(xS - Xref(:,k));
        if all(isfinite(feat))
            du = surrogate_predict_du(sur, feat); csAll(k) = surrogate_predict_cs(sur, feat);
        else, du = zeros(4,1); end
        u = min(max(uLk + du, lo), hi);
        stateHist = [stateHist(:,2:end), xS]; inputHist = [inputHist(:,2:end), u];
        xS = quad_step_rk4(0, xS, u, Ts, theta, []); xSt(:,k) = xS;
        if ~all(isfinite(xS)) || norm(xS(1:3)) > 1e4, break; end
    end
    Tv = find(all(isfinite(xSt),1), 1, 'last'); if isempty(Tv), Tv = 1; end
    pe = vecnorm(xSt(1:3,1:Tv) - Xref(1:3,2:Tv+1));
    mcs = mean(csAll(isfinite(csAll)));
    D(ci).groupId = kase.groupId; D(ci).Xref = Xref(:,2:Tv+1); D(ci).xS = xSt(:,1:Tv);
    D(ci).posErr = pe; D(ci).meanCS = mcs;
    fprintf('SURR %s: posErr med=%.3f max=%.3f Tv=%d/%d | meanCS=%.2f\n', ...
        kase.groupId, median(pe), max(pe), Tv, T, mcs);
end
if ~isfolder(cfg.runDir), mkdir(cfg.runDir); end
save(fullfile(cfg.runDir, sprintf('surr_eval_seed%d.mat', cfg.seed)), 'D', '-v7.3');
fprintf('SURR_EVAL_DONE %d cases\n', numel(D));
end

% ---- 3-controller comparison flight ----------------------------------------
function compare_flight(cfg, teacher, lqr, cases, sur, sac, conf)
% One representative case per trajectory FAMILY; fly all controllers on the SAME
% reference realization and dump full position trajectories + tracking error:
%   (a) LQR-only        u = uh - K e
%   (b) teacher NMPC    = SAC-NMPC: scenario-M5 acados with the SAC-TUNED Q,R
%       (deterministic SAC policy a=tanh(mu)); apply the NMPC iterate every step,
%       NO LQR fallback (teacher is pure NMPC).
%   (c) surrogate alpha=1  u = sat(u_LQR + Delta_u_hat)  (full residual, no gate).
%   (d) proposed BLEND     u = sat(u_LQR + alpha*Delta_u_hat),
%                          alpha = c_S * g_L(c_LQR),
%                          g_L = clip((c_high - c_LQR)/(c_high - c_low), 0, 1).
%       c_S = surrogate cs head (recent tracking quality); c_LQR = logistic.
%       Optional alpha_safe = min{alpha, (1-eps)*alpha_bar} when D1_ALPHA_SAFE=1.
% Pick one case per family (mid speed/accel = middle of the family block).
fams = cell(1, numel(cases));
for i = 1:numel(cases)
    g = cases(i).groupId; p = find(g=='|', 1); fams{i} = g(1:p-1);
end
uf = unique(fams, 'stable');
hard = strcmp(getenv_str('D1_HARD','0'), '1');        % 1 = hardest (max v,a) per family
sel = zeros(1, numel(uf));
for f = 1:numel(uf)
    ids = find(strcmp(fams, uf{f}));
    if hard
        sc = zeros(numel(ids),1);
        for t = 1:numel(ids)
            tk = regexp(cases(ids(t)).groupId, 'v([\d.]+)\|a([\d.]+)', 'tokens', 'once');
            sc(t) = str2double(tk{1})*100 + str2double(tk{2});   % speed dominates accel
        end
        [~, jj] = max(sc); sel(f) = ids(jj);          % hardest case of the family
    else
        sel(f) = ids(max(1, round(numel(ids)/2)));    % middle case of the family
    end
end
% teacher = SAC-NMPC: use the deterministic SAC policy weights (a = tanh(mu)),
% set ONCE (persist on the solver). This is the frozen SAC-tuned teacher.
aMean = tanh(extractdata(sac.mu));
[Qt, Rt] = action_to_QR(aMean, cfg); set_teacher_weights(teacher, Qt, Rt, cfg);
fprintf('COMPARE teacher weights = SAC-NMPC (a=tanh(mu)); diag(Q)=[%s]\n', ...
    strtrim(sprintf('%.3g ', diag(Qt))));
haveConf = ~isempty(conf) && isfield(conf,'LQR') && ~isempty(conf.LQR.w);
fprintf('COMPARE blend: u=sat(u_LQR+alpha*Du), alpha=c_S*g_L(c_LQR); c_LQR %s, alphaSafe=%d\n', ...
    ternary(haveConf, 'trained', 'MISSING(gL=0)'), cfg.alphaSafe);
% Flight plant: nominal, or OFF-NOMINAL (same perturbed plant for ALL controllers;
% LQR gain and the teacher model stay nominal-designed -> a fair robustness test).
pscale = getenv_num('D1_PLANT_PERTURB', 0);
testTheta = perturb_plant(cfg.plant.nominal, pscale, cfg.seed);
if pscale > 0
    fprintf('COMPARE flight plant = OFF-NOMINAL (perturb scale %.2f x train rho)\n', pscale);
else
    fprintf('COMPARE flight plant = nominal\n');
end
D = struct('groupId',{},'family',{},'Xref',{},'xL',{},'xN',{},'xS',{},'xB',{}, ...
    'peL',{},'peN',{},'peS',{},'peB',{},'alpha',{},'okN',{});
for ci = 1:numel(sel)
    kase = cases(sel(ci));
    [Xr, xL, xN, xS, xB, okN, alp] = fly_compare(teacher, lqr, sur, kase, cfg, conf);
    peL = vecnorm(xL(1:3,:) - Xr(1:3,:));
    peN = vecnorm(xN(1:3,:) - Xr(1:3,:));
    peS = vecnorm(xS(1:3,:) - Xr(1:3,:));            % pure surrogate (no safety)
    peB = vecnorm(xB(1:3,:) - Xr(1:3,:));
    D(ci).groupId = kase.groupId; D(ci).family = uf{ci};
    D(ci).Xref = Xr; D(ci).xL = xL; D(ci).xN = xN; D(ci).xS = xS; D(ci).xB = xB;
    D(ci).peL = peL; D(ci).peN = peN; D(ci).peS = peS; D(ci).peB = peB;
    D(ci).alpha = alp; D(ci).okN = okN;
    fprintf(['CMP %-18s | LQR rmse=%.3f max=%.3f | NMPC rmse=%.3f max=%.3f | ' ...
        'SUR rmse=%.3f max=%.3f | BLEND rmse=%.3f max=%.3f | meanA=%.2f okN=%.2f\n'], ...
        kase.groupId, rmse_(peL), max_(peL), rmse_(peN), max_(peN), ...
        rmse_(peS), max_(peS), rmse_(peB), max_(peB), mean(alp(isfinite(alp))), mean(okN));
end
if ~isfolder(cfg.runDir), mkdir(cfg.runDir); end
save(fullfile(cfg.runDir, sprintf('compare_seed%d.mat', cfg.seed)), 'D', '-v7.3');
fprintf('COMPARE_DONE %d families\n', numel(D));
end

function r = rmse_(pe)
pe = pe(isfinite(pe)); if isempty(pe), r = NaN; else, r = sqrt(mean(pe.^2)); end
end
function m = max_(pe)
pe = pe(isfinite(pe)); if isempty(pe), m = NaN; else, m = max(pe); end
end
function V = lyapV(x, xref, P)
e = x - xref; V = e.' * P * e;                        % V=e'Pe, full 12-state error
end
function s = ternary(c, a, b); if c, s = a; else, s = b; end; end

function [Xr, xL, xN, xS, xB, okN, alphaTraj] = fly_compare(teacher, lqr, sur, kase, cfg, conf, theta)
if nargin < 6, conf = []; end
if nargin < 7 || isempty(theta), theta = cfg.plant.nominal; end  % flight plant
haveConf = ~isempty(conf) && isfield(conf,'LQR') && ~isempty(conf.LQR.w);
Ts = cfg.Ts; N = cfg.N; M = cfg.M;
Xref = kase.Xref; T = min(cfg.stepsPerCase, size(Xref,2)-N-1);
uh = [cfg.plant.m*cfg.plant.g; 0; 0; 0];
lo = [0;-0.5;-0.5;-0.25]; hi = [cfg.plant.Tmax;0.5;0.5;0.25];
P = lqr.P;
% x?(:,k) is the state AFTER integrating step k -> it must be compared with the
% reference one step ahead, Xref(:,k+1) (matches the training reward alignment).
Xr = Xref(:, 2:T+1);
xL = nan(12,T); xN = nan(12,T); xS = nan(12,T); xB = nan(12,T);
okN = zeros(1,T); alphaTraj = nan(1,T);

% (a) LQR-only ----------------------------------------------------------------
x = Xref(:,1);
for k = 1:T
    u = uh - lqr.K*(x - Xref(:,k)); u = min(max(u,lo),hi);
    x = quad_step_rk4(0, x, u, Ts, theta, []); xL(:,k) = x;
    if ~all(isfinite(x)) || norm(x(1:3)) > 1e4, break; end
end

% (b) teacher = SAC-NMPC. Weights already set to the SAC policy by the caller.
% Apply the NMPC iterate EVERY step (converged or max-iter); on a rare numerical
% failure hold the last NMPC control. NO LQR anywhere in the teacher. okN records
% the true SQP convergence rate as a diagnostic only.
x = Xref(:,1); uprev = uh; warmstart_ref(teacher, Xref, 1, uh, cfg);
for k = 1:T
    for s = 0:N-1, teacher.set('cost_y_ref', [repmat(Xref(:,k+s),M,1); uh], s); end
    teacher.set('cost_y_ref_e', repmat(Xref(:,k+N),M,1));
    teacher.set('constr_x0', [repmat(x,M,1); uprev]);
    teacher.solve(); okN(k) = (teacher.get('status')==0); du0 = teacher.get('u',0);
    if all(isfinite(du0)), u = uprev + du0; else, u = uprev; end
    u = min(max(u,lo),hi); uprev = u;
    x = quad_step_rk4(0, x, u, Ts, theta, []); xN(:,k) = x;
    if ~all(isfinite(x)) || norm(x(1:3)) > 1e4, break; end
end

% (c) surrogate at alpha=1: u = sat(u_LQR + Delta_u_hat) -- full residual, no gate
x = Xref(:,1); stateHist = repmat(x,1,4); inputHist = repmat(uh,1,4);
for k = 1:T
    feat = surrogate_build_feature(stateHist, inputHist, Xref(:,k:k+10), zeros(12,1));
    uLk = uh - lqr.K*(x - Xref(:,k));
    if all(isfinite(feat)), du = surrogate_predict_du(sur, feat); else, du = zeros(4,1); end
    u = min(max(uLk + du, lo), hi);
    stateHist = [stateHist(:,2:end), x]; inputHist = [inputHist(:,2:end), u];
    x = quad_step_rk4(0, x, u, Ts, theta, []); xS(:,k) = x;
    if ~all(isfinite(x)) || norm(x(1:3)) > 1e4, break; end
end

% (d) proposed blend: u = sat(u_LQR + alpha*Delta_u), alpha = c_S * g_L(c_LQR) ---
x = Xref(:,1); stateHist = repmat(x,1,4); inputHist = repmat(uh,1,4);
for k = 1:T
    e = x - Xref(:,k); uLk = uh - lqr.K*e;
    feat = surrogate_build_feature(stateHist, inputHist, Xref(:,k:k+10), zeros(12,1));
    if all(isfinite(feat))
        du = surrogate_predict_du(sur, feat); cS = surrogate_predict_cs(sur, feat);
    else
        du = zeros(4,1); cS = 0;
    end
    if haveConf, cLp = predict_logistic(conf.LQR, conf_feature_online(e).'); else, cLp = 0; end
    gL = min(max((cfg.cHigh - cLp)/(cfg.cHigh - cfg.cLow), 0), 1);
    alpha = cS * gL;
    if cfg.alphaSafe
        alpha = min(alpha, (1-cfg.epsSafe)*alpha_bar_est(e, du, lqr));
    end
    u = min(max(uLk + alpha*du, lo), hi); alphaTraj(k) = alpha;
    stateHist = [stateHist(:,2:end), x]; inputHist = [inputHist(:,2:end), u];
    x = quad_step_rk4(0, x, u, Ts, theta, []); xB(:,k) = x;
    if ~all(isfinite(x)) || norm(x(1:3)) > 1e4, break; end
end
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

% ============================================================================
% CONFIDENCE: c_S and c_LQR = P(next H=20 steps contract | current error state).
% Learned as logistic classifiers over closed-loop rollouts on the SAME case bank
% / test conditions. Label at step k = isContracting(k) (V_{k+H} < V_k in V=e'Pe,
% P = LQR Riccati); feature = the error state e_k (+ per-group magnitudes).
% ============================================================================
function consolidate_confidence(cfg, lqr, cases, sur, st, ckptPath) %#ok<INUSD>
% Phase B: train c_S head (surrogate closed-loop alpha=1 tracking RMS) + c_LQR
% logistic (LQR V-contraction). Append updated surrogate + conf to checkpoint.
sur = train_cS_head(cfg, lqr, cases, sur);
confLQR = train_cLQR(cfg, lqr, cases);
conf.LQR = confLQR; conf.epsP = cfg.epsP; conf.H = cfg.H;
conf.def = ['c_S=exp(-(RMS_pastH_pos/epsP)^2) predicted by surrogate cs head; ' ...
    'c_LQR=logistic P(V_{k+H}<V_k)'];
save(ckptPath, 'sur', 'conf', '-append');
save(fullfile(cfg.runDir, sprintf('conf_seed%d.mat', cfg.seed)), 'conf', '-v7.3');
fprintf('CONF_DONE c_LQR(acc=%.2f base=%.2f n=%d) epsP=%.3g\n', ...
    confLQR.acc, confLQR.base, confLQR.n, cfg.epsP);
end

function sur = train_cS_head(cfg, lqr, cases, sur)
% Fly surrogate closed-loop at alpha=1: u=sat(u_LQR+Delta_u_hat). Collect
% (feature z_k, s_k), s_k=exp(-(RMS pos err over PAST H steps / epsP)^2). Train
% ONLY the cs head (freeze trunk + du head).
Ts=cfg.Ts; theta=cfg.plant.nominal; H=cfg.H; uh=[cfg.plant.m*cfg.plant.g;0;0;0];
lo=[0;-0.5;-0.5;-0.25]; hi=[cfg.plant.Tmax;0.5;0.5;0.25];
nSel = min(cfg.csCasesPerCall, numel(cases));
sel = unique(round(linspace(1, numel(cases), nSel)));
Zall = zeros(208,0,'single'); Sall = zeros(1,0,'single');
for ci = 1:numel(sel)
    kase = cases(sel(ci)); Xref = kase.Xref; T = min(cfg.stepsPerCase, size(Xref,2)-11);
    x = Xref(:,1); stateHist = repmat(x,1,4); inputHist = repmat(uh,1,4);
    feats = zeros(208, T); perr = nan(1,T); nok = 0;
    for k = 1:T
        feat = surrogate_build_feature(stateHist, inputHist, Xref(:,k:k+10), zeros(12,1));
        if ~all(isfinite(feat)), break; end
        uLk = uh - lqr.K*(x - Xref(:,k));
        du  = surrogate_predict_du(sur, feat);
        u = min(max(uLk + du, lo), hi);                  % alpha = 1
        feats(:,k) = single(feat) ./ sur.featScale;
        stateHist = [stateHist(:,2:end), x]; inputHist = [inputHist(:,2:end), u];
        x = quad_step_rk4(0, x, u, Ts, theta, []);
        if ~all(isfinite(x)) || norm(x(1:3)) > 1e4, break; end
        perr(k) = norm(x(1:3) - Xref(1:3,k+1)); nok = k;
    end
    for k = H:nok
        wv = perr(k-H+1:k); wv = wv(isfinite(wv));
        if isempty(wv), continue; end
        E20 = sqrt(mean(wv.^2)); sk = exp(-(E20/cfg.epsP)^2);
        Zall(:,end+1) = feats(:,k); Sall(end+1) = single(sk); %#ok<AGROW>
    end
end
nAll = numel(Sall);
if nAll == 0, fprintf('CS_TRAIN: no samples\n'); return; end
for ep = 1:cfg.csEpochs
    idx = randi(nAll, 1, min(cfg.surBatch, nAll));
    Xb = dlarray(Zall(:,idx),'CB'); Sb = dlarray(Sall(idx),'CB');
    [g,~] = dlfeval(@cs_loss, sur.net, Xb, Sb);
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
% feature (16-dim) + binary contract label per step with a full finite window
if isempty(E) || size(E,2) <= H, X = zeros(16,0); y = zeros(1,0); return; end
o = d1_finite_horizon_contraction(E, P, H, struct());
m = o.validMask & o.windowFinite;
Ew = E; Ew(4:6,:) = mod(Ew(4:6,:)+pi, 2*pi) - pi;
F = conf_feature(Ew);
X = F(:, m); y = double(o.isContracting(m));
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

function p = predict_logistic(clf, X)
% X: n x d -> p: n x 1 = P(contract)
if isempty(clf) || isempty(clf.w), p = 0.5*ones(size(X,1),1); return; end
Z = (X - clf.mu)./clf.sg; p = 1./(1+exp(-(Z*clf.w + clf.b)));
end

function E = rollout_error_lqr(cfg, lqr, kase)
Ts = cfg.Ts; theta = cfg.plant.nominal; Xref = kase.Xref;
T = min(cfg.stepsPerCase, size(Xref,2)-1);
uh = [cfg.plant.m*cfg.plant.g;0;0;0];
lo = [0;-0.5;-0.5;-0.25]; hi = [cfg.plant.Tmax;0.5;0.5;0.25];
x = Xref(:,1); X = nan(12,T);
for k = 1:T
    u = uh - lqr.K*(x - Xref(:,k)); u = min(max(u,lo),hi);
    x = quad_step_rk4(0, x, u, Ts, theta, []); X(:,k) = x;
    if ~all(isfinite(x)) || norm(x(1:3)) > 1e4, break; end
end
Tv = find(all(isfinite(X),1), 1, 'last'); if isempty(Tv), Tv = 1; end
E = X(:,1:Tv) - Xref(:,2:Tv+1);             % align state_k with ref_{k+1}
end
