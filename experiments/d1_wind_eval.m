function d1_wind_eval()
%D1_WIND_EVAL Teacher-free real-wind evaluation of LQR/LQI and the proposed blend.
%
% Every controller is designed on the NOMINAL model only; the true plant (D1_PLANTSET)
% is unknown to it:  nom = nominal | train / ood = repo LHS plants (step1_plant_config,
% default seed), cycled over the flights.
%
% Flights: every wind series in D1_WIND_DIR (t,Fx,Fy,Fz [N], world frame, produced by
% tools/wind/prepare_wind_series.py from Neural-Fly and SWUF-3D) x 15 in-distribution
% references (5 families x {v4 a2, v8 a5, v12 a9}), nominal-model reference generator.
%
% Controllers (D1_CTRLS, comma list):
%   LQR     u = sat(uh - K e)                                   (Bryson, nominal)
%   LQI     u = sat(uh - Kx e - KI z),  z = int(p - p_ref) dt   (joint dlqr, nominal)
%   Ptight / P104        u = sat(uh - K e + alpha*Du)           (proposed, pipeline blend)
%   PtightI / P104I      u = sat(uh - K e - KI z + alpha*Du)    (proposed + integral)
%   PtightJ / P104J      u = sat(uh - Kx e - KI z + alpha*Du)   (proposed on the LQI base)
% alpha = c_S * g_L(c_LQR) and Du come from the surrogate checkpoints D1_CKPT_T/_R
% (+ D1_CONF_T/_R). Integral states use clamping anti-windup (no integration on steps
% where any input saturates). LQI: augmented z_{k+1} = z_k + Ts*e_pos, Bryson weight
% for z with z_allow = e_allow_pos (0.10 m, d1_bryson_weights) x 1 s.
%
% D1_SHARD = 'k/K' flies flights with mod(i-1,K) == k-1. Results: one FL line per
% flight on stdout and D1_OUT/wind_eval_<plantset>_<k>of<K>.csv.
cfg = joint_config(); rng(cfg.seed, 'twister');
lqr = build_lqr(cfg);
[KI, KxLQI] = lqi_setup(cfg, lqr);
St = load(getenv('D1_CKPT_T')); Ct = pick_conf(St, getenv_str('D1_CONF_T', ''));
Sr = load(getenv('D1_CKPT_R')); Cr = pick_conf(Sr, getenv_str('D1_CONF_R', ''));
ctrls = strtrim(strsplit(getenv_str('D1_CTRLS', 'LQI,PtightJ,P104J'), ','));
pset = getenv_str('D1_PLANTSET', 'nom');
F = build_wind_flights(cfg, pset);
sh = sscanf(getenv_str('D1_SHARD', '1/1'), '%d/%d');
sel = find(mod((1:numel(F)) - 1, sh(2)) == sh(1) - 1);
fprintf('WIND_EVAL plantset=%s flights=%d shard=%d/%d selected=%d ctrls=%s\n', ...
    pset, numel(F), sh(1), sh(2), numel(sel), strjoin(ctrls, ','));
rows = {};
for i = sel
    f = F(i); ln = sprintf('FL %4d %-5s %-44s', i, f.tier, f.id);
    for c = 1:numel(ctrls)
        switch ctrls{c}
            case 'LQR',     [X, tm, aux] = fly_ctrl('L', f, cfg, lqr, [], [], KI, KxLQI);
            case 'LQI',     [X, tm, aux] = fly_ctrl('Q', f, cfg, lqr, [], [], KI, KxLQI);
            case 'Ptight',  [X, tm, aux] = fly_ctrl('P', f, cfg, lqr, St.sur, Ct, KI, KxLQI);
            case 'P104',    [X, tm, aux] = fly_ctrl('P', f, cfg, lqr, Sr.sur, Cr, KI, KxLQI);
            case 'PtightI', [X, tm, aux] = fly_ctrl('I', f, cfg, lqr, St.sur, Ct, KI, KxLQI);
            case 'P104I',   [X, tm, aux] = fly_ctrl('I', f, cfg, lqr, Sr.sur, Cr, KI, KxLQI);
            case 'PtightJ', [X, tm, aux] = fly_ctrl('J', f, cfg, lqr, St.sur, Ct, KI, KxLQI);
            case 'P104J',   [X, tm, aux] = fly_ctrl('J', f, cfg, lqr, Sr.sur, Cr, KI, KxLQI);
            otherwise, error('d1_wind_eval:ctrl', 'unknown controller %s', ctrls{c});
        end
        m = metr(X, f.Xref(:, 2:size(X,2)+1), tm); a = mean(aux(isfinite(aux)));
        ln = [ln sprintf(' | %s %s pos=%.4f vel=%.4f pmax=%.3f spd=%.4f tmed=%.1f tp99=%.1f tmax=%.1f aux=%.2f', ...
            ctrls{c}, okc(m.ok), m.pos, m.vel, m.pmax, m.spd, m.tmed, m.tp99, m.tmax, a)]; %#ok<AGROW>
        rows(end+1, :) = {i, f.tier, f.series, f.ref, f.plant, ctrls{c}, double(m.ok), ...
            m.pos, m.vel, m.pmax, m.spd, m.tmed, m.tp99, m.tmax, a}; %#ok<AGROW>
    end
    fprintf('%s\n', ln);
end
out = getenv_str('D1_OUT', fullfile('results', 'wind_eval'));
if ~isfolder(out), mkdir(out); end
T = cell2table(rows, 'VariableNames', {'flight','source','series','ref','plant','ctrl','ok', ...
    'pos_rmse','vel_rmse','pos_max','speed_ratio','t_med_us','t_p99_us','t_max_us','alpha_mean'});
writetable(T, fullfile(out, sprintf('wind_eval_%s_%dof%d.csv', pset, sh(1), sh(2))));
fprintf('DONE %d flights\n', numel(sel));
end

% ---- flights ------------------------------------------------------------------
function F = build_wind_flights(cfg, pset)
D = getenv('D1_WIND_DIR'); nom = cfg.plant.nominal; cases = generate_teacher_dev_cases(cfg);
va = zeros(numel(cases), 2);
for i = 1:numel(cases)
    tk = regexp(cases(i).groupId, 'v([\d.]+)\|a([\d.]+)', 'tokens', 'once');
    va(i,:) = [str2double(tk{1}), str2double(tk{2})];
end
rid = find((va(:,1)==4 & va(:,2)==2) | (va(:,1)==8 & va(:,2)==5) | (va(:,1)==12 & va(:,2)==9)).';
fl = [dir(fullfile(D, 'nf_*.csv')); dir(fullfile(D, 'swuf_*.csv'))];
assert(~isempty(fl), 'd1_wind_eval:nowind', 'no wind series in %s', D);
if strcmp(pset, 'nom')
    pl = nom;
else
    pc = step1_plant_config(); pl = quad_sample_uncertainty(pc, 5, pset, pc.uncertainty.defaultSeed, 'lhs');
end
F = struct('tier',{},'id',{},'series',{},'ref',{},'plant',{},'Xref',{},'theta',{},'ds',{});
n = 0;
for j = 1:numel(fl)
    Tb = readtable(fullfile(D, fl(j).name));
    ds = make_wind_ds(Tb.t, [Tb.Fx, Tb.Fy, Tb.Fz]);
    [~, nm] = fileparts(fl(j).name); src = extractBefore(nm, '_');
    for i = rid
        n = n + 1; ip = mod(n-1, numel(pl)) + 1;
        if strcmp(pset, 'nom'), pid = 'nom'; id = sprintf('%s %s', nm, cases(i).groupId);
        else, pid = sprintf('p%d', ip); id = sprintf('%s %s %s', nm, cases(i).groupId, pid); end
        F(end+1) = struct('tier', src, 'id', id, 'series', nm, 'ref', cases(i).groupId, ...
            'plant', pid, 'Xref', cases(i).Xref, 'theta', pl(ip), 'ds', ds); %#ok<AGROW>
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
function [X, tm, aux] = fly_ctrl(kind, f, cfg, lqr, sur, conf, KI, KxLQI)
% Blend logic copied from the pipeline's fly_gate branch (d); real time (k-1)*Ts is
% passed to the plant so the time-varying wind acts.
Ts = cfg.Ts; N = cfg.N; Xref = f.Xref; T = min(cfg.stepsPerCase, size(Xref,2)-N-1);
uh = [cfg.plant.m*cfg.plant.g; 0; 0; 0];
lo = [0;-0.5;-0.5;-0.25]; hi = [cfg.plant.Tmax;0.5;0.5;0.25];
X = nan(12,T); tm = nan(1,T); aux = nan(1,T);
haveConf = ~isempty(conf) && isfield(conf,'LQR') && ~isempty(conf.LQR.w);
x = Xref(:,1); stateHist = repmat(x,1,4); inputHist = repmat(uh,1,4); zI = zeros(3,1);
for k = 1:T
    t0 = tic;
    e = x - Xref(:,k);
    switch kind
        case 'L'
            u = min(max(uh - lqr.K*e, lo), hi);
        case 'Q'
            uu = uh - KxLQI*e - KI*zI; u = min(max(uu, lo), hi);
        case {'P', 'I', 'J'}
            if kind == 'J'
                uLk = uh - KxLQI*e - KI*zI;                % LQI base [Kx, KI]
            elseif kind == 'I'
                uLk = uh - lqr.K*e - KI*zI;                % LQR base + integral
            else
                uLk = uh - lqr.K*e;                        % pipeline blend base
            end
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
            uu = uLk + alpha*du; u = min(max(uu, lo), hi); aux(k) = alpha;
    end
    if any(kind == 'QIJ') && all(u == uu), zI = zI + Ts*e(1:3); end   % clamping anti-windup
    tm(k) = toc(t0);
    if any(kind == 'PIJ'), stateHist = [stateHist(:,2:end), x]; inputHist = [inputHist(:,2:end), u]; end
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
fprintf('LQI z_allow=%.2f m*s | spectral radius: LQI %.5f, [K_LQR KI] %.5f\n', zAllow, ...
    max(abs(eig(Aa - Ba*Ka))), max(abs(eig(Aa - Ba*[lqr.K, KI]))));
end

% ---- metrics / misc -------------------------------------------------------------
function m = metr(X, Xr, tm)
tv = tm(isfinite(tm))*1e6; m.tmed = median(tv); m.tp99 = prctile(tv, 99); m.tmax = max(tv);
m.ok = all(isfinite(X(:)));
if ~m.ok, m.pos = NaN; m.vel = NaN; m.pmax = NaN; m.spd = NaN; return; end
ep = vecnorm(X(1:3,:) - Xr(1:3,:)); ev = vecnorm(X(7:9,:) - Xr(7:9,:));
m.pos = sqrt(mean(ep.^2)); m.vel = sqrt(mean(ev.^2)); m.pmax = max(ep);
m.spd = mean(vecnorm(X(7:9,:))) / mean(vecnorm(Xr(7:9,:)));
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
