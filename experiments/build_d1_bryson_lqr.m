function selectedLqr = build_d1_bryson_lqr(varargin)
%BUILD_D1_BRYSON_LQR Generate the D1 baseline/Lyapunov LQR from Bryson-normalized
% Q/R (NO grid retune). Discrete linearization of the RK4 one-step map about hover,
% then [K,S] = dlqr(A,B,Q,R). Verifies controllability, S>0, closed-loop poles
% inside the unit circle, and saturation at a hover perturbation. Saves the
% artifact selectedLqr.S (=P) + K + provenance to cfg.contraction.lqrArtifactPath.
opts = struct('save', true);
for k = 1:2:numel(varargin); opts.(varargin{k}) = varargin{k + 1}; end
here = fileparts(mfilename('fullpath')); projectRoot = fileparts(here);
addpath(genpath(fullfile(projectRoot, 'src'))); addpath(fullfile(projectRoot, 'configs'));

w = targeted_lqr_weak_config();
nmpc = step2_nmpc_config();           % Bryson-normalized Q/R base
Q = nmpc.weights.Q; R = nmpc.weights.R;
Ts = w.sampleTime; nom = w.plant.nominal;

% Bryson sources (for provenance; must match step2_nmpc_config).
eAllow = [0.10;0.10;0.10; deg2rad(5)*[1;1;1]; 0.30;0.30;0.30; 2.0;2.0;2.0];
duAllow = [nom.m*nom.g; 0.5; 0.5; 0.25];
assert(max(abs(diag(Q) - 1./eAllow.^2)) < 1e-9, 'Q not Bryson(eAllow)');
assert(max(abs(diag(R) - 1./duAllow.^2)) < 1e-9, 'R not Bryson(duAllow)');

% Discrete linearization about hover (same scheme as retune_targeted_lqr_baseline).
xEq = zeros(12, 1); uEq = quad_hover_input(nom);
map = @(x, u) quad_step_rk4(0, x, u, Ts, nom, []);
A = zeros(12, 12); B = zeros(12, 4);
for i = 1:12
    d = zeros(12, 1); d(i) = w.lqr.statePerturbation(i);
    A(:, i) = (map(xEq + d, uEq) - map(xEq - d, uEq)) / (2 * d(i));
end
for j = 1:4
    d = zeros(4, 1); d(j) = w.lqr.inputPerturbation(j);
    B(:, j) = (map(xEq, uEq + d) - map(xEq, uEq - d)) / (2 * d(j));
end

% --- verification gates (fail-fast; do not save a bad artifact) ---
ctrbRank = rank(ctrb(A, B));
if ctrbRank < 12
    error('build_d1_bryson_lqr:NotControllable', ...
        'ctrb rank %d < 12 (uncontrollable linearization).', ctrbRank);
end
[K, S, poles] = dlqr(A, B, Q, R);
S = (S + S.') / 2;
minEigS = min(eig(S));
if minEigS <= 0
    error('build_d1_bryson_lqr:PNotPosDef', 'S(=P) min eig %.3e <= 0.', minEigS);
end
clPoles = eig(A - B * K); maxAbsCl = max(abs(clPoles));
if maxAbsCl >= 1
    error('build_d1_bryson_lqr:Unstable', ...
        'closed-loop pole magnitude %.4f >= 1 (not Schur-stable).', maxAbsCl);
end
% Saturation check at a representative hover perturbation (0.1 m / 5 deg / small).
xp = [0.1;0.1;0.1; deg2rad(5)*[1;1;1]; 0.2;0.2;0.2; deg2rad(10)*[1;1;1]];
uSat = uEq - K * xp;
lo = [nom.inputLimits.T(1); nom.inputLimits.tau(:,1)];
hi = [nom.inputLimits.T(2); nom.inputLimits.tau(:,2)];
satMargin = min([uSat - lo; hi - uSat]);   % >0 means inside limits

fprintf('== build_d1_bryson_lqr ==\n');
fprintf('ctrb rank = %d/12\n', ctrbRank);
fprintf('min eig(P=S) = %.6g\n', minEigS);
fprintf('max |closed-loop pole| = %.6f  (Schur stable: %d)\n', maxAbsCl, maxAbsCl < 1);
fprintf('hover-perturb saturation margin = %.4g N/Nm (>0 = within limits)\n', satMargin);
fprintf('diag(Q) = %s\n', mat2str(diag(Q).', 6));
fprintf('diag(R) = %s\n', mat2str(diag(R).', 6));

selectedLqr = struct();
selectedLqr.S = S; selectedLqr.K = K; selectedLqr.closedLoopPoles = poles;
selectedLqr.Q = Q; selectedLqr.R = R; selectedLqr.uEquilibrium = uEq;
selectedLqr.positionVelocityScale = 1; selectedLqr.attitudeRateScale = 1;
selectedLqr.inputPenaltyScale = 1; selectedLqr.designModel = 'nominal';
selectedLqr.linearizationPoint = 'hover'; selectedLqr.sampleTime = Ts;
selectedLqr.artifactVersion = 'd1_bryson_lqr_v1';
selectedLqr.selectionCandidateId = 'bryson';
selectedLqr.designMethod = 'bryson_rule';
selectedLqr.eAllow = eAllow; selectedLqr.duAllow = duAllow;
selectedLqr.controllabilityRank = ctrbRank;
selectedLqr.maxAbsClosedLoopPole = maxAbsCl;
selectedLqr.minEigP = minEigS;
selectedLqr.saturationMarginAtHoverPerturb = satMargin;

if opts.save
    outPath = fullfile(projectRoot, w.contraction.lqrArtifactPath);
    outDir = fileparts(outPath);
    if ~exist(outDir, 'dir'); mkdir(outDir); end
    save(outPath, 'selectedLqr', 'A', 'B', '-v7');
    fprintf('saved %s\n', outPath);
end
end
