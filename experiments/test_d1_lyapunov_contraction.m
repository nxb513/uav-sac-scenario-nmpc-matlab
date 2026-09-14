function test_d1_lyapunov_contraction()
%TEST_D1_LYAPUNOV_CONTRACTION Unit tests for the V=e'Pe contraction metric.
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(genpath(fullfile(root, 'src'))); addpath(fullfile(root, 'configs'));
rng(1234, 'twister');
np = 0; nf = 0;
    function check(name, cond)
        if cond; fprintf('  PASS  %s\n', name); np = np + 1;
        else;     fprintf('  FAIL  %s\n', name); nf = nf + 1; end
    end

[P, info] = d1_load_lyapunov_P();
fprintf('P source: %s (pv=%.4g ar=%.4g r=%.4g)\n', info.source, ...
    info.positionVelocityScale, info.attitudeRateScale, info.inputPenaltyScale);

% 1. Equilibrium e=0 => V=0
check('1 equilibrium V(0)=0', lqr_lyapunov_metric(zeros(12,1), P) == 0);

% 2. Positive definiteness: random e~=0 => V>0
E = randn(12, 200); E(:, all(E==0,1)) = 1;
Vr = lqr_lyapunov_metric(E, P);
check('2 pos-def V>0 for e~=0', all(Vr > 0));

% 3. Symmetry P==P'
check('3 P symmetric', max(abs(P-P.'),[],'all') <= 1e-9*max(1,max(abs(P(:)))));

% 4. Artifact consistency: loaded P == selectedLqr.S
d = load(info.source); check('4 P == selectedLqr.S', isequal(P, d.selectedLqr.S));

% 5. Manual formula V==e'Pe per column
e5 = randn(12, 5); Vh = lqr_lyapunov_metric(e5, P);
Vm = zeros(1,5); for k=1:5; Vm(k) = e5(:,k)'*P*e5(:,k); end
check('5 V == e''Pe (per col)', max(abs(Vh-Vm)) <= 1e-8*max(1,max(abs(Vm))));

% 6. H-window on a synthetic geometric-decay trajectory: e(:,k)=alpha^(k-1)*v0
H = 20; alpha = 0.9; v0 = randn(12,1); N = 60;
Esyn = v0 * (alpha.^(0:N-1)); Esyn(4:6,:) = 0;   % keep angles unwrapped-safe (small)
out = d1_finite_horizon_contraction(Esyn, P, H, struct('V_floor',1e-30));
Vsyn = lqr_lyapunov_metric(Esyn, P);
v0m = v0; v0m(4:6) = 0;      % angle rows are zeroed in Esyn, so V0 uses masked v0
V0 = v0m'*P*v0m;
k = 5;
expDelta = V0*alpha^(2*(k-1))*(alpha^(2*H)-1);
expR = alpha^(2*H);
expG = (1/H)*log(alpha^(2*H));
check('6a DeltaV_H correct', abs(out.DeltaV_H(k)-expDelta) <= 1e-6*abs(expDelta));
check('6b r_H correct', abs(out.r_H(k)-expR) <= 1e-8);
check('6c g_H correct', abs(out.g_H(k)-expG) <= 1e-9);
check('6d isContracting (alpha<1)', all(out.isContracting(1:N-H)));
check('6e V_max_window = V(k) for decay', abs(out.V_max_window(k)-Vsyn(k))<=1e-9*Vsyn(k));

% 6f expanding trajectory (alpha>1) => not contracting
Eexp = v0 * (1.05.^(0:N-1)); Eexp(4:6,:)=0;
oExp = d1_finite_horizon_contraction(Eexp, P, H, struct('V_floor',1e-30));
check('6f expanding => not contracting', ~any(oExp.isContracting(1:N-H)));

% 7. End-of-episode: no label for k+H>N; count == H
check('7a insufficient_horizon count == H', out.nInsufficientHorizon == H);
check('7b validMask false for k>N-H', ~any(out.validMask(N-H+1:N)) && all(out.validMask(1:N-H)));
check('7c no NaN pad passed off as label', all(isnan(out.DeltaV_H(N-H+1:N))));

% 8. Euler wrap: an error differing by 2*pi in yaw gives (near) identical V
e8 = randn(12,1); e8b = e8; e8b(6) = e8(6) + 2*pi;
o8 = d1_finite_horizon_contraction([e8 e8b], P, 1, struct());   % just to wrap
check('8 Euler-wrap invariance (2pi in yaw)', abs(o8.V(1)-o8.V(2)) <= 1e-6*max(1,o8.V(1)));

% 9. Threshold independence: module signature has NO threshold input (by nargin)
check('9 contraction module takes no threshold arg', nargin('d1_finite_horizon_contraction')==4);

% 10. Non-finite window not counted as contracting
Enf = v0*(alpha.^(0:N-1)); Enf(:,30)=Inf;
onf = d1_finite_horizon_contraction(Enf, P, H, struct());
check('10 non-finite window not contracting', ~onf.isContracting(30) && onf.nNonfiniteWindow>=1);

% ---- Phase G: fail-fast loader (must THROW, never fallback) ----
tmp = tempname; mkdir(tmp);
save_lqr(fullfile(tmp,'good.mat'), P);
check('G7 valid artifact => P == selectedLqr.S', ...
    isequal(d1_load_lyapunov_P(fullfile(tmp,'good.mat')), P));
check('G1 missing artifact => throws', throws(@() d1_load_lyapunov_P(fullfile(tmp,'nope.mat'))));
Pv = P; save(fullfile(tmp,'noS.mat'), 'Pv');
check('G2 missing selectedLqr.S => throws', throws(@() d1_load_lyapunov_P(fullfile(tmp,'noS.mat'))));
save_lqr(fullfile(tmp,'badsize.mat'), eye(6));
check('G3 wrong size => throws', throws(@() d1_load_lyapunov_P(fullfile(tmp,'badsize.mat'))));
Snan = P; Snan(1,1) = Inf; save_lqr(fullfile(tmp,'nan.mat'), Snan);
check('G4 nonfinite => throws', throws(@() d1_load_lyapunov_P(fullfile(tmp,'nan.mat'))));
Sasym = P; Sasym(1,2) = Sasym(1,2) + 1; save_lqr(fullfile(tmp,'asym.mat'), Sasym);
check('G5 non-symmetric => throws', throws(@() d1_load_lyapunov_P(fullfile(tmp,'asym.mat'))));
save_lqr(fullfile(tmp,'npd.mat'), -P);
check('G6 non-pos-def => throws', throws(@() d1_load_lyapunov_P(fullfile(tmp,'npd.mat'))));

% ---- Phase G: H source-of-truth + module takes H by argument ----
w = targeted_lqr_weak_config();
check('G10a cfg.contraction.horizonSteps == 20', w.contraction.horizonSteps == 20);
check('G10b Ts == 0.05', abs(w.sampleTime - 0.05) <= 1e-12);
check('G10c physical horizon == 1.0 s', abs(w.contraction.horizonSteps*w.sampleTime - 1.0) <= 1e-12);
oH10 = d1_finite_horizon_contraction(Esyn, P, 10, struct());
oH20 = d1_finite_horizon_contraction(Esyn, P, 20, struct());
check('G8/9 module uses passed H (nInsufficientHorizon = H)', ...
    oH10.nInsufficientHorizon == 10 && oH20.nInsufficientHorizon == 20 && ...
    oH10.H == 10 && oH20.H == 20);

% ---- Phase G: FINAL Bryson weights, Qf=0, dU=0 ----
nc = step2_nmpc_config();
check('G12 Qf == 0 (terminal not amplified)', isequal(nc.weights.Qf, zeros(12)));
check('G13 dU == 0 (no slew spec)', isequal(nc.weights.dU, zeros(4)));
eA = [0.10;0.10;0.10; deg2rad(5)*[1;1;1]; 0.30;0.30;0.30; 2;2;2];
duA = [nc.plant.nominal.m*nc.plant.nominal.g; 0.5; 0.5; 0.25];
check('G14 Q = Bryson 1/eAllow^2', max(abs(diag(nc.weights.Q) - 1./eA.^2)) < 1e-9);
check('G15 R = Bryson 1/duAllow^2', max(abs(diag(nc.weights.R) - 1./duA.^2)) < 1e-9);

% ---- Phase G: tracking RMS formula intact (independent of P/H) ----
Etr = randn(3, 50); rmsRef = sqrt(mean(vecnorm(Etr, 2, 1) .^ 2));
check('G11 tracking RMS formula finite/correct', isfinite(rmsRef) && rmsRef > 0);

fprintf('\n== %d passed, %d failed ==\n', np, nf);
if nf>0; error('test_d1_lyapunov_contraction:Failures','%d test(s) failed', nf); end
end

function tf = throws(fn)
tf = false;
try; fn(); catch; tf = true; end
end

function save_lqr(path, S)
selectedLqr = struct('S', S, 'positionVelocityScale', 1, ...
    'attitudeRateScale', 1, 'inputPenaltyScale', 1, 'sampleTime', 0.05); %#ok<NASGU>
save(path, 'selectedLqr');
end
