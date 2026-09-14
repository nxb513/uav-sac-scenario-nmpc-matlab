function out = d1_finite_horizon_contraction(E, P, H, opts)
%D1_FINITE_HORIZON_CONTRACTION Primary D1 finite-horizon contraction metric on the
% Lyapunov function V = e' P e (P = discrete LQR Riccati selectedLqr.S).
%
%   out = d1_finite_horizon_contraction(E, P, H, opts)
%
% Inputs:
%   E    : 12-by-N raw tracking error (x - x_ref), ordered like the plant state.
%          Euler-angle rows (4:6) are wrapped to (-pi, pi] internally to match
%          nmpc_state_error (radians; NOT degrees, NOT normalized).
%   P    : 12-by-12 symmetric positive-definite Lyapunov matrix (load via
%          d1_load_lyapunov_P; = dlqr Riccati S).
%   H    : forward window in steps, PASSED IN by the caller from the single
%          source-of-truth cfg.contraction.horizonSteps (D1 frozen = 20 => 1.0 s
%          at Ts=0.05). This module never hard-codes the horizon.
%   opts : struct, optional. Field V_floor (default 1e-12) is a NUMERICAL-SAFETY
%          floor for the ratios/logs only; it is NOT a success/fail threshold.
%
% Per state k the finite-horizon label uses V_k and V_{k+H}:
%   DeltaV_H(k) = V(k+H) - V(k)
%   r_H(k)      = V(k+H) / max(V(k), V_floor)
%   g_H(k)      = (1/H) * log( max(V(k+H),V_floor) / max(V(k),V_floor) )
%   V_max_win(k)= max(V(k), ..., V(k+H))            (transient blow-up)
%   r_max(k)    = V_max_win(k) / max(V(k), V_floor)
% isContracting is a RAW finite-horizon contraction DIAGNOSTIC (strict, no frozen
% residual margin, so no tuned tolerance is introduced):
%   isContracting(k) = DeltaV_H(k) < 0   (equivalently r_H<1, g_H<0)
% A state whose window is non-finite (V_k or V_{k+H} not finite) is NOT counted as
% contracting. NOTE: this is NOT the final S7 confidence target -- the startup
% transient (episode starts at e~0, V grows to steady state) makes DeltaV_H>0 even
% when tracking is good, so the practical-contraction rule that maps
% {DeltaV_H, r_H, g_H, transient context} -> confidence is decided later at
% S7/Gate-dev. No rho_H / delta_H / practical threshold is introduced here.
%
% End-of-trajectory: labels are only produced for k with k+H <= N. States with
% k+H > N are marked insufficient_horizon and EXCLUDED (no padding, no reference
% wrap, no extrapolation, not called divergence). out.nInsufficientHorizon reports
% how many were excluded.

if nargin < 4 || isempty(opts); opts = struct(); end
if ~isfield(opts, 'V_floor') || isempty(opts.V_floor); opts.V_floor = 1e-12; end
Vfloor = opts.V_floor;
if ~(isscalar(H) && H >= 1 && H == floor(H))
    error('d1_finite_horizon_contraction:BadH', 'H must be a positive integer.');
end
if size(E, 1) ~= 12
    error('d1_finite_horizon_contraction:BadError', 'E must have 12 rows.');
end

% Wrap Euler-angle error rows to (-pi,pi], matching nmpc_state_error convention.
Ew = E;
Ew(4:6, :) = mod(Ew(4:6, :) + pi, 2 * pi) - pi;

V = lqr_lyapunov_metric(Ew, P);        % 1-by-N Lyapunov trajectory
N = numel(V);

kValid = 1:(N - H);                    % states with a full forward window
nInsufficient = N - numel(kValid);     % states with k+H > N (excluded)

DeltaV_H = nan(1, N); r_H = nan(1, N); g_H = nan(1, N);
V_max_win = nan(1, N); r_max = nan(1, N);
isContracting = false(1, N);
windowFinite = false(1, N);
validMask = false(1, N);

for k = kValid
    Vk = V(k); VkH = V(k + H);
    win = V(k:k + H);
    validMask(k) = true;
    windowFinite(k) = isfinite(Vk) && isfinite(VkH) && all(isfinite(win));
    baseK = max(Vk, Vfloor);
    DeltaV_H(k) = VkH - Vk;
    r_H(k) = VkH / baseK;
    g_H(k) = log(max(VkH, Vfloor) / baseK) / H;
    V_max_win(k) = max(win);
    r_max(k) = V_max_win(k) / baseK;
    isContracting(k) = windowFinite(k) && (DeltaV_H(k) < 0);
end

nValid = numel(kValid);
finiteValid = validMask & windowFinite;

out = struct();
out.H = H;
out.V = V;
out.validMask = validMask;                 % label defined (k+H <= N)
out.windowFinite = windowFinite;           % window all finite
out.nStates = N;
out.nValid = nValid;
out.nInsufficientHorizon = nInsufficient;  % excluded (k+H > N)
out.nNonfiniteWindow = sum(validMask & ~windowFinite);
out.DeltaV_H = DeltaV_H;
out.r_H = r_H;
out.g_H = g_H;
out.V_max_window = V_max_win;
out.r_max = r_max;
out.isContracting = isContracting;
% Episode-level aggregates over VALID + FINITE windows only (never silently drop
% non-finite: they are reported via nNonfiniteWindow and counted as non-contracting).
out.fractionContracting = frac(isContracting(validMask), validMask(validMask));
out.fractionExpanding = frac(validMask & windowFinite & ~isContracting, finiteValid);
out.meanG_H = mean_finite(g_H(finiteValid));
out.medianG_H = median_finite(g_H(finiteValid));
out.meanDeltaV_H = mean_finite(DeltaV_H(finiteValid));
out.maxRmax = max_finite(r_max(finiteValid));
out.Vstart = first_val(V);
out.Vend = last_val(V);
end

function f = frac(mask, denomMask)
d = sum(denomMask);
if d == 0; f = NaN; else; f = sum(mask) / d; end
end
function m = mean_finite(x); x = x(isfinite(x)); if isempty(x); m = NaN; else; m = mean(x); end; end
function m = median_finite(x); x = x(isfinite(x)); if isempty(x); m = NaN; else; m = median(x); end; end
function m = max_finite(x); x = x(isfinite(x)); if isempty(x); m = NaN; else; m = max(x); end; end
function v = first_val(x); if isempty(x); v = NaN; else; v = x(1); end; end
function v = last_val(x); if isempty(x); v = NaN; else; v = x(end); end; end
