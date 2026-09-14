function V = lqr_lyapunov_metric(e, P)
%LQR_LYAPUNOV_METRIC Common quadratic tracking-error Lyapunov metric V = e' P e.
%
%   V = LQR_LYAPUNOV_METRIC(e, P)
%
% Inputs:
%   e : 12-by-N (or 12-by-1) tracking error, ordered exactly like the plant/LQR
%       state [px py pz phi theta psi vx vy vz p q r]. Angles are in RADIANS and
%       must already be wrapped to (-pi, pi] the same way nmpc_state_error /
%       d1_wrap_state_error do. NO state normalization is applied before P.
%   P : 12-by-12 symmetric positive-definite matrix. For D1 this is the discrete
%       Riccati solution selectedLqr.S returned by dlqr(A,B,Q_LQR,R_LQR); load it
%       with d1_load_lyapunov_P (never recompute / hard-code it here).
%
% Output:
%   V : 1-by-N with V(k) = e(:,k)' * P * e(:,k). Columns whose error is
%       non-finite yield a non-finite V (propagated, not silently zeroed).
%
% The metric is not normalized and does not use any fixed error threshold.

if size(e, 1) ~= 12
    error('lqr_lyapunov_metric:BadError', 'e must have 12 rows.');
end
assert_valid_P(P);

% V(k) = e_k' P e_k, computed column-wise without forming e'*P*e for all pairs.
V = sum((P * e) .* e, 1);
end

function assert_valid_P(P)
if ~isequal(size(P), [12, 12])
    error('lqr_lyapunov_metric:BadP', 'P must be 12-by-12.');
end
if ~all(isfinite(P(:)))
    error('lqr_lyapunov_metric:NonfiniteP', 'P must be finite.');
end
symTol = 1e-9 * max(1, max(abs(P(:))));
if max(abs(P - P.'), [], 'all') > symTol
    error('lqr_lyapunov_metric:AsymmetricP', ...
        'P must be symmetric within tolerance (max asym %.3e > %.3e).', ...
        max(abs(P - P.'), [], 'all'), symTol);
end
% Positive definiteness via Cholesky on the symmetric part (robust, cheap).
[~, notPosDef] = chol((P + P.') / 2);
if notPosDef ~= 0
    error('lqr_lyapunov_metric:NotPosDef', ...
        'P must be positive definite (Cholesky failed).');
end
end
