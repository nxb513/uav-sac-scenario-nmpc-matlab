function stu = d1_ridge_fit(stats, cfg)
%D1_RIDGE_FIT Ridge regression of the normalized label y = Du*./resHalf on the student
% features phi, from per-flight sufficient statistics (exact, Follow-the-Leader on all data):
%   stats(c).G = sum phi*phi',  .H = sum phi*y',  .Q = sum y*y',  .n = #labels,  .fold
% Features are scaled by their RMS over all data (divided only, never centered, so that
% Du(0) = 0 is kept). lambda (relative to n) is chosen on cfg.ridgeGrid by case-grouped
% cfg.cvFolds-fold cross-validation (held-out MSE computed exactly from the statistics),
% then the model is refit on all data. Returns stu.W (4 x nPhi, physical units: Du = W*phi),
% stu.lambda, stu.cv (CV loss per lambda), stu.rms, stu.n, stu.nFlights.
P = cfg.nPhi; K = cfg.cvFolds;
G = zeros(P); H = zeros(P, 4); n = 0;
for c = 1:numel(stats), G = G + stats(c).G; H = H + stats(c).H; n = n + stats(c).n; end
assert(n > P, 'd1_ridge_fit:data', 'not enough labels (%d)', n);
rms = sqrt(diag(G)/n); rms(rms < 1e-12) = 1; D = diag(1./rms);
folds = [stats.fold];
cv = nan(1, numel(cfg.ridgeGrid));
for li = 1:numel(cfg.ridgeGrid)
    lam = cfg.ridgeGrid(li); loss = 0; nh = 0;
    for f = 1:K
        tr = folds ~= f; he = ~tr;
        if ~any(he) || ~any(tr), continue; end
        [Gtr, Htr, ~, ntr] = sum_stats(stats(tr), P);
        [Ghe, Hhe, Qhe, nhe] = sum_stats(stats(he), P);
        Wt = (D*Gtr*D + lam*ntr*eye(P)) \ (D*Htr);          % P x 4, scaled features
        loss = loss + trace(Qhe) - 2*trace(Wt.'*(D*Hhe)) + trace(Wt.'*(D*Ghe*D)*Wt);
        nh = nh + nhe;
    end
    cv(li) = loss / max(nh, 1);
end
[~, best] = min(cv);
lam = cfg.ridgeGrid(best);
Wt = (D*G*D + lam*n*eye(P)) \ (D*H);
stu.W = diag(cfg.resHalf) * Wt.' * D;                     % Du = resHalf .* (Wt' * (D*phi))
stu.lambda = lam; stu.cv = cv; stu.rms = rms; stu.n = n; stu.nFlights = numel(stats);
end

function [G, H, Q, n] = sum_stats(s, P)
G = zeros(P); H = zeros(P, 4); Q = zeros(4); n = 0;
for c = 1:numel(s), G = G + s(c).G; H = H + s(c).H; Q = Q + s(c).Q; n = n + s(c).n; end
end
