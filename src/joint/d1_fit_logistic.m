function clf = d1_fit_logistic(X, y, soft)
%D1_FIT_LOGISTIC L2 logistic regression (lambda = 1e-3) on standardized features, 800 full
% gradient steps (lr 0.5). X: n x d, y: n x 1.
%   soft = false: y in {0,1}, class-balanced weights (c_LQR: P(contract)).
%   soft = true : y in [0,1] (soft labels, cross-entropy), unweighted (c_S: predicts s_k).
% acc = mean((p>0.5) == (y>0.5)), base = mean(y).
if nargin < 3, soft = false; end
if isempty(y)
    clf = struct('w',[],'b',0,'mu',[],'sg',[],'acc',NaN,'base',NaN,'n',0); return;
end
y = y(:); mu = mean(X,1); sg = std(X,0,1) + 1e-6; Z = (X - mu)./sg;
[n, d] = size(Z); w = zeros(d,1); b = 0; lr = 0.5; lam = 1e-3;
if soft
    sw = ones(n,1);
else
    p1 = mean(y); wpos = 1/max(p1,1e-3); wneg = 1/max(1-p1,1e-3);
    sw = y*wpos + (1-y)*wneg; sw = sw/mean(sw);
end
for it = 1:800
    p = 1./(1+exp(-(Z*w + b)));
    g = Z.'*((p - y).*sw)/n + lam*w; gb = mean((p - y).*sw);
    w = w - lr*g; b = b - lr*gb;
end
p = 1./(1+exp(-(Z*w + b)));
clf = struct('w',w,'b',b,'mu',mu,'sg',sg,'acc',mean((p>0.5)==(y>0.5)),'base',mean(y),'n',n);
end
