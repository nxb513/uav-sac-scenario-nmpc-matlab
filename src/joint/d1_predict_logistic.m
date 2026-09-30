function p = d1_predict_logistic(clf, X)
%D1_PREDICT_LOGISTIC P(contract) of a fitted logistic (X: n x d -> p: n x 1);
% an untrained classifier returns 0.5.
if isempty(clf) || isempty(clf.w), p = 0.5*ones(size(X,1),1); return; end
Z = (X - clf.mu)./clf.sg; p = 1./(1+exp(-(Z*clf.w + clf.b)));
end
