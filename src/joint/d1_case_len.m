function T = d1_case_len(Xref, cfg)
%D1_CASE_LEN Number of control steps of a flight on reference Xref (common flight rule):
% every flight -- teacher, LQR, surrogate, blend, consolidation, evaluation -- flies the
% same length, limited so that the teacher horizon Xref(:,k:k+N) stays inside the
% reference (979 steps for the 1000-sample references).
T = min(cfg.stepsPerCase, size(Xref,2) - cfg.N - 1);
end
