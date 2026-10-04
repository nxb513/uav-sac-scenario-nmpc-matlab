function [P, info] = d1_load_lyapunov_P(lqrPath)
%D1_LOAD_LYAPUNOV_P Load the D1 Lyapunov matrix P = selectedLqr.S.
%
%   [P, info] = d1_load_lyapunov_P()          % default selected-LQR artifact
%   [P, info] = d1_load_lyapunov_P(lqrPath)   % explicit artifact path
%
% P is taken DIRECTLY from selectedLqr.S (the discrete Riccati solution returned
% by dlqr(A,B,Q_LQR,R_LQR) inside the frozen selected LQR artifact). It is NOT
% recomputed from Q_NMPC/R_NMPC, NOT Qf, NOT hard-coded, and NOT re-tuned.
%
% info records provenance (source path, the LQR scale factors, sample time).

if nargin < 1 || isempty(lqrPath)
    lqrPath = default_lqr_path();
end
if exist(lqrPath, 'file') ~= 2
    error('d1_load_lyapunov_P:NotFound', ...
        ['Selected LQR artifact not found: %s\n(commit/provide the selected LQR ' ...
         'so P = selectedLqr.S can be loaded.)'], lqrPath);
end

data = load(lqrPath);
if ~isfield(data, 'selectedLqr') || ~isfield(data.selectedLqr, 'S')
    error('d1_load_lyapunov_P:NoRiccati', ...
        'Artifact %s has no selectedLqr.S (discrete Riccati matrix).', lqrPath);
end
L = data.selectedLqr;
P = L.S;

% Validation (same contract as lqr_lyapunov_metric): 12x12, finite, symmetric, PD.
% Every message names the artifact path so a fail-fast is self-describing.
if ~isequal(size(P), [12, 12])
    error('d1_load_lyapunov_P:BadSize', ...
        'selectedLqr.S must be 12-by-12 (got %s) in %s.', mat2str(size(P)), lqrPath);
end
if ~all(isfinite(P(:)))
    error('d1_load_lyapunov_P:Nonfinite', ...
        'selectedLqr.S contains NaN/Inf in %s.', lqrPath);
end
symErr = max(abs(P - P.'), [], 'all');
if symErr > 1e-9 * max(1, max(abs(P(:))))
    error('d1_load_lyapunov_P:Asymmetric', ...
        'selectedLqr.S not symmetric (max asym %.3e) in %s.', symErr, lqrPath);
end
[~, notPD] = chol((P + P.') / 2);
if notPD ~= 0
    error('d1_load_lyapunov_P:NotPosDef', ...
        'selectedLqr.S not positive definite in %s.', lqrPath);
end

info = struct();
info.source = lqrPath;
info.field = 'selectedLqr.S';
info.minEig = min(eig((P + P.') / 2));
info.maxAsym = symErr;
info.positionVelocityScale = field_or(L, 'positionVelocityScale', NaN);
info.attitudeRateScale = field_or(L, 'attitudeRateScale', NaN);
info.inputPenaltyScale = field_or(L, 'inputPenaltyScale', NaN);
info.sampleTime = field_or(L, 'sampleTime', NaN);
info.selectionCandidateId = field_or(L, 'selectionCandidateId', '');
info.sha256 = read_sidecar_sha(lqrPath);   % '' if no <name>.sha256 sidecar
info.loadedAt = datestr(now, 'yyyy-mm-dd HH:MM:SS');
end

function sha = read_sidecar_sha(lqrPath)
sha = '';
[d, base, ~] = fileparts(lqrPath);
shaFile = fullfile(d, [base, '.sha256']);
if exist(shaFile, 'file') == 2
    fid = fopen(shaFile, 'r');
    if fid >= 0
        line = fgetl(fid); fclose(fid);
        if ischar(line)
            tok = strsplit(strtrim(line));
            sha = tok{1};
        end
    end
end
end

function p = default_lqr_path()
here = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(here));      % src/common -> project root
cfg = targeted_lqr_weak_config();
% D1 Lyapunov/baseline LQR = the Bryson artifact (cfg.contraction.lqrArtifactPath).
if isfield(cfg, 'contraction') && isfield(cfg.contraction, 'lqrArtifactPath') ...
        && ~isempty(cfg.contraction.lqrArtifactPath)
    rel = cfg.contraction.lqrArtifactPath;
else
    error('d1_load_lyapunov_P:NoDefault', ...
        'targeted_lqr_weak_config has no contraction.lqrArtifactPath.');
end
p = fullfile(projectRoot, rel);
end

function v = field_or(s, name, default)
if isfield(s, name) && ~isempty(s.(name)); v = s.(name); else; v = default; end
end
