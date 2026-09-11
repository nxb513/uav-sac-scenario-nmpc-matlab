function [thetaSamples, Xi, info] = quad_sample_uncertainty(cfg, sampleCount, domain, seed, method)
%QUAD_SAMPLE_UNCERTAINTY Sample physical uncertainty scenarios.
%
% domain is 'train' or 'ood'. method is 'lhs' or 'uniform'.
%
% The three inertia coordinates (columns 2:4 of Xi = the Ix, Iy, Iz scalings)
% are drawn by REJECTION so that every sampled plant has a physically
% realizable inertia (positive principal moments satisfying the triangle
% inequalities, quad_inertia_consistent). The remaining 11 coordinates keep
% their LHS/uniform design; inertia rejection is drawn AFTER the base design and
% only overwrites columns 2:4, so it never perturbs the mass, drag, gain or any
% downstream (disturbance/reference) draws. Optional third output INFO reports
% the acceptance statistics. See docs/notes/
% d1_physical_validity_fix_recommendations_20260912.md.

if nargin < 3 || isempty(domain)
    domain = 'train';
end
if nargin < 4
    seed = [];
end
if nargin < 5 || isempty(method)
    method = cfg.uncertainty.defaultMethod;
end

if sampleCount <= 0 || sampleCount ~= floor(sampleCount)
    error('quad_sample_uncertainty:BadSampleCount', ...
          'sampleCount must be a positive integer.');
end

if ~isfield(cfg.uncertainty, domain)
    error('quad_sample_uncertainty:BadDomain', 'Unknown domain: %s', domain);
end

if ~isempty(seed)
    rng(seed, 'twister');
end

rho = cfg.uncertainty.(domain).rho(:);
dim = numel(rho);

switch lower(method)
    case 'lhs'
        Xi = lhs_unit_box(sampleCount, dim);
    case 'uniform'
        Xi = -1 + 2 * rand(sampleCount, dim);
    otherwise
        error('quad_sample_uncertainty:BadMethod', 'Unknown method: %s', method);
end

% Overwrite the inertia coordinates (Ix, Iy, Iz = columns 2:4) with rejection
% samples that keep every plant's inertia physically realizable.
[Xi(:, 2:4), info] = sample_consistent_inertia(cfg.nominal.J, rho(2:4), ...
    sampleCount);

thetaSamples(1, sampleCount) = cfg.nominal;
for k = 1:sampleCount
    thetaSamples(k) = quad_apply_uncertainty(cfg.nominal, Xi(k, :).', rho);
end
end

function [xiJ, info] = sample_consistent_inertia(Jnom, rhoJ, sampleCount)
%SAMPLE_CONSISTENT_INERTIA Rejection-sample physically valid inertia scalings.
Jd = diag(Jnom);
Jd = Jd(:);
rhoJ = rhoJ(:);
retryCap = 1000;
xiJ = zeros(sampleCount, 3);
attempts = zeros(sampleCount, 1);
for k = 1:sampleCount
    accepted = false;
    for attempt = 1:retryCap
        candidate = -1 + 2 * rand(3, 1);
        J = Jd .* (1 + rhoJ .* candidate);
        if quad_inertia_consistent(J)
            xiJ(k, :) = candidate.';
            attempts(k) = attempt;
            accepted = true;
            break;
        end
    end
    if ~accepted
        error('quad_sample_uncertainty:InertiaRejection', ...
            ['Could not draw a physically consistent inertia within %d ' ...
             'attempts (sample %d). Check the inertia envelope rho.'], ...
            retryCap, k);
    end
end
info = struct();
info.inertiaMeanAttempts = mean(attempts);
info.inertiaMaxAttempts = max(attempts);
info.inertiaAcceptanceRate = 1 / mean(attempts);
end

function Xi = lhs_unit_box(sampleCount, dim)
Xi = zeros(sampleCount, dim);
for j = 1:dim
    strata = ((0:sampleCount-1).' + rand(sampleCount, 1)) / sampleCount;
    Xi(:, j) = -1 + 2 * strata(randperm(sampleCount));
end
end
