function [thetaSamples, Xi] = quad_sample_uncertainty(cfg, sampleCount, domain, seed, method)
%QUAD_SAMPLE_UNCERTAINTY Sample physical uncertainty scenarios.
%
% domain is 'train' or 'ood'. method is 'lhs' or 'uniform'.

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

thetaSamples(1, sampleCount) = cfg.nominal;
for k = 1:sampleCount
    thetaSamples(k) = quad_apply_uncertainty(cfg.nominal, Xi(k, :).', rho);
end
end

function Xi = lhs_unit_box(sampleCount, dim)
Xi = zeros(sampleCount, dim);
for j = 1:dim
    strata = ((0:sampleCount-1).' + rand(sampleCount, 1)) / sampleCount;
    Xi(:, j) = -1 + 2 * strata(randperm(sampleCount));
end
end
