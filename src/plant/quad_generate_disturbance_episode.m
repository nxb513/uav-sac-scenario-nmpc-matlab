function spec = quad_generate_disturbance_episode(plantCfg, disturbanceCfg, type, domain, levelIndex, sampleTime, stepCount, seed)
%QUAD_GENERATE_DISTURBANCE_EPISODE Generate a reproducible hidden realization.

if nargin < 4 || isempty(domain)
    domain = 'train';
end
if nargin < 5 || isempty(levelIndex)
    levelIndex = 1;
end
if nargin < 8 || isempty(seed)
    seed = 0;
end
if sampleTime <= 0 || stepCount <= 0 || stepCount ~= floor(stepCount)
    error('quad_generate_disturbance_episode:BadTimeGrid', ...
          'sampleTime and integer stepCount must be positive.');
end
if ~isfield(disturbanceCfg, domain)
    error('quad_generate_disturbance_episode:BadDomain', ...
          'Unknown disturbance domain: %s', domain);
end

domainCfg = disturbanceCfg.(domain);
if levelIndex < 1 || levelIndex > numel(domainCfg.forceFractionHover) || ...
        levelIndex > numel(domainCfg.torqueFractionLimit)
    error('quad_generate_disturbance_episode:BadLevel', ...
          'levelIndex is outside the candidate level range.');
end

previousRng = rng;
cleanup = onCleanup(@() rng(previousRng));
rng(seed, 'twister');

time = (0:stepCount) * sampleTime;
episodeDuration = stepCount * sampleTime;
forceSeries = zeros(3, stepCount + 1);
torqueSeries = zeros(3, stepCount + 1);

forceBound = plantCfg.nominal.m * plantCfg.nominal.g * ...
             domainCfg.forceFractionHover(levelIndex);
torqueLimits = max(abs(plantCfg.nominal.inputLimits.tau), [], 2);
torqueBound = torqueLimits * domainCfg.torqueFractionLimit(levelIndex);
amplitudeRange = [0.50, 1.00];
if isfield(domainCfg, 'amplitudeFractionRange')
    amplitudeRange = domainCfg.amplitudeFractionRange;
end
forceAmplitude = forceBound * uniform_between(amplitudeRange);
torqueAmplitude = torqueBound .* (amplitudeRange(1) + ...
    diff(amplitudeRange) .* rand(3, 1));
forceDirection = random_unit_vector();
torqueDirection = random_unit_vector();

onsetRange = domainCfg.onsetFraction * episodeDuration;
durationRange = domainCfg.durationFraction * episodeDuration;
eventStartTime = uniform_between(onsetRange);
eventDuration = uniform_between(durationRange);
eventEndTime = min(episodeDuration, eventStartTime + eventDuration);

switch lower(type)
    case 'zero'
        eventStartTime = NaN;
        eventEndTime = NaN;

    case 'constant'
        active = time >= eventStartTime;
        forceSeries(:, active) = repmat(forceAmplitude * forceDirection, 1, nnz(active));
        torqueSeries(:, active) = repmat(torqueAmplitude .* torqueDirection, 1, nnz(active));
        eventEndTime = episodeDuration;

    case 'gust'
        active = time >= eventStartTime & time <= eventEndTime;
        forceSeries(:, active) = repmat(forceAmplitude * forceDirection, 1, nnz(active));
        torqueSeries(:, active) = repmat(torqueAmplitude .* torqueDirection, 1, nnz(active));

    case 'sinusoidal'
        frequency = uniform_between(domainCfg.sinusoidFrequencyHz);
        forcePhase = 2 * pi * rand();
        torquePhase = 2 * pi * rand();
        active = time >= eventStartTime & time <= eventEndTime;
        localTime = time(active) - eventStartTime;
        forceWave = sin(2 * pi * frequency * localTime + forcePhase);
        torqueWave = sin(2 * pi * frequency * localTime + torquePhase);
        forceSeries(:, active) = (forceAmplitude * forceDirection) .* forceWave;
        torqueSeries(:, active) = (torqueAmplitude .* torqueDirection) .* torqueWave;

    case 'stochastic'
        correlation = uniform_between(domainCfg.stochasticCorrelation);
        active = time >= eventStartTime;
        forceNoise = colored_noise(3, nnz(active), correlation);
        torqueNoise = colored_noise(3, nnz(active), correlation);
        forceNoise = normalize_vector_series(forceNoise);
        torqueNoise = torqueNoise / max(1.0, max(abs(torqueNoise), [], 'all'));
        forceSeries(:, active) = forceAmplitude * forceNoise;
        torqueSeries(:, active) = torqueAmplitude .* torqueNoise;
        eventEndTime = episodeDuration;

    otherwise
        error('quad_generate_disturbance_episode:BadType', ...
              'Unknown disturbance type: %s', type);
end

spec.type = lower(type);
spec.domain = domain;
spec.levelIndex = levelIndex;
spec.seed = seed;
spec.sampleTime = sampleTime;
spec.time = time;
spec.forceSeries = forceSeries;
spec.torqueSeries = torqueSeries;
spec.interpolation = 'previous';
spec.eventStartTime = eventStartTime;
spec.eventEndTime = eventEndTime;
spec.forceCandidateBound = forceBound;
spec.torqueCandidateBound = torqueBound;
spec.forcePeak = max(sqrt(sum(forceSeries.^2, 1)));
spec.torquePeak = max(sqrt(sum(torqueSeries.^2, 1)));
spec.hiddenFromController = true;
spec.status = 'candidate_not_final';
end

function direction = random_unit_vector()
direction = randn(3, 1);
direction = direction / max(norm(direction), eps);
end

function value = uniform_between(range)
range = range(:);
if numel(range) ~= 2 || range(2) < range(1)
    error('quad_generate_disturbance_episode:BadRange', ...
          'Candidate ranges must contain [minimum, maximum].');
end
value = range(1) + (range(2) - range(1)) * rand();
end

function noise = colored_noise(dimension, sampleCount, correlation)
noise = zeros(dimension, sampleCount);
if sampleCount == 0
    return;
end
noise(:, 1) = randn(dimension, 1);
innovationScale = sqrt(max(0.0, 1.0 - correlation^2));
for k = 2:sampleCount
    noise(:, k) = correlation * noise(:, k - 1) + ...
                  innovationScale * randn(dimension, 1);
end
end

function series = normalize_vector_series(series)
norms = sqrt(sum(series.^2, 1));
peakNorm = max(norms);
if peakNorm > 1.0
    series = series / peakNorm;
end
end
