function run_step1_plant_tests()
%RUN_STEP1_PLANT_TESTS Smoke tests for pipeline step 1 plant.

projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(projectRoot, 'configs'));
addpath(genpath(fullfile(projectRoot, 'src')));

cfg = step1_plant_config();
theta = cfg.nominal;

test_hover_derivative(theta);
test_hover_rk4(theta);
test_thrust_sign(theta);
test_torque_signs(theta);
test_input_saturation(theta);
test_uncertainty_sampling(cfg);
test_disturbance_generation(cfg);
test_model_shift(cfg);

fprintf('Step 1 plant tests passed.\n');
end

function test_disturbance_generation(plantCfg)
disturbanceCfg = step3_disturbance_config();
types = {'zero', 'constant', 'gust', 'sinusoidal', 'stochastic'};
sampleTime = 0.05;
stepCount = 40;

for i = 1:numel(types)
    first = quad_generate_disturbance_episode(plantCfg, disturbanceCfg, ...
        types{i}, 'train', 2, sampleTime, stepCount, 9100 + i);
    second = quad_generate_disturbance_episode(plantCfg, disturbanceCfg, ...
        types{i}, 'train', 2, sampleTime, stepCount, 9100 + i);

    assert(isequal(size(first.forceSeries), [3, stepCount + 1]));
    assert(isequal(size(first.torqueSeries), [3, stepCount + 1]));
    assert(max(abs(first.forceSeries(:) - second.forceSeries(:))) == 0);
    assert(max(abs(first.torqueSeries(:) - second.torqueSeries(:))) == 0);
    assert(first.hiddenFromController);
    assert(max(sqrt(sum(first.forceSeries.^2, 1))) <= ...
           first.forceCandidateBound + 1e-12);
    assert(all(abs(first.torqueSeries) <= first.torqueCandidateBound + 1e-12, 'all'));

    queryIndex = min(5, stepCount + 1);
    sampled = quad_disturbance(first.time(queryIndex), first);
    assert(max(abs(sampled.force - first.forceSeries(:, queryIndex))) < 1e-12);
    assert(max(abs(sampled.torque - first.torqueSeries(:, queryIndex))) < 1e-12);
end

zeroSpec = quad_generate_disturbance_episode(plantCfg, disturbanceCfg, ...
    'zero', 'train', 1, sampleTime, stepCount, 99);
assert(all(zeroSpec.forceSeries == 0, 'all'));
assert(all(zeroSpec.torqueSeries == 0, 'all'));

x0 = zeros(12, 1);
u0 = quad_hover_input(plantCfg.nominal);
constantSpec = quad_generate_disturbance_episode(plantCfg, disturbanceCfg, ...
    'constant', 'train', 2, sampleTime, stepCount, 101);
activeIndex = find(sqrt(sum(constantSpec.forceSeries.^2, 1)) > 0, 1);
[dx, aux] = quad_dynamics(constantSpec.time(activeIndex), x0, u0, ...
                          plantCfg.nominal, constantSpec);
assert(max(abs(aux.disturbance.force - constantSpec.forceSeries(:, activeIndex))) < 1e-12);
assert(norm(dx(7:9)) > 0, 'External force must affect translational acceleration.');
end

function test_hover_derivative(theta)
x0 = zeros(12, 1);
u0 = quad_hover_input(theta);
dx = quad_dynamics(0, x0, u0, theta, []);
assert(max(abs(dx)) < 1e-10, 'Hover derivative is not zero.');
end

function test_hover_rk4(theta)
x = zeros(12, 1);
u = quad_hover_input(theta);
dt = 0.005;
for k = 1:round(2 / dt)
    x = quad_step_rk4((k - 1) * dt, x, u, dt, theta, []);
end
assert(norm(x) < 1e-9, 'RK4 hover state drifted.');
end

function test_thrust_sign(theta)
x0 = zeros(12, 1);
u = quad_hover_input(theta);
u(1) = 1.1 * u(1);
dx = quad_dynamics(0, x0, u, theta, []);
assert(dx(9) > 0, 'With z-up convention, extra thrust must accelerate upward.');
end

function test_torque_signs(theta)
x0 = zeros(12, 1);
u = quad_hover_input(theta);

uRoll = u; uRoll(2) = 0.01;
dxRoll = quad_dynamics(0, x0, uRoll, theta, []);
assert(dxRoll(10) > 0, 'Positive tau_phi must increase p_dot.');

uPitch = u; uPitch(3) = 0.01;
dxPitch = quad_dynamics(0, x0, uPitch, theta, []);
assert(dxPitch(11) > 0, 'Positive tau_theta must increase q_dot.');

uYaw = u; uYaw(4) = 0.01;
dxYaw = quad_dynamics(0, x0, uYaw, theta, []);
assert(dxYaw(12) > 0, 'Positive tau_psi must increase r_dot.');
end

function test_input_saturation(theta)
u = [Inf; 99; -99; 99];
uSat = quad_saturate_input(u, theta);
assert(uSat(1) == theta.inputLimits.T(2), 'T upper saturation failed.');
assert(uSat(2) == theta.inputLimits.tau(1, 2), 'tau_phi upper saturation failed.');
assert(uSat(3) == theta.inputLimits.tau(2, 1), 'tau_theta lower saturation failed.');
assert(uSat(4) == theta.inputLimits.tau(3, 2), 'tau_psi upper saturation failed.');
end

function test_uncertainty_sampling(cfg)
[s1, xi1] = quad_sample_uncertainty(cfg, 8, 'train', 1234, 'lhs');
[s2, xi2] = quad_sample_uncertainty(cfg, 8, 'train', 1234, 'lhs');

assert(max(abs(xi1(:) - xi2(:))) == 0, 'Sampling is not reproducible.');
assert(all(abs(xi1(:)) <= 1), 'Sample xi is outside [-1,1].');

mValues = [s1.m];
rhoM = cfg.uncertainty.train.rho(1);
assert(all(mValues >= cfg.nominal.m * (1 - rhoM) - 1e-12), 'm below train range.');
assert(all(mValues <= cfg.nominal.m * (1 + rhoM) + 1e-12), 'm above train range.');

assert(abs(s1(1).m - s2(1).m) < eps, 'Struct sample reproducibility failed.');
assert(all(arrayfun(@(theta) isequal(theta.inputLimits, ...
    cfg.nominal.inputLimits), s1)), ...
    'Parametric uncertainty must preserve fixed hardware input limits.');
end

function test_model_shift(cfg)
theta0 = cfg.nominal;
shift = cfg.modelShift.payload25;
before = quad_apply_model_shift(theta0, shift, shift.startTime - 1e-3);
after = quad_apply_model_shift(theta0, shift, shift.startTime);

assert(abs(before.m - theta0.m) < eps, 'Model shift applied too early.');
assert(abs(after.m - 1.25 * theta0.m) < 1e-12, 'Payload mass shift failed.');
assert(isequal(after.inputLimits, theta0.inputLimits), ...
    'Payload shift must preserve fixed hardware input limits.');
end
