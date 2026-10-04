function cfg = step2_nmpc_config()
%STEP2_NMPC_CONFIG Configuration for fixed and scenario NMPC teachers.

plantCfg = step1_plant_config();

cfg.name = 'step2_uncertain_scenario_nmpc_teacher';
cfg.plant = plantCfg;
cfg.sampleTime = 0.05;
cfg.predictionHorizon = 8;
cfg.controlHorizon = 8;

cfg.scenario.count = 7;
cfg.scenario.domain = 'train';
cfg.scenario.method = plantCfg.uncertainty.defaultMethod;
cfg.scenario.seed = 26082602;

% Bryson's rule (Bryson & Ho, Applied Optimal Control; Okyere et al. 2019 for
% quadrotor LQR): Q_ii = 1/e_allow_i^2, R_jj = 1/du_allow_j^2. Physically
% normalized -> no heuristic hand-picked weights. Sources:
%   e_allow = preregistered 20-step tracking tolerances (d1_preregistration_
%     20260912 §5): position 0.10 m, attitude 5 deg, velocity 0.30 m/s, rate 2 rad/s.
%   du_allow = actuator limits (step1_plant_config): thrust deviation ~ hover m*g,
%     tau_phi/theta 0.5 N*m, tau_psi 0.25 N*m.
nom = plantCfg.nominal;
eAllow = [0.10; 0.10; 0.10; ...            % position [m]
          deg2rad(5.0) * [1; 1; 1]; ...    % attitude [rad] (5 deg)
          0.30; 0.30; 0.30; ...            % velocity [m/s]
          2.0; 2.0; 2.0];                  % body rate [rad/s]
duAllow = [nom.m * nom.g; ...              % thrust deviation ~ hover m*g [N]
           0.5; 0.5; 0.25];                % torques [N*m] (actuator limits)
cfg.weights.Q = diag(1 ./ eAllow .^ 2);
cfg.weights.R = diag(1 ./ duAllow .^ 2);
% Terminal cost: the stage loop already penalizes the terminal state e_N with Q
% (it sums e_1..e_N), so Qf=0 means the terminal state is NOT amplified. (Qf=Q
% would double-weight e_N to 2Q; Qf=4Q to 5Q.)
cfg.weights.Qf = zeros(12);
% No actuator slew/rate-limit specification exists in the repo, so no input-rate
% penalty is invented: dU = 0.
cfg.weights.dU = zeros(4);
cfg.weights.inputReference = 'hover';

cfg.constraints.enableStateBounds = true;
cfg.constraints.enforceScenarioStateBounds = true;
cfg.constraints.stateLower = [-100; -100; -10; ...
                              -1.35; -1.35; -pi; ...
                              -25; -25; -25; ...
                              -10; -10; -10];
cfg.constraints.stateUpper = [100; 100; 100; ...
                              1.35; 1.35; pi; ...
                              25; 25; 25; ...
                              10; 10; 10];
cfg.constraints.maxTilt = deg2rad(70);

cfg.rollout.disturbance = [];
cfg.rollout.startTime = 0.0;

cfg.solver.name = 'fmincon';
cfg.solver.algorithm = 'sqp';
cfg.solver.display = 'none';
cfg.solver.maxIterations = 60;
cfg.solver.maxFunctionEvaluations = 6000;
cfg.solver.maxWallSeconds = Inf;
cfg.solver.constraintTolerance = 1e-5;
cfg.solver.optimalityTolerance = 1e-4;
cfg.solver.stepTolerance = 1e-7;
end
