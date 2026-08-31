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

qDiag = [20; 20; 30; ...
         5; 5; 3; ...
         2; 2; 3; ...
         0.25; 0.25; 0.20];
cfg.weights.Q = diag(qDiag);
cfg.weights.Qf = 4.0 * cfg.weights.Q;
cfg.weights.R = diag([0.02; 0.01; 0.01; 0.006]);
cfg.weights.dU = diag([0.002; 0.0005; 0.0005; 0.0003]);
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
