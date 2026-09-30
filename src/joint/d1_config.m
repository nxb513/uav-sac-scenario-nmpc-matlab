function cfg = d1_config()
%D1_CONFIG The single D1 configuration, used by training (run_d1_joint_pipeline), its
% consolidation / diagnostic modes and the final evaluation (d1_final_eval), so every
% flight uses the same constants. Environment overrides are listed in the pipeline header.
cfg.seed = d1_getenv_num('D1_SEED', 260914001);
cfg.runDir = d1_getenv_str('D1_RUN_DIR', fullfile('results','d1_joint', ...
    sprintf('seed%d', d1_getenv_num('D1_SEED',260914001))));
cfg.wallSeconds = d1_getenv_num('D1_WALL_SECONDS', 300);
cfg.stopIter = d1_getenv_num('D1_STOP_ITER', 0);     % stop exactly at this SAC iter (0 = off)
cfg.resume = strcmp(d1_getenv_str('D1_RESUME','0'),'1');
cfg.Ts = 0.05; cfg.N = 20; cfg.Nc = 5; cfg.H = 20; cfg.Qf = 0; cfg.dU = 0;
cfg.M = 5;                                          % robust scenarios (frozen)
cfg.solverType = d1_getenv_str('D1_SOLVER', 'SQP');  % 'SQP' (used by every run) | 'SQP_RTI'
cfg.stepsPerCase = d1_getenv_num('D1_STEPS', 1000);
cfg.casesPerEval = d1_getenv_num('D1_CASES_PER_EVAL', 20);
cfg.plant = d1_joint_plant_params();
lim = cfg.plant.nominal.inputLimits;                 % fixed nominal hardware limits
cfg.uLo = [lim.T(1); lim.tau(:,1)];                  % [0; -0.5; -0.5; -0.25]
cfg.uHi = [lim.T(2); lim.tau(:,2)];                  % [4 m g; 0.5; 0.5; 0.25]
cfg.uh = [cfg.plant.m*cfg.plant.g; 0; 0; 0];         % hover input
cfg.refYaw = 0;                                      % heading of every D1 reference
                                                     % (quad_sample_targeted_reference_options)
cfg.actionDim = 6;                                  % Q:{pos,att,vel,rate}, R:{T,tau}
% SAC Q,R search half-width in decades around the base (mult in 10^[-dec, +dec]).
% Default 1.5 (0.03x..32x, wide). Smaller = SAC stays CLOSER to the base.
cfg.logMultDec = d1_getenv_num('D1_LOGMULT_DEC', 1.5);
cfg.logMultBounds = [10^(-cfg.logMultDec), 10^(cfg.logMultDec)];
% Q,R base for the SAC-tuned teacher: 0 = Bryson warm-start (default), 1 = RANDOM
% (no Bryson) log-uniform diag weights, deterministic per seed. The LQR baseline stays
% Bryson in BOTH (fixed yardstick).
cfg.randomQR = strcmp(d1_getenv_str('D1_RANDOM_QR','0'),'1');
cfg.rqrLog   = [-2, 2];                              % random base: 10^[-2,2] per weight
% surrogate (2-head: residual Delta_u + confidence c_S)
cfg.surHidden = 128; cfg.surLR = 1e-3; cfg.surBatch = 256;
cfg.surBufferCap = 1e5; cfg.surRecentFrac = 0.5;
cfg.resHalf = cfg.uHi - cfg.uLo;                     % residual normalization = actuator range
% blend / confidence design params (fixed, disclosed)
cfg.epsP   = d1_getenv_num('D1_EPS_P',  0.5);        % c_S error scale (m): s=exp(-(RMS/epsP)^2)
cfg.cLow   = d1_getenv_num('D1_C_LOW',  0.3);        % g_L gate low threshold on c_LQR
cfg.cHigh  = d1_getenv_num('D1_C_HIGH', 0.7);        % g_L gate high threshold on c_LQR
cfg.epsSafe= d1_getenv_num('D1_EPS_SAFE', 0.1);      % Lyapunov safeguard margin (alpha_safe mode)
cfg.alphaSafe = strcmp(d1_getenv_str('D1_ALPHA_SAFE','0'),'1');
cfg.csCasesPerCall = d1_getenv_num('D1_CS_CASES', 40);  % surrogate closed-loop cases for c_S labels
cfg.csEpochs = d1_getenv_num('D1_CS_EPOCHS', 300);
% sac
cfg.sacLR = 3e-4; cfg.sacBatch = 256; cfg.sacBufferCap = 5e4;
cfg.sacGamma = 0.0;                                  % 1-step bandit (done each ep)
cfg.sacTargetEntropy = -cfg.actionDim;
cfg.logEvery = 1; cfg.checkpointEverySec = 120;
% frozen milestone checkpoints (+ their own confidences) every N SAC iterations; 0 = off
cfg.ckptEvery = d1_getenv_num('D1_CKPT_EVERY_ITER', 50);
% training wind (random, synthetic; NO measured wind data is used): see d1_sample_wind
cfg.windOn   = strcmp(d1_getenv_str('D1_WIND','1'),'1');
cfg.windMin  = d1_getenv_num('D1_WIND_MIN', 1);      % mean wind speed range [m/s]
cfg.windMax  = d1_getenv_num('D1_WIND_MAX', 10);
cfg.windHft  = 20;                                   % Dryden reference height [ft] (MIL-HDBK-1797)
cfg.windDrag = [0.425; 0.256; 0];                    % mass-normalized rotor drag [1/s], Faessler et al. RA-L 2018
end
