function cfg = step1_plant_config()
%STEP1_PLANT_CONFIG Configuration for pipeline step 1.
%
% State: x = [px; py; pz; phi; theta; psi; vx; vy; vz; p; q; r]
% Input: u = [T; tau_phi; tau_theta; tau_psi]
%
% Convention: inertial frame is z-up. Hover at zero attitude uses T = m*g.

cfg.name = 'step1_quadrotor_plant_uncertainty';
cfg.frame = 'z_up_enu';
cfg.stateOrder = {'px','py','pz','phi','theta','psi','vx','vy','vz','p','q','r'};
cfg.inputOrder = {'T','tau_phi','tau_theta','tau_psi'};

nom.g = 9.81;
nom.m = 0.486;
nom.J = diag([3.8278e-3, 3.8278e-3, 7.6566e-3]);
nom.arm = 0.25;
nom.Dv = [5.5670e-4; 5.5670e-4; 6.3540e-4];
nom.Domega = [5.5670e-4; 5.5670e-4; 6.3540e-4];
nom.alphaT = 1.0;
nom.alphaTau = [1.0; 1.0; 1.0];
nom.Jr = 2.8385e-5;
nom.rotorGyroOmega = 0.0;
nom.actuatorCapacityPolicy = 'fixed_nominal_hardware';
nom.notes = ['Nominal values follow the 12-state quadrotor paper in step 1 ' ...
             'where available; generalized-input dynamics omit rotor-speed ' ...
             'allocation until mixer/actuator modeling is needed.'];

nom.inputLimits.T = [0.0; 4.0 * nom.m * nom.g];
nom.inputLimits.tau = [-0.5, 0.5;
                       -0.5, 0.5;
                       -0.25, 0.25];

cfg.nominal = nom;

cfg.uncertainty.names = {'m', ...
                         'Ix', 'Iy', 'Iz', ...
                         'Dv_x', 'Dv_y', 'Dv_z', ...
                         'Domega_p', 'Domega_q', 'Domega_r', ...
                         'alphaT', ...
                         'alphaTau_phi', 'alphaTau_theta', 'alphaTau_psi'};

cfg.uncertainty.train.rho = [0.10; ...
                             0.05; 0.05; 0.05; ...
                             0.30; 0.30; 0.30; ...
                             0.30; 0.30; 0.30; ...
                             0.15; ...
                             0.15; 0.15; 0.15];

cfg.uncertainty.ood.rho = [0.25; ...
                           0.15; 0.15; 0.15; ...
                           0.60; 0.60; 0.60; ...
                           0.60; 0.60; 0.60; ...
                           0.30; ...
                           0.30; 0.30; 0.30];

cfg.uncertainty.defaultMethod = 'lhs';
cfg.uncertainty.defaultSeed = 26082601;

cfg.disturbance.zero.force = zeros(3, 1);
cfg.disturbance.zero.torque = zeros(3, 1);

cfg.disturbance.training.forceSineAmp = [0.05; 0.05; 0.08];
cfg.disturbance.training.forceSineFreq = [0.7; 0.5; 0.3];
cfg.disturbance.training.torqueSineAmp = [0.002; 0.002; 0.001];
cfg.disturbance.training.torqueSineFreq = [0.9; 0.6; 0.4];

cfg.modelShift.none.enabled = false;
cfg.modelShift.payload25.enabled = true;
cfg.modelShift.payload25.startTime = 5.0;
cfg.modelShift.payload25.mScale = 1.25;
cfg.modelShift.payload25.JScale = [1.10; 1.10; 1.15];
cfg.modelShift.payload25.DvScale = [1.0; 1.0; 1.0];
cfg.modelShift.payload25.DomegaScale = [1.0; 1.0; 1.0];
cfg.modelShift.payload25.alphaTScale = 1.0;
cfg.modelShift.payload25.alphaTauScale = [1.0; 1.0; 1.0];
end
