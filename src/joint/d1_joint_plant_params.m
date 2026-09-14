function P = d1_joint_plant_params()
%D1_JOINT_PLANT_PARAMS Nominal plant params for the joint pipeline, pulled from
% the single source of truth step1_plant_config (no hardcoded duplicates).
nom = step1_plant_config().nominal;
P.g = nom.g; P.m = nom.m;
P.Jd = diag(nom.J);
P.Dv = nom.Dv(:); P.Domega = nom.Domega(:);
P.alphaT = nom.alphaT; P.alphaTau = nom.alphaTau(:);
P.Tmax = nom.inputLimits.T(2);
P.nominal = nom;                    % keep full struct for quad_step_rk4 / dlqr
end
