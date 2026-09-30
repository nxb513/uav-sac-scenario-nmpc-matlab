function scen = d1_sample_scenarios(cfg)
%D1_SAMPLE_SCENARIOS The teacher's M uncertain model realizations: scenario 1 = nominal,
% the rest scaled by (1 + xi.*rho_train), xi ~ U[-1,1]^14 from the GLOBAL rng (callers
% draw them first after rng(seed), so a chain has the same scenarios in every job).
nom = cfg.plant.nominal;
rho = step1_plant_config().uncertainty.train.rho;   % 14x1
Jnom = diag(nom.J);
scen = struct('m',{},'Jd',{},'Dv',{},'Domega',{},'alphaT',{},'alphaTau',{});
for i = 1:cfg.M
    if i==1, xi = zeros(14,1); else, xi = 2*rand(14,1)-1; end
    f = 1 + xi.*rho;
    scen(i).m = nom.m*f(1);
    scen(i).Jd = Jnom.*f(2:4);
    scen(i).Dv = nom.Dv(:).*f(5:7);
    scen(i).Domega = nom.Domega(:).*f(8:10);
    scen(i).alphaT = nom.alphaT*f(11);
    scen(i).alphaTau = nom.alphaTau(:).*f(12:14);
end
end
