function cases = d1_train_cases(cfg)
%D1_TRAIN_CASES The training reference bank: deterministic stratified sweep
% family x acceleration x speed (5 x 3 x 8) until 120 cases, nominal flat references.
ref = targeted_lqr_weak_config().reference;
theta = cfg.plant.nominal;
families = ref.families; speeds = ref.approvedIdSpeedAnchors;
accels = ref.approvedAccelerationTargets;
cases = struct('Xref',{},'Uref',{},'groupId',{});
count = 0; target = 120;
for fi = 1:numel(families)
  for ai = 1:numel(accels)
    for si = 1:numel(speeds)
      if count >= target, break; end
      gid = sprintf('%s|v%g|a%g', families{fi}, speeds(si), accels(ai));
      rng(d1_case_seed(gid), 'twister');
      try
        opt = quad_sample_targeted_reference_options(families{fi}, ref, ...
            speeds(si), accels(ai));
        [Xref,~,Uref] = quad_targeted_reference_trajectory(families{fi}, ...
            cfg.Ts, cfg.stepsPerCase, opt, theta);
        if all(isfinite(Xref(:)))
          count = count + 1;
          cases(count).Xref = Xref; cases(count).Uref = Uref;
          cases(count).groupId = gid;
        end
      catch
      end
    end
  end
end
assert(count > 0, 'No teacher-dev cases generated.');
end
