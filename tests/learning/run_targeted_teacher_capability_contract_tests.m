function run_targeted_teacher_capability_contract_tests()
%RUN_TARGETED_TEACHER_CAPABILITY_CONTRACT_TESTS Validate Task-02 design.

projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(projectRoot, 'experiments'));
audit = validate_targeted_teacher_closed_loop('contract_only');
assert(strcmp(audit.ProtocolVersion, ...
    'targeted_teacher_capability_h20_v1'));
assert(audit.ContextCount == 15 && audit.FamilyCount == 5);
assert(audit.LeadRankStratumCount == 3 && audit.CaseCount == 90);
assert(audit.RolloutSteps == 20 && audit.LqrEventCount == 15);
assert(audit.FullSacPairGridRetained);
fprintf(['Targeted teacher capability contract passed: ' ...
    '15 contexts, 90 H=20 tasks.\n']);
end
