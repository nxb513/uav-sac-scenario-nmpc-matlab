function [stu, conf] = d1_load_deployed(studentFile, required)
%D1_LOAD_DEPLOYED The deployed controller = the DAgger student (stu.W) and its confidences
% (conf.S, conf.LQR), saved together in student_seed<s>[_iter<NNNN>].mat by d1_dagger_run.
% required = true (final evaluation) -> a missing file is an error; otherwise returns empty
% (callers then fly LQR only) with a warning.
if nargin < 2, required = false; end
stu = []; conf = [];
if ~isempty(studentFile) && isfile(studentFile)
    C = load(studentFile); stu = C.stu; conf = C.conf;
elseif required
    error('d1_load_deployed:nostudent', 'deployed student file missing: %s', studentFile);
else
    fprintf('WARN no student file (%s): the proposed controller is not available\n', studentFile);
end
end
