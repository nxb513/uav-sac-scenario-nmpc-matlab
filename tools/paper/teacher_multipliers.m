function teacher_multipliers(root)
%TEACHER_MULTIPLIERS Group multipliers 10.^mu of the teacher used in the final test.
% Reads every checkpoint_seed*_iter0100.mat under ROOT (for example the extracted
% d1_final_controllers_iter0100 folder of the release) and prints the SAC mean action mu and
% 10.^mu for the six weight groups (position, attitude, velocity, angular rate, thrust,
% torques); the teacher of the final test uses a = mu.
% Usage (MATLAB, Deep Learning Toolbox for dlarray): teacher_multipliers('<data_root>')
files = dir(fullfile(root, '**', 'checkpoint_seed*_iter0100.mat'));
for i = 1:numel(files)
    S = load(fullfile(files(i).folder, files(i).name));
    mu = double(extractdata(S.sac.mu));
    fprintf('%s\n  mu       %s\n  10.^mu   %s\n', files(i).name, mat2str(mu(:)', 4), mat2str(10.^mu(:)', 4));
end
end
