function d1_local_timing()
%D1_LOCAL_TIMING Computation time of the deployable controllers on ONE documented machine.
%
% Computation time is not taken from the CI jobs (shared virtual runners, CPU model not
% specified). This script runs experiments/d1_final_eval.m unchanged -- same flights, same
% controller code, same tic/toc around the control computation only -- for LQR and MPC
% (tag base, D1_SEED as in the CI baseline job) and for P of every chain in
% D1_TIMING_CHAINS, in both conditions, with one computational thread. The teacher is not
% timed (oracle, not deployable; acados is not used here).
%
% Start MATLAB pinned to one logical processor (Windows: start /affinity 1 ...), from the
% repository root:
%   matlab -batch "addpath(genpath('src')); addpath('configs'); addpath('experiments'); d1_local_timing"
% Environment:
%   D1_WIND_DIR       wind/series built by tools/wind (validation data, never committed)
%   D1_OUT            output folder (default results/local_timing)
%   D1_TIMING_CHAINS  "seed|random_qr|checkpoint.mat|student.mat;..." (the final-test files)
% Writes the CSVs of d1_final_eval and machine.txt; summarize with
%   python tools/eval/summarize_final_eval.py --timing <D1_OUT>
maxNumCompThreads(1);
out = d1_getenv_str('D1_OUT', fullfile('results', 'local_timing'));
if ~isfolder(out), mkdir(out); end
setenv('D1_OUT', out); setenv('D1_SHARD', '1/1');
write_machine(fullfile(out, 'machine.txt'));
chains = strsplit(d1_getenv_str('D1_TIMING_CHAINS', ''), ';');
chains = chains(~cellfun(@isempty, chains));
for cond = {'train', 'ood'}
    setenv('D1_COND', cond{1});
    setenv('D1_SEED', '260914201'); setenv('D1_RANDOM_QR', '0');   % as the CI baseline job
    setenv('D1_TAG', 'base'); setenv('D1_CTRLS', 'LQR,MPC');
    setenv('D1_CKPT', ''); setenv('D1_STUDENT', '');
    d1_final_eval();
    for c = 1:numel(chains)
        p = strsplit(chains{c}, '|');            % seed | random_qr | ckpt | student
        assert(numel(p) == 4, 'd1_local_timing:chain', 'bad D1_TIMING_CHAINS entry %s', chains{c});
        ck = p{3}; stu = p{4};
        setenv('D1_SEED', p{1}); setenv('D1_RANDOM_QR', p{2});
        setenv('D1_TAG', ['c' p{1}]); setenv('D1_CTRLS', 'P');
        setenv('D1_CKPT', ck); setenv('D1_STUDENT', stu);
        d1_final_eval();
    end
end
end

function write_machine(f)
fid = fopen(f, 'w');
fprintf(fid, 'MATLAB: %s\n', version);
fprintf(fid, 'computational threads: %d\n', maxNumCompThreads);
fprintf(fid, 'computer: %s\n', computer);
if ispc
    [~, cpu] = system('powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).Name"');
    [~, os] = system('powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).Caption"');
    [~, aff] = system(sprintf('powershell -NoProfile -Command "(Get-Process -Id %d).ProcessorAffinity"', ...
        feature('getpid')));
    [~, pw] = system('powershell -NoProfile -Command "(Get-CimInstance Win32_Battery).BatteryStatus"');
    fprintf(fid, 'CPU: %s\nOS: %s\nprocess affinity mask: %s\nbattery status (2 = on AC): %s\n', ...
        strtrim(cpu), strtrim(os), strtrim(aff), strtrim(pw));
else
    [~, cpu] = system('grep -m1 "model name" /proc/cpuinfo');
    fprintf(fid, 'CPU: %s\n', strtrim(cpu));
end
fprintf(fid, 'date: %s\n', char(datetime('now', 'TimeZone', 'UTC'), 'yyyy-MM-dd HH:mm:ss ''UTC'''));
fclose(fid);
end
