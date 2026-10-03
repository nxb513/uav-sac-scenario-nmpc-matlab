#!/usr/bin/env bash
# Hang watchdog for the D1 training step (diagnostics only; it never changes training).
#
# Usage: hang_watchdog.sh <run_dir> <stale_seconds> (the workflow uses 3600 s)
# Every 60 s it takes the newest modification time of any file in <run_dir> (the pipeline
# writes a checkpoint after every SAC / DAgger iteration, normally every few minutes). If
# nothing was written for <stale_seconds>, the run is considered hung: the stacks of all
# threads of every MATLAB process (client and parallel workers) are dumped with gdb to
# <run_dir>/hang_backtrace_<time>.txt, and the MATLAB processes are terminated so the job
# uploads the last checkpoint instead of waiting for the step timeout. The chain resumes
# from the last completed iteration.
set -u
dir=$1; stale=${2:-3600}
start=$(date +%s)
echo "watchdog: dir=$dir stale=${stale}s start=$start"
while true; do
  sleep 60
  pids=$(pgrep -x MATLAB || true)
  [ -z "$pids" ] && continue
  newest=$start
  if [ -d "$dir" ]; then
    m=$(find "$dir" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
    [ -n "$m" ] && [ "$m" -gt "$newest" ] && newest=$m
  fi
  now=$(date +%s)
  if [ $((now - newest)) -gt "$stale" ]; then
    out="$dir/hang_backtrace_$(date -u +%Y%m%dT%H%M%SZ).txt"
    mkdir -p "$dir"
    {
      echo "MATLAB pids=$(echo $pids), no file written in $dir for $((now - newest)) s"
      free -m
      for pid in $pids; do
        echo "================ pid $pid"
        ps -o pid,stat,pcpu,pmem,rss,etime,cmd -p "$pid"
        sudo gdb -p "$pid" -batch -ex "set pagination off" -ex "info threads" -ex "thread apply all bt 40"
      done
    } > "$out" 2>&1
    echo "watchdog: hang detected, backtrace -> $out; terminating MATLAB"
    for pid in $pids; do kill -TERM "$pid" 2>/dev/null; done
    sleep 30
    for pid in $pids; do kill -KILL "$pid" 2>/dev/null; done
    exit 0
  fi
done
