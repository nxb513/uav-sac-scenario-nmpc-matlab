#!/usr/bin/env bash
# Hang watchdog for the D1 training step (diagnostics only; it never changes training).
#
# Usage: hang_watchdog.sh <run_dir> <stale_seconds>
# Every 60 s it takes the newest modification time of any file in <run_dir> (the pipeline
# writes a checkpoint after every case once 120 s have passed, so in normal operation the
# gap is at most a few minutes). If nothing was written for <stale_seconds>, the MATLAB
# process is considered hung: the stacks of all its threads are dumped with gdb to
# <run_dir>/hang_backtrace_<time>.txt, and MATLAB is terminated so the job uploads the last
# checkpoint instead of waiting for the step timeout. The in-progress case is lost; the
# chain resumes from the last saved case.
set -u
dir=$1; stale=${2:-1800}
start=$(date +%s)
echo "watchdog: dir=$dir stale=${stale}s start=$start"
while true; do
  sleep 60
  pid=$(pgrep -x MATLAB | head -1 || true)
  [ -z "$pid" ] && continue
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
      echo "MATLAB pid=$pid, no file written in $dir for $((now - newest)) s"
      ps -o pid,stat,pcpu,pmem,rss,etime,cmd -p "$pid"
      free -m
      sudo gdb -p "$pid" -batch -ex "set pagination off" -ex "info threads" -ex "thread apply all bt 40"
    } > "$out" 2>&1
    echo "watchdog: hang detected, backtrace -> $out; terminating MATLAB"
    kill -TERM "$pid" 2>/dev/null; sleep 30; kill -KILL "$pid" 2>/dev/null
    exit 0
  fi
done
