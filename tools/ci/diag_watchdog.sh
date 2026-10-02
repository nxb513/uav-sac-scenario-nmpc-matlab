#!/usr/bin/env bash
# DIAGNOSTIC only (not part of the method), used by d1-diag-replay.yml.
#
# Usage: diag_watchdog.sh <heartbeat_file> <stale_seconds> <out_dir> [snapshots] [gap_seconds]
# experiments/d1_diag_replay.m rewrites <heartbeat_file> before every teacher solve (a normal
# solve takes < 1 s). If the file is older than <stale_seconds> while MATLAB runs, the solve
# is stalled: the script takes <snapshots> gdb snapshots of the stalled thread
# (tools/ci/diag_gdb.py), <gap_seconds> apart, into <out_dir>/gdb_snapshot_<i>.txt, then
# terminates MATLAB.
set -u
hb=$1; stale=$2; out=$3; nsnap=${4:-3}; gap=${5:-30}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
echo "diag_watchdog: heartbeat=$hb stale=${stale}s snapshots=$nsnap gap=${gap}s"
while true; do
  sleep 10
  pid=$(pgrep -x MATLAB | head -1 || true)
  [ -z "$pid" ] && continue
  [ -f "$hb" ] || continue
  age=$(( $(date +%s) - $(stat -c %Y "$hb") ))
  if [ "$age" -gt "$stale" ]; then
    echo "diag_watchdog: heartbeat $age s old: $(head -1 "$hb")"
    cp "$hb" "$out/heartbeat_at_stall.txt"
    for i in $(seq 1 "$nsnap"); do
      {
        echo "snapshot $i at $(date -u +%FT%TZ), heartbeat age $(( $(date +%s) - $(stat -c %Y "$hb") )) s"
        ps -o pid,stat,pcpu,etime -p "$pid"
        sudo gdb -p "$pid" -batch -x "$here/diag_gdb.py"
      } > "$out/gdb_snapshot_$i.txt" 2>&1
      echo "diag_watchdog: snapshot $i written"
      [ "$i" -lt "$nsnap" ] && sleep "$gap"
    done
    kill -TERM "$pid" 2>/dev/null; sleep 20; kill -KILL "$pid" 2>/dev/null
    exit 0
  fi
done
