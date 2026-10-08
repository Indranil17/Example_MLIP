#!/usr/bin/env bash
# One-screen status of the am26_* jobs and of what they have written so far.
# Usage, from the repository directory on the cluster:
#   bash tools/track_jobs.sh            # once
#   watch -n 120 bash tools/track_jobs.sh   # every two minutes, Ctrl-C to stop
# PBS copies a job's .o file back only when the job ends, so live progress is read from the
# files the scripts write directly to /storage (per-frame CSVs, MACE training logs).
set -u
cd "$(dirname "$0")/.."

echo "== $(date '+%F %T')  $(hostname) =="
echo
echo "-- queue --"
qstat -u "$USER" 2>/dev/null | tail -n +4 | grep -E 'am26_' || echo "no am26_ jobs in the queue"
echo
echo "-- finished job outputs (appear only when a job ends) --"
for f in am26_*.o*; do
  [ -e "$f" ] || continue
  printf '%s\n' "$f:"
  tail -n 3 "$f" | sed 's/^/    /'
done
echo
echo "-- M0 triage: per-model files appear as each model finishes --"
ls -l modules/M0_foundation_triage/outputs/ 2>/dev/null | grep -E 'per_frame|metrics|spread|summary' | awk '{print "    " $6, $7, $8, $5, $9}' || echo "    (none yet)"
echo
echo "-- M2 support criterion --"
ls -l modules/M2_ensemble_support/outputs/ 2>/dev/null | awk 'NR>1 {print "    " $6, $7, $8, $5, $9}' || echo "    (none yet)"
echo
echo "-- M1 fine-tune: last lines of the MACE log and latest checkpoint --"
for log in modules/M1_finetune_asio2/logs/*.log; do
  [ -e "$log" ] || { echo "    (no training log yet)"; break; }
  printf '    %s\n' "$log"
  grep -E 'Epoch|RMSE|Loading|Started|Done|Error|error' "$log" | tail -n 4 | sed 's/^/      /'
done
ls -lt modules/M1_finetune_asio2/checkpoints/ 2>/dev/null | head -n 3 | awk 'NR>1 {print "    " $6, $7, $8, $5, $9}'
echo
echo "-- held-out evaluation (written at the very end of the GPU job) --"
[ -e modules/M1_finetune_asio2/outputs/heldout_metrics.csv ] && cat modules/M1_finetune_asio2/outputs/heldout_metrics.csv || echo "    (not yet)"
