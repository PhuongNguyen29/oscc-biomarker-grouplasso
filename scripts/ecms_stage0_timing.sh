#!/usr/bin/env bash
# ==============================================================================
# ECMS Stage 0 timing driver. Run from the project root:
#   nohup bash scripts/ecms_stage0_timing.sh > results/ecms_stage0_timing_driver.log 2>&1 &
#   tail -f results/ecms_stage0_timing_driver.log
#
# Phase A: one call ALONE at nPermutations 50, 100, 200 (nothing else running
#          on the server). Stops early if the 50-permutation call alone already
#          exceeds the reuse threshold (5,040 s), so no day is wasted.
# Phase B: 30 calls CONCURRENTLY at nPermutations 50 (the decision number).
# Then:    summarize -> results/ecms_stage0_timing/summary.txt
# ==============================================================================
set -u
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export ECMS_WGCNA_THREADS=2

OUT=results/ecms_stage0_timing
LOG=$OUT/logs
mkdir -p "$LOG"
CONCURRENCY=30
T_REUSE=5040

TIME_BIN=/usr/bin/time
if [ ! -x "$TIME_BIN" ]; then
  echo "WARNING: /usr/bin/time not found; peak memory will not be recorded"
  TIME_BIN=""
fi
export TIME_BIN LOG

run_one() {
  local nperm=$1 rep=$2 phase=$3 rc
  local tag; tag=$(printf "%s_nperm%03d_rep%02d" "$phase" "$nperm" "$rep")
  echo "$(date '+%F %T') start  $tag"
  if [ -n "$TIME_BIN" ]; then
    "$TIME_BIN" -v -o "$LOG/$tag.time" \
      Rscript scripts/ecms_stage0_timing.R run "$nperm" "$rep" "$phase" > "$LOG/$tag.log" 2>&1
  else
    Rscript scripts/ecms_stage0_timing.R run "$nperm" "$rep" "$phase" > "$LOG/$tag.log" 2>&1
  fi
  rc=$?
  echo "$(date '+%F %T') finish $tag (exit $rc)"
  [ $rc -ne 0 ] && echo "  see $LOG/$tag.log"
  return $rc
}
export -f run_one

echo "=== Phase A: alone ==="
for np in 50 100 200; do
  run_one "$np" 1 alone || { echo "Phase A failed at nperm $np; stopping."; exit 1; }
  if [ "$np" -eq 50 ]; then
    t50=$(Rscript -e "cat(read.csv('$OUT/timing_alone_nperm050_rep01.csv')\$total_sec)")
    echo "nPerm 50 alone: ${t50} s"
    if awk "BEGIN{exit !($t50 > $T_REUSE)}"; then
      echo "Alone call already exceeds ${T_REUSE} s: stop and switch to the fallback."
      Rscript scripts/ecms_stage0_timing.R summarize | tee "$OUT/summary.txt"
      exit 0
    fi
  fi
done

echo "=== Phase B: ${CONCURRENCY} concurrent, nPerm 50 ==="
seq 1 "$CONCURRENCY" | xargs -P "$CONCURRENCY" -I{} bash -c 'run_one 50 {} concurrent'

echo "=== Summary ==="
Rscript scripts/ecms_stage0_timing.R summarize | tee "$OUT/summary.txt"
