#!/usr/bin/env bash
# InterpBench DORMANCY sweep over the classification (Cls) tracr cases x seeds.
#
# For each (case, seed): run the WRITER (run_dormancy_pipeline.py) into its own results tree
# results/sweep/{case}_seed{seed}, then the READER (off-distribution-analyses.py, mahalanobis
# off-metric by default) to emit dormancy.json. Failures (a case with no InterpBench model, an
# OOM, a degenerate fit) are logged and SKIPPED so one bad case can't abort the sweep.
#
# Aggregate at the end (combine across cases per seed, then average across seeds):
#     python off-distribution-analyses.py --aggregate results/sweep
#
# Only CLASSIFICATION (Cls) tracr cases: the per-token class is the manifold's `result`
# variable, so the fit is well-posed. --no-residual because for tracr the residual stream is
# the carrier bus (guaranteed hits), NOT a labelled circuit node.
#
#   source <causalab venv>; cd off_manifold && tmux new -s dorm && ./run_dormancy_sweep.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

CASES="${CASES:-8 11 13 18 19 20 21 26 29 33 34 35 36 37}"
SEEDS="${SEEDS:-0 1 2}"
POS="${POS:-last_token}"
PATCH="${PATCH:-corr}"
N="${N:-400}"
MODES="${MODES:-STR Gaussian}"
ROOT="${ROOT:-results/sweep}"

mkdir -p "$ROOT"
LOG="$ROOT/sweep.log"; : > "$LOG"
ok=0; fail=0

for case in $CASES; do
  for seed in $SEEDS; do
    tree="$ROOT/${case}_seed${seed}"
    echo "=== case $case | seed $seed -> $tree ===" | tee -a "$LOG"

    if OFFDIST_RESULTS_DIR="$tree" python "$HERE/run_dormancy_pipeline.py" \
         --case "$case" --seed "$seed" --position "$POS" --patch-at "$PATCH" \
         --n "$N" --modes $MODES --no-residual >>"$LOG" 2>&1
    then
      if OFFDIST_RESULTS_DIR="$tree" python "$HERE/off-distribution-analyses.py" \
           --case "$case" --modes $MODES --offmetric mahalanobis >>"$LOG" 2>&1
      then
        echo "    ok" | tee -a "$LOG"; ok=$((ok+1))
      else
        echo "    [reader FAILED] case $case seed $seed (see $LOG)" | tee -a "$LOG"; fail=$((fail+1))
      fi
    else
      echo "    [writer FAILED] case $case seed $seed — likely no InterpBench model; skipping" | tee -a "$LOG"; fail=$((fail+1))
    fi
  done
done

echo "" | tee -a "$LOG"
echo "SWEEP DONE: $ok ok, $fail failed. Aggregate with:" | tee -a "$LOG"
echo "    python $HERE/off-distribution-analyses.py --aggregate $ROOT" | tee -a "$LOG"
