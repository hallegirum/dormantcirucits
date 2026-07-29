#!/usr/bin/env bash
# Multi-seed sweep for the off-manifold pipeline. Run from this directory
# (off_manifold/) with the causalab venv active, on a CUDA pod.
#
#   source /workspace/causalab/.venv/bin/activate   # or wherever causalab lives
#   cd off_manifold && tmux new -s run              # so an SSH drop won't kill it
#   ./run_sweep.sh
#
# Each seed writes to its OWN results dir (results/seed_<s>) via OFFDIST_RESULTS_DIR,
# so seeds never overwrite each other. Aggregate across seeds afterwards.
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
SEEDS=(${SEEDS:-0 1 2 3 4})
NUM_EXAMPLES="${NUM_EXAMPLES:-700}"
NUM_SITES="${NUM_SITES:-8}"
K_FEATURES="${K_FEATURES:-64}"
N_PATHFIND="${N_PATHFIND:-256}"           # subsample for site RANKING only (full set still used for manifolds/reps)
LAYERS="${LAYERS:-4 6 10 16 20 22 24 26 28 30}"                       # e.g. LAYERS="8 12 16 20 24 28" to scan a subset; empty = all layers
DECODING="${DECODING:-}"   # empty = tier-C off; set DECODING="--decoding-eval" to enable

layers_arg=""; [ -n "$LAYERS" ] && layers_arg="--layers $LAYERS"
echo "model=$MODEL seeds=${SEEDS[*]} num_examples=$NUM_EXAMPLES num_sites=$NUM_SITES k=$K_FEATURES n_pathfind=$N_PATHFIND layers='${LAYERS:-all}' decoding='$DECODING'"

RESULTS_ROOT="${RESULTS_ROOT:-results}"   # RESULTS_ROOT=results_v2 to start clean without deleting old runs
failed=()

for s in "${SEEDS[@]}"; do
  outdir="${RESULTS_ROOT}/seed_${s}"
  logf="${outdir}/run.log"
  mkdir -p "$outdir"
  echo "================  seed $s  (logging to $logf)  ================"
  OFFDIST_RESULTS_DIR="$outdir" \
    python -m activation_manifold \
      --model-name "$MODEL" \
      --device cuda \
      --dtype bfloat16 \
      --seed "$s" \
      --num-examples "$NUM_EXAMPLES" \
      --num-sites "$NUM_SITES" \
      --k-features "$K_FEATURES" \
      --n-pathfind "$N_PATHFIND" \
      $layers_arg \
      $DECODING > "$logf" 2>&1 || rc=$?
  rc=${rc:-0}
  if [ "$rc" -ne 0 ]; then
    # Don't let one bad seed kill the sweep (set -e would abort otherwise).
    failed+=("$s")
    echo "seed $s FAILED (exit $rc) -> $logf   last lines:"
    tail -n 20 "$logf" | sed 's/^/    | /'
  else
    echo "seed $s done -> $outdir  (log: $logf)"
  fi
  unset rc
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "ALL SEEDS DONE — FAILED: ${failed[*]} (see <root>/seed_<s>/run.log)"
else
  echo "ALL SEEDS DONE — all succeeded"
fi
