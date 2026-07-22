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
NUM_SITES="${NUM_SITES:-10}"
K_FEATURES="${K_FEATURES:-64}"
DECODING="${DECODING:---decoding-eval}"   # set DECODING="" to disable the tier-C test

echo "model=$MODEL  seeds=${SEEDS[*]}  num_examples=$NUM_EXAMPLES  num_sites=$NUM_SITES  k=$K_FEATURES  decoding='$DECODING'"

for s in "${SEEDS[@]}"; do
  outdir="results/seed_${s}"
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
      $DECODING > "$logf" 2>&1
  echo "seed $s done (exit $?) -> $outdir  (log: $logf)"
done
echo "ALL SEEDS DONE"
