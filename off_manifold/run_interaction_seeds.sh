#!/usr/bin/env bash
# run_interaction_seeds.sh — run_interaction_effect over three seeds, each into its
# own results tree (results/{tree}/int_seed{S}) so int.json rows never overwrite.
#
# Usage (from the repo root, poly-venv active):
#   bash dormant-circuit-analysis/off_manifold/run_interaction_seeds.sh                 # step 0 only
#   STEP=both bash dormant-circuit-analysis/off_manifold/run_interaction_seeds.sh      # gate + full map
#   POLY_COMB=cases_1+2 SEEDS="0 1 2" STEP=both bash .../run_interaction_seeds.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CASE="${CASE:-poly}"
POLY_COMB="${POLY_COMB:-cases_0+1}"
STEP="${STEP:-0}"                 # 0 | full | both
SEEDS="${SEEDS:-0 1 2}"
N="${N:-200}"
ALPHAS="${ALPHAS:-0.3 1.0 3.0}"
DEVICE="${DEVICE:-cuda}"
EXTRA="${EXTRA:-}"                # e.g. EXTRA="--push-site L1_mlp --layers 0 1"

if [ "$CASE" = "poly" ]; then TREE="poly_${POLY_COMB}"; else TREE="$CASE"; fi

for S in $SEEDS; do
  export OFFDIST_RESULTS_DIR="results/${TREE}/int_seed${S}"
  echo "==== seed $S -> $OFFDIST_RESULTS_DIR (step $STEP) ===="
  # shellcheck disable=SC2086
  python "$HERE/run_interaction_effect.py" \
    --case "$CASE" --poly-comb "$POLY_COMB" --step "$STEP" --seed "$S" \
    --n "$N" --alphas $ALPHAS --device "$DEVICE" $EXTRA
done

echo "==== all seeds done ===="
echo "gate verdicts:"
for S in $SEEDS; do
  F="results/${TREE}/int_seed${S}/interaction/step0_gate.json"
  [ -f "$F" ] && python -c "import json;d=json.load(open('$F'));print('  seed $S: max|INT|', round(d['max_abs_INT'],4), '->', d['verdict'])"
done
