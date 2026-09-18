#!/usr/bin/env bash
# run_local_gap_sweep.sh — re-run interaction effects with --local-gap on all combos x seeds
# This repopulates the existing int.json files, adding local_relu_gap / local_relu_gap_frac /
# local_relu_ncross fields to every row. Existing fields (NIE, PIE, INT, gap_frac, ...) are
# recomputed identically (same seed, same n).
#
# Run from: dormant-circuit-analysis/off_manifold/
# Assumes:  poly_results_new/results/ contains the original runs at the same --n (200).

set -euo pipefail

RESULTS_BASE="poly_results_new/results"
COMBOS=(
    cases_0+1 cases_0+2 cases_0+3 cases_1+2 cases_1+3 cases_2+3
    cases_0+1+2 cases_0+1+3 cases_0+2+3 cases_1+2+3
    cases_0+1+2+3
)
SEEDS=(0 1 2)
N=200
DEVICE=cuda
STEP=both

for combo in "${COMBOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        dir="${RESULTS_BASE}/poly_${combo}/int_seed${seed}"
        if [ ! -d "$dir" ]; then
            echo "[skip] $dir does not exist"
            continue
        fi
        echo "========================================"
        echo "  combo=${combo}  seed=${seed}"
        echo "  OFFDIST_RESULTS_DIR=${dir}"
        echo "========================================"
        OFFDIST_RESULTS_DIR="${dir}" python off_manifold/run_interaction_effect.py \
            --case poly \
            --poly-comb "${combo}" \
            --seed "${seed}" \
            --step "${STEP}" \
            --local-gap \
            --n "${N}" \
            --device "${DEVICE}" \
            --batch-size 32
    done
done

echo ""
echo "=== local-gap sweep complete ==="
