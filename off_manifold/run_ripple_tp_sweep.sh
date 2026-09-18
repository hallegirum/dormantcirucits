#!/usr/bin/env bash
# run_ripple_tp_sweep.sh — run ripple with --include-tp on all combos x seeds
# Adds ripple_xint_in_orth to BOTH off-circuit (FP) and in-circuit (TP) rows
# so the FP-vs-TP regression in collect_interaction.py has coverage.
#
# Run from: dormant-circuit-analysis/
# Assumes:  poly_results_new/results/ contains the original interaction runs.

set -euo pipefail

RESULTS_BASE="poly_results_new/results"
COMBOS=(
    cases_0+1 cases_0+2 cases_0+3 cases_1+2 cases_1+3 cases_2+3
    cases_0+1+2 cases_0+1+3 cases_0+2+3 cases_1+2+3
    cases_0+1+2+3
)
SEEDS=(0 1 2)
N=200
TAU=0.0
DEVICE=cuda

for combo in "${COMBOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        dir="${RESULTS_BASE}/poly_${combo}/int_seed${seed}"
        if [ ! -d "$dir/interaction" ]; then
            echo "[skip] $dir/interaction does not exist"
            continue
        fi
        echo "========================================"
        echo "  combo=${combo}  seed=${seed}  tau=${TAU}"
        echo "========================================"
        python off_manifold/run_ripple.py \
            --results-root "${RESULTS_BASE}" \
            --combo "${combo}" \
            --seed "${seed}" \
            --case poly \
            --include-tp \
            --gate-by none \
            --kind all \
            --n "${N}" \
            --device "${DEVICE}" \
            --batch-size 32
    done
done

echo ""
echo "=== ripple TP sweep complete ==="
