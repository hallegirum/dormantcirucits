#!/usr/bin/env bash
# Run endpoint density v2 (per-task distances) for all 11 combos x 3 seeds.
# Single combined script processes both FP and TP in one model-loading pass.
#
# Usage on pod:
#   cd /workspace/dormantcirucits/dormant-circuit-analysis
#   bash off_manifold/run_density_v2.sh

set -euo pipefail

RESULTS_ROOT="./poly_results_new/results"
FP_DUMP="./endpoint_density_v2.jsonl"
TP_DUMP="./endpoint_density_tp_v2.jsonl"
N=150
DEVICE="cuda"

COMBOS=(
  cases_0+1 cases_0+2 cases_0+3
  cases_1+2 cases_1+3 cases_2+3
  cases_0+1+2 cases_0+1+3 cases_0+2+3 cases_1+2+3
  cases_0+1+2+3
)

# clear any previous partial run
rm -f "$FP_DUMP" "$TP_DUMP"

TOTAL=$(( ${#COMBOS[@]} * 3 ))
I=0

for combo in "${COMBOS[@]}"; do
  for seed in 0 1 2; do
    I=$((I + 1))
    echo "=== [$I/$TOTAL] $combo  seed=$seed ==="

    python off_manifold/check_endpoint_density_v2.py \
      --results-root "$RESULTS_ROOT" \
      --combo "$combo" --seed "$seed" \
      --case poly --n "$N" --device "$DEVICE" \
      --dump-fp "$FP_DUMP" \
      --dump-tp "$TP_DUMP" \
    || echo "[WARN] density failed: $combo seed=$seed"
  done
done

echo ""
echo "Done. Output:"
echo "  FP: $FP_DUMP ($(wc -l < "$FP_DUMP") records)"
echo "  TP: $TP_DUMP ($(wc -l < "$TP_DUMP") records)"
