#!/usr/bin/env bash
# Run check_endpoint_density_tp.py for all 11 combos x 3 seeds.
# Outputs TP records to endpoint_density_tp.jsonl (append mode).
#
# Usage on pod:
#   cd /workspace/dormantcirucits/dormant-circuit-analysis/off_manifold
#   bash run_endpoint_density_tp.sh

set -euo pipefail

RESULTS_ROOT="./poly_results_new/results"
DUMP="./endpoint_density_tp.jsonl"
N=150
DEVICE="cuda"

COMBOS=(
  cases_0+1 cases_0+2 cases_0+3
  cases_1+2 cases_1+3 cases_2+3
  cases_0+1+2 cases_0+1+3 cases_0+2+3 cases_1+2+3
  cases_0+1+2+3
)

# clear any previous partial run
rm -f "$DUMP"

for combo in "${COMBOS[@]}"; do
  for seed in 0 1 2; do
    echo "=== $combo  seed=$seed ==="
    python off_manifold/check_endpoint_density_tp.py \
      --results-root "$RESULTS_ROOT" \
      --combo "$combo" --seed "$seed" \
      --case poly --n "$N" --device "$DEVICE" \
      --dump "$DUMP" \
    || echo "FAILED: $combo seed=$seed"
  done
done

wc -l "$DUMP"
echo "Done."
