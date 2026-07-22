#!/usr/bin/env bash
# One-command pod setup. Idempotent: re-running skips anything already present.
# If /workspace is a NETWORK VOLUME, run this ONCE ever — causalab, the venv, and
# the model cache persist across pods, so future pods skip straight to running.
#
# Flow:
#   1. (Mac) scp off_manifold up:  scp -i ~/.ssh/runpod_key -P <port> -r off_manifold root@<ip>:/workspace/
#   2. (pod) bash /workspace/off_manifold/setup_pod.sh
#   3. (pod) source /workspace/causalab/.venv/bin/activate
#   4. (pod) huggingface-cli login          # or export HF_TOKEN=... before step 2
#   5. (pod) cd /workspace/off_manifold && ./run_sweep.sh
set -euo pipefail

ROOT="${ROOT:-/workspace}"
export HF_HOME="${HF_HOME:-$ROOT/hf}"      # model cache — persists if $ROOT is a network volume
mkdir -p "$HF_HOME"

# causalab as a SIBLING of off_manifold (off_manifold doesn't need to be nested;
# it just needs causalab importable + to be run from its own dir).
if [ ! -d "$ROOT/causalab/.git" ]; then
  echo "cloning causalab..."
  git clone https://github.com/goodfire-ai/causalab.git "$ROOT/causalab"
  git -C "$ROOT/causalab" checkout bf15b35
else
  echo "causalab already present — skipping clone"
fi
cd "$ROOT/causalab"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source "$HOME/.local/bin/env"
fi
uv sync                                     # no-op if the venv is already built

echo ""
echo "setup done. next:"
echo "  source $ROOT/causalab/.venv/bin/activate"
echo "  huggingface-cli login    # if HF_TOKEN not already exported"
echo "  export HF_HOME=$HF_HOME"
echo "  cd $ROOT/off_manifold && ./run_sweep.sh"
