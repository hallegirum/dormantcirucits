#!/usr/bin/env bash
# setup_poly_env.sh — poly-venv on a pod that ALREADY has off_manifold/ and
# interp_bench/ side by side (whatever you scp'd up). Run it from the directory
# that CONTAINS those two folders (e.g. /workspace). Idempotent.
#
# It does NOT clone or need a causalab CHECKOUT on disk — causalab is pip-installed
# INTO the venv from the fork. The code in off_manifold/ finds interp_bench/ on its
# own (sys.path uses the sibling dir), so the flat layout you have is all it needs.
#
# WHY A SEPARATE VENV: poly_bench needs transformers ~4.44.1 + the evanhanders/iit
# fork. The ib-venv pins transformers 4.43.4 and a DIFFERENT iit under the same
# name — they can't coexist. tracr/ioi stay in ib-venv; poly uses this venv.
# WHY --no-deps ON CAUSALAB: its declared deps drag pyvene (git) + unpinned
# transformers, which would fight poly's 4.44 pin. The poly/tracr path is the
# NATIVE (pyvene-free) TL-hook backend, so causalab never needs pyvene at run time.
#
# Do NOT also run setup_pod.sh (old uv venv) or setup_pod_env.sh (ib-venv).
#
# Usage (from /workspace, which holds off_manifold/ and interp_bench/):
#   bash off_manifold/setup_poly_env.sh
set -euo pipefail

ROOT="${ROOT:-$PWD}"                 # must contain off_manifold/ and interp_bench/
for d in off_manifold interp_bench; do
  [ -d "$ROOT/$d" ] || {
    echo "ERROR: '$ROOT/$d' not found." >&2
    echo "       cd to the directory that holds off_manifold/ and interp_bench/," >&2
    echo "       or pass it:  ROOT=/workspace bash off_manifold/setup_poly_env.sh" >&2
    exit 1; }
done
VENV="${VENV:-$ROOT/poly-venv}"
PY="${PY:-python3.10}"               # poly_bench wants >=3.10
# The forked causalab (guard changes + tasks/interpbench). Public fork -> no token.
CAUSALAB_SPEC="${CAUSALAB_SPEC:-git+https://github.com/hallegirum/causalab.git@interpbench-integration}"
export HF_HOME="${HF_HOME:-$ROOT/hf}"   # model cache; persists if $ROOT is a network volume
mkdir -p "$HF_HOME"

echo "== poly-venv at $VENV  (code root: $ROOT) =="
"$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip

# Order matters: numpy<2 FIRST, then the iit fork, then poly_bench (pulls
# transformers ~4.44.1, transformer_lens, torch).
pip install "numpy<2"
pip install "git+https://github.com/evanhanders/iit@training_improvements"
pip install "git+https://github.com/evanhanders/polysemantic-benchmark"

# transformer_lens companions that its own install sometimes fails to pull in
# (poly_bench imports TL at module load, so a missing one aborts before any run).
pip install typeguard jaxtyping einops fancy_einsum

# The light runtime deps the native path imports — installed BEFORE causalab so a
# flaky git install (set -e aborts the script) can never strand these.
pip install safetensors scikit-learn scipy tqdm networkx matplotlib pandas hydra-core rich

# causalab bare (its full deps drag pyvene/git + unpinned transformers; the native
# TL-hook path never needs pyvene). If the smoke below dies with ModuleNotFoundError:
# <x>, `pip install <x>` and rerun — do NOT re-add causalab's full deps.
pip install --no-deps "$CAUSALAB_SPEC"

echo "== smoke: transformers/torch + poly_bench + the whole runner import chain =="
cd "$ROOT"
python - <<'EOF'
import transformers, torch
print("transformers", transformers.__version__, "| torch", torch.__version__,
      "| cuda", torch.cuda.is_available())
from poly_bench.io import load_from_hf
import causalab.neural.pipeline                    # the fork must import w/o pyvene
import os, sys
os.environ.setdefault("OFFDIST_RESULTS_DIR", "results/_smoke")
sys.path.insert(0, "off_manifold"); sys.path.insert(0, "interp_bench")
import run_interaction_effect                      # exercises the full import chain
print("run_interaction_effect imports OK")
EOF

echo ""
echo "== ready. next =="
echo "  source $VENV/bin/activate && export HF_HOME=$HF_HOME && cd $ROOT/off_manifold"
echo "  bash run_interaction_seeds.sh                 # step 0, seeds 0 1 2"
echo "  STEP=both bash run_interaction_seeds.sh"
# NOTE (torch 2.9 pods): if transformers import fails on a stray torchvision,
# `pip uninstall -y torchvision` — see memory: torchvision-stray-breaks-venv.
