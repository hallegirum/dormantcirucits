#!/usr/bin/env bash
# Set up an ISOLATED InterpBench environment on a fresh Ubuntu pod and run the
# feasibility spike. Kept separate from the causalab venv because InterpBench pins
# transformer_lens==1.19.0 / transformers ^4.37 (incompatible with causalab's newer
# transformers). On Linux the whole dep tree resolves cleanly — no --no-deps needed.
#
# Usage:
#   1. (Mac) scp the spike up:
#        scp -i ~/.ssh/runpod_key -P <port> -r dormant-circuit-analysis/interp_bench \
#            root@<ip>:/workspace/
#   2. (pod) bash /workspace/interp_bench/setup_interpbench_pod.sh
#   3. (pod) source ~/ib-venv/bin/activate
#            python /workspace/interp_bench/spike_load_interpbench.py --case 8 --layer 0 --device cpu
set -euo pipefail

echo "=== 1/4  system deps (graphviz + PYTHON headers for pygraphviz, git) ==="
apt-get update
# python3.10-dev provides Python.h (the build looks in /usr/include/python3.10);
# graphviz-dev provides cgraph.h. Both are needed to compile pygraphviz.
apt-get install -y graphviz graphviz-dev python3.10-dev pkg-config git build-essential

echo "=== 2/4  uv + isolated python 3.10 venv ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source "$HOME/.local/bin/env"
fi
uv venv "$HOME/ib-venv" --python 3.10
source "$HOME/ib-venv/bin/activate"

echo "=== 3/4  InterpBench + all deps (Linux wheels build cleanly) ==="
# full install: resolves circuits-benchmark, iit, tracr, transformer_lens==1.19.0,
# transformers, pygraphviz, lxml, jax, etc. in one shot.
uv pip install "circuits-benchmark @ git+https://github.com/FlyingPumba/InterpBench.git"

echo "=== 4/4  pin torch < 2.4 (newer torch's strict __getattr__ breaks iit's ll_model) ==="
uv pip install "torch>=2.2,<2.4"

echo ""
echo "SETUP DONE. Run the spike:"
echo "  source $HOME/ib-venv/bin/activate"
echo "  python /workspace/interp_bench/spike_load_interpbench.py --case 8 --layer 0 --device cpu"
echo ""
echo "(the InterpBench models are tiny — CPU is instant, and it avoids any CUDA/"
echo " torch-wheel mismatch on this pod. Use --device cuda only if you want to test GPU.)"
