#!/usr/bin/env bash
# =============================================================================
# Pod setup for the InterpBench integration.
#
# THE CONFLICT: circuits_benchmark (via `iit`) requires torch < 2.4, but the
# integrated causalab stack runs on torch 2.9 / transformers 4.57. They cannot
# coexist in one interpreter.
#
# THE SOLUTION: two isolated venvs that NEVER share a Python process.
#
#   ~/ib-venv            circuits_benchmark loader  (torch<2.4, jax 0.4.20, TL 1.19)
#                        -> ONE job: load model + hl_ll_corr + encoders and
#                           SERIALISE them to plain files (torch/json). Then done.
#
#   $CAUSALAB_DIR/.venv  causalab + our TracrPipeline  (torch 2.9, transformers 4.57)
#                        -> loads those plain files and runs the off-manifold
#                           analysis. NEVER imports circuits_benchmark.
#
# The bridge is on-disk artifacts (see extract step at the bottom), so no
# dependency ever has to satisfy both trees at once.
#
# Usage (set CAUSALAB_DIR to your existing causalab checkout on the pod):
#     CAUSALAB_DIR=/workspace/causalab \
#       bash dormant-circuit-analysis/off_manifold/setup_pod_env.sh
# =============================================================================
set -euo pipefail

# Point this at your EXISTING causalab checkout on the pod (the old setup already
# put one there). Defaults to $PWD/causalab, NOT $PWD — never clone into /workspace.
CAUSALAB_DIR="${CAUSALAB_DIR:-$PWD/causalab}"
IB_VENV="${IB_VENV:-$HOME/ib-venv}"
# Your forked causalab (guard-line changes + tasks/interpbench package):
CAUSALAB_REPO="${CAUSALAB_REPO:-https://github.com/hallegirum/causalab.git}"
CAUSALAB_BRANCH="${CAUSALAB_BRANCH:-interpbench-integration}"

echo "== 0/4  uv =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "$HOME/.local/bin/env"
fi

echo "== 1/4  system deps (pygraphviz needs Python.h + graphviz headers) =="
apt-get update
apt-get install -y graphviz graphviz-dev python3.10-dev pkg-config git build-essential

echo "== 2/4  ib-venv: circuits_benchmark LOAD/SERIALISE env =="
# Full tree first (resolves iit, tracr, transformer_lens, jax, haiku, pygraphviz…),
# then force the pins we discovered. Order matters: pin AFTER the tree resolves so
# these override whatever it pulled.
uv venv "$IB_VENV" --python 3.10
# shellcheck disable=SC1091
source "$IB_VENV/bin/activate"

uv pip install "circuits-benchmark @ git+https://github.com/FlyingPumba/InterpBench.git"

# --- the pins, each with the error it fixes ---
uv pip install "torch>=2.2,<2.4"                  # torch>=2.4 strict __getattr__ breaks iit ll_model (t.Any)
uv pip install "transformer_lens==1.19.0"         # newer TL is incompatible with the tree
uv pip install "jax==0.4.20" "jaxlib==0.4.20"     # jax>=0.4.21 removed jax.lib.xe
uv pip install "dm-haiku==0.0.11"                 # tracr's haiku pin
# flax<0.8 avoids jax.tree_util.register_dataclass, which needs jax>=0.4.24 (we're on 0.4.20).
# This is the one triad that fought earlier — if resolution still fails, nudge these two.
uv pip install "flax==0.7.5"
uv pip install "absl-py"                          # import name is 'absl', package is absl-py

# --- forked causalab, INTO the ib-venv, so verify_against_model can run here ---
# (causalab.neural imports pyvene; installing causalab pulls torch 2.9, so we
#  re-pin torch<2.4 afterwards for iit. Editable clone so you can iterate on the pod.)
# Idempotent: reuse an existing causalab checkout (the old setup left one), and
# just pull your fork's branch onto it. Only clone when the dir doesn't exist.
if [ -f "$CAUSALAB_DIR/pyproject.toml" ]; then
  echo "   using existing causalab at $CAUSALAB_DIR"
  if [ -d "$CAUSALAB_DIR/.git" ] && [ "${CAUSALAB_SYNC_FORK:-1}" = "1" ]; then
    git -C "$CAUSALAB_DIR" remote get-url fork >/dev/null 2>&1 \
      || git -C "$CAUSALAB_DIR" remote add fork "$CAUSALAB_REPO"
    git -C "$CAUSALAB_DIR" fetch fork "$CAUSALAB_BRANCH"
    git -C "$CAUSALAB_DIR" checkout "$CAUSALAB_BRANCH" 2>/dev/null \
      || git -C "$CAUSALAB_DIR" checkout -b "$CAUSALAB_BRANCH" "fork/$CAUSALAB_BRANCH"
    echo "   checked out fork/$CAUSALAB_BRANCH (set CAUSALAB_SYNC_FORK=0 to skip)"
  fi
elif [ -e "$CAUSALAB_DIR" ] && [ -n "$(ls -A "$CAUSALAB_DIR" 2>/dev/null)" ]; then
  echo "!! $CAUSALAB_DIR is non-empty but has no pyproject.toml — refusing to clone into it." >&2
  echo "   Set CAUSALAB_DIR to your causalab checkout, or to an empty/new path." >&2
  exit 1
else
  git clone --branch "$CAUSALAB_BRANCH" "$CAUSALAB_REPO" "$CAUSALAB_DIR"
fi
uv pip install -e "$CAUSALAB_DIR"
# causalab (unpinned) pulls a bleeding-edge transformers that needs torch>=2.4 AND
# dropped TRANSFORMERS_CACHE. Pin the narrow window that has BOTH gemma2 (pyvene)
# and TRANSFORMERS_CACHE (transformer_lens 1.19) and still supports torch 2.3:
uv pip install "transformers==4.43.4"
uv pip install "typeguard"                         # transformer_lens 1.19 import needs it
uv pip install -U wandb                            # TL 1.19 imports wandb; fix its protobuf-stub skew
uv pip install "torch>=2.2,<2.4"                   # RESTORE: causalab bumped torch to 2.9; iit needs <2.4
# RESTORE jax LAST: flax/causalab/chex/optax bump jax past 0.4.20, but haiku 0.0.11
# references jax.interpreters.xla.xe (removed in newer jax). Pin after everything.
uv pip install --reinstall "jax==0.4.20" "jaxlib==0.4.20"
# jaxlib 0.4.20's C-extension was built against NumPy 1.x; NumPy 2.0's ABI break
# (_ARRAY_API not found) crashes it. This whole tracr/jax era needs numpy<2.
uv pip install "numpy<2"
uv pip uninstall torchvision >/dev/null 2>&1 || true

echo "   -- ib-venv import smoke test (verify chain, pyvene-free) --"
python - <<'PY' || echo "   (import failed — see the jax/flax/haiku triad + torch re-pin notes)"
import torch, transformer_lens, transformers
print("   torch", torch.__version__, "| transformers", transformers.__version__,
      "| TL", getattr(transformer_lens, "__version__", "1.19.0"))
import circuits_benchmark  # noqa: F401
# pyvene is NOT imported: TracrPipeline + the interpbench task load pyvene-free
# (causalab lazy-imports pyvene). pyvene-git-main can't import on transformers<4.44,
# and we don't need it for verify.
import causalab.neural.pipeline           # noqa: F401
from causalab.tasks.interpbench.causal_models import CASES  # noqa: F401
print("   circuits_benchmark + causalab (pyvene-free verify chain) import OK")
PY
deactivate

echo "== 3/4  [OPTIONAL] causalab .venv: torch-2.9 analysis env =="
# NOT needed for verify_against_model or interventions (those run in the ib-venv
# above, which now has causalab). Build this only if you later want the analysis
# on torch 2.9 — then it pairs with extract_artifacts.py as the two-venv bridge.
# Skip with: SKIP_CAUSALAB_VENV=1 bash setup_pod_env.sh
if [ "${SKIP_CAUSALAB_VENV:-0}" = "1" ]; then
  echo "   skipped (SKIP_CAUSALAB_VENV=1) — verify runs in the ib-venv"
else
  cd "$CAUSALAB_DIR"
  uv sync
  # transformer_lens is NOT a declared causalab dep, but TracrPipeline needs it to
  # rebuild the HookedTransformer from the serialised cfg+weights. Same version as
  # the ib-venv so the serialised HookedTransformerConfig dict maps cleanly.
  uv pip install "transformer_lens==1.19.0" "typeguard"   # typeguard: TL 1.19 import dep
  # stray torchvision 0.18.1 (pairs with torch 2.3) breaks transformers on torch 2.9 —
  # not a declared dep, nothing imports it. Removing it repairs the pyvene stack.
  uv pip uninstall torchvision >/dev/null 2>&1 || true
  echo "   -- causalab .venv import smoke test --"
  uv run python -c "import causalab, pyvene, transformer_lens; print('   causalab + pyvene + TL import OK')"
fi

echo "== 4/4  DONE =="
cat <<'NEXT'

PRIMARY workflow — everything in the ib-venv (it now has circuits_benchmark AND
your forked causalab), so verify_against_model runs directly on the loaded model:

  source ~/ib-venv/bin/activate
  # load model + hl_ll_corr + encoders (circuits_benchmark loader), then:
  #   from interpbench_loading import build_tracr_pipeline
  #   pipeline, bos = build_tracr_pipeline(model)
  #   from causalab.tasks.interpbench.causal_models import CASES, verify_against_model
  #   CASES["18"].bos = bos
  #   verify_against_model(CASES["18"], pipeline)

FALLBACK (only if causalab won't run on torch<2.4 in the ib-venv): build the
optional torch-2.9 .venv above and bridge with a small extract_artifacts.py that
dumps PLAIN data — cfg+state_dict (model), encoder dicts+bos_token (JSON), and
hl_ll_corr as {hl_label, ll_name, ll_index} (JSON) — so the .venv never imports
circuits_benchmark/tracr.
NEXT
