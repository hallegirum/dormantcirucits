"""
parity_check.py
===============
Prove the native-hooks ``TracrPipeline`` reproduces causalab's **pyvene** backend.

The idea
--------
pyvene can't wrap a HookedTransformer, so we can't run both backends on the same
object. Instead we use **gpt2**, which loads both ways with matching numerics:

    HF  AutoModelForCausalLM("gpt2")                 -> LMPipeline  (pyvene)
    HookedTransformer.from_pretrained_no_processing  -> TracrPipeline (native)

``from_pretrained_no_processing`` keeps the raw weights (no LN folding / centring),
so the two models' residual streams and logits agree up to float error. To be
robust to even that small drift, every logit check compares the **intervention
delta** (intervened - clean) between the two backends: the shared baseline drift
cancels, leaving only what the intervention did. If the native backend computes
the same swap / interpolation / collection as pyvene, the deltas match.

What it checks
--------------
  collect          : gathered featurized activations           (raw, vs drift)
  interchange      : whole-vector residual swap                (delta)
  interchange DAS  : subspace swap via a SubspaceFeaturizer     (delta)
  interpolation    : linear interp alpha=0.5 (the seam Gaussian uses)  (delta)
  gaussian (info)  : seeded add-noise fn — informational, drift-sensitive

Each example is run as its own batch of 1, so there is no padding and positions
are unambiguous.

Run (needs causalab + pyvene + transformer_lens importable together):
    uv run python dormant-circuit-analysis/interp_bench/parity_check.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# allow "import tracr_pipeline" when run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tracr_pipeline import TracrPipeline  # noqa: E402

from causalab.neural.pipeline import LMPipeline  # noqa: E402
from causalab.neural.units import AtomicModelUnit, InterchangeTarget  # noqa: E402
from causalab.neural.featurizer import Featurizer  # noqa: E402
from causalab.methods.trained_subspace.subspace import SubspaceFeaturizer  # noqa: E402
from causalab.neural.activations.intervenable_model import (  # noqa: E402
    prepare_intervenable_model,
    delete_intervenable_model,
)
from causalab.neural.activations.interchange_mode import (  # noqa: E402
    prepare_intervenable_inputs,
)
from causalab.neural.activations.collect import collect_features  # noqa: E402
from causalab.neural.activations.interpolate import (  # noqa: E402
    set_interventions_interpolation,
)


# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #
def _trace(text: str) -> dict:
    return {"raw_input": text}


# base / counterfactual pairs, each chosen so both sides tokenize to the same
# length (>= POS+1). Run one at a time -> no padding.
EXAMPLES = [
    {"input": _trace("The cat sat on the mat"),
     "counterfactual_inputs": [_trace("A dog ran in the park")]},
    {"input": _trace("She quickly opened the red door"),
     "counterfactual_inputs": [_trace("He slowly closed the blue gate")]},
    {"input": _trace("Paris is the capital of France"),
     "counterfactual_inputs": [_trace("Tokyo is the centre of Japan")]},
]
LAYER = 4
POS = 3


def build_pipelines(device: str) -> tuple[LMPipeline, TracrPipeline]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer

    tok = AutoTokenizer.from_pretrained("gpt2")
    hf = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()
    lm = LMPipeline(hf, max_new_tokens=1, max_length=None, device=device)

    ht = HookedTransformer.from_pretrained_no_processing("gpt2").to(device).eval()
    tp = TracrPipeline(
        ht,
        encode=lambda text: tok.encode(text),
        decode=lambda ids: tok.decode(ids),
        device=device,
    )
    return lm, tp


def make_target(featurizer: Featurizer, feature_indices=None) -> AtomicModelUnit:
    """One resid_post unit at (LAYER, POS). Returned as a bare unit; wrap as
    InterchangeTarget([[unit]]) for interventions, or pass [unit] to collect."""
    return AtomicModelUnit(
        layer=LAYER,
        component_type="block_output",
        indices_func=[POS],  # static -> constant indexer
        unit="pos",
        featurizer=featurizer,
        feature_indices=feature_indices,
        id=f"resid{LAYER}_p{POS}",
    )


# --------------------------------------------------------------------------- #
#  Clean baselines (for delta cancellation)
# --------------------------------------------------------------------------- #
def hf_clean(lm: LMPipeline, ex: dict) -> torch.Tensor:
    with torch.no_grad():
        out = lm.model(**lm.load([ex["input"]]))
    return out.logits.detach().cpu()


def tl_clean(tp: TracrPipeline, ex: dict) -> torch.Tensor:
    with torch.no_grad():
        logits = tp.model(tp.load([ex["input"]])["tokens"])
    return logits.detach().cpu()


# --------------------------------------------------------------------------- #
#  Intervened logits, both backends
# --------------------------------------------------------------------------- #
def _pyvene_logits(lm, examples, target, itype, interp=None) -> torch.Tensor:
    iv = prepare_intervenable_model(lm, target, itype)  # real pyvene (no build_intervention)
    base, cfs, locs, feat = prepare_intervenable_inputs(lm, examples, target)
    if interp is not None:
        set_interventions_interpolation(iv, interp[0], **interp[1])
    with torch.no_grad():
        res = iv(base, cfs, unit_locations=locs, subspaces=feat)
    cf_out = res[1] if isinstance(res, (tuple, list)) else res
    delete_intervenable_model(iv)
    return cf_out.logits.detach().cpu()


def _native_logits(tp, examples, target, itype, interp=None) -> torch.Tensor:
    plan = tp.build_intervention(target, itype)
    base, cfs, locs, feat = prepare_intervenable_inputs(tp, examples, target)
    if interp is not None:
        set_interventions_interpolation(plan, interp[0], **interp[1])  # -> plan.set_interpolation
    res = tp.intervenable_generate(plan, base, cfs, locs, feat, output_scores=True)
    tp.release_intervention(plan)
    return torch.as_tensor(res["scores"]).detach().cpu()


# --------------------------------------------------------------------------- #
#  Comparators
# --------------------------------------------------------------------------- #
def _check(name: str, maxdiff: float, atol: float) -> bool:
    ok = maxdiff <= atol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:22s} max|Δ| = {maxdiff:.2e}  (atol {atol:.0e})")
    return ok


def compare_logits(name, lm, tp, target, itype, atol, interp=None, seed=None) -> bool:
    worst = 0.0
    for ex in EXAMPLES:
        if seed is not None:
            torch.manual_seed(seed)
        p = _pyvene_logits(lm, [ex], target, itype, interp) - hf_clean(lm, ex)
        if seed is not None:
            torch.manual_seed(seed)
        n = _native_logits(tp, [ex], target, itype, interp) - tl_clean(tp, ex)
        worst = max(worst, (p - n).abs().max().item())
    return _check(name, worst, atol)


def compare_collect(lm, tp, atol) -> bool:
    worst = 0.0
    for ex in EXAMPLES:
        unit_p = make_target(Featurizer())
        unit_n = make_target(Featurizer())
        fp = list(collect_features([ex], lm, [unit_p]).values())[0].detach().cpu()
        fn = list(collect_features([ex], tp, [unit_n]).values())[0].detach().cpu()
        worst = max(worst, (fp - fn).abs().max().item())
    return _check("collect", worst, atol)


def report_clean_drift(lm, tp) -> float:
    worst = max((hf_clean(lm, ex) - tl_clean(tp, ex)).abs().max().item() for ex in EXAMPLES)
    print(f"  baseline HF vs TL(no_processing) logit drift: {worst:.2e}  "
          f"(deltas below cancel this)")
    return worst


def compare_attn_result(tp, atol: float) -> bool:
    """Validate the native attn_result (per-head hook_result) path against a manual
    TL hook_result swap. pyvene's head_attention_value_output is hook_z (a different
    tensor), so this is native-vs-TL-reference — the head-slab logic itself is already
    pyvene-validated by the 'interchange head' (hook_z) case. Native vs manual on the
    SAME model, so this should match near-exactly, not just at the drift floor."""
    from causalab.neural.LM_units import AttentionHead

    tp.model.set_use_attn_result(True)  # hook_result only exists when this is on
    head, hook = 3, f"blocks.{LAYER}.attn.hook_result"
    worst = 0.0
    for ex in EXAMPLES:
        base = tp.load([ex["input"]])["tokens"]
        src = tp.load([ex["counterfactual_inputs"][0]])["tokens"]
        with torch.no_grad():
            _, src_cache = tp.model.run_with_cache(src)
        src_slab = src_cache[hook][:, :, head, :]

        def ref_hook(act, hook, _s=src_slab):  # manual: swap head's slab at POS
            act = act.clone()
            act[:, POS, head, :] = _s[:, POS, :]
            return act

        with torch.no_grad():
            ref = tp.model.run_with_hooks(base, fwd_hooks=[(hook, ref_hook)]).detach().cpu()
        unit = AttentionHead(LAYER, head, [POS], location="attn_result", featurizer=Featurizer())
        native = _native_logits(tp, [ex], InterchangeTarget([[unit]]), "interchange")
        worst = max(worst, (native - ref).abs().max().item())
    return _check("attn_result (vs TL ref)", worst, atol)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--atol", type=float, default=1e-3)
    args = ap.parse_args()

    print("Loading gpt2 as HF (pyvene) and HookedTransformer (native)…")
    lm, tp = build_pipelines(args.device)

    print("\nParity: native TracrPipeline vs pyvene, on the same gpt2\n")
    drift = report_clean_drift(lm, tp)
    atol = max(args.atol, 20 * drift)  # keep atol above pure float drift

    results = []
    results.append(compare_collect(lm, tp, atol))

    identity_target = InterchangeTarget([[make_target(Featurizer())]])
    results.append(compare_logits("interchange", lm, tp, identity_target, "interchange", atol))

    das_feat = SubspaceFeaturizer(shape=(tp.model.cfg.d_model, 16))
    das_unit = make_target(das_feat, feature_indices=list(range(8)))
    results.append(
        compare_logits("interchange DAS", lm, tp, InterchangeTarget([[das_unit]]), "interchange", atol)
    )

    # attention-head interchange (per-head hook_z path)
    from causalab.neural.LM_units import AttentionHead
    head_unit = AttentionHead(layer=LAYER, head=3, token_indices=[POS])
    results.append(
        compare_logits("interchange head", lm, tp, InterchangeTarget([[head_unit]]), "interchange", atol)
    )

    def linear(f_base, f_src, alpha):
        return (1 - alpha) * f_base + alpha * f_src

    interp_target = InterchangeTarget([[make_target(Featurizer())]])
    results.append(
        compare_logits("interpolation", lm, tp, interp_target, "interpolation",
                       atol, interp=(linear, {"alpha": 0.5}))
    )

    # informational: the same seam with a stochastic fn. std is read off f_base,
    # which drifts slightly between HF/TL, so this is drift-sensitive by nature —
    # not a gate. linear-interp above already proves the interpolation plumbing.
    def add_noise(f_base, f_src, sigma):
        return f_base + sigma * f_base.std() * torch.randn_like(f_base)

    gauss_target = InterchangeTarget([[make_target(Featurizer())]])
    compare_logits("gaussian (info)", lm, tp, gauss_target, "interpolation",
                   atol=float("inf"), interp=(add_noise, {"sigma": 3.0}), seed=0)

    # last: enables use_attn_result on tp.model, so keep it after the others
    results.append(compare_attn_result(tp, atol))

    print("\n" + ("ALL GATED CHECKS PASSED ✅" if all(results)
                   else "SOME CHECKS FAILED ❌ — inspect Δ above"))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
