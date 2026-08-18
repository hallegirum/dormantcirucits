"""
interaction_effects.py
======================
Interaction-term (INT) analysis over the dormancy pipeline's grid — the
mediation 2x2 of arXiv:2606.27510, per component, with the push machinery for
graded off-manifold corruptions.

The 2x2 (mediator M = one component c; bypass R = everything feeding the
readout NOT through c):

                        M clean (h)          M corrupted (h')
    bypass clean (r)    Phi(r,h)  = Y(a)     Phi(r,h')   <- NOISING
    bypass corr. (r')   Phi(r',h) <- DENOIS. Phi(r',h') = Y(a')

Orientation (all quantities "toward the corruption's answer", so recruitment is
positive regardless of A->B vs B->A direction):

    Phi  = logit[source_result] - logit[base_result]  (per example, at target_pos)
    NIE  = mean( Phi(r,h')  - Y(a)  )   M's effect at the CLEAN bypass
    PIE  = mean( Y(a')      - Phi(r',h) )   M's effect at the CORRUPTED bypass
    INT  = NIE - PIE                    how much M's effect CHANGES with the bypass

Affine network -> NIE == PIE -> INT = 0 (the Step-0 curvature gate). A task-B
component recruited by an off-context state shows PIE ~ 0 with INT > 0.

Two corruption kinds (CFSpec.kind):

  "swap"  the corrupted world is a second INPUT a' (within-task or cross-task
          counterfactual). Cells run through causalab's interchange /
          interpolation interventions; denoising is the same pair with
          base/source roles swapped.
  "push"  the corrupted world is the SAME input A with an additive activation
          push alpha*delta at push_unit (no a' token sequence exists — the push
          occupies its role). Cells run through TracrPipeline.forward_cache /
          .patch, with the push composed into the denoising forward's hooks.
          push_unit is None for SELF-push (push each measured component
          itself): there the bypass never changes (r' == r), so the 2x2
          degenerates — NIE = PIE = the total effect and INT = 0 by
          construction. Self-push rows therefore report TE (the direct
          off-manifold effect of pushing c) and are flagged degenerate=True;
          INT is only informative when push_unit != measured component.

Every row also carries the manifold split of the mediator divergence
delta_c = h'_c - h_c via manifold_decompose (along = component inside the
reference cloud's tangent subspace, orth = perpendicular remainder), the
ground-truth circuit label, and poly shared/task membership — the table is
simultaneously the Step-1 result and the Step-3 calibration set.

Directed axis: a scan is a (base_task -> target_task) cell (make_counterfactual with
base_task/target_task rebuilds pairs from prep["pool"] via create_directed_pairs). Each
row logs base/target, in_base_circuit/in_target_circuit (vs the site's home_tasks), the
attribution effect NIE, and the mediator divergence decomposed against EVERY task cloud
(orth_base = off task-A's manifold, along_target = into task-B's subspace) — the fields
the dormancy path model orth_base -> INT -> |NIE|(FP) needs.

Artifacts (results_layout, case-scoped RESULTS_DIR) — direction + alpha in the path so the
sweep never overwrites itself:
    interaction/{component_id}/{strategy}/{push_site}/{base-target}/a{alpha}/int.json
    intervened_reps/int-{strategy}__{base-target}-a{alpha}__{push_site}/
        {SWAP|PUSH}_intervened_reps.safetensors                    {site: h'_c}
so off-distribution-analyses.py can run its full off-manifold metric suite on the reps.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from safetensors.torch import save_file

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "interp_bench"))

from causalab.neural.activations.interchange_mode import run_interchange_interventions
from causalab.neural.activations.interpolate import run_interpolation_interventions
from causalab.neural.featurizer import Featurizer

from results_layout import INTERVENED_REPS, interaction_dir, label_category_dir


# --------------------------------------------------------------------------- #
#  Phi — the readout
# --------------------------------------------------------------------------- #
def make_readout(cf_pairs: list[dict], target_pos: Any = -1) -> Callable:
    """Directional Phi for one paired counterfactual set: per example i,
    Phi_i(scores) = logit[src_result_i] - logit[base_result_i] at target_pos.

    "Toward the source": positive = the model prefers the corruption's answer, so
    NIE/PIE/INT read as recruitment in whichever direction (A->B or B->A) the
    pairs encode — no per-run sign flag. The SAME readout is applied to all four
    cells (cf_denoise is the same pairs with roles swapped, order preserved, so
    the columns still line up).

    target_pos: scalar int (broadcast; -1 = last position) or per-example (N,)
    positions (e.g. poly's last non-PAD content position).
    Accepts a scores tensor (N,S,V) or the list of per-batch tensors the
    intervention runners return. Runs no model — pure indexing.
    """
    pos_col = torch.tensor([int(ex["counterfactual_inputs"][0]["result"]) for ex in cf_pairs])
    neg_col = torch.tensor([int(ex["input"]["result"]) for ex in cf_pairs])
    idx = torch.arange(len(cf_pairs))
    tp = torch.as_tensor(target_pos, dtype=torch.long)

    def phi(scores: Any) -> torch.Tensor:
        L = (torch.cat([torch.as_tensor(s) for s in scores], 0)
             if isinstance(scores, list) else torch.as_tensor(scores))
        L = L.float()
        rows = L[idx, tp] if tp.dim() else L[:, int(tp), :]   # (N, V)
        return rows[idx, pos_col] - rows[idx, neg_col]

    phi.readout_target = "toward_source(logit[src]-logit[base])"
    phi.pos_col, phi.neg_col, phi.target_pos = pos_col, neg_col, tp
    return phi


def poly_target_positions(cf_pairs: list[dict], pad: int = 1) -> torch.Tensor:
    """Per-example last non-PAD content position of the BASE input (poly layout:
    [task, BOS, tok..., PAD...]; content starts at pos 2). Mirrors poly_records."""
    out = []
    for ex in cf_pairs:
        toks = list(ex["input"]["raw_input"])
        content = [p for p, t in enumerate(toks) if p >= 2 and t != pad]
        out.append(content[-1] if content else len(toks) - 1)
    return torch.tensor(out, dtype=torch.long)


# --------------------------------------------------------------------------- #
#  Manifold split of a divergence
# --------------------------------------------------------------------------- #
def manifold_decompose(delta: torch.Tensor, cloud: torch.Tensor,
                       d: int = 8, var_keep: float = 0.99) -> tuple[torch.Tensor, torch.Tensor]:
    """Split divergence DIRECTIONS delta (N, D) against a reference cloud (M, D)
    of raw clean reps: along = norm of the projection onto the cloud's tangent
    subspace, orth = norm of the perpendicular remainder. Returns ((N,), (N,)).

    The tangent basis is the centered SVD of the cloud, effective-rank capped
    exactly like _cloud_offmanifold (top PCs holding var_keep of the variance,
    max d) so dead dims never count as "along". Which cloud you pass defines the
    semantics: the SOURCE-context clean cloud at the site gives along-B (the
    B-subspace component of the divergence — the quantity INT is regressed on);
    the base-context cloud gives along-A. Works on any site's reps — nothing
    here is specific to the MLP or to B.

    Direction decomposition (no mean-shift of delta): delta is a displacement,
    not a point, so only the subspace matters, not the cloud's center.
    """
    X = cloud.double()
    Xc = X - X.mean(0)
    _u, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    D = delta.double()
    if float(S[0]) <= 1e-8:  # degenerate cloud: no tangent — everything is orth
        return torch.zeros(D.shape[0]), D.norm(dim=-1).float()
    energy = (S ** 2).cumsum(0) / (S ** 2).sum().clamp_min(1e-12)
    d_eff = int(min(d, int((energy < var_keep).sum().item()) + 1, S.shape[0]))
    Vd = Vt[:d_eff]
    onto = D @ Vd.T
    along = onto.norm(dim=-1)
    orth = (D - onto @ Vd).norm(dim=-1)
    return along.float(), orth.float()


# --------------------------------------------------------------------------- #
#  Counterfactual spec
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class CFSpec:
    """One corruption design, shared by every component in a scan.

    strategy   how the corruption direction delta is built:
                 "STR"  delta = paired source rep - base rep (a real
                                 counterfactual divergence; within-task pairs via
                                 category="context_matched", cross-task via
                                 "context_mismatched")
                 "Gaussian"      delta = 3*std(base)*randn (the pipeline's GN
                                 convention), push kind only
               (tangent_pc / along-orth-projected / subspace strategies slot in
                here later: they only add new delta builders — nothing in
                component_int or run_layer_scans changes.)
    kind       "swap" (input corruption) | "push" (activation corruption)
    alpha      strength scalar: h' = h + alpha*delta. Sweep it; STR-at-1 is
               alpha=1 STR.
    push_unit  None = self-push (each measured component pushed itself,
               degenerate INT — see module docstring) | an AtomicModelUnit =
               two-site mediation (push there, measure downstream).
    positions  scalar or per-example (N,) intervention positions.
    """
    strategy: str
    kind: str
    cf_noise: list
    cf_denoise: list
    alpha: float = 1.0
    push_unit: Any = None
    positions: Any = -1
    seed: int = 0
    base_task: Any = None      # the directed cell: patch/push base_task -> target_task
    target_task: Any = None    # None for Gaussian (no direction) or legacy category scans
    meta: dict = dataclasses.field(default_factory=dict)

    @property
    def push_name(self) -> str:
        if self.kind == "swap":
            return "input"
        return "self" if self.push_unit is None else str(self.push_unit.id)

    @property
    def direction(self) -> str:
        """Path/label token for the (base,target) cell. Gaussian has no target ->
        '{base}-rand'; legacy (no base_task) falls back to the category name."""
        if self.base_task is None:
            return str(self.meta.get("category", "na"))
        tt = "rand" if self.target_task is None else self.target_task
        return f"{self.base_task}-{tt}"

    @property
    def mode_name(self) -> str:  # filename prefix for the saved reps
        return "SWAP" if self.kind == "swap" else "PUSH"


def make_counterfactual(prep: dict, strategy: str, *, base_task: Any = None,
                        target_task: Any = None, category: str = "context_mismatched",
                        alpha: float = 1.0, push_site: Any = None, n: int | None = None,
                        positions: Any = None, seed: int = 0) -> CFSpec:
    """Build a CFSpec from a PrepareData.generate() dict.

    DIRECTED (preferred): pass base_task AND target_task -> rebuild pairs from
    prep["pool"] via create_directed_pairs (base ∈ base_task, source ∈ target_task).
    base==target is the within-task control. This is the axis the regression needs;
    Gaussian ignores target_task (random push) but keeps base_task for labelling.

    LEGACY: omit base_task/target_task -> filter prep["cf_dataset"] by `category`
    (the old mixed-direction behaviour; kept so the screen/older callers still run).

    cf_denoise = the same pairs with base/source swapped (order preserved so one
    readout serves all four 2x2 cells). kind: push_site given -> "push" at that unit;
    Gaussian -> "push" at self; else "swap". positions default: per-example poly
    last-non-PAD, else scalar -1.
    """
    if strategy not in ("STR", "Gaussian"):
        raise NotImplementedError(
            f"strategy {strategy!r} not implemented yet (STR | Gaussian). "
            f"tangent_pc / along-orth-projected / subspace go in as new delta builders here.")

    if base_task is not None:
        from context_datasets import create_directed_pairs
        pairs = create_directed_pairs(prep["pool"], base_task,
                                      target_task if target_task is not None else base_task,
                                      n=n, seed=seed)
        cat = "context_matched" if base_task == target_task else "context_mismatched"
    else:
        pairs = [ex for ex in prep["cf_dataset"] if ex.get("category", category) == category]
        cat = category
        if n is not None and n < len(pairs):
            rng = torch.Generator().manual_seed(seed)
            keep = torch.randperm(len(pairs), generator=rng)[:n].tolist()
            pairs = [pairs[i] for i in sorted(keep)]
    if not pairs:
        raise ValueError(f"no cf pairs (base={base_task}, target={target_task}, category={cat})")

    def _swapped(ex):
        out = dict(ex)
        out["input"], out["counterfactual_inputs"] = ex["counterfactual_inputs"][0], [ex["input"]]
        return out

    cf_denoise = [_swapped(ex) for ex in pairs]

    if positions is None:
        try:
            positions = poly_target_positions(pairs)
        except Exception:
            positions = -1

    kind = "push" if (push_site is not None or strategy == "Gaussian") else "swap"
    if strategy == "Gaussian" and kind == "swap":
        raise ValueError("Gaussian is an activation push; it has no input-swap form")
    # Gaussian has no task direction — record base for labelling, drop target.
    tt = None if strategy == "Gaussian" else target_task

    return CFSpec(strategy=strategy, kind=kind, cf_noise=pairs, cf_denoise=cf_denoise,
                  alpha=alpha, push_unit=push_site, positions=positions, seed=seed,
                  base_task=base_task, target_task=tt, meta={"category": cat})


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _units_of(target) -> list:
    return [u for group in target for u in group]


def _gather(slab: torch.Tensor, positions: Any) -> torch.Tensor:
    """(N, S, D) slab -> (N, D) at scalar or per-example positions."""
    p = torch.as_tensor(positions, dtype=torch.long)
    if p.dim() == 0:
        return slab[:, int(p), :]
    return slab[torch.arange(slab.shape[0]), p]


def _scores_of(raw: dict) -> list:
    return raw["scores"]


# --------------------------------------------------------------------------- #
#  component_int — the 2x2 for one target
# --------------------------------------------------------------------------- #
def component_int(pipeline, spec: CFSpec, target, readout: Callable,
                  Ya: torch.Tensor, Yap: torch.Tensor, batch_size: int = 32,
                  caches: dict | None = None) -> dict:
    """NIE/PIE/INT for one grid target under one CFSpec. The two SHARED forwards
    (Ya = Phi on the clean run, Yap = Phi on the corrupted run) are computed once
    by the driver and passed in; this runs only the two PER-COMPONENT cells.

    swap kind: noising = interchange on cf_noise (base a, c <- a'), denoising =
    interchange on cf_denoise (base a', c <- a); alpha != 1 routes through the
    interpolation runner with f_base + alpha*(f_src - f_base).

    push kind (two-site): needs `caches` from the driver — {"clean": slabs,
    "pushed": slabs, "push": (unit, delta, pos)}. noising = patch(A, c <- h'_c)
    with NO push (the clean bypass must stay clean — the push's downstream effect
    is already baked into the cached h'_c); denoising = patch(A, c <- h_c) WITH
    the push composed in (the corrupted bypass r' is realised by actually running
    under the push while only c is restored).
    """
    if spec.kind == "swap":
        for unit in _units_of(target):
            unit.set_featurizer(Featurizer())  # raw-space swap, like compute_patching_scores
        if spec.alpha == 1.0:
            raw_n = run_interchange_interventions(pipeline, spec.cf_noise, target,
                                                  batch_size=batch_size, output_scores=True)
            raw_d = run_interchange_interventions(pipeline, spec.cf_denoise, target,
                                                  batch_size=batch_size, output_scores=True)
        else:
            def scaled(f_base, f_src, **_):
                return f_base + spec.alpha * (f_src - f_base)
            raw_n = run_interpolation_interventions(pipeline, spec.cf_noise, target,
                                                    fn=scaled, params={},
                                                    batch_size=batch_size, output_scores=True)
            raw_d = run_interpolation_interventions(pipeline, spec.cf_denoise, target,
                                                    fn=scaled, params={},
                                                    batch_size=batch_size, output_scores=True)
        phi_n = readout(_scores_of(raw_n))
        phi_d = readout(_scores_of(raw_d))
    else:
        units = _units_of(target)
        bases = [ex["input"] for ex in spec.cf_noise]
        pos = spec.positions
        h = {u.id: _gather(caches["clean"][u.id], pos) for u in units}
        hp = {u.id: _gather(caches["pushed"][u.id], pos) for u in units}
        writes_n = [(u, hp[u.id], pos) for u in units]
        writes_d = [(u, h[u.id], pos) for u in units]
        phi_n = readout(pipeline.patch(bases, writes_n, push=None, batch_size=batch_size))
        phi_d = readout(pipeline.patch(bases, writes_d, push=caches["push"], batch_size=batch_size))

    nie = float((phi_n - Ya).mean())
    pie = float((Yap - phi_d).mean())
    return {"NIE": nie, "PIE": pie, "INT": nie - pie,
            "TE": float((Yap - Ya).mean()),
            "phi_noise": float(phi_n.mean()), "phi_denoise": float(phi_d.mean()),
            "Ya": float(Ya.mean()), "Yap": float(Yap.mean())}


def build_task_clouds(pipeline, pool, units, batch_size: int = 32, pad: int = 1) -> dict:
    """Per-task clean-rep clouds at each unit: {task: {unit.id: (M, D)}}. Each task's
    reps are its own pool traces run clean and gathered at THEIR last-non-PAD positions
    — i.e. task K's within-task manifold at the site. Computed ONCE per combo and reused
    across every direction/strategy/alpha; this is what makes orth_base = 'off task-A's
    manifold' and along_target = 'into task-B's subspace' well defined."""
    clouds = {}
    for task, traces in pool.items():
        tr = list(traces)
        pos = poly_target_positions([{"input": t} for t in tr], pad=pad)
        _, slabs = pipeline.forward_cache(tr, units, batch_size=batch_size)
        clouds[task] = {u.id: _gather(slabs[u.id], pos) for u in units}
    return clouds


# --------------------------------------------------------------------------- #
#  run_layer_scans — the driver
# --------------------------------------------------------------------------- #
def run_layer_scans(prep: dict, components: dict, spec: CFSpec, *,
                    batch_size: int = 32, gt_labels: dict | None = None,
                    poly_mem: dict | None = None, home_tasks: dict | None = None,
                    clouds: dict | None = None, save: bool = True) -> list[dict]:
    """Per-component (NIE, PIE, INT) + PER-TASK geometry + directed labels for one
    (base_task -> target_task, strategy, alpha) scan over a grid subset.

    prep        the PrepareData.generate() dict.
    components  {key: InterchangeTarget} — a build_full_grid subset.
    spec        one make_counterfactual output (directed).
    gt_labels   {site_id: 'circuit'|'none'} (circuit_specs); poly_mem {site_id: label}.
    home_tasks  {site_id: [task ints]} — the tasks whose circuit includes the site
                (poly_membership 'tasks'); gives in_base_circuit / in_target_circuit.
    clouds      {task: {site_id: (M,D)}} per-task clean clouds (build_task_clouds),
                computed ONCE per combo and passed in. The mediator divergence
                delta_c = h'_c - h_c is decomposed against EVERY task's cloud, so a
                row carries orth_base (off task-A's manifold), along_target (into
                task-B's subspace), and along_nontarget_max — the competing drivers
                the dormancy regression (orth_base -> INT -> |NIE| FP) needs.

    Attribution signal: NIE IS the method's apparent causal effect; with
    in_base_circuit it defines the base task's false positives. Push scans also log
    the PUSH's own per-task split (push_along/orth_by_task at X) for the mediation
    term. Saves collision-free: interaction/{site}/{strategy}/{push}/{base-target}/a{alpha}/.
    """
    pipeline = prep["pipeline"]
    readout = make_readout(spec.cf_noise, spec.positions)
    bases = [ex["input"] for ex in spec.cf_noise]
    sources = [ex["counterfactual_inputs"][0] for ex in spec.cf_noise]
    all_units = [u for t in components.values() for u in _units_of(t)]
    for u in all_units:
        u.set_featurizer(Featurizer())
    pos = spec.positions
    tasks = sorted(clouds.keys()) if clouds else None

    base_scores, base_slabs = pipeline.forward_cache(bases, all_units, batch_size=batch_size)
    Ya = readout(base_scores)
    push_geo = None  # {task: (along, orth)} of the push's OWN divergence at X

    if spec.kind == "swap":
        src_scores, src_slabs = pipeline.forward_cache(sources, all_units, batch_size=batch_size)
        Yap = readout(src_scores)
        caches = None
        h = {u.id: _gather(base_slabs[u.id], pos) for u in all_units}
        h_src = {u.id: _gather(src_slabs[u.id], pos) for u in all_units}
        hp = {uid: h[uid] + spec.alpha * (h_src[uid] - h[uid]) for uid in h}
        legacy_cloud = h_src
    elif spec.push_unit is not None:
        X = spec.push_unit
        _, x_slabs = pipeline.forward_cache(bases, [X], batch_size=batch_size)
        x_base = _gather(x_slabs[X.id], pos)
        if spec.strategy == "STR":
            _, xs_slabs = pipeline.forward_cache(sources, [X], batch_size=batch_size)
            delta = spec.alpha * (_gather(xs_slabs[X.id], pos) - x_base)
        else:  # Gaussian: no task direction — the pipeline's GN convention scaled by alpha
            g = torch.Generator().manual_seed(spec.seed)
            delta = spec.alpha * 3.0 * x_base.std() * torch.randn(x_base.shape, generator=g)
        push = (X, delta, pos)
        pushed_scores, pushed_slabs = pipeline.forward_cache(
            bases, all_units, push=push, batch_size=batch_size)
        Yap = readout(pushed_scores)
        caches = {"clean": base_slabs, "pushed": pushed_slabs, "push": push}
        h = {u.id: _gather(base_slabs[u.id], pos) for u in all_units}
        hp = {u.id: _gather(pushed_slabs[u.id], pos) for u in all_units}
        legacy_cloud = None
        if clouds and all(X.id in clouds[K] for K in clouds):  # push-site mediation geometry
            push_geo = {str(K): [float(v.mean()) for v in manifold_decompose(delta, clouds[K][X.id])]
                        for K in clouds}
        if float((Yap - Ya).abs().mean()) < 1e-6:  # would flatten every row -> the earlier bug
            print(f"[int][warn] push at {X.id} left the readout unchanged (Yap==Ya): the push "
                  f"may not reach the readout position; INT will be ~0 at every site.")
    else:
        caches, Yap = None, None  # self-push: per-component pushed world in the loop
        h = {u.id: _gather(base_slabs[u.id], pos) for u in all_units}
        hp, legacy_cloud = {}, None
        g = torch.Generator().manual_seed(spec.seed)
        deltas = {uid: spec.alpha * 3.0 * h[uid].std() * torch.randn(h[uid].shape, generator=g)
                  for uid in h}

    def _fmt(v):
        return f"{v:+.3f}" if v is not None else "  n/a"

    rows, reps_out = [], {}
    for key, target in components.items():
        units = _units_of(target)
        site_id = units[0].id
        degenerate = False
        if spec.kind == "push" and spec.push_unit is None:
            delta_c = deltas[site_id]
            pushed_scores_c, _ = pipeline.forward_cache(
                bases, [], push=(units[0], delta_c, pos), batch_size=batch_size)
            Yap_c = readout(pushed_scores_c)
            te = float((Yap_c - Ya).mean())
            cells = {"NIE": te, "PIE": te, "INT": 0.0, "TE": te,
                     "phi_noise": float(Yap_c.mean()), "phi_denoise": float(Ya.mean()),
                     "Ya": float(Ya.mean()), "Yap": float(Yap_c.mean())}
            hp[site_id] = h[site_id] + delta_c
            degenerate = True
        else:
            if spec.kind == "push" and site_id == spec.push_unit.id:
                degenerate = True  # measuring the pushed site itself: r' == r
            cells = component_int(pipeline, spec, target, readout, Ya, Yap,
                                  batch_size=batch_size, caches=caches)

        delta_c = hp[site_id] - h[site_id]
        if clouds:
            dec = {K: manifold_decompose(delta_c, clouds[K][site_id]) for K in tasks}
            along_K = {str(K): float(a.mean()) for K, (a, o) in dec.items()}
            orth_K = {str(K): float(o.mean()) for K, (a, o) in dec.items()}
        else:  # legacy single-cloud fallback (old callers without per-task clouds)
            a, o = manifold_decompose(delta_c, (legacy_cloud or h)[site_id])
            along_K, orth_K = {"legacy": float(a.mean())}, {"legacy": float(o.mean())}

        bt, tt = spec.base_task, spec.target_task
        home = list((home_tasks or {}).get(site_id, []))
        along_target = along_K.get(str(tt)) if tt is not None else None
        orth_base = orth_K.get(str(bt)) if bt is not None else None
        nontarget = [v for k, v in along_K.items() if k != str(tt)]
        row = {
            "site": site_id, "key": str(key),
            "strategy": spec.strategy, "kind": spec.kind, "push_site": spec.push_name,
            "direction": spec.direction, "base_task": bt, "target_task": tt, "alpha": spec.alpha,
            "category": spec.meta.get("category"), "n": len(spec.cf_noise),
            **cells,
            # --- M-layer: attribution + ground truth (dormancy outcome) ---
            "home_tasks": home,
            "in_base_circuit": (bt in home) if bt is not None else None,
            "in_target_circuit": (tt in home) if tt is not None else None,
            "gt_label": (gt_labels or {}).get(site_id),
            "poly_membership": (poly_mem or {}).get(site_id),
            # --- per-task geometry (competing drivers) ---
            "along_by_task": along_K, "orth_by_task": orth_K,
            "along_target": along_target, "orth_base": orth_base,
            "along_nontarget_max": (max(nontarget) if nontarget else None),
            # legacy fields the collector reads -> now the base/target-relative geometry
            "along": along_target if along_target is not None else next(iter(along_K.values())),
            "orth": orth_base if orth_base is not None else next(iter(orth_K.values())),
            # --- push-site mediation geometry (push scans only) ---
            "push_geo_by_task": push_geo,
            "push_along_target": (push_geo[str(tt)][0] if (push_geo and str(tt) in push_geo) else None),
            "push_orth_base": (push_geo[str(bt)][1] if (push_geo and str(bt) in push_geo) else None),
            "degenerate": degenerate,
            "readout_target": readout.readout_target,
            "reps_ref": _reps_dirname(spec),
        }
        rows.append(row)
        reps_out[site_id] = hp[site_id].detach().contiguous().cpu()

        if save:
            d = interaction_dir(site_id, spec.strategy, spec.push_name, spec.direction, spec.alpha)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "int.json"), "w") as f:
                json.dump(row, f, indent=2)
        print(f"[int] {spec.direction:>7} a{spec.alpha:<4g} {site_id:<32} INT {cells['INT']:+.4f}  "
              f"orth_b {_fmt(orth_base)}  along_t {_fmt(along_target)}"
              f"{'  (deg)' if degenerate else ''}")

    if save:
        rd = label_category_dir(INTERVENED_REPS, f"int-{spec.strategy}",
                                f"{spec.direction}-a{spec.alpha:g}", spec.push_name)
        os.makedirs(rd, exist_ok=True)
        save_file(reps_out, os.path.join(rd, f"{spec.mode_name}_intervened_reps.safetensors"))
        print(f"[int] saved {len(rows)} rows ({spec.strategy}/{spec.direction}/a{spec.alpha:g})")
    return rows


def _reps_dirname(spec: CFSpec) -> str:
    return f"int-{spec.strategy}__{spec.direction}__a{spec.alpha:g}__{spec.push_name}"
