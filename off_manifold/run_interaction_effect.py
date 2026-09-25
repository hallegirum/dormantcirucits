"""
run_interaction_effect.py
=========================
CLI driver for interaction_effects.py over a prepared dormancy case — Step 0
(the curvature gate) and the full strategy x push map.

    --step 0     STR swap on WITHIN-task pairs (category=context_matched) over the
                 grid. If the substrate is near-affine, INT ~ 0 everywhere and the
                 borrowing mechanism cannot exist — the gate decides whether
                 anything downstream can work. Verdict = max |INT| over
                 non-degenerate rows vs --gate-tol.
    --step full  the four-row map (see interaction_effects.py docstring):
                   1. STR swap, CROSS-task pairs        (existence: PIE~0, INT>0)
                   2. STR push at the shared site        (mediation, alpha-sweep)
                   3. Gaussian push at the shared site   (direction control, alpha-sweep)
                   4. Gaussian self-push                 (per-component TE, alpha-sweep)
    --step both  0 then full.

Reuses PrepareData / build_full_grid / circuit_specs / poly_membership from the
dormancy writer, so ground-truth + shared/disjoint labels ride on every row.
Artifacts land in the results_layout tree (interaction/..., intervened_reps/int-...)
plus a per-step summary json under interaction/.

Run in the poly-venv (poly) / ib-venv (tracr, ioi); seed-sweep via the wrapper:
    OFFDIST_RESULTS_DIR=results/poly_cases_0+1/int_seed0 \
      python run_interaction_effect.py --case poly --poly-comb cases_0+1 --step both --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "interp_bench"))


# results tree BEFORE results_layout import (same convention as run_dormancy_pipeline,
# plus a seed suffix so the three-seed sweep never overwrites int.json rows).
def _prearg(flag, default):
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default


_case_arg = str(_prearg("--case", "poly"))
_tree = (f"poly_{_prearg('--poly-comb', 'cases_0+1')}" if _case_arg.lower() == "poly" else _case_arg)
os.environ.setdefault("OFFDIST_RESULTS_DIR",
                      os.path.join("results", _tree, f"int_seed{_prearg('--seed', '0')}"))

from results_layout import RESULTS_DIR, INTERACTION  # noqa: E402
from interaction_effects import make_counterfactual, run_layer_scans  # noqa: E402
from run_dormancy_pipeline import PrepareData, _site_to_spec  # noqa: E402
from path_finder import build_full_grid  # noqa: E402
from interpbench_loading import circuit_specs  # noqa: E402


# --------------------------------------------------------------------------- #
#  labels + push-site selection
# --------------------------------------------------------------------------- #
def site_labels(grid, d):
    """(gt, mem, home): gt {site_id: 'circuit'|'none'} from circuit_specs; mem
    {site_id: 'shared'|'task{i}'|'none'} label; home {site_id: [task ints]} — the
    tasks whose circuit includes the site (poly_membership 'tasks'), which gives
    in_base_circuit / in_target_circuit per directed row."""
    union = [t.flatten()[0] for t in grid.values()]
    circuit = circuit_specs(d["hl_ll_corr"])
    gt = {s.id: ("circuit" if _site_to_spec(s) in circuit else "none") for s in union}
    mem, home = {}, {}
    pm = d.get("poly_membership") or {}
    for s in union:
        info = pm.get(_site_to_spec(s))
        mem[s.id] = info["label"] if info else "none"
        home[s.id] = list(info["tasks"]) if info else []
    return gt, mem, home


def pick_push_unit(grid, poly_mem, arg):
    """The push site X for the mediation rows. --push-site:
      'auto'    first grid MLP site labelled 'shared' by poly_membership (the
                shared-circuit MLP — the hypothesised recruiter); falls back to
                MLP at layer 0 when no 'shared' labels exist (GPT-2 IOI);
      a site_id exact match against the grid's units."""
    units = {t.flatten()[0].id: t.flatten()[0] for t in grid.values()}
    if arg != "auto":
        if arg not in units:
            raise SystemExit(f"--push-site {arg!r} not in grid; sites: {sorted(units)}")
        return units[arg]
    shared_mlp = [u for uid, u in units.items()
                  if poly_mem.get(uid) == "shared" and "mlp" in u.component_type]
    if shared_mlp:
        return shared_mlp[0]
    shared = [u for uid, u in units.items() if poly_mem.get(uid) == "shared"]
    if shared:
        return shared[0]
    mlp0 = [(uid, u) for uid, u in units.items()
            if "mlp" in u.component_type and int(u.layer) == 0]
    if mlp0:
        print(f"[push] no 'shared' labels; falling back to MLP layer 0: {mlp0[0][0]}")
        return mlp0[0][1]
    raise SystemExit("--push-site auto: no shared-labelled or MLP-layer-0 site in grid; "
                     "pass an explicit site_id.")


def _summary(rows):
    live = [r for r in rows if not r["degenerate"]]
    top = sorted(live, key=lambda r: -abs(r["INT"]))[:15]
    keys = ("site", "strategy", "push_site", "direction", "base_task", "target_task",
            "alpha", "NIE", "PIE", "INT", "orth_base", "along_target", "in_base_circuit",
            "in_target_circuit", "gt_label", "poly_membership")
    return {
        "n_rows": len(rows), "n_degenerate": len(rows) - len(live),
        "max_abs_INT": max((abs(r["INT"]) for r in live), default=0.0),
        "mean_INT": (sum(r["INT"] for r in live) / len(live)) if live else 0.0,
        "top_by_abs_INT": [{k: r.get(k) for k in keys} for r in top],
    }


def _save_summary(name, payload):
    os.makedirs(INTERACTION, exist_ok=True)
    path = os.path.join(INTERACTION, name)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[int] summary -> {path}")


def _scan(prep, grid, gt, mem, home, clouds, args, **mk):
    """One make_counterfactual + run_layer_scans, threading the shared labels/clouds."""
    spec = make_counterfactual(prep, n=args.n, seed=args.seed, **mk)
    return run_layer_scans(prep, grid, spec, batch_size=args.batch_size,
                           gt_labels=gt, poly_mem=mem, home_tasks=home, clouds=clouds,
                           graddrop_k=getattr(args, "graddrop", 0),
                           compute_local_gap=getattr(args, "local_gap", False))


# --------------------------------------------------------------------------- #
#  directed sweep — the (base x target) matrix over all strategies
# --------------------------------------------------------------------------- #
def directed_sweep(prep, grid, gt, mem, home, clouds, X, args, mode="full"):
    """Full base x target x strategy x alpha map (or just the within-task diagonal
    control when diagonal_only). See the strategy x direction table:
      STR swap  base->base   : within-task control (affine gate), single alpha
      STR swap  base->target : cross-task existence (PIE~0, INT>0), alpha-swept
      STR push@X base->target : mediation (downstream recruitment), alpha-swept
      Gaussian push@X base    : direction-null control, alpha-swept (no target)
      Gaussian self-push base : per-component TE (degenerate INT), alpha-swept
    """
    tasks = prep["tasks"]
    all_rows = []
    for base in tasks:
        if mode in ("diagonal", "full"):
            print(f"== base task {base} : within-task control (STR swap {base}->{base}) ==")
            all_rows += _scan(prep, grid, gt, mem, home, clouds, args,
                              strategy="STR", base_task=base, target_task=base, alpha=1.0)
        if mode == "diagonal":
            continue
        # within-task STR push @ shared MLP, alpha-swept: the push analog of the within-task
        # swap. base==target draws DISTINCT same-task example pairs, so the injection
        # alpha*(x_src - x_base) at X is a real near-manifold perturbation (non-degenerate INT
        # for downstream c). Closes the swap/push grid on the within-task diagonal.
        print(f"== within-task {base}->{base} : STR push@{X.id} (alphas {args.alphas}) ==")
        for a in args.alphas:
            all_rows += _scan(prep, grid, gt, mem, home, clouds, args, strategy="STR",
                              base_task=base, target_task=base, alpha=a, push_site=X)
        if mode == "within-push":                              # ONLY the new within-task push
            continue
        for a in args.alphas:  # Gaussian controls: no target, once per base
            all_rows += _scan(prep, grid, gt, mem, home, clouds, args,
                              strategy="Gaussian", base_task=base, push_site=X, alpha=a)
            all_rows += _scan(prep, grid, gt, mem, home, clouds, args,
                              strategy="Gaussian", base_task=base, push_site=None, alpha=a)
        for target in tasks:
            if target == base:
                continue
            print(f"== direction {base}->{target} : STR swap + STR push@{X.id} (alphas {args.alphas}) ==")
            for a in args.alphas:
                all_rows += _scan(prep, grid, gt, mem, home, clouds, args, strategy="STR",
                                  base_task=base, target_task=target, alpha=a)
                all_rows += _scan(prep, grid, gt, mem, home, clouds, args, strategy="STR",
                                  base_task=base, target_task=target, alpha=a, push_site=X)
    return all_rows


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="poly", help="'poly', 'ioi', 'gpt2-ioi', or a tracr case id")
    ap.add_argument("--poly-comb", default="cases_0+1")
    ap.add_argument("--target-pos", type=int, default=-1)
    ap.add_argument("--step", default="0", choices=["0", "full", "both", "within-push"],
                    help="0=within-swap gate | full=all | both | within-push=ONLY within-task STR push@MLP")
    ap.add_argument("--push-site", default="auto",
                    help="mediation push site: 'auto' = first shared-labelled MLP "
                         "(poly), or an explicit grid site_id")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.3, 1.0, 3.0])
    ap.add_argument("--gate-tol", type=float, default=0.05,
                    help="step-0 affine verdict threshold on max|INT| (logit units)")
    ap.add_argument("--patch-at", default="internal", choices=["corr", "write", "internal"],
                    help="internal = mlp.hook_post + attn.hook_z (poly correspondence sites)")
    ap.add_argument("--position", default="last_token")
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--graddrop", type=int, default=0,
                    help="AtP* GradDrop passes for the gap measure (0 = raw AtP; ~5 de-cancels)")
    ap.add_argument("--local-gap", action="store_true",
                    help="compute local gate-crossing at each component's own activation function")
    ap.add_argument("--no-residual", dest="include_residual", action="store_false")
    ap.set_defaults(include_residual=False)  # INT is a component story; residual = the bypass
    args = ap.parse_args()

    from activation_manifold import set_seed
    set_seed(args.seed)

    print(f"== prepare (case {args.case}, seed {args.seed}) | results -> {RESULTS_DIR} ==")
    d = PrepareData(args.case, n=args.n, seed=args.seed, device=args.device,
                    position=args.position, poly_comb=args.poly_comb,
                    target_pos=args.target_pos).generate()

    layers = args.layers if args.layers is not None else list(range(d["n_layers"]))
    grid = build_full_grid(d["pipeline"], d["task"], layers, positions=[args.position],
                           patch_at=args.patch_at, include_residual=args.include_residual)
    gt, mem, home = site_labels(grid, d)
    print(f"== grid: {len(grid)} sites | circuit: {sum(v == 'circuit' for v in gt.values())} "
          f"| shared: {sum(v == 'shared' for v in mem.values())} | tasks: {d['tasks']} ==")

    # per-task clean clouds at every grid site — ONCE per combo, reused across the sweep.
    from interaction_effects import build_task_clouds
    cloud_units = [t.flatten()[0] for t in grid.values()]
    print(f"== fitting per-task clouds ({len(d['tasks'])} tasks x {len(cloud_units)} sites) ==")
    clouds = build_task_clouds(d["pipeline"], d["pool"], cloud_units, batch_size=args.batch_size)

    X = pick_push_unit(grid, mem, args.push_site)
    print(f"== push site X = {X.id} ==")
    _mode = {"0": "diagonal", "within-push": "within-push"}.get(args.step, "full")
    rows = directed_sweep(d, grid, gt, mem, home, clouds, X, args, mode=_mode)
    s = _summary(rows)
    s.update(push_site=X.id, alphas=args.alphas, tasks=d["tasks"], step=args.step)
    _save_summary("directed_map.json" if args.step != "0" else "gate.json", s)
    print(f"DONE. {len(rows)} rows | max|INT| {s['max_abs_INT']:.4f} -> "
          f"{'directed_map.json' if args.step != '0' else 'gate.json'}")


if __name__ == "__main__":
    main()
