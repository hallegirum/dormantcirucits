"""check_endpoint_density_v2.py — endpoint density for FPs AND TPs with PER-TASK distances.

Single-pass script: loads model and task clouds once, then processes both
off-circuit (FP/TN) and in-circuit (TP) components from the same safetensors.

New v2 fields (both FP and TP records):
  - dist_per_task: {task_id: normalised_dist} for EVERY task with a fit at that site
  - dist_base:     normalised distance to the BASE task's cloud
  - nearest_task:  which task's cloud the endpoint is closest to
  - K:             most-aligned non-base task (argmax of along_by_task excl base)
  - along_by_task: {task_id: alignment} per task

FP-only fields:
  - base_ref:      distance of base cloud to K's fit (reference baseline)

Usage:
    python check_endpoint_density_v2.py --results-root poly_results_density/poly_results_new/results \
        --combo cases_0+1 --seed 0 --case poly --n 150 --device cuda \
        --dump-fp endpoint_density_v2.jsonl \
        --dump-tp endpoint_density_tp_v2.jsonl
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from safetensors.torch import load_file

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE.parent / "interp_bench"))
from interaction_effects import build_task_clouds, _units_of


def parse_reps_dir(name):
    m = re.match(r"int-([^_]+)__(.+)-a([0-9.]+)__(.+)$", name)
    if not m:
        return None
    strat, direction, alpha, push = m.group(1), m.group(2), float(m.group(3)), m.group(4)
    parts = direction.split("-")
    return {"strategy": strat, "direction": direction, "alpha": alpha, "push_site": push,
            "base_task": int(parts[0]) if parts[0].isdigit() else None,
            "target_task": (int(parts[1]) if (len(parts) == 2 and parts[1].isdigit()) else None)}


def load_scan_rows(results_dir):
    idx = {}
    for p in glob.glob(os.path.join(results_dir, "interaction", "**", "int.json"), recursive=True):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if "strategy" in r:
            idx[(r["strategy"], r["push_site"], r["direction"], float(r["alpha"]), r["site"])] = r
    return idx


def whiten_fit(C, d=8):
    C = np.asarray(C, dtype=np.float64)
    mu = C.mean(0)
    _, S, Vt = np.linalg.svd(C - mu, full_matrices=False)
    d = int(min(d, (S > 1e-6).sum()))
    V = Vt[:d]
    std = S[:d] / np.sqrt(max(len(C) - 1, 1))
    return mu, V, std


def wdist(X, fit):
    mu, V, std = fit
    Z = (np.asarray(X, dtype=np.float64) - mu) @ V.T / (std + 1e-6)
    return np.linalg.norm(Z, axis=1)


def anie(r):
    v = r.get("abs_NIE")
    if v is None:
        v = r.get("NIE")
    return abs(float(v)) if isinstance(v, (int, float)) and np.isfinite(v) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--reps-dir", default=None,
                    help="path to the intervened_reps dir directly")
    ap.add_argument("--dump-fp", default=None,
                    help="append FP/TN records (off-circuit) to this JSONL file")
    ap.add_argument("--dump-tp", default=None,
                    help="append TP records (in-circuit) to this JSONL file")
    ap.add_argument("--combo", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--case", default="poly")
    ap.add_argument("--n", type=int, default=200, help="MUST match the original run")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--position", default="last_token")
    ap.add_argument("--patch-at", default="internal")
    ap.add_argument("--pca-dim", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    if not args.dump_fp and not args.dump_tp:
        raise SystemExit("at least one of --dump-fp or --dump-tp is required")

    # ── Locate reps directory ──
    if args.reps_dir:
        reps_root = os.path.normpath(args.reps_dir)
        if not os.path.isdir(reps_root):
            raise SystemExit(f"--reps-dir not found: {reps_root}")
    else:
        hits = glob.glob(os.path.join(args.results_root, "**", f"poly_{args.combo}",
                                      f"int_seed{args.seed}", "intervened_reps"), recursive=True)
        if not hits:
            raise SystemExit(f"no intervened_reps for poly_{args.combo}/int_seed{args.seed} "
                             f"under {args.results_root}")
        reps_root = hits[0]
    results_dir = os.path.dirname(reps_root)

    # ── Load model and build task clouds (expensive, done ONCE) ──
    from run_dormancy_pipeline import PrepareData
    from path_finder import build_full_grid
    from activation_manifold import set_seed
    set_seed(args.seed)
    d = PrepareData(args.case, n=args.n, seed=args.seed, device=args.device,
                    position=args.position, poly_comb=args.combo).generate()
    grid = build_full_grid(d["pipeline"], d["task"], list(range(d["n_layers"])),
                           positions=[args.position], patch_at=args.patch_at, include_residual=False)
    unit_lists = [_units_of(t) for t in grid.values()]
    all_units = [u for ul in unit_lists for u in ul]
    clouds = build_task_clouds(d["pipeline"], d["pool"], all_units)
    rows = load_scan_rows(results_dir)
    n_repsdirs = len(glob.glob(os.path.join(reps_root, "int-*")))
    print(f"[diag] reps_root={reps_root}")
    print(f"[diag] results_dir={results_dir}  int.json rows loaded={len(rows)}  reps int-* dirs={n_repsdirs}")
    if not rows:
        alt = glob.glob(os.path.join(results_dir, "**", "int.json"), recursive=True)
        print(f"[diag] load_scan_rows empty; fallback int.json count = {len(alt)}")
        for p in alt:
            try:
                r = json.load(open(p))
            except Exception:
                continue
            if "strategy" in r:
                rows[(r["strategy"], r["push_site"], r["direction"], float(r["alpha"]), r["site"])] = r
        print(f"[diag] fallback rows now = {len(rows)}")

    # ── Whiten fits ──
    fits = defaultdict(dict); insc = defaultdict(dict)
    for K, sites in clouds.items():
        for sid, C in sites.items():
            Cn = np.asarray(C.detach().cpu() if hasattr(C, "detach") else C, dtype=np.float64)
            if len(Cn) < args.pca_dim + 2:
                continue
            fit = whiten_fit(Cn, args.pca_dim)
            fits[K][sid] = fit
            insc[K][sid] = float(np.median(wdist(Cn, fit)) + 1e-9)

    # ── Process all components ──
    fp_agg = defaultdict(lambda: {"epotic": [], "borrowed": []})
    base_ref_agg = defaultdict(list)
    diag = {"dirs": 0, "no_thr": 0, "off_rows": 0, "fp": 0, "tp": 0}
    fp_records = []
    tp_records = []

    for dpath in sorted(glob.glob(os.path.join(reps_root, "int-*"))):
        info = parse_reps_dir(os.path.basename(dpath))
        if info is None or info["push_site"] == "self" or info["base_task"] is None:
            continue
        diag["dirs"] += 1
        strat, push, direc, alpha = info["strategy"], info["push_site"], info["direction"], info["alpha"]
        base = info["base_task"]
        if base not in clouds:
            continue
        kind = "swap" if push == "input" else "push"
        within = (info["base_task"] == info["target_task"])
        regime = ("swap" if kind == "swap" else "STR_push") + ("_within" if within else "_cross")
        sf = glob.glob(os.path.join(dpath, "*_intervened_reps.safetensors"))
        if not sf:
            continue
        hp = load_file(sf[0])

        # FP threshold: median abs_NIE of in-circuit components for this intervention
        incirc_vals = []
        for k in rows:
            if k[:4] == (strat, push, direc, alpha) and rows[k].get("in_base_circuit") is True:
                a = anie(rows[k])
                if a is not None:
                    incirc_vals.append(a)
        thr = float(np.quantile(incirc_vals, 0.5)) if incirc_vals else None

        for sid, Hp in hp.items():
            r = rows.get((strat, push, direc, alpha, sid))
            if r is None:
                continue

            Hpn = np.asarray(Hp.detach().cpu() if hasattr(Hp, "detach") else Hp, dtype=np.float64)

            # Per-task distances (shared computation)
            dist_per_task = {}
            for K2 in fits:
                if sid in fits[K2]:
                    dist_per_task[int(K2)] = float(
                        np.median(wdist(Hpn, fits[K2][sid])) / insc[K2][sid]
                    )

            if dist_per_task:
                nearest_task = min(dist_per_task, key=dist_per_task.get)
                dmin = dist_per_task[nearest_task]
            else:
                nearest_task = None
                dmin = None

            dist_base = dist_per_task.get(int(base))

            # K task and subset (shared logic)
            abt = {k: v for k, v in (r.get("along_by_task") or {}).items()
                   if k != str(base) and isinstance(v, (int, float))}
            K_task = int(max(abt, key=abt.get)) if abt else None
            home = {int(t) for t in (r.get("home_tasks") or [])}

            along_by_task = {k: round(v, 4) for k, v in (r.get("along_by_task") or {}).items()
                            if isinstance(v, (int, float))}

            in_circuit = r.get("in_base_circuit")

            # ── OFF-CIRCUIT (FP/TN) ──
            if in_circuit is False and args.dump_fp:
                a = anie(r)
                if a is None:
                    continue
                diag["off_rows"] += 1
                is_fp = thr is not None and a >= thr
                if is_fp:
                    diag["fp"] += 1

                g = r.get("gap_frac"); g = float(g) if isinstance(g, (int, float)) else None
                subset = dist_K = base_ref_v = None
                if is_fp and K_task is not None and K_task in fits and sid in fits[K_task]:
                    subset = "borrowed" if K_task in home else "epotic"
                    dist_K = dist_per_task.get(K_task)
                    if sid in clouds[base]:
                        Cb = np.asarray(clouds[base][sid].detach().cpu() if hasattr(clouds[base][sid], "detach")
                                        else clouds[base][sid], dtype=np.float64)
                        base_ref_v = float(np.median(wdist(Cb, fits[K_task][sid]))) / insc[K_task][sid]

                if subset is not None:
                    fp_agg[regime][subset].append(dist_K)
                    if base_ref_v is not None:
                        base_ref_agg[regime].append(base_ref_v)

                fp_records.append({
                    "combo": args.combo, "seed": args.seed, "regime": regime,
                    "is_fp": int(is_fp),
                    "off_dist": dmin, "gap_frac": g, "subset": subset,
                    "dist_K": dist_K, "base_ref": base_ref_v,
                    "dist_base": dist_base,
                    "nearest_task": nearest_task,
                    "K": K_task if is_fp else None,
                    "dist_per_task": {str(k): round(v, 4) for k, v in dist_per_task.items()} if is_fp else None,
                    "along_by_task": along_by_task if is_fp else None,
                })

            # ── IN-CIRCUIT (TP) ──
            elif in_circuit is True and args.dump_tp:
                diag["tp"] += 1
                subset = None
                dist_K = None
                if K_task is not None:
                    subset = "borrowed" if K_task in home else "epotic"
                    dist_K = dist_per_task.get(K_task)

                tp_records.append({
                    "combo": args.combo, "seed": args.seed, "regime": regime,
                    "is_fp": 0, "in_circuit": 1, "site": sid,
                    "off_dist": dmin, "dist_K": dist_K, "subset": subset,
                    "dist_base": dist_base,
                    "nearest_task": nearest_task,
                    "K": K_task,
                    "dist_per_task": {str(k): round(v, 4) for k, v in dist_per_task.items()},
                    "along_by_task": along_by_task,
                    "gap_frac": float(r["gap_frac"]) if isinstance(r.get("gap_frac"), (int, float)) else None,
                })

    # ── Summary ──
    print(f"[diag] processed int-* dirs={diag['dirs']}  dirs_with_no_threshold={diag['no_thr']}  "
          f"off-circuit rows={diag['off_rows']}  FPs={diag['fp']}  TPs={diag['tp']}")

    if fp_records:
        def med(x): return float(np.median(x)) if x else float("nan")
        print(f"\n== FP endpoint density to aligned task K, normalised ==")
        print(f"   combo {args.combo} seed {args.seed}")
        print(f"   {'regime':16s} {'epotic':>18s} {'borrowed':>18s} {'base-reps->K (ref)':>20s}")
        for regime in sorted(fp_agg):
            e, b = fp_agg[regime]["epotic"], fp_agg[regime]["borrowed"]
            print(f"   {regime:16s} {med(e):8.2f} (n={len(e):4d}) {med(b):8.2f} (n={len(b):4d}) "
                  f"{med(base_ref_agg[regime]):12.2f}")

    if args.dump_fp and fp_records:
        with open(args.dump_fp, "a") as fh:
            for rec in fp_records:
                fh.write(json.dumps(rec) + "\n")
        print(f"[dump] appended {len(fp_records)} FP records -> {args.dump_fp}")

    if args.dump_tp and tp_records:
        with open(args.dump_tp, "a") as fh:
            for rec in tp_records:
                fh.write(json.dumps(rec) + "\n")
        print(f"[dump] appended {len(tp_records)} TP records -> {args.dump_tp}")

    if not fp_records and not tp_records:
        print("[warn] no records produced")


if __name__ == "__main__":
    main()
