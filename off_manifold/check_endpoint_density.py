"""check_endpoint_density.py — is the INTERVENED endpoint in a task's DATA, or slid past it?
==================================================================================
Resolves the "no-man's land" question that orth_by_task cannot: orth_by_task decomposes the
DIVERGENCE against a task's tangent SUBSPACE (a linear-angle measure). It says nothing about
whether the intervened rep h'_c LANDS where task K's data actually lies, or slides far ALONG
K's subspace beyond the data. This measures the endpoint density directly.

For every off-circuit FALSE POSITIVE, let K = the non-base task it most aligns with (its home
task if borrowed, a non-member task if epotic). We whiten each task's clean-rep cloud (PCA +
per-PC std) and report the Mahalanobis-style distance of h'_c to K's cloud, normalised by K's
own in-distribution scale:

    dist_norm(h'_c, K) = ||whiten_K(h'_c - mu_K)|| / median ||whiten_K(clean_K - mu_K)||

    ~1  -> endpoint sits in K's data density        (impersonation of a real region)
    >>1 -> endpoint slid far along K's subspace      (no-man's land: 'along but far off')

Prints, per regime, the median normalised endpoint distance for EPOTIC vs BORROWED FPs, plus
two references: K's own points (=1 by construction) and base clean reps scored against K (how
far an ordinary non-K rep sits). One (combo, seed) per call, like run_ripple.

    python check_endpoint_density.py --results-root poly_results_final2 --combo cases_0+2 \
        --seed 0 --case poly --n 150 --device cuda
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
    V = Vt[:d]                                  # (d, D)
    std = S[:d] / np.sqrt(max(len(C) - 1, 1))   # per-PC std
    return mu, V, std


def wdist(X, fit):                              # X: (n, D) -> (n,) whitened norms
    mu, V, std = fit
    Z = (np.asarray(X, dtype=np.float64) - mu) @ V.T / (std + 1e-6)
    return np.linalg.norm(Z, axis=1)


def anie(r):                                    # raw int.json has NIE, not abs_NIE (that is derived in collect)
    v = r.get("abs_NIE")
    if v is None:
        v = r.get("NIE")
    return abs(float(v)) if isinstance(v, (int, float)) and np.isfinite(v) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--reps-dir", default=None,
                    help="path to the intervened_reps dir directly (bypasses the combo/seed glob)")
    ap.add_argument("--dump", default=None,
                    help="append one JSON line per off-circuit component to this file (for post-hoc aggregation)")
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

    if args.reps_dir:
        reps_root = os.path.normpath(args.reps_dir)
        if not os.path.isdir(reps_root):
            raise SystemExit(f"--reps-dir not found: {reps_root}")
    else:
        hits = glob.glob(os.path.join(args.results_root, "**", f"poly_{args.combo}",
                                      f"int_seed{args.seed}", "intervened_reps"), recursive=True)
        if not hits:
            raise SystemExit(f"no intervened_reps for poly_{args.combo}/int_seed{args.seed} "
                             f"under {args.results_root} (pass --reps-dir <path> to point at it directly)")
        reps_root = hits[0]
    results_dir = os.path.dirname(reps_root)                # the int_seed dir (holds interaction/ + intervened_reps/)

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
    clouds = build_task_clouds(d["pipeline"], d["pool"], all_units)      # {task: {site: (M,D)}}
    rows = load_scan_rows(results_dir)
    n_repsdirs = len(glob.glob(os.path.join(reps_root, "int-*")))
    print(f"[diag] reps_root={reps_root}")
    print(f"[diag] results_dir={results_dir}  int.json rows loaded={len(rows)}  reps int-* dirs={n_repsdirs}")
    if not rows:                                             # interaction/ not where load_scan_rows looked
        alt = glob.glob(os.path.join(results_dir, "**", "int.json"), recursive=True)
        print(f"[diag] load_scan_rows empty; int.json anywhere under results_dir (recursive) = {len(alt)}"
              f"{' e.g. '+alt[0] if alt else ''}")
        for p in alt:                                        # same combo/seed scope, just a different nesting
            try:
                r = json.load(open(p))
            except Exception:
                continue
            if "strategy" in r:
                rows[(r["strategy"], r["push_site"], r["direction"], float(r["alpha"]), r["site"])] = r
        print(f"[diag] fallback rows now = {len(rows)}")

    # per-task, per-site whitening fit from the CLEAN clouds, + in-distribution scale
    fits = defaultdict(dict); insc = defaultdict(dict)
    for K, sites in clouds.items():
        for sid, C in sites.items():
            Cn = np.asarray(C.detach().cpu() if hasattr(C, "detach") else C, dtype=np.float64)
            if len(Cn) < args.pca_dim + 2:
                continue
            fit = whiten_fit(Cn, args.pca_dim)
            fits[K][sid] = fit
            insc[K][sid] = float(np.median(wdist(Cn, fit)) + 1e-9)   # K's own in-distribution scale

    tasks = list(clouds.keys())
    agg = defaultdict(lambda: {"epotic": [], "borrowed": []})   # regime -> normalised endpoint dists (to aligned K)
    base_ref = defaultdict(list)                                # regime -> base-reps-scored-against-K
    # extrapolation test: endpoint's min distance to ANY task's data + link to gap_frac
    extrap = defaultdict(lambda: {"fp_d": [], "harm_d": [], "pairs": []})
    diag = {"dirs": 0, "no_thr": 0, "off_rows": 0, "fp": 0}
    records = []

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
        hp = load_file(sf[0])                                   # {site: (N, D)} = the ENDPOINTS h'_c

        # in-circuit |NIE| threshold for this scan (per combo already fixed -> use these rows)
        incirc = []
        for k in rows:
            if k[:4] == (strat, push, direc, alpha) and rows[k].get("in_base_circuit") is True:
                a = anie(rows[k])
                if a is not None:
                    incirc.append(a)
        if not incirc:
            diag["no_thr"] += 1
            continue
        thr = float(np.quantile(incirc, 0.5))

        for sid, Hp in hp.items():
            r = rows.get((strat, push, direc, alpha, sid))
            a = anie(r) if r is not None else None
            if r is None or r.get("in_base_circuit") is not False or a is None:
                continue
            diag["off_rows"] += 1
            is_fp = a >= thr
            if is_fp:
                diag["fp"] += 1
            Hpn = np.asarray(Hp.detach().cpu() if hasattr(Hp, "detach") else Hp, dtype=np.float64)
            # off-distribution: min normalised endpoint distance to ANY task's data at this site
            dmins = [float(np.median(wdist(Hpn, fits[K2][sid]))) / insc[K2][sid]
                     for K2 in fits if sid in fits[K2]]
            dmin = min(dmins) if dmins else None
            g = r.get("gap_frac"); g = float(g) if isinstance(g, (int, float)) else None
            subset = dist_K = base_ref_v = None
            if is_fp:                                              # aligned-task (K) density, FPs only
                abt = {k: v for k, v in (r.get("along_by_task") or {}).items()
                       if k != str(base) and isinstance(v, (int, float))}
                if abt:
                    K = int(max(abt, key=abt.get))                 # most-aligned non-base task
                    if K in fits and sid in fits[K]:
                        home = {int(t) for t in (r.get("home_tasks") or [])}
                        subset = "borrowed" if K in home else "epotic"
                        dist_K = float(np.median(wdist(Hpn, fits[K][sid]))) / insc[K][sid]
                        if sid in clouds[base]:
                            Cb = np.asarray(clouds[base][sid].detach().cpu() if hasattr(clouds[base][sid], "detach")
                                            else clouds[base][sid], dtype=np.float64)
                            base_ref_v = float(np.median(wdist(Cb, fits[K][sid]))) / insc[K][sid]
            # in-memory aggregates for the printed per-run tables
            if dmin is not None:
                extrap[regime]["fp_d" if is_fp else "harm_d"].append(dmin)
                if is_fp and g is not None:
                    extrap[regime]["pairs"].append((dmin, g))
            if subset is not None:
                agg[regime][subset].append(dist_K)
                if base_ref_v is not None:
                    base_ref[regime].append(base_ref_v)
            # one flat record per off-circuit component, for post-hoc aggregation across combos
            records.append({"combo": args.combo, "seed": args.seed, "regime": regime, "is_fp": int(is_fp),
                            "off_dist": dmin, "gap_frac": g, "subset": subset,
                            "dist_K": dist_K, "base_ref": base_ref_v})

    print(f"[diag] processed int-* dirs={diag['dirs']}  dirs_with_no_threshold={diag['no_thr']}  "
          f"off-circuit rows matched={diag['off_rows']}  false positives={diag['fp']}")
    def med(x): return float(np.median(x)) if x else float("nan")
    print(f"\n== endpoint density to the ALIGNED task K, normalised (1=in K's data, >>1=no-man's land) ==")
    print(f"   combo {args.combo} seed {args.seed}   (K's own points =1.0 by construction)")
    print(f"   {'regime':16s} {'epotic':>18s} {'borrowed':>18s} {'base-reps->K (ref)':>20s}")
    for regime in sorted(agg):
        e, b = agg[regime]["epotic"], agg[regime]["borrowed"]
        print(f"   {regime:16s} {med(e):8.2f} (n={len(e):4d}) {med(b):8.2f} (n={len(b):4d}) "
              f"{med(base_ref[regime]):12.2f}")

    print(f"\n== endpoint OFF-DISTRIBUTION (min dist to ANY task's data) — does gate-crossing == extrapolation? ==")
    print(f"   {'regime':16s} {'FP off-dist':>12s} {'harmless':>12s} {'corr(off-dist, gap_frac)':>26s}")
    for regime in sorted(extrap):
        fp_d, harm_d, pairs = extrap[regime]["fp_d"], extrap[regime]["harm_d"], extrap[regime]["pairs"]
        c = float("nan")
        if len(pairs) > 5:
            a = np.array([p[0] for p in pairs]); g = np.array([p[1] for p in pairs])
            m = np.isfinite(a) & np.isfinite(g)
            if m.sum() > 5 and a[m].std() > 1e-9 and g[m].std() > 1e-9:
                c = float(np.corrcoef(a[m], g[m])[0, 1])
        print(f"   {regime:16s} {med(fp_d):12.2f} {med(harm_d):12.2f} {c:26.2f}")

    if args.dump:
        with open(args.dump, "a") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        print(f"[dump] appended {len(records)} records -> {args.dump}")


if __name__ == "__main__":
    main()
