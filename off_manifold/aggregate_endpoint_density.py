"""aggregate_endpoint_density.py — pool the per-run JSONL dumps from check_endpoint_density.py
across all combos/seeds into the two summary tables (run locally after syncing the .jsonl).

    python aggregate_endpoint_density.py endpoint_density.jsonl
"""
import json, sys, math
from collections import defaultdict
import numpy as np

REGIMES = ["swap_cross", "swap_within", "STR_push_cross", "STR_push_within"]


def med(x):
    return float(np.median(x)) if len(x) else float("nan")


def boot_med_ci(vals, combos, B=2000, seed=0):
    """cluster bootstrap of the median over combos -> (est, lo, hi)."""
    if not vals:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for v, c in zip(vals, combos):
        by[c].append(v)
    cs = list(by)
    est = med(vals)
    draws = []
    for _ in range(B):
        samp = [v for c in rng.choice(cs, len(cs)) for v in by[c]]
        if samp:
            draws.append(np.median(samp))
    lo, hi = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))) if draws else (float("nan"),) * 2
    return est, lo, hi


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "endpoint_density.jsonl"
    recs = [json.loads(l) for l in open(path) if l.strip()]
    print(f"loaded {len(recs)} records | combos {sorted({r['combo'] for r in recs})} "
          f"| seeds {sorted({r['seed'] for r in recs})}")

    # -------- Table 1: aligned-task (K) endpoint density, epotic vs borrowed (FPs only) --------
    print("\n== endpoint density to the ALIGNED task K, normalised (1=in K's data, >>1=no-man's land) ==")
    print(f"   {'regime':16s} {'epotic (median [CI])':>28s} {'borrowed (median [CI])':>28s} {'base->K ref':>12s}")
    for rg in REGIMES:
        ep = [(r["dist_K"], r["combo"]) for r in recs if r["regime"] == rg and r.get("subset") == "epotic" and r.get("dist_K") is not None]
        bo = [(r["dist_K"], r["combo"]) for r in recs if r["regime"] == rg and r.get("subset") == "borrowed" and r.get("dist_K") is not None]
        br = [r["base_ref"] for r in recs if r["regime"] == rg and r.get("base_ref") is not None]
        e_est, e_lo, e_hi = boot_med_ci([v for v, _ in ep], [c for _, c in ep])
        b_est, b_lo, b_hi = boot_med_ci([v for v, _ in bo], [c for _, c in bo])
        print(f"   {rg:16s} {f'{e_est:.2f} [{e_lo:.2f},{e_hi:.2f}] n={len(ep)}':>28s} "
              f"{f'{b_est:.2f} [{b_lo:.2f},{b_hi:.2f}] n={len(bo)}':>28s} {med(br):12.2f}")

    # -------- Table 2: off-distribution (min dist to ANY task) + link to gap_frac --------
    print("\n== endpoint OFF-DISTRIBUTION (min dist to ANY task's data) — does gate-crossing == extrapolation? ==")
    print(f"   {'regime':16s} {'FP off-dist [CI]':>26s} {'harmless':>12s} {'corr(off-dist, gap)':>20s}")
    for rg in REGIMES:
        fp = [(r["off_dist"], r["combo"]) for r in recs if r["regime"] == rg and r["is_fp"] == 1 and r.get("off_dist") is not None]
        hm = [r["off_dist"] for r in recs if r["regime"] == rg and r["is_fp"] == 0 and r.get("off_dist") is not None]
        f_est, f_lo, f_hi = boot_med_ci([v for v, _ in fp], [c for _, c in fp])
        pairs = [(r["off_dist"], r["gap_frac"]) for r in recs
                 if r["regime"] == rg and r["is_fp"] == 1 and r.get("off_dist") is not None and r.get("gap_frac") is not None]
        c = float("nan")
        if len(pairs) > 5:
            a = np.array([p[0] for p in pairs]); g = np.array([p[1] for p in pairs])
            if a.std() > 1e-9 and g.std() > 1e-9:
                c = float(np.corrcoef(a, g)[0, 1])
        print(f"   {rg:16s} {f'{f_est:.2f} [{f_lo:.2f},{f_hi:.2f}] n={len(fp)}':>26s} {med(hm):12.2f} {c:20.2f}")


if __name__ == "__main__":
    main()
