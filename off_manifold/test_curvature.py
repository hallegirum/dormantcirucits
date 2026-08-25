"""
test_curvature.py  —  TEST 1: alpha-curvature (Type-R vs Type-G, no model needed)
=================================================================================
Fits NIE(alpha) per (component x intervention) over the alpha sweep and asks whether
the effect is LINEAR from the origin (Type-R: linear dormant read) or a HOCKEY-STICK
onset (Type-G: gated dormancy). Pure post-hoc on the collected int.json rows; reuses
collect_interaction.load_rows so the schema stays in one place.

Discriminator (robust with the {0.3, 1, 3} sweep):
    per-unit effect  e(a) = |NIE(a)| / a
    r_small = e(a_min) / e(a_max)          in [0, ~1]
      r_small ~ 1  -> effect is already there at small alpha, scales linearly -> TYPE-R
      r_small ~ 0  -> effect only appears at large alpha (onset)             -> TYPE-G
    plus a through-origin quadratic fit  NIE ~ c1*a + c2*a^2 :
      c1 ~ 0, c2 > 0 (same sign as the effect) -> onset / convex             -> TYPE-G
      c1 != 0, c2 ~ 0                            -> linear                    -> TYPE-R

Strategy axis: every table is broken out by strategy (STR swap / STR push / Gaussian
push) and, for swap, by within- vs cross-task, plus FP (off-circuit) vs in-circuit —
so you can read off which interventions carry which dormancy type.

    python test_curvature.py --results-root poly_results_new
    python test_curvature.py --results-root poly_results_new --out curvature_out
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np

from collect_interaction import load_rows, _num   # same loader/schema as the path model


# --------------------------------------------------------------------------- #
#  grouping: one curve = one component under one intervention, varying alpha
# --------------------------------------------------------------------------- #
def _series_key(r):
    """Everything that identifies a single NIE(alpha) curve EXCEPT alpha."""
    return (r["combo"], r["seed"], r["site"], r["strategy"], r["push_site"], r["direction"])


def _strategy_bucket(r):
    """Coarse strategy label for the differentiation axis."""
    strat, push = r["strategy"], r["push_site"]
    if push == "input":
        within = r["base_task"] == r["target_task"]
        return f"STR_swap_{'within' if within else 'cross'}"
    if push == "self":
        return f"{strat}_self"
    return f"{strat}_push"          # STR_push / Gaussian_push (component push site)


def build_curves(rows):
    """{series_key: {'meta':..., 'pts':[(alpha, NIE), ...]}} over non-degenerate rows."""
    curves = defaultdict(lambda: {"meta": None, "pts": []})
    for r in rows:
        if not _num(r, "NIE") or r.get("alpha") is None:
            continue
        k = _series_key(r)
        curves[k]["pts"].append((float(r["alpha"]), float(r["NIE"])))
        if curves[k]["meta"] is None:
            curves[k]["meta"] = {
                "strategy_bucket": _strategy_bucket(r),
                "in_base_circuit": r.get("in_base_circuit"),
                "ctype": r.get("ctype"), "layer": r.get("layer"),
                "combo": r["combo"], "seed": r["seed"], "site": r["site"],
                "direction": r["direction"], "push_site": r["push_site"],
            }
    return curves


# --------------------------------------------------------------------------- #
#  per-curve curvature metrics
# --------------------------------------------------------------------------- #
def curve_metrics(pts):
    """pts: list of (alpha, NIE). Returns dict of curvature features or None."""
    pts = sorted(set(pts))
    if len(pts) < 3:
        return None
    a = np.array([p[0] for p in pts], float)
    y = np.array([p[1] for p in pts], float)
    if not np.all(np.isfinite(y)) or a.min() <= 0:
        return None

    # through-origin quadratic fit  y ~ c1*a + c2*a^2  (NIE(0)=0 by construction)
    X = np.column_stack([a, a ** 2])
    (c1, c2), *_ = np.linalg.lstsq(X, y, rcond=None)

    # per-unit effect and the small-vs-large ratio (sign-robust: use magnitudes)
    e = np.abs(y) / a
    e_min, e_max = e[a.argmin()], e[a.argmax()]
    r_small = float(e_min / e_max) if e_max > 1e-9 else float("nan")

    # normalized curvature: c2 relative to the initial slope c1 (accelerating if > 0)
    kappa = float(c2 / c1) if abs(c1) > 1e-9 else float("inf") if abs(c2) > 1e-9 else 0.0

    # classify
    if not math.isfinite(r_small):
        typ = "degenerate"
    elif r_small >= 0.6:
        typ = "R"          # effect present at small alpha, ~linear
    elif r_small <= 0.30:
        typ = "G"          # onset: effect only at large alpha
    else:
        typ = "mixed"

    return {"c1": float(c1), "c2": float(c2), "kappa": kappa, "r_small": r_small,
            "type": typ, "n_alpha": len(pts),
            "NIE_min_a": float(y[a.argmin()]), "NIE_max_a": float(y[a.argmax()]),
            "a_min": float(a.min()), "a_max": float(a.max())}


# --------------------------------------------------------------------------- #
#  aggregation over the strategy axis
# --------------------------------------------------------------------------- #
def aggregate(curve_rows, split_circuit=True):
    """curve_rows: list of dicts (meta + metrics). Group by strategy_bucket [x circuit]."""
    groups = defaultdict(list)
    for cr in curve_rows:
        circ = cr["in_base_circuit"]
        circ_lbl = ("in_circuit" if circ is True else "off_circuit" if circ is False else "unknown")
        key = (cr["strategy_bucket"], circ_lbl) if split_circuit else (cr["strategy_bucket"], "all")
        groups[key].append(cr)

    out = {}
    for (strat, circ), items in sorted(groups.items()):
        types = [it["type"] for it in items]
        n = len(items)
        finite_k = [it["kappa"] for it in items if math.isfinite(it["kappa"])]
        out[f"{strat} | {circ}"] = {
            "n": n,
            "frac_typeR": round(sum(t == "R" for t in types) / n, 3),
            "frac_typeG": round(sum(t == "G" for t in types) / n, 3),
            "frac_mixed": round(sum(t == "mixed" for t in types) / n, 3),
            "median_r_small": round(float(np.median([it["r_small"] for it in items
                                                     if math.isfinite(it["r_small"])] or [float("nan")])), 3),
            "median_kappa": round(float(np.median(finite_k)) if finite_k else float("nan"), 3),
        }
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--combos", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--out", default="curvature_analysis")
    args = ap.parse_args()

    rows = load_rows(args.results_root, args.combos, args.seeds)
    print(f"loaded {len(rows)} rows | combos {sorted({r['combo'] for r in rows})}")

    curves = build_curves(rows)
    curve_rows = []
    for k, v in curves.items():
        m = curve_metrics(v["pts"])
        if m is None:
            continue
        curve_rows.append({**v["meta"], **m})
    print(f"fitted {len(curve_rows)} alpha-curves (>=3 alphas each)")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "curves.csv"), "w", newline="") as f:
        cols = ["combo", "seed", "site", "ctype", "layer", "strategy_bucket", "direction",
                "push_site", "in_base_circuit", "type", "r_small", "kappa", "c1", "c2",
                "n_alpha", "a_min", "a_max", "NIE_min_a", "NIE_max_a"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(curve_rows)

    agg = aggregate(curve_rows, split_circuit=True)
    json.dump(agg, open(os.path.join(args.out, "curvature_by_strategy.json"), "w"), indent=2)

    print("\n== TYPE-R vs TYPE-G by strategy (r_small~1 => Type-R linear; ~0 => Type-G onset) ==")
    print(f"  {'strategy | circuit':<34} {'n':>5} {'%R':>6} {'%G':>6} {'%mix':>6} "
          f"{'med r_small':>12} {'med kappa':>10}")
    for name, d in agg.items():
        print(f"  {name:<34} {d['n']:>5} {d['frac_typeR']*100:>5.0f}% {d['frac_typeG']*100:>5.0f}% "
              f"{d['frac_mixed']*100:>5.0f}% {d['median_r_small']:>12} {d['median_kappa']:>10}")
    print(f"\nwrote curves.csv + curvature_by_strategy.json under {args.out}/")


if __name__ == "__main__":
    main()
