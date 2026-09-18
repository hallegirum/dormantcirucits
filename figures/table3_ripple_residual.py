"""Table 3 — Ripple AUC on the residual of curvature models.

For each curvature metric (local, downstream, excess):
  1. Fit base + curvature model
  2. Show AUC gain when adding each ripple metric (and ripple block)

Demonstrates that ripple captures information orthogonal to curvature.
Both FP-vs-TN and FP-vs-TP (moderate band) decompositions shown.

Usage:
    python figures/table3_ripple_residual.py [--csv PATH] [--out PATH]
"""
import argparse, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

RIPPLE_FEATS = ["ripple_in_orth", "ripple_off_orth", "ripple_frac", "ripple_xint_in_orth"]
RIPPLE_DISPLAY = {
    "ripple_in_orth": "ripple (in-task orth)",
    "ripple_off_orth": "ripple (off-task orth)",
    "ripple_frac": "ripple frac",
    "ripple_xint_in_orth": "ripple (xint in-task)",
    "block": "ripple block",
}
CURV_DISPLAY = {
    "local_gap_frac": "local curv.",
    "gap_frac": "downstream",
    "excess_local_gap": "excess curv.",
}


def _strat_bucket(row):
    push = str(row["push_site"]) if pd.notna(row.get("push_site")) else ""
    strategy = str(row["strategy"]) if pd.notna(row.get("strategy")) else ""
    base = row.get("base_task")
    target = row.get("target_task")
    within = (pd.notna(target) and base == target)
    if push == "input":
        return f"STR_swap_{'within' if within else 'cross'}"
    if push == "self":
        return f"{strategy}_self"
    if strategy == "STR":
        return f"STR_push_{'within' if within else 'cross'}"
    return f"{strategy}_push"


def build_fptn_population(df, bucket):
    """Off-circuit: FP (above threshold) vs TN (below)."""
    sub = df[df["bucket"] == bucket].copy()
    in_circ = sub[sub["in_base_circuit"] == True]
    off_circ = sub[sub["in_base_circuit"] == False].copy()
    thresholds = in_circ.groupby("combo")["abs_NIE"].median().to_dict()
    def is_fp(row):
        thr = thresholds.get(row["combo"])
        return thr is not None and pd.notna(row["abs_NIE"]) and row["abs_NIE"] >= thr
    off_circ["label"] = off_circ.apply(is_fp, axis=1).astype(float)
    return off_circ


def build_fptp_moderate(df, bucket):
    """Moderate-|NIE| band: FP vs TP."""
    sub = df[df["bucket"] == bucket].copy()
    in_circ = sub[sub["in_base_circuit"] == True]
    off_circ = sub[sub["in_base_circuit"] == False]
    thresholds = in_circ.groupby("combo")["abs_NIE"].median().to_dict()
    def above(row):
        thr = thresholds.get(row["combo"])
        return thr is not None and pd.notna(row["abs_NIE"]) and row["abs_NIE"] >= thr
    pool = pd.concat([in_circ, off_circ])
    pool = pool[pool.apply(above, axis=1)].copy()
    pool_median = pool["abs_NIE"].median()
    pool = pool[pool["abs_NIE"] <= pool_median].copy()
    pool["label"] = (~pool["in_base_circuit"]).astype(float)
    return pool


def zscore_within_combo(df, feats):
    out = df.copy()
    for f in feats:
        out[f"z_{f}"] = out.groupby("combo")[f].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-12))
    return out


def fit_auc(df, feats):
    z_feats = [f"z_{f}" for f in feats]
    sub = df.dropna(subset=z_feats + ["label"])
    X = sub[z_feats].values.astype(float)
    y = sub["label"].values
    mask = np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    if len(np.unique(y)) < 2 or len(y) < 20:
        return None
    lr = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0, class_weight="balanced")
    lr.fit(X, y)
    return roc_auc_score(y, lr.predict_proba(X)[:, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="interaction_analysis_ripplefull/rows.csv")
    ap.add_argument("--out", default="figures/table3_ripple_residual.tex")
    ap.add_argument("--txt", default="figures/table3_ripple_residual.txt")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["bucket"] = df.apply(_strat_bucket, axis=1)

    BUCKETS = ["STR_swap_within", "STR_swap_cross",
                "STR_push_within", "STR_push_cross", "Gaussian_push"]
    BUCKET_LABELS = {
        "STR_swap_within": "swap (within)", "STR_swap_cross": "swap (cross)",
        "STR_push_within": "STR push (within)", "STR_push_cross": "STR push (cross)",
        "Gaussian_push": "Gaussian push",
    }
    CURVATURES = ["local_gap_frac", "gap_frac", "excess_local_gap"]

    results = {}

    for bucket in BUCKETS:
        for decomp_name, build_fn in [("FP vs TN", build_fptn_population),
                                       ("FP vs TP", build_fptp_moderate)]:
            pool = build_fn(df, bucket)
            is_cross = "cross" in bucket
            base_feats = ["orth_base", "along_nonbase_max"]
            if is_cross:
                base_feats.append("along_target")

            all_needed = base_feats + CURVATURES + RIPPLE_FEATS
            clean = pool.dropna(subset=all_needed + ["label", "combo"])
            for f in all_needed:
                clean = clean[np.isfinite(clean[f])]
            clean = zscore_within_combo(clean, all_needed)

            n_pos = int(clean["label"].sum())
            n_neg = len(clean) - n_pos
            key = (bucket, decomp_name)
            results[key] = {"n": len(clean), "n_pos": n_pos, "n_neg": n_neg, "rows": {}}

            for curv in CURVATURES:
                curv_feats = base_feats + [curv]
                auc_curv = fit_auc(clean, curv_feats)

                row_data = {"auc_base_curv": auc_curv}

                for rf in RIPPLE_FEATS:
                    auc_with = fit_auc(clean, curv_feats + [rf])
                    gain = (auc_with - auc_curv) if (auc_with and auc_curv) else None
                    row_data[f"auc_{rf}"] = auc_with
                    row_data[f"gain_{rf}"] = gain

                auc_block = fit_auc(clean, curv_feats + RIPPLE_FEATS)
                gain_block = (auc_block - auc_curv) if (auc_block and auc_curv) else None
                row_data["auc_block"] = auc_block
                row_data["gain_block"] = gain_block

                results[key]["rows"][curv] = row_data

            print(f"\n{BUCKET_LABELS[bucket]} — {decomp_name}  "
                  f"(n={len(clean)}, +={n_pos}, -={n_neg})")
            for curv in CURVATURES:
                rd = results[key]["rows"][curv]
                print(f"  base+{CURV_DISPLAY[curv]:12s} AUC={rd['auc_base_curv']:.3f}")
                for rf in RIPPLE_FEATS:
                    g = rd.get(f"gain_{rf}")
                    gs = f"{g:+.3f}" if g is not None else "n/a"
                    print(f"    +{RIPPLE_DISPLAY[rf]:25s}  AUC={rd[f'auc_{rf}']:.3f}  gain={gs}")
                g = rd.get("gain_block")
                gs = f"{g:+.3f}" if g is not None else "n/a"
                print(f"    +{RIPPLE_DISPLAY['block']:25s}  AUC={rd['auc_block']:.3f}  gain={gs}")

    # ── LaTeX ──
    with open(args.out, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Ripple AUC gain over curvature models. "
                "Each cell shows AUC when a ripple metric is added to "
                "base $+$ curvature; gain in parentheses. "
                "Ripple block $=$ all four ripple metrics together. "
                "All strategies with pairwise site interactions (swap, STR push, Gaussian push).}\n")
        f.write("\\label{tab:ripple-residual}\n\\small\n")

        for bucket in BUCKETS:
            f.write(f"\n\\medskip\n\\textbf{{{BUCKET_LABELS[bucket]}}}\n\n")

            for decomp_name in ["FP vs TN", "FP vs TP"]:
                key = (bucket, decomp_name)
                res = results[key]
                label_parts = decomp_name.split(" vs ")

                f.write(f"\n\\smallskip\n\\textit{{{decomp_name}}} "
                        f"($n={res['n']}$, "
                        f"{label_parts[0]}$={res['n_pos']}$, "
                        f"{label_parts[1]}$={res['n_neg']}$)\n\n")

                n_curv = len(CURVATURES)
                col_spec = "l " + " ".join(["c"] * n_curv)
                f.write(f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n")

                header = [""] + [f"$+${CURV_DISPLAY[c]}" for c in CURVATURES]
                f.write(" & ".join(header) + " \\\\\n")
                f.write("\\midrule\n")

                # base+curv AUC row
                parts = ["base $+$ curv."]
                for curv in CURVATURES:
                    rd = res["rows"][curv]
                    parts.append(f"{rd['auc_base_curv']:.3f}")
                f.write(" & ".join(parts) + " \\\\\n\\midrule\n")

                # each ripple metric
                for rf in RIPPLE_FEATS:
                    parts = [RIPPLE_DISPLAY[rf]]
                    for curv in CURVATURES:
                        rd = res["rows"][curv]
                        auc = rd[f"auc_{rf}"]
                        gain = rd[f"gain_{rf}"]
                        if auc is not None and gain is not None:
                            parts.append(f"{auc:.3f} ({gain:+.3f})")
                        else:
                            parts.append("--")
                    f.write(" & ".join(parts) + " \\\\\n")

                # ripple block
                parts = [RIPPLE_DISPLAY["block"]]
                for curv in CURVATURES:
                    rd = res["rows"][curv]
                    auc = rd["auc_block"]
                    gain = rd["gain_block"]
                    if auc is not None and gain is not None:
                        parts.append(f"{auc:.3f} ({gain:+.3f})")
                    else:
                        parts.append("--")
                f.write(" & ".join(parts) + " \\\\\n")
                f.write("\\bottomrule\n\\end{tabular}\n")

        f.write("\n\\end{table}\n")
    print(f"\nSaved {args.out}")

    # ── plain text ──
    with open(args.txt, "w") as f:
        for bucket in BUCKETS:
            for decomp_name in ["FP vs TN", "FP vs TP"]:
                key = (bucket, decomp_name)
                res = results[key]
                f.write(f"\n{'='*80}\n")
                f.write(f"{BUCKET_LABELS[bucket]} — {decomp_name}  "
                        f"(n={res['n']}, +={res['n_pos']}, -={res['n_neg']})\n\n")
                header = f"{'':>28s}" + "".join(f"  {CURV_DISPLAY[c]:>16s}" for c in CURVATURES)
                f.write(header + "\n")
                f.write("-" * len(header) + "\n")
                parts = f"{'base+curv AUC':>28s}"
                for curv in CURVATURES:
                    parts += f"  {res['rows'][curv]['auc_base_curv']:>16.3f}"
                f.write(parts + "\n\n")
                for rf in RIPPLE_FEATS + ["block"]:
                    label = RIPPLE_DISPLAY[rf]
                    row = f"{label:>28s}"
                    for curv in CURVATURES:
                        rd = res["rows"][curv]
                        if rf == "block":
                            auc, gain = rd["auc_block"], rd["gain_block"]
                        else:
                            auc = rd[f"auc_{rf}"]
                            gain = rd[f"gain_{rf}"]
                        if auc is not None and gain is not None:
                            row += f"  {auc:.3f} ({gain:+.3f})"
                        else:
                            row += f"  {'--':>16s}"
                    f.write(row + "\n")
    print(f"Saved {args.txt}")


if __name__ == "__main__":
    main()
