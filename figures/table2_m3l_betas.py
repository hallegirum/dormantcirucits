"""Table 2 — M3L: FP-vs-TN betas (off-circuit false positives vs true negatives).

Population: off-circuit components only.
  FP = off-circuit with |NIE| >= per-combo median of in-circuit |NIE|
  TN = off-circuit with |NIE| < threshold

Base model: orth_base, along_nonbase_max (+ along_target for cross).
Then add one of: local_gap_frac, gap_frac, excess_local_gap.
All features z-scored within combo before fitting.
class_weight='balanced' to handle FP:TN imbalance.

Usage:
    python figures/table2_m3l_betas.py [--csv PATH] [--out PATH]
"""
import argparse, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


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


def build_m3l_population(df, bucket):
    """Off-circuit only: FP (above per-combo median) vs TN (below)."""
    sub = df[df["bucket"] == bucket].copy()
    in_circ = sub[sub["in_base_circuit"] == True]
    off_circ = sub[sub["in_base_circuit"] == False].copy()
    thresholds = in_circ.groupby("combo")["abs_NIE"].median().to_dict()

    def is_fp(row):
        thr = thresholds.get(row["combo"])
        return thr is not None and pd.notna(row["abs_NIE"]) and row["abs_NIE"] >= thr

    off_circ["label"] = off_circ.apply(is_fp, axis=1).astype(float)  # FP=1, TN=0
    return off_circ


def zscore_within_combo(df, feats):
    out = df.copy()
    for f in feats:
        out[f"z_{f}"] = out.groupby("combo")[f].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-12))
    return out


def fit_model(df, feats, B=200):
    z_feats = [f"z_{f}" for f in feats]
    X = df[z_feats].values.astype(float)
    y = df["label"].values

    lr = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0,
                            class_weight="balanced")
    lr.fit(X, y)
    auc = roc_auc_score(y, lr.predict_proba(X)[:, 1])
    betas = lr.coef_[0]

    combos = df["combo"].values
    unique_combos = np.unique(combos)
    rng = np.random.RandomState(42)
    beta_boots, auc_boots = [], []
    for _ in range(B):
        idx_c = rng.choice(unique_combos, size=len(unique_combos), replace=True)
        idx = np.concatenate([np.where(combos == c)[0] for c in idx_c])
        yb, Xb = y[idx], X[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            lrb = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0,
                                     class_weight="balanced")
            lrb.fit(Xb, yb)
            beta_boots.append(lrb.coef_[0])
            auc_boots.append(roc_auc_score(yb, lrb.predict_proba(Xb)[:, 1]))
        except Exception:
            continue

    beta_boots = np.array(beta_boots)
    beta_lo = np.percentile(beta_boots, 2.5, axis=0) if len(beta_boots) else betas
    beta_hi = np.percentile(beta_boots, 97.5, axis=0) if len(beta_boots) else betas
    auc_lo = np.percentile(auc_boots, 2.5) if auc_boots else auc
    auc_hi = np.percentile(auc_boots, 97.5) if auc_boots else auc

    result = {"auc": auc, "auc_lo": auc_lo, "auc_hi": auc_hi}
    for i, f in enumerate(feats):
        result[f"beta_{f}"] = betas[i]
        result[f"lo_{f}"] = beta_lo[i]
        result[f"hi_{f}"] = beta_hi[i]
    return result


DISPLAY = {
    "orth_base": "$\\mathrm{orth}_b(\\delta)$",
    "along_nonbase_max": "$\\mathrm{along}_s(\\delta)$",
    "along_target": "$\\mathrm{along}_t(\\delta)$",
    "local_gap_frac": "local curvature",
    "gap_frac": "downstream curv.",
    "excess_local_gap": "excess curv.",
}

DISPLAY_TXT = {
    "orth_base": "orth_b(d)",
    "along_nonbase_max": "along_s(d)",
    "along_target": "along_t(d)",
    "local_gap_frac": "local_curv",
    "gap_frac": "downstream_curv",
    "excess_local_gap": "excess_curv",
}


def fmt_beta(est, lo, hi):
    sig = "*" if (lo > 0 or hi < 0) else ""
    return f"{est:+.2f}{sig}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="interaction_analysis_allcombos_fullgap_trio/rows.csv")
    ap.add_argument("--out", default="figures/table2_m3l_betas.tex")
    ap.add_argument("--txt", default="figures/table2_m3l_betas.txt")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["bucket"] = df.apply(_strat_bucket, axis=1)

    BUCKETS = ["STR_swap_within", "STR_swap_cross", "STR_push_within",
               "STR_push_cross", "Gaussian_push"]
    BUCKET_LABELS = {
        "STR_swap_within": "swap (within)",
        "STR_swap_cross": "swap (cross)",
        "STR_push_within": "push (within)",
        "STR_push_cross": "push (cross)",
        "Gaussian_push": "Gauss. push",
    }

    CURVATURE_VARIANTS = [
        ("Model 0 (no curv.)",   None),
        ("+ local curvature",    "local_gap_frac"),
        ("+ downstream curv.",   "gap_frac"),
        ("+ excess curv.",       "excess_local_gap"),
    ]

    all_results = {}

    for bucket in BUCKETS:
        pool = build_m3l_population(df, bucket)
        is_cross = "cross" in bucket

        base_feats = ["orth_base", "along_nonbase_max"]
        if is_cross:
            base_feats.append("along_target")

        required = base_feats + ["local_gap_frac", "gap_frac", "excess_local_gap"]
        pool_clean = pool.dropna(subset=required + ["label", "combo"])
        for f in required:
            pool_clean = pool_clean[np.isfinite(pool_clean[f])]
        pool_clean = zscore_within_combo(pool_clean, required)

        n_fp = int(pool_clean["label"].sum())
        n_tn = len(pool_clean) - n_fp

        print(f"\n{bucket}: n={len(pool_clean)}, FP={n_fp}, TN={n_tn}")

        bucket_results = {"n": len(pool_clean), "n_fp": n_fp, "n_tn": n_tn,
                          "base_feats": base_feats, "is_cross": is_cross, "models": []}

        for model_label, curv_feat in CURVATURE_VARIANTS:
            feats = list(base_feats)
            if curv_feat:
                feats.append(curv_feat)
            res = fit_model(pool_clean, feats)
            res["model_label"] = model_label
            res["feats"] = feats

            beta_strs = []
            for f in feats:
                b = res[f"beta_{f}"]
                lo = res[f"lo_{f}"]
                hi = res[f"hi_{f}"]
                beta_strs.append(f"{DISPLAY_TXT.get(f,f)}={fmt_beta(b,lo,hi)}")
            print(f"  {model_label:22s}  AUC={res['auc']:.3f}  {', '.join(beta_strs)}")

            bucket_results["models"].append(res)
        all_results[bucket] = bucket_results

    # ── write plain-text table ──
    MODEL_SHORT = {
        "Model 0 (no curv.)": "base",
        "+ local curvature":  "+local",
        "+ downstream curv.": "+downstr.",
        "+ excess curv.":     "+excess",
    }

    with open(args.txt, "w") as f:
        for bucket in BUCKETS:
            br = all_results[bucket]
            base = br["base_feats"]
            all_display = base + ["local_gap_frac", "gap_frac", "excess_local_gap"]
            models = br["models"]

            f.write(f"\n{'='*80}\n")
            f.write(f"{BUCKET_LABELS[bucket]}  (n={br['n']}, FP={br['n_fp']}, TN={br['n_tn']})\n\n")

            col_labels = [MODEL_SHORT.get(m["model_label"], m["model_label"]) for m in models]
            header = f"{'':>18s}" + "".join(f"  {c:>12s}" for c in col_labels)
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            row = f"{'AUC':>18s}"
            for m in models:
                row += f"  {m['auc']:>12.3f}"
            f.write(row + "\n")

            for feat in all_display:
                row = f"{DISPLAY_TXT.get(feat,feat):>18s}"
                for m in models:
                    if feat in m["feats"]:
                        b = m[f"beta_{feat}"]
                        lo = m[f"lo_{feat}"]
                        hi = m[f"hi_{feat}"]
                        row += f"  {fmt_beta(b,lo,hi):>12s}"
                    else:
                        row += f"  {'--':>12s}"
                f.write(row + "\n")
    print(f"\nSaved {args.txt}")

    # ── write LaTeX table ──
    with open(args.out, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{M3L: FP vs.\\ TN logistic regression (off-circuit population). "
                "FP $=$ off-circuit components with $|$NIE$|$ above per-combo median of "
                "in-circuit $|$NIE$|$; TN $=$ off-circuit below threshold. "
                "Standardised $\\beta$ from base model "
                "($\\mathrm{orth}_b$, $\\mathrm{along}_s$) "
                "$+$ one curvature feature at a time. "
                "* $=$ 95\\% cluster-bootstrap CI excludes 0.}\n")
        f.write("\\label{tab:m3l-curv}\n\\small\n")

        MODEL_COL = {
            "Model 0 (no curv.)": "base",
            "+ local curvature":  "$+$local",
            "+ downstream curv.": "$+$downstream",
            "+ excess curv.":     "$+$excess",
        }

        for bucket in BUCKETS:
            br = all_results[bucket]
            base = br["base_feats"]
            all_display = base + ["local_gap_frac", "gap_frac", "excess_local_gap"]
            models = br["models"]
            n_models = len(models)

            f.write(f"\n\\medskip\n")
            f.write(f"\\textbf{{{BUCKET_LABELS[bucket]}}} "
                    f"($n={br['n']}$, FP$={br['n_fp']}$, TN$={br['n_tn']}$)\n\n")
            col_spec = "l " + " ".join(["r"] * n_models)
            f.write(f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n")

            col_heads = [""]
            for m in models:
                col_heads.append(MODEL_COL.get(m["model_label"], m["model_label"]))
            f.write(" & ".join(col_heads) + " \\\\\n\\midrule\n")

            auc_parts = ["AUC"]
            for m in models:
                auc_parts.append(f"{m['auc']:.3f}")
            f.write(" & ".join(auc_parts) + " \\\\\n")

            for feat in all_display:
                parts = [DISPLAY.get(feat, feat)]
                for m in models:
                    if feat in m["feats"]:
                        b = m[f"beta_{feat}"]
                        lo = m[f"lo_{feat}"]
                        hi = m[f"hi_{feat}"]
                        sig = "*" if (lo > 0 or hi < 0) else ""
                        parts.append(f"${b:+.2f}${sig}")
                    else:
                        parts.append("--")
                f.write(" & ".join(parts) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")

        f.write("\n\\end{table}\n")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
