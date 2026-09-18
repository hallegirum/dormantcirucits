"""Table 5 — FN-vs-TN: ripple betas and AUC.

FN = in-circuit below per-combo median threshold (dormant circuit members).
TN = off-circuit below threshold (genuinely inactive).

Do ripple metrics distinguish dormant circuit components from truly inactive ones?
Swap strategies only (ripple requires pairwise site interactions).

Usage:
    python figures/table5_fntn_ripple.py [--csv PATH] [--out PATH]
"""
import argparse, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

RIPPLE_FEATS = ["ripple_in_orth", "ripple_off_orth", "ripple_frac", "ripple_xint_in_orth"]
DISPLAY = {
    "orth_base": "$\\mathrm{orth}_b(\\delta)$",
    "along_nonbase_max": "$\\mathrm{along}_s(\\delta)$",
    "along_target": "$\\mathrm{along}_t(\\delta)$",
    "local_gap_frac": "local curvature",
    "ripple_in_orth": "ripple (in-task)",
    "ripple_off_orth": "ripple (off-task)",
    "ripple_frac": "ripple frac",
    "ripple_xint_in_orth": "ripple (xint in-task)",
    "block": "ripple block (all 4)",
}
DISPLAY_TXT = {
    "orth_base": "orth_b",
    "along_nonbase_max": "along_s",
    "along_target": "along_t",
    "local_gap_frac": "local_curv",
    "ripple_in_orth": "ripple_in",
    "ripple_off_orth": "ripple_off",
    "ripple_frac": "ripple_frac",
    "ripple_xint_in_orth": "ripple_xint_in",
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


def build_fntn_population(df, bucket):
    """Below-threshold: FN (in-circuit below) vs TN (off-circuit below)."""
    sub = df[df["bucket"] == bucket].copy()
    in_circ = sub[sub["in_base_circuit"] == True]
    off_circ = sub[sub["in_base_circuit"] == False]
    thresholds = in_circ.groupby("combo")["abs_NIE"].median().to_dict()

    def below(row):
        thr = thresholds.get(row["combo"])
        return thr is not None and pd.notna(row["abs_NIE"]) and row["abs_NIE"] < thr

    pool = pd.concat([in_circ, off_circ])
    pool = pool[pool.apply(below, axis=1)].copy()
    # FN=1 (in-circuit below threshold), TN=0 (off-circuit below threshold)
    pool["label"] = pool["in_base_circuit"].astype(float)
    return pool


def zscore_within_combo(df, feats):
    out = df.copy()
    for f in feats:
        out[f"z_{f}"] = out.groupby("combo")[f].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-12))
    return out


def fit_model(df, feats, B=200):
    z_feats = [f"z_{f}" for f in feats]
    sub = df.dropna(subset=z_feats + ["label"]).copy()
    X = sub[z_feats].values.astype(float)
    y = sub["label"].values
    mask = np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    if len(np.unique(y)) < 2 or len(y) < 20:
        return None

    lr = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0, class_weight="balanced")
    lr.fit(X, y)
    auc = roc_auc_score(y, lr.predict_proba(X)[:, 1])
    betas = lr.coef_[0]

    combos = sub["combo"].values
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


def fmt_beta(est, lo, hi):
    sig = "*" if (lo > 0 or hi < 0) else ""
    return f"${est:+.2f}${sig}"


def fmt_beta_txt(est, lo, hi):
    sig = "*" if (lo > 0 or hi < 0) else ""
    return f"{est:+.2f}{sig}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="interaction_analysis_ripplefull/rows.csv")
    ap.add_argument("--out", default="figures/table5_fntn_ripple.tex")
    ap.add_argument("--txt", default="figures/table5_fntn_ripple.txt")
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

    MODELS = [
        ("base (no ripple)",    None),
        ("+ ripple (in-task)",  "ripple_in_orth"),
        ("+ ripple (off-task)", "ripple_off_orth"),
        ("+ ripple frac",       "ripple_frac"),
        ("+ ripple (xint)",     "ripple_xint_in_orth"),
        ("+ ripple block",      RIPPLE_FEATS),
    ]

    all_results = {}

    for bucket in BUCKETS:
        is_cross = "cross" in bucket
        base_feats = ["orth_base", "along_nonbase_max"]
        if is_cross:
            base_feats.append("along_target")

        pool = build_fntn_population(df, bucket)
        all_needed = base_feats + RIPPLE_FEATS
        clean = pool.dropna(subset=all_needed + ["label", "combo"])
        for f in all_needed:
            clean = clean[np.isfinite(clean[f])]
        clean = zscore_within_combo(clean, all_needed)

        n_fn = int(clean["label"].sum())
        n_tn = len(clean) - n_fn
        print(f"\n{BUCKET_LABELS[bucket]}  (n={len(clean)}, FN={n_fn}, TN={n_tn})")

        bucket_models = []
        for model_label, ripple_add in MODELS:
            feats = list(base_feats)
            if ripple_add is not None:
                if isinstance(ripple_add, list):
                    feats.extend(ripple_add)
                else:
                    feats.append(ripple_add)
            res = fit_model(clean, feats)
            if res:
                res["model_label"] = model_label
                res["feats"] = feats
                beta_strs = []
                for f in feats:
                    b = res[f"beta_{f}"]
                    lo = res[f"lo_{f}"]
                    hi = res[f"hi_{f}"]
                    beta_strs.append(f"{DISPLAY_TXT.get(f,f)}={fmt_beta_txt(b,lo,hi)}")
                print(f"  {model_label:25s}  AUC={res['auc']:.3f}  {', '.join(beta_strs)}")
            else:
                res = {"model_label": model_label, "feats": feats, "auc": None}
                print(f"  {model_label:25s}  FAILED")
            bucket_models.append(res)
        all_results[bucket] = {"n": len(clean), "n_fn": n_fn, "n_tn": n_tn,
                               "base_feats": base_feats, "models": bucket_models}

    # ── LaTeX ──
    with open(args.out, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{FN vs.\\ TN logistic regression (below-threshold population). "
                "FN $=$ in-circuit components with $|$NIE$|$ below per-combo median "
                "(dormant circuit members); TN $=$ off-circuit below threshold. "
                "Base model: $\\mathrm{orth}_b$, $\\mathrm{along}_s$ "
                "($+\\ \\mathrm{along}_t$ for cross). "
                "All strategies with pairwise interactions. * $=$ 95\\% cluster-bootstrap CI excludes 0.}\n")
        f.write("\\label{tab:fntn-ripple}\n\\small\n")

        for bucket in BUCKETS:
            br = all_results[bucket]
            base = br["base_feats"]
            models = br["models"]
            n_models = len(models)
            # display all features that appear in any model
            all_display = base + RIPPLE_FEATS

            f.write(f"\n\\medskip\n")
            f.write(f"\\textbf{{{BUCKET_LABELS[bucket]}}} "
                    f"($n={br['n']}$, FN$={br['n_fn']}$, TN$={br['n_tn']}$)\n\n")

            MODEL_SHORT = {
                "base (no ripple)":    "base",
                "+ ripple (in-task)":   "$+$in",
                "+ ripple (off-task)":  "$+$off",
                "+ ripple frac":        "$+$frac",
                "+ ripple (xint)":      "$+$xint",
                "+ ripple block":       "$+$block",
            }

            col_spec = "l " + " ".join(["r"] * n_models)
            f.write(f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n")

            col_heads = [""]
            for m in models:
                col_heads.append(MODEL_SHORT.get(m["model_label"], m["model_label"]))
            f.write(" & ".join(col_heads) + " \\\\\n\\midrule\n")

            # AUC row
            auc_parts = ["AUC"]
            for m in models:
                if m.get("auc") is not None:
                    auc_parts.append(f"{m['auc']:.3f}")
                else:
                    auc_parts.append("--")
            f.write(" & ".join(auc_parts) + " \\\\\n")

            # beta rows
            for feat in all_display:
                parts = [DISPLAY.get(feat, feat)]
                for m in models:
                    if m.get("auc") is None:
                        parts.append("--")
                    elif feat in m["feats"]:
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
    print(f"\nSaved {args.out}")

    # ── plain text ──
    with open(args.txt, "w") as f:
        MODEL_SHORT_TXT = {
            "base (no ripple)":    "base",
            "+ ripple (in-task)":   "+in",
            "+ ripple (off-task)":  "+off",
            "+ ripple frac":        "+frac",
            "+ ripple (xint)":      "+xint",
            "+ ripple block":       "+block",
        }
        for bucket in BUCKETS:
            br = all_results[bucket]
            base = br["base_feats"]
            models = br["models"]
            all_display = base + RIPPLE_FEATS

            f.write(f"\n{'='*90}\n")
            f.write(f"{BUCKET_LABELS[bucket]}  (n={br['n']}, FN={br['n_fn']}, TN={br['n_tn']})\n\n")

            col_labels = [MODEL_SHORT_TXT.get(m["model_label"], m["model_label"]) for m in models]
            header = f"{'':>18s}" + "".join(f"  {c:>10s}" for c in col_labels)
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            row = f"{'AUC':>18s}"
            for m in models:
                if m.get("auc") is not None:
                    row += f"  {m['auc']:>10.3f}"
                else:
                    row += f"  {'--':>10s}"
            f.write(row + "\n")

            for feat in all_display:
                row = f"{DISPLAY_TXT.get(feat,feat):>18s}"
                for m in models:
                    if m.get("auc") is None:
                        row += f"  {'--':>10s}"
                    elif feat in m["feats"]:
                        b = m[f"beta_{feat}"]
                        lo = m[f"lo_{feat}"]
                        hi = m[f"hi_{feat}"]
                        row += f"  {fmt_beta_txt(b,lo,hi):>10s}"
                    else:
                        row += f"  {'--':>10s}"
                f.write(row + "\n")
    print(f"Saved {args.txt}")


if __name__ == "__main__":
    main()
