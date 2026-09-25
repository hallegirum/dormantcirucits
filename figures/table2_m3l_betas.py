"""Table 2 — M3L: FP-vs-TN nested betas (off-circuit population).

Swap regimes: base (orth, along) → +curv — shows curv absorbing orth/along variance.
Push/Gauss regimes: curv alone → +orth → +along → +orth,along — shows along absorbing curv.
Per-strategy best curvature. C=10.0, no class_weight.

Usage:
    python figures/table2_m3l_betas.py [--csv PATH]
"""
import argparse, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

CURV_MAP = {
    "STR_swap_within": ("gap_frac",       "downstream curv."),
    "STR_swap_cross":  ("gap_frac",       "downstream curv."),
    "STR_push_within": ("local_gap_frac", "local curvature"),
    "STR_push_cross":  ("local_gap_frac", "local curvature"),
    "Gaussian_push":   ("gap_frac",       "downstream curv."),
}

BUCKETS = ["STR_swap_within", "STR_swap_cross",
           "STR_push_within", "STR_push_cross", "Gaussian_push"]
BUCKET_LABELS = {
    "STR_swap_within": "swap (within)",
    "STR_swap_cross":  "swap (cross)",
    "STR_push_within": "push (within)",
    "STR_push_cross":  "push (cross)",
    "Gaussian_push":   "Gauss.\\ push",
}
BUCKET_LABELS_TXT = {
    "STR_swap_within": "swap (within)",
    "STR_swap_cross":  "swap (cross)",
    "STR_push_within": "push (within)",
    "STR_push_cross":  "push (cross)",
    "Gaussian_push":   "Gauss. push",
}

DISPLAY = {
    "orth_base": r"$\mathrm{orth}_b(\delta)$",
    "along_nonbase_max": r"$\mathrm{along}_s(\delta)$",
    "along_target": r"$\mathrm{along}_t(\delta)$",
    "local_gap_frac": "local curv.",
    "gap_frac": "downstream curv.",
}
DISPLAY_TXT = {
    "orth_base": "orth_b(d)",
    "along_nonbase_max": "along_s(d)",
    "along_target": "along_t(d)",
    "local_gap_frac": "local_curv",
    "gap_frac": "dwnstrm_curv",
}


def _strat_bucket(row):
    push = str(row["push_site"]) if pd.notna(row.get("push_site")) else ""
    strategy = str(row["strategy"]) if pd.notna(row.get("strategy")) else ""
    base, target = row.get("base_task"), row.get("target_task")
    within = pd.notna(target) and base == target
    if push == "input": return f"STR_swap_{'within' if within else 'cross'}"
    if push == "self": return f"{strategy}_self"
    if strategy == "STR": return f"STR_push_{'within' if within else 'cross'}"
    return f"{strategy}_push"


def build_fptn(df, bucket):
    sub = df[df["bucket"] == bucket].copy()
    ic = sub[sub["in_base_circuit"] == True]
    oc = sub[sub["in_base_circuit"] == False].copy()
    thr = ic.groupby("combo")["abs_NIE"].median().to_dict()
    def is_fp(row):
        t = thr.get(row["combo"])
        return t is not None and pd.notna(row["abs_NIE"]) and row["abs_NIE"] >= t
    oc["label"] = oc.apply(is_fp, axis=1).astype(float)
    return oc


def zscore(df, feats):
    out = df.copy()
    for f in feats:
        out[f"z_{f}"] = out.groupby("combo")[f].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-12))
    return out


def fit_model(df, feats, B=300):
    z = [f"z_{f}" for f in feats]
    X = df[z].values.astype(float)
    y = df["label"].values
    lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    lr.fit(X, y)
    auc = roc_auc_score(y, lr.predict_proba(X)[:, 1])
    betas = lr.coef_[0]
    combos = df["combo"].values
    uc = np.unique(combos)
    rng = np.random.RandomState(0)
    bb, ab = [], []
    for _ in range(B):
        idx = np.concatenate([np.where(combos == c)[0]
                              for c in rng.choice(uc, len(uc), replace=True)])
        yb = y[idx]
        if len(np.unique(yb)) < 2: continue
        try:
            lrb = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
            lrb.fit(X[idx], yb)
            bb.append(lrb.coef_[0])
            ab.append(roc_auc_score(yb, lrb.predict_proba(X[idx])[:, 1]))
        except: continue
    bb = np.array(bb)
    res = {"auc": auc,
           "auc_ci": (np.percentile(ab, 2.5), np.percentile(ab, 97.5)) if ab else (auc, auc)}
    for i, f in enumerate(feats):
        res[f"beta_{f}"] = betas[i]
        res[f"lo_{f}"] = np.percentile(bb[:, i], 2.5) if len(bb) else betas[i]
        res[f"hi_{f}"] = np.percentile(bb[:, i], 97.5) if len(bb) else betas[i]
    return res


def fmt(res, feat):
    b = res[f"beta_{feat}"]
    lo, hi = res[f"lo_{feat}"], res[f"hi_{feat}"]
    sig = "*" if (lo > 0 or hi < 0) else ""
    return f"{b:+.2f}", sig, f"[{lo:+.2f}, {hi:+.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="interaction_analysis_ripplefull/rows.csv")
    ap.add_argument("--out", default="figures/table2_m3l_betas.tex")
    ap.add_argument("--txt", default="figures/table2_m3l_betas.txt")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["bucket"] = df.apply(_strat_bucket, axis=1)

    txt_lines = []
    tex_lines = []

    tex_lines.append(r"\begin{table}[t]")
    tex_lines.append(r"\centering")
    tex_lines.append(r"\caption{FP vs.\ TN nested logistic regression (off-circuit population). "
                     r"Swap regimes: base model (orth, along) $\to$ $+$curvature shows curvature "
                     r"absorbing along/orth variance. Push/Gaussian regimes: curvature alone $\to$ "
                     r"$+$orth $\to$ $+$along $\to$ $+$orth,along shows along absorbing curvature. "
                     r"Per-strategy best curvature metric. "
                     r"Standardised $\beta$; * $=$ 95\% cluster-bootstrap CI excludes 0.}")
    tex_lines.append(r"\label{tab:m3l-curv}")
    tex_lines.append(r"\small")

    for bucket in BUCKETS:
        curv_feat, curv_label = CURV_MAP[bucket]
        is_swap = "swap" in bucket
        is_cross = "cross" in bucket

        pool = build_fptn(df, bucket)
        base_feats = ["orth_base", "along_nonbase_max"]
        if is_cross:
            base_feats.append("along_target")
        all_needed = base_feats + [curv_feat]
        pool = pool.dropna(subset=all_needed + ["label", "combo"])
        for f in all_needed:
            pool = pool[np.isfinite(pool[f])]
        pool = zscore(pool, all_needed)

        n_fp = int(pool["label"].sum())
        n_tn = len(pool) - n_fp

        if is_swap:
            models = [
                ("base",   base_feats),
                ("+curv",  base_feats + [curv_feat]),
            ]
            all_display_feats = base_feats + [curv_feat]
            col_labels_tex = ["base", f"$+${curv_label}"]
            col_labels_txt = ["base", f"+{curv_label[:8]}"]
        else:
            along_feats = ["along_nonbase_max"] + (["along_target"] if is_cross else [])
            models = [
                ("curv",        [curv_feat]),
                ("+orth",       [curv_feat, "orth_base"]),
                ("+along",      [curv_feat] + along_feats),
                ("+orth,along", [curv_feat] + base_feats),
            ]
            all_display_feats = [curv_feat] + base_feats
            col_labels_tex = [f"{curv_label}", "$+$orth", "$+$along", "$+$orth,along"]
            col_labels_txt = [curv_label[:10], "+orth", "+along", "+orth,along"]

        results = []
        for mname, mfeats in models:
            res = fit_model(pool, mfeats)
            res["feats"] = mfeats
            res["name"] = mname
            results.append(res)

        # ── plain text ──
        blab = BUCKET_LABELS_TXT[bucket]
        txt_lines.append("")
        txt_lines.append(f"{blab} (n={len(pool)}, FP={n_fp}, TN={n_tn}, curv={curv_label})")
        header = f"{'':>18s}" + "".join(f"  {c:>12s}" for c in col_labels_txt)
        txt_lines.append(header)
        txt_lines.append("-" * len(header))
        row = f"{'AUC':>18s}"
        for r in results:
            row += f"  {r['auc']:>12.3f}"
        txt_lines.append(row)
        for feat in all_display_feats:
            row = f"{DISPLAY_TXT.get(feat, feat):>18s}"
            for r in results:
                if feat in r["feats"]:
                    b, sig, ci = fmt(r, feat)
                    row += f"  {b+sig:>12s}"
                else:
                    row += f"  {'--':>12s}"
            txt_lines.append(row)

        # ── LaTeX ──
        n_cols = len(results)
        tex_lines.append("")
        tex_lines.append(r"\medskip")
        tex_lines.append(f"\\textbf{{{BUCKET_LABELS[bucket]}}} "
                         f"($n={len(pool)}$, FP$={n_fp}$, TN$={n_tn}$; "
                         f"curv $=$ {curv_label})")
        tex_lines.append("")
        tex_lines.append(f"\\begin{{tabular}}{{l {'r ' * n_cols}}}")
        tex_lines.append(r"\toprule")
        tex_lines.append(" & ".join([""] + col_labels_tex) + r" \\")
        tex_lines.append(r"\midrule")

        auc_row = ["AUC"]
        for r in results:
            auc_row.append(f"{r['auc']:.3f}")
        tex_lines.append(" & ".join(auc_row) + r" \\")

        for feat in all_display_feats:
            parts = [DISPLAY.get(feat, feat)]
            for r in results:
                if feat in r["feats"]:
                    b, sig, ci = fmt(r, feat)
                    cell = f"${b}${'^*' if sig else ''}"
                    parts.append(cell)
                else:
                    parts.append("--")
            tex_lines.append(" & ".join(parts) + r" \\")

        tex_lines.append(r"\bottomrule")
        tex_lines.append(r"\end{tabular}")

    tex_lines.append("")
    tex_lines.append(r"\end{table}")

    with open(args.txt, "w") as f:
        f.write("\n".join(txt_lines) + "\n")
    print(f"Saved {args.txt}")

    with open(args.out, "w") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
