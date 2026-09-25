"""Table R (FP vs TN) — Full model betas: nested logistic regressions.

Nested columns:
  base (orth, along) → +curv → +ripple → +curv,ripple → curv+ripple (no along)

The last column drops along to reveal ripple's effect in push regimes
where along absorbs curvature variance.

Shows best ripple (by |beta|) + ripple_xint (sign-flip metric).
Per-strategy best curvature. C=10.0, B=500.

Usage:
    python figures/tableR_fptn_full_model_betas.py [--csv PATH]
"""
import argparse, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

BUCKETS = [
    ("STR_swap_within",  "swap (within)"),
    ("STR_swap_cross",   "swap (cross)"),
    ("STR_push_within",  "push (within)"),
    ("STR_push_cross",   "push (cross)"),
    ("Gaussian_push",    "Gauss.\\ push"),
]

CURV_MAP = {
    "STR_swap_within": ("gap_frac",       "downstream curv."),
    "STR_swap_cross":  ("gap_frac",       "downstream curv."),
    "STR_push_within": ("local_gap_frac", "local curvature"),
    "STR_push_cross":  ("local_gap_frac", "local curvature"),
    "Gaussian_push":   ("gap_frac",       "downstream curv."),
}

RIPPLE_ALL = ["ripple_in_orth", "ripple_off_orth", "ripple_frac", "ripple_xint_in_orth"]
XINT = "ripple_xint_in_orth"

RIPPLE_DISPLAY = {
    "ripple_in_orth":      r"$\mathrm{ripple}_\mathrm{in}$",
    "ripple_off_orth":     r"$\mathrm{ripple}_\mathrm{off}$",
    "ripple_frac":         r"$\mathrm{ripple}_\mathrm{frac}$",
    "ripple_xint_in_orth": r"$\mathrm{ripple}_\mathrm{xint}$",
}

RIPPLE_DISPLAY_TXT = {
    "ripple_in_orth":      "r_in",
    "ripple_off_orth":     "r_off",
    "ripple_frac":         "r_frac",
    "ripple_xint_in_orth": "r_xint_in",
}


def _strat_bucket(row):
    push = str(row["push_site"]) if pd.notna(row.get("push_site")) else ""
    strategy = str(row["strategy"]) if pd.notna(row.get("strategy")) else ""
    base, target = row.get("base_task"), row.get("target_task")
    within = pd.notna(target) and base == target
    if push == "input":
        return f"STR_swap_{'within' if within else 'cross'}"
    if push == "self":
        return f"{strategy}_self"
    if strategy == "STR":
        return f"STR_push_{'within' if within else 'cross'}"
    return f"{strategy}_push"


def build_fptn_population(df, bucket):
    sub = df[df["bucket"] == bucket].copy()
    ic = sub[sub["in_base_circuit"] == True]
    oc = sub[sub["in_base_circuit"] == False].copy()
    thresholds = ic.groupby("combo")["abs_NIE"].median().to_dict()
    def is_fp(row):
        thr = thresholds.get(row["combo"])
        return thr is not None and pd.notna(row["abs_NIE"]) and row["abs_NIE"] >= thr
    oc["label"] = oc.apply(is_fp, axis=1).astype(float)
    return oc


def zscore(df, feats):
    out = df.copy()
    for f in feats:
        out[f"z_{f}"] = out.groupby("combo")[f].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-12))
    return out


def fmt_beta(b, lo, hi):
    sig = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
    star = "*" if sig else ""
    return f"${b:+.2f}${'^*' if sig else ''}$", f"{b:+.2f}{star}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="interaction_analysis_ripplefull/rows.csv")
    ap.add_argument("--out-txt", default="figures/tableR_fptn_full_model_betas.txt")
    ap.add_argument("--out-tex", default="figures/tableR_fptn_full_model_betas.tex")
    ap.add_argument("--B", type=int, default=500)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["bucket"] = df.apply(_strat_bucket, axis=1)

    txt_lines = []
    tex_blocks = []

    for bkey, blabel in BUCKETS:
        curv_feat, curv_label = CURV_MAP[bkey]
        is_cross = "cross" in bkey
        pool = build_fptn_population(df, bkey)

        base_feats = ["orth_base", "along_nonbase_max"]
        if is_cross:
            base_feats.append("along_target")

        all_feats = base_feats + [curv_feat] + RIPPLE_ALL
        no_along_feats = ["orth_base", curv_feat] + RIPPLE_ALL
        pool = zscore(pool, list(set(all_feats)))

        z_all = [f"z_{f}" for f in all_feats]
        clean = pool.dropna(subset=z_all + ["label", "combo"])
        X = clean[z_all].values
        y = clean["label"].values
        combos = clean["combo"].values
        mask = np.all(np.isfinite(X), axis=1)
        X, y, combos = X[mask], y[mask], combos[mask]
        clean = clean[mask].reset_index(drop=True)
        uc = np.unique(combos)

        n_fp = int((y == 1).sum())
        n_tn = int((y == 0).sum())

        model_configs = [
            ("base",           base_feats),
            ("+curv",          base_feats + [curv_feat]),
            ("+ripple",        base_feats + RIPPLE_ALL),
            ("+curv,ripple",   all_feats),
            ("curv+rip\\,(-al)", no_along_feats),
        ]

        model_results = {}
        for mname, mfeats in model_configs:
            z = [f"z_{f}" for f in mfeats]
            Xm = clean[z].values
            lrm = LogisticRegression(max_iter=2000, C=10.0)
            lrm.fit(Xm, y)
            aucm = roc_auc_score(y, lrm.predict_proba(Xm)[:, 1])
            betas_m = {mfeats[i]: lrm.coef_[0][i] for i in range(len(mfeats))}

            boots_m = []
            rng = np.random.RandomState(0)
            for _ in range(args.B):
                idx_c = rng.choice(uc, size=len(uc), replace=True)
                idx = np.concatenate([np.where(combos == c)[0] for c in idx_c])
                Xb, yb = Xm[idx], y[idx]
                if len(np.unique(yb)) < 2:
                    continue
                try:
                    lrb = LogisticRegression(max_iter=2000, C=10.0)
                    lrb.fit(Xb, yb)
                    boots_m.append(lrb.coef_[0])
                except:
                    continue
            boots_m = np.array(boots_m)
            model_results[mname] = (betas_m, boots_m, mfeats, aucm)

        full_betas = model_results["+curv,ripple"][0]
        best_rip = max(RIPPLE_ALL, key=lambda f: abs(full_betas[f]))

        ripple_show = [(best_rip, True)]
        if best_rip != XINT:
            ripple_show.append((XINT, False))

        display_feats = [
            ("orth_base",          r"$\mathrm{orth}_b(\delta)$", "orth_b(d)"),
            ("along_nonbase_max",  r"$\mathrm{along}_s(\delta)$", "along_s(d)"),
        ]
        if is_cross:
            display_feats.append(("along_target", r"$\mathrm{along}_t(\delta)$", "along_t(d)"))
        display_feats.append((curv_feat, curv_label,
                              "local_curv" if curv_feat == "local_gap_frac" else "dwnstrm_curv"))

        def get_beta_ci(mname, feat):
            betas_m, boots_m, mf, _ = model_results[mname]
            if feat not in betas_m:
                return None, None, None
            idx_f = mf.index(feat)
            b = betas_m[feat]
            lo = np.percentile(boots_m[:, idx_f], 2.5)
            hi = np.percentile(boots_m[:, idx_f], 97.5)
            return b, lo, hi

        # --- TXT ---
        full_auc = model_results["+curv,ripple"][3]
        no_along_auc = model_results["curv+rip\\,(-al)"][3]
        txt_lines.append(f"\n{blabel} (n={len(y)}, FP={n_fp}, TN={n_tn}, "
                         f"AUC={full_auc:.3f}, curv={curv_label})")
        col_names = ["base", "+curv", "+ripple", "+curv,rip", "c+r(-al)"]
        header = f"{'':>18s}" + "".join(f"  {c:>10s}" for c in col_names)
        txt_lines.append(header)
        txt_lines.append("-" * len(header))

        for feat, _, dname_txt in display_feats:
            row = f"{dname_txt:>18s}"
            for mname, _ in model_configs:
                b, lo, hi = get_beta_ci(mname, feat)
                if b is not None:
                    _, short = fmt_beta(b, lo, hi)
                    row += f"  {short:>10s}"
                else:
                    row += f"  {'--':>10s}"
            txt_lines.append(row)

        for rf, is_best in ripple_show:
            tag = " (best)" if is_best else ""
            dname = RIPPLE_DISPLAY_TXT[rf] + tag
            row = f"{dname:>18s}"
            for mname, _ in model_configs:
                b, lo, hi = get_beta_ci(mname, rf)
                if b is not None:
                    _, short = fmt_beta(b, lo, hi)
                    row += f"  {short:>10s}"
                else:
                    row += f"  {'--':>10s}"
            txt_lines.append(row)

        # --- TEX ---
        tex_rows = []
        tex_rows.append(f"\\medskip\n\\textbf{{{blabel}}} "
                        f"($n={len(y)}$, FP$={n_fp}$, TN$={n_tn}$; "
                        f"curv $=$ {curv_label})\n")
        tex_rows.append("\\begin{tabular}{l r r r r r}")
        tex_rows.append("\\toprule")
        tex_rows.append(
            f" & base & $+$curv & $+$ripple & $+$curv,rip & curv$+$rip ($-$al) \\\\")
        tex_rows.append("\\midrule")

        for feat, dname_tex, _ in display_feats:
            cells = [dname_tex]
            for mname, _ in model_configs:
                b, lo, hi = get_beta_ci(mname, feat)
                if b is not None:
                    tex_val, _ = fmt_beta(b, lo, hi)
                    cells.append(tex_val)
                else:
                    cells.append("--")
            tex_rows.append(" & ".join(cells) + " \\\\")

        tex_rows.append("\\midrule")
        for rf, is_best in ripple_show:
            label = RIPPLE_DISPLAY[rf]
            rip_cells = [label]
            for mname, _ in model_configs:
                b, lo, hi = get_beta_ci(mname, rf)
                if b is not None:
                    tex_val, _ = fmt_beta(b, lo, hi)
                    rip_cells.append(tex_val)
                else:
                    rip_cells.append("--")
            tex_rows.append(" & ".join(rip_cells) + " \\\\")

        tex_rows.append("\\bottomrule")
        tex_rows.append("\\end{tabular}")
        tex_blocks.append("\n".join(tex_rows))

    with open(args.out_txt, "w") as f:
        f.write("\n".join(txt_lines) + "\n")
    print(f"Saved {args.out_txt}")

    tex_full = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Full model: standardised $\\beta$ from nested logistic regressions "
        "(FP vs.\\ TN, off-circuit population). "
        "Base model includes $\\mathrm{orth}_b$ and $\\mathrm{along}_s$. "
        "Per-strategy best curvature metric. "
        "Best ripple (largest $|\\beta|$) and $\\mathrm{ripple}_\\mathrm{xint}$ "
        "(cross-subspace interference, sign-flip across regimes) shown per strategy. "
        "Final column drops $\\mathrm{along}$ to isolate curv$+$ripple complementarity. "
        "* $=$ 95\\% cluster-bootstrap CI excludes 0.}\n"
        "\\label{tab:full-model-fptn}\n"
        "\\small\n\n"
        + "\n\n".join(tex_blocks)
        + "\n\n\\end{table}\n"
    )
    with open(args.out_tex, "w") as f:
        f.write(tex_full)
    print(f"Saved {args.out_tex}")


if __name__ == "__main__":
    main()
