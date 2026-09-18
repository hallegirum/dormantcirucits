"""Table R (FP vs TN) — Full model betas: orth + along + local curvature + ripple block.

FP-vs-TN: off-circuit only. FP = |NIE| >= per-combo median of in-circuit |NIE|;
TN = off-circuit below threshold. C=10.0, seed=0. Bootstrap CIs over combos, B=500.

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

RIPPLE_ALL = ["ripple_in_orth", "ripple_off_orth", "ripple_frac", "ripple_xint_in_orth"]

RIPPLE_DISPLAY = {
    "ripple_in_orth":     r"$\mathrm{ripple}_\mathrm{in}$",
    "ripple_off_orth":    r"$\mathrm{ripple}_\mathrm{off}$",
    "ripple_frac":        r"$\mathrm{ripple}_\mathrm{frac}$",
    "ripple_xint_in_orth": r"$\mathrm{ripple}_\mathrm{xint}$",
}

BASE_FEATS_DISPLAY = [
    ("orth_base",     r"$\mathrm{orth}_b(\delta)$"),
    ("along_target",  r"$\mathrm{along}_s(\delta)$"),
    ("gap_frac",      "downstream curv."),
]


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
    return f"${b:+.2f}${star}", f"{b:+.2f} [{lo:+.2f}, {hi:+.2f}]{star}"


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
        pool = build_fptn_population(df, bkey)
        base_feats = ["orth_base", "along_target"] if bkey != "Gaussian_push" else ["orth_base"]
        all_feats = base_feats + ["gap_frac"] + RIPPLE_ALL
        pool = zscore(pool, all_feats)

        z_all = [f"z_{f}" for f in all_feats]
        clean = pool.dropna(subset=z_all + ["label", "combo"])
        X = clean[z_all].values
        y = clean["label"].values
        combos = clean["combo"].values
        mask = np.all(np.isfinite(X), axis=1)
        X, y, combos = X[mask], y[mask], combos[mask]

        lr = LogisticRegression(max_iter=2000, C=10.0)
        lr.fit(X, y)
        auc = roc_auc_score(y, lr.predict_proba(X)[:, 1])
        betas = {all_feats[i]: lr.coef_[0][i] for i in range(len(all_feats))}

        rng = np.random.RandomState(0)
        uc = np.unique(combos)
        beta_boots, auc_boots = [], []
        for _ in range(args.B):
            idx_c = rng.choice(uc, size=len(uc), replace=True)
            idx = np.concatenate([np.where(combos == c)[0] for c in idx_c])
            Xb, yb = X[idx], y[idx]
            if len(np.unique(yb)) < 2:
                continue
            try:
                lrb = LogisticRegression(max_iter=2000, C=10.0)
                lrb.fit(Xb, yb)
                beta_boots.append(lrb.coef_[0])
                auc_boots.append(roc_auc_score(yb, lrb.predict_proba(Xb)[:, 1]))
            except Exception:
                continue

        beta_boots = np.array(beta_boots)

        best_rip = max(RIPPLE_ALL, key=lambda f: abs(betas[f]))
        best_rip_label = RIPPLE_DISPLAY[best_rip]

        n_fp = int((y == 1).sum())
        n_tn = int((y == 0).sum())

        txt_lines.append(f"\n{blabel} (n={len(y)}, FP={n_fp}, TN={n_tn}, "
                         f"AUC={auc:.3f})")
        for feat, dname in BASE_FEATS_DISPLAY:
            if feat not in betas:
                txt_lines.append(f"  {dname:>20s}: ---")
                continue
            idx_f = all_feats.index(feat)
            b = betas[feat]
            lo = np.percentile(beta_boots[:, idx_f], 2.5)
            hi = np.percentile(beta_boots[:, idx_f], 97.5)
            _, txt_val = fmt_beta(b, lo, hi)
            txt_lines.append(f"  {dname:>20s}: {txt_val}")

        rip_short = best_rip.replace("ripple_", "r_").replace("_orth", "")
        idx_f = all_feats.index(best_rip)
        b = betas[best_rip]
        lo = np.percentile(beta_boots[:, idx_f], 2.5)
        hi = np.percentile(beta_boots[:, idx_f], 97.5)
        _, txt_rip = fmt_beta(b, lo, hi)
        txt_lines.append(f"  {'best ripple (' + rip_short + ')':>20s}: {txt_rip}")

        # --- TEX ---
        model_configs = [
            ("base", base_feats),
            ("+curv", base_feats + ["gap_frac"]),
            ("+ripple", base_feats + RIPPLE_ALL),
            ("+curv,ripple", all_feats),
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
            rng2 = np.random.RandomState(0)
            for _ in range(args.B):
                idx_c = rng2.choice(uc, size=len(uc), replace=True)
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

        tex_label = blabel
        tex_rows = []
        tex_rows.append(f"\\medskip\n\\textbf{{{tex_label}}} "
                        f"($n={len(y)}$, FP$={n_fp}$, TN$={n_tn}$)\n")
        tex_rows.append("\\begin{tabular}{l r r r r r}")
        tex_rows.append("\\toprule")
        tex_rows.append(
            f" & base & $+$curv & $+$ripple & $+$curv,ripple & AUC \\\\")
        tex_rows.append("\\midrule")

        for feat, dname in BASE_FEATS_DISPLAY:
            if feat not in betas:
                continue
            cells = [dname]
            for mname, mfeats in model_configs:
                betas_m, boots_m, mf, _ = model_results[mname]
                if feat in betas_m:
                    idx_f = mf.index(feat)
                    b = betas_m[feat]
                    lo = np.percentile(boots_m[:, idx_f], 2.5)
                    hi = np.percentile(boots_m[:, idx_f], 97.5)
                    tex_val, _ = fmt_beta(b, lo, hi)
                    cells.append(tex_val)
                else:
                    cells.append("--")
            cells.append("")
            tex_rows.append(" & ".join(cells) + " \\\\")

        rip_cells = [best_rip_label]
        for mname, mfeats in model_configs:
            betas_m, boots_m, mf, _ = model_results[mname]
            if best_rip in betas_m:
                idx_f = mf.index(best_rip)
                b = betas_m[best_rip]
                lo = np.percentile(boots_m[:, idx_f], 2.5)
                hi = np.percentile(boots_m[:, idx_f], 97.5)
                tex_val, _ = fmt_beta(b, lo, hi)
                rip_cells.append(tex_val)
            else:
                rip_cells.append("--")
        rip_cells.append("")
        tex_rows.append(" & ".join(rip_cells) + " \\\\")

        dauc_cells = ["ripple block $\\Delta$AUC"]
        base_auc = model_results["base"][3]
        curv_auc = model_results["+curv"][3]
        for mname, _ in model_configs:
            _, _, _, aucm = model_results[mname]
            if mname == "+ripple":
                dauc_cells.append(f"$+{aucm - base_auc:.3f}$")
            elif mname == "+curv,ripple":
                dauc_cells.append(f"$+{aucm - curv_auc:.3f}$")
            else:
                dauc_cells.append("--")
        dauc_cells.append("")
        tex_rows.append(" & ".join(dauc_cells) + " \\\\")

        auc_cells = ["AUC"]
        for mname, _ in model_configs:
            _, _, _, aucm = model_results[mname]
            auc_cells.append(f"{aucm:.3f}")
        auc_cells.append("")
        tex_rows.append(" & ".join(auc_cells) + " \\\\")

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
        "FP $=$ off-circuit $|$NIE$|$ above per-combo in-circuit median; "
        "TN $=$ off-circuit below. "
        "Base model includes $\\mathrm{orth}_b$ and $\\mathrm{along}_s$ "
        "(except Gaussian). "
        "Best ripple metric (largest $|\\beta|$) shown per strategy. "
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
