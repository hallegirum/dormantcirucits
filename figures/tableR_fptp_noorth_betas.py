"""Supplementary table: how betas shift when orth is removed (FP vs TP).

Three nested models per strategy:
  Full     = orth + along_s + local_curv + ripple block
  No-orth  = along_s + local_curv + ripple block
  Curv+rip = local_curv + ripple block  (no orth, no along)

Shows suppression/enhancement effects — curvature and ripple betas
can change substantially when geometry instruments are dropped.

FP-vs-TP moderate-|NIE| band. C=10.0, no class_weight.
Bootstrap CIs (B=500) over combos.

Usage:
    python figures/tableR_fptp_noorth_betas.py [--csv PATH]
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
ALONG = "along_nonbase_max"
CURV = "local_gap_frac"

RIPPLE_DISPLAY = {
    "ripple_in_orth":      r"r$_\mathrm{in}$",
    "ripple_off_orth":     r"r$_\mathrm{off}$",
    "ripple_frac":         r"r$_\mathrm{frac}$",
    "ripple_xint_in_orth": r"r$_\mathrm{xint}$",
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


def build_moderate_band(df, bucket):
    sub = df[df["bucket"] == bucket].copy()
    ic = sub[sub["in_base_circuit"] == True]
    oc = sub[sub["in_base_circuit"] == False]
    thresholds = ic.groupby("combo")["abs_NIE"].median().to_dict()
    def above(row):
        thr = thresholds.get(row["combo"])
        return thr is not None and pd.notna(row["abs_NIE"]) and row["abs_NIE"] >= thr
    pool = pd.concat([ic, oc])
    pool = pool[pool.apply(above, axis=1)].copy()
    pool_median = pool["abs_NIE"].median()
    pool = pool[pool["abs_NIE"] <= pool_median].copy()
    pool["label"] = (~pool["in_base_circuit"]).astype(float)
    return pool


def zscore(df, feats):
    out = df.copy()
    for f in feats:
        out[f"z_{f}"] = out.groupby("combo")[f].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-12))
    return out


def fmt_beta(b, lo, hi):
    sig = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
    star = "*" if sig else ""
    return f"{b:+.2f}{star}", f"${b:+.2f}${star}"


def fit_and_bootstrap(X, y, combos, B, rng):
    lr = LogisticRegression(max_iter=2000, C=10.0)
    lr.fit(X, y)
    auc = roc_auc_score(y, lr.predict_proba(X)[:, 1])
    betas = lr.coef_[0]

    uc = np.unique(combos)
    boots = []
    for _ in range(B):
        idx_c = rng.choice(uc, size=len(uc), replace=True)
        idx = np.concatenate([np.where(combos == c)[0] for c in idx_c])
        Xb, yb = X[idx], y[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            lrb = LogisticRegression(max_iter=2000, C=10.0)
            lrb.fit(Xb, yb)
            boots.append(lrb.coef_[0])
        except Exception:
            continue
    boots = np.array(boots)
    return betas, boots, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="interaction_analysis_ripplefull/rows.csv")
    ap.add_argument("--out-txt", default="figures/tableR_fptp_noorth_betas.txt")
    ap.add_argument("--out-tex", default="figures/tableR_fptp_noorth_betas.tex")
    ap.add_argument("--B", type=int, default=500)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["bucket"] = df.apply(_strat_bucket, axis=1)

    txt_lines = []
    tex_blocks = []

    for bkey, blabel in BUCKETS:
        pool = build_moderate_band(df, bkey)
        is_cross = "cross" in bkey

        base_feats = ["orth_base", ALONG]
        if is_cross:
            base_feats.append("along_target")
        all_feats = base_feats + [CURV] + RIPPLE_ALL

        pool = zscore(pool, all_feats)
        z_all = [f"z_{f}" for f in all_feats]
        clean = pool.dropna(subset=z_all + ["label", "combo"])
        X_full = clean[z_all].values
        mask = np.all(np.isfinite(X_full), axis=1)
        clean = clean[mask].reset_index(drop=True)
        y = clean["label"].values
        combos = clean["combo"].values

        # Three model specs
        full_feats = all_feats[:]
        noorth_feats = [f for f in all_feats if f != "orth_base"]
        curvrip_feats = [CURV] + RIPPLE_ALL

        models = [
            ("full", full_feats),
            ("no-orth", noorth_feats),
            ("curv+rip", curvrip_feats),
        ]

        model_results = {}
        for mname, mfeats in models:
            z = [f"z_{f}" for f in mfeats]
            Xm = clean[z].values
            rng = np.random.RandomState(0)
            betas, boots, auc = fit_and_bootstrap(Xm, y, combos, args.B, rng)
            model_results[mname] = {
                "feats": mfeats,
                "betas": {mfeats[i]: betas[i] for i in range(len(mfeats))},
                "boots": boots,
                "auc": auc,
            }

        n_fp = int((y == 1).sum())
        n_tp = int((y == 0).sum())

        # Best ripple by |beta| in full model
        full_betas = model_results["full"]["betas"]
        best_rip = max(RIPPLE_ALL, key=lambda f: abs(full_betas.get(f, 0)))

        # --- TXT ---
        txt_lines.append(f"\n{blabel} (n={len(y)}, FP={n_fp}, TP={n_tp})")
        txt_lines.append(f"  {'instrument':>22s}  {'full':>10s}  {'no-orth':>10s}  {'curv+rip':>10s}")
        txt_lines.append(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*10}")

        display_rows = [
            ("orth_base", "orth_b"),
            (ALONG, "along_s"),
        ]
        if is_cross:
            display_rows.append(("along_target", "along_t"))
        display_rows.append((CURV, "local curv"))
        display_rows.append((best_rip, f"best rip ({best_rip.replace('ripple_','r_').replace('_orth','')})"))

        for feat, dname in display_rows:
            cells = [f"  {dname:>22s}"]
            for mname, mfeats in models:
                mr = model_results[mname]
                if feat in mr["betas"]:
                    idx_f = mr["feats"].index(feat)
                    b = mr["betas"][feat]
                    lo = np.percentile(mr["boots"][:, idx_f], 2.5)
                    hi = np.percentile(mr["boots"][:, idx_f], 97.5)
                    txt_val, _ = fmt_beta(b, lo, hi)
                    cells.append(f"{txt_val:>10s}")
                else:
                    cells.append(f"{'--':>10s}")
            txt_lines.append("  ".join(cells))

        auc_cells = [f"  {'AUC':>22s}"]
        for mname, _ in models:
            auc_cells.append(f"{model_results[mname]['auc']:>10.3f}")
        txt_lines.append("  ".join(auc_cells))

        # --- TEX ---
        tex_rows = []
        tex_label = blabel
        tex_rows.append(f"\\medskip\n\\textbf{{{tex_label}}} "
                        f"($n={len(y)}$, FP$={n_fp}$, TP$={n_tp}$)\n")
        tex_rows.append("\\begin{tabular}{l r r r}")
        tex_rows.append("\\toprule")
        tex_rows.append(" & full & no-orth & curv$+$rip \\\\")
        tex_rows.append("\\midrule")

        tex_display = [
            ("orth_base", r"$\mathrm{orth}_b(\delta)$"),
            (ALONG, r"$\mathrm{along}_s(\delta)$"),
        ]
        if is_cross:
            tex_display.append(("along_target", r"$\mathrm{along}_t(\delta)$"))
        tex_display.append((CURV, "local curvature"))
        tex_display.append((best_rip, f"best ripple ({RIPPLE_DISPLAY[best_rip]})"))

        for feat, dname in tex_display:
            cells = [dname]
            for mname, mfeats in models:
                mr = model_results[mname]
                if feat in mr["betas"]:
                    idx_f = mr["feats"].index(feat)
                    b = mr["betas"][feat]
                    lo = np.percentile(mr["boots"][:, idx_f], 2.5)
                    hi = np.percentile(mr["boots"][:, idx_f], 97.5)
                    _, tex_val = fmt_beta(b, lo, hi)
                    cells.append(tex_val)
                else:
                    cells.append("--")
            tex_rows.append(" & ".join(cells) + " \\\\")

        auc_row = ["AUC"]
        for mname, _ in models:
            auc_row.append(f"{model_results[mname]['auc']:.3f}")
        tex_rows.append(" & ".join(auc_row) + " \\\\")

        tex_rows.append("\\bottomrule")
        tex_rows.append("\\end{tabular}")
        tex_blocks.append("\n".join(tex_rows))

    with open(args.out_txt, "w") as f:
        f.write("\n".join(txt_lines) + "\n")
    print(f"Saved {args.out_txt}")

    tex_full = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Suppression effects: standardised $\\beta$ as geometry "
        "instruments are removed (FP vs.\\ TP, moderate-$|$NIE$|$ band). "
        "Full = all instruments; no-orth = drop $\\mathrm{orth}_b$; "
        "curv$+$rip = local curvature and ripple block only. "
        "Best ripple metric (largest $|\\beta|$) shown per strategy. "
        "$C=10.0$, no class weight. "
        "* $=$ 95\\% cluster-bootstrap CI excludes 0.}\n"
        "\\label{tab:fptp-noorth}\n"
        "\\small\n\n"
        + "\n\n".join(tex_blocks)
        + "\n\n\\end{table}\n"
    )
    with open(args.out_tex, "w") as f:
        f.write(tex_full)
    print(f"Saved {args.out_tex}")


if __name__ == "__main__":
    main()
