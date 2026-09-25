# FP vs TP section — context handoff

## Purpose

Distinguish actual in-circuit components (TP) from dormant components that mimic in-circuit behavior (FP). The practical dormancy detection problem — unlike FP vs TN (off-circuit characterisation), FP vs TP asks whether you can tell apart genuine contributors from impostors within the set of components that *look* active (high |NIE|).

## Existing figures/tables

| File | Description |
|------|-------------|
| `table1_fptp_nested_betas.py/.txt/.tex` | Nested betas: base (orth, along_s) + each curvature variant separately |
| `tableR_full_model_betas.py/.txt/.tex` | Full model betas: nested orth+along → +curv → +ripple → +curv,ripple → curv+rip† (no along). C=10.0, local_gap_frac throughout |
| `tableR_fptp_noorth_betas.py/.txt/.tex` | Suppression table: full → no-orth → curv+rip only. Shows how betas shift when geometry instruments removed |
| `fig3_fptp_auc_bars.py/.pdf` | AUC bar chart |
| `fig4_along_beta_reduction_fptp.py/.pdf` | Along beta reduction |
| `fig5_along_vs_curv_scatter_fptp.py/.jpeg` | Along vs curvature scatter |
| `fig_orth_beta_reduction_fptp.py/.pdf` | Orth beta reduction |
| `fig_standalone_auc_fptp.py/.txt/.jpeg` | Standalone AUC for ripple selection |
| `fig_suppression_scatter.py/.pdf` | Suppression scatter: orth vs local curv with three decision boundaries (orth-only dashed, curv-only dotted, joint solid) |
| `fig_fptp_incremental_r2_waterfall.py/.pdf/.txt` | Incremental R² waterfall: 4 pathways (A–D) showing curv-ripple complementarity. Local curv, C=10.0 |

## Population construction

- **FP** = off-circuit with |NIE| >= per-combo in-circuit median (same threshold as FP vs TN)
- **TP** = in-circuit components
- Both FP and TP have high |NIE| — the question is what geometric signatures distinguish them

## Table config

- **FP vs TP (table1 nested)**: C=1.0, class_weight="balanced"
- **FP vs TP (tableR full model / noorth / waterfall)**: C=10.0, no class_weight
- **FP vs TN** (previous section): C=10.0, no class_weight

## Along metrics (verified correct)

- `along_nonbase_max` (along_s) — used throughout, already correct
- `along_target` (along_t) — added for cross strategies only

## Curvature

FP vs TP table currently tests all three curvature variants (local, downstream, excess) rather than per-strategy best. Per-strategy best from FP vs TN work: `gap_frac` (downstream) for swap/Gaussian, `local_gap_frac` for push.

## Data and buckets

- **Data source**: `interaction_analysis_ripplefull/rows.csv`
- **Strategy buckets**: swap(within), swap(cross), push(within), push(cross), Gaussian push
- **Ripple metrics**: ripple_in_orth, ripple_off_orth, ripple_frac, ripple_xint_in_orth

## Key findings (new tables/figures)

- **Along is largely redundant in FP vs TP**: along_s betas mostly non-significant; incremental R² from along is tiny (0.001–0.050) across all regimes
- **Curv-ripple complementarity varies by regime**: ripple dominates in swap(within) and Gaussian; curvature dominates in push
- **Suppression effects**: removing orth liberates curvature in push (push within: +0.96 ns → +1.78*) and ripple in Gaussian (-3.83 ns → -4.00*); but in swap within, removing orth collapses curvature (+1.67* → +0.36 ns)
- **No ripple sign-flip in FP vs TP** (unlike FP vs TN where ripple_xint flips sign across swap/push)

## Connection from previous section

The FP vs TN prose ends: "a practical detector must distinguish actual in-circuit components from dormant ones (FP vs TP) — a distinction we turn to in the next section."
