"""
collect_interaction.py
======================
Collector + DIRECTED path-model analysis over the interaction-effect runs. Walks the
results tree, tidies every per-component int.json (the directed schema: base_task,
target_task, in_base_circuit, in_target_circuit, orth_base, along_target, NIE, INT, ...)
into rows.csv, then fits the dormancy chain

    orth_base ──► INT ──► |NIE| (false-positive attribution)

as three regressions, plus the α dose-response and the Gaussian direction-null.

  M1  geometry -> interaction        INT ~ orth_base + along_target        (which move
        drives interaction: departure from the BASE manifold, or alignment with the
        TARGET subspace? dormancy predicts orth_base > 0, along_target ~ 0)
  M2  interaction -> false attrib.   |NIE| ~ orth_base [+ INT] [+ in_target] [+ cross_task]
        Same outcome model run on EACH corruption's FP population (¬in_base_circuit),
        so the strategies can be compared side by side:
          M2_STR_swap      STR swap (input), within+cross task  (full predictor set)
          M2_STR_push_MLP  STR push @ shared MLP, downstream components
          M2_Gaussian_push Gaussian push @ MLP, downstream
          M2_Gaussian_self Gaussian self-push (INT degenerate -> |NIE| ~ orth_base = TE)
        Degenerate/absent predictors are dropped per strategy automatically.
  M2b borrowed FPs                   |NIE| ~ INT + orth_base + along_target
        (on fp_borrowed only = ¬in_base ∧ in another task's circuit; tests whether
        along_target specifically drives borrowing FPs)
  M3  separation                     |NIE| ~ orth_base + INT + is_borrowed
        (on nonbase_all = TN+FP together, all ¬in_base_circuit; can INT discriminate
        which non-circuit components get falsely attributed?)
  MED mediation                      does INT mediate orth_base -> |NIE|?  (on fp_all;
        indirect a*b vs total c; a high proportion-mediated is the dormancy claim)
  P5  push mediation                 INT_c ~ push geometry  (downstream components only)
          P5_STR_push       INT ~ push_along_target (target-directed)
          P5_Gaussian_push  INT ~ push_orth_base (no target)
  DOSE  mean INT / |NIE|_FP per alpha + cross-alpha |NIE|_FP ~ alpha + orth_base, run on
        BOTH DOSE_STR_swap and DOSE_Gaussian_push (the recruitment curve per strategy)
  NULL  STR-push vs Gaussian-push INT (task-structured vs random, matched site)

Predictors are z-scored (within combo when >1) so betas are comparable partial
correlations. CIs are a cluster bootstrap over COMBOS (over rows when only one combo).
numpy for the OLS; stdlib for the rest; optional scipy unused. Runs on the pod or
after pulling results/ to a laptop.

    python collect_interaction.py --results-root poly_results_new
    python collect_interaction.py --results-root results --combos cases_0+1 cases_1+2 ...

Outputs under --out: rows.csv, audit.json, path_model.json, and a printed report.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np


# --------------------------------------------------------------------------- #
#  load + tidy
# --------------------------------------------------------------------------- #
def component_type(site: str) -> str:
    s = str(site).lower()
    return "mlp" if "mlp" in s else ("head" if ("head" in s or "attention" in s) else "other")


def _layer(site: str):
    m = re.search(r"layer[-_ ]?(\d+)", str(site), re.I)
    return int(m.group(1)) if m else None


def membership_class(label) -> str:
    if not label or label == "none":
        return "none"
    return "shared" if label == "shared" else "disjoint"


def load_rows(results_root: str, combos=None, seeds=None) -> list[dict]:
    """Every int.json under any {..}/poly_{combo}/int_seed{S}/interaction/ found beneath
    results_root (searched recursively, so an extra nesting level like poly_results_new/
    results/poly_*/ is fine) -> flat rows tagged combo/seed with derived fields.
    combos/seeds, when given, filter by the parsed combo/seed. Requires the directed
    schema; rows without base_task are skipped."""
    rows, skipped = [], 0
    # recursive so poly_*/int_seed*/interaction is found at whatever depth it lives.
    globs = glob.glob(os.path.join(results_root, "**", "poly_*", "int_seed*", "interaction"),
                      recursive=True)
    globs += glob.glob(os.path.join(results_root, "poly_*", "int_seed*", "interaction"))  # depth-0
    globs = sorted(set(globs))
    want_combos = set(combos) if combos else None
    want_seeds = set(seeds) if seeds else None
    for base in globs:
        if not os.path.isdir(base):
            continue
        m = re.search(r"poly_([^/\\]+)[/\\]int_seed(\d+)", base)
        combo, seed = (m.group(1), int(m.group(2))) if m else ("?", 0)
        if want_combos is not None and combo not in want_combos:
            continue
        if want_seeds is not None and seed not in want_seeds:
            continue
        for path in glob.glob(os.path.join(base, "**", "int.json"), recursive=True):
            try:
                r = json.load(open(path))
            except Exception:
                continue
            if "base_task" not in r:            # legacy (pre-directed) file
                skipped += 1
                continue
            r["combo"], r["seed"] = combo, seed
            r["ctype"], r["layer"] = component_type(r.get("site", "")), _layer(r.get("site", ""))
            r["mclass"] = membership_class(r.get("poly_membership"))
            r["abs_NIE"] = abs(r.get("NIE", 0.0)) if r.get("NIE") is not None else None
            # borrowing feature that works for ANY strategy/direction: the divergence's
            # max alignment with a NON-base task's subspace. For cross-task swap this is
            # ~along_target; for within-task swap or Gaussian (no target) it's the honest
            # "does the displacement point into some OTHER task" — the realistic-method test.
            abt = r.get("along_by_task") or {}
            bt = r.get("base_task")
            nonbase = [v for k, v in abt.items()
                       if str(k) != str(bt) and isinstance(v, (int, float)) and math.isfinite(v)]
            r["along_nonbase_max"] = max(nonbase) if nonbase else None
            rows.append(r)
    if skipped:
        print(f"[warn] skipped {skipped} legacy int.json without base_task (pre-directed run)")
    return rows


def audit(rows) -> dict:
    per = defaultdict(lambda: defaultdict(int))
    for r in rows:
        per[(r["combo"], r["seed"])][f"{r['strategy']}/{r['push_site']}"] += 1
    return {f"{c}|seed{s}": dict(tags) for (c, s), tags in sorted(per.items())}


# --------------------------------------------------------------------------- #
#  populations
# --------------------------------------------------------------------------- #
def _num(r, *ks):
    """All requested fields present AND finite, and the row is non-degenerate. A few
    scans can produce non-finite NIE/INT (a saturated logit); those must not poison the
    OLS (float overflow), so they're excluded from every model here."""
    if r.get("degenerate"):
        return False
    for k in ks:
        v = r.get(k)
        if v is None or not math.isfinite(v):
            return False
    return True


def swap_cross(rows):
    """cross-task STR swap rows with the geometry present — M1 geometry driver."""
    return [r for r in rows if r["strategy"] == "STR" and r["push_site"] == "input"
            and r["base_task"] != r["target_task"] and _num(r, "INT", "orth_base", "along_target", "NIE")]


def swap_all(rows):
    """ALL STR swap rows (within + cross task) with geometry — broadened FP base."""
    return [r for r in rows if r["strategy"] == "STR" and r["push_site"] == "input"
            and _num(r, "INT", "orth_base", "along_target", "NIE")]


def swap_stratum(rows, within):
    """ALL STR-swap rows (in- AND off-circuit, for the separation ref/pop split) of one
    direction stratum: within=True keeps base==target (the realistic method), within=False
    keeps base!=target (the artificial cross-task base case)."""
    return [r for r in rows if r["strategy"] == "STR" and r["push_site"] == "input"
            and ((r["base_task"] == r["target_task"]) == within)]


def _is_downstream(r):
    """component's layer is strictly after the push site's layer."""
    pl, cl = _layer(r.get("push_site", "")), r.get("layer")
    return pl is not None and cl is not None and cl > pl


def push_comp(rows, strategy="STR"):
    """Component-push rows (push at X, measure c downstream) for one strategy. STR push
    is directed (cross-task, has push_along_target); Gaussian push has no target."""
    out = [r for r in rows if r["strategy"] == strategy
           and r["push_site"] not in ("input", "self") and _num(r, "INT", "orth_base")]
    if strategy == "STR":
        out = [r for r in out if r["base_task"] != r["target_task"]]
    return out


def push_downstream(rows, strategy="STR"):
    """push_comp filtered to components DOWNSTREAM of the push site (layer > push layer)."""
    pop = push_comp(rows, strategy)
    out = [r for r in pop if _is_downstream(r)]
    return out if out else pop


def self_push(rows, strategy="Gaussian"):
    """self-push rows (push_site='self'): INT is degenerate (=0), so |NIE| = TE = the
    DIRECT off-manifold effect of pushing the component itself off its base manifold."""
    return [r for r in rows if r["strategy"] == strategy and r["push_site"] == "self"
            and _num(r, "orth_base", "NIE")]


def fp_population(rows):
    """swap-cross rows NOT truly in the base circuit -> any big |NIE| is a false positive."""
    return [r for r in swap_cross(rows) if r.get("in_base_circuit") is False]


def fp_all(rows):
    """ALL STR swap rows (within + cross) NOT in base circuit — the full FP population.
    Within-task FPs test whether standard STR can produce recruitment."""
    return [r for r in swap_all(rows) if r.get("in_base_circuit") is False]


def _borrowed_row(r) -> bool:
    """This FP component belongs to ANOTHER task's circuit (not the base). Cross-task
    swap/push: in_target_circuit=True. No explicit target (within-task swap, Gaussian
    push): home_tasks non-empty means it's in SOME other task's circuit."""
    if r.get("in_target_circuit") is True:
        return True
    if r.get("home_tasks") and (r.get("target_task") is None
                                or r.get("base_task") == r.get("target_task")):
        return True
    return False


def fp_borrowed(rows):
    """STR-swap FP rows where the component is in another task's circuit (borrowing)."""
    return [r for r in fp_all(rows) if _borrowed_row(r)]


def borrowed_of(pop):
    """Borrowed subset of any already-FP population (¬in_base ∧ in another circuit)."""
    return [r for r in pop if r.get("in_base_circuit") is False and _borrowed_row(r)]


def fp_geometric(rows):
    """FP rows where the component is in NO task's circuit — purely geometric FPs."""
    return [r for r in fp_all(rows) if not r.get("home_tasks")]


def nonbase_all(rows):
    """All STR swap rows NOT in base circuit (TN + FP together) — for M3 separation.
    TN = low |NIE| despite being off-circuit; FP = high |NIE| off-circuit."""
    return [r for r in swap_all(rows) if r.get("in_base_circuit") is False]


# --- per-strategy FP populations (¬in_base_circuit) — for the cross-strategy M2 sweep ---
def _fp(pop):
    """FP filter: keep rows not truly in the base circuit."""
    return [r for r in pop if r.get("in_base_circuit") is False]


def fp_str_swap(rows):    return fp_all(rows)                          # STR swap (input)
def fp_str_push(rows):    return _fp(push_downstream(rows, "STR"))      # STR push @ MLP, downstream
def fp_gauss_push(rows):  return _fp(push_downstream(rows, "Gaussian")) # Gaussian push, downstream
def fp_gauss_self(rows):  return _fp(self_push(rows, "Gaussian"))       # Gaussian self-push (TE)


# --------------------------------------------------------------------------- #
#  OLS + cluster bootstrap
# --------------------------------------------------------------------------- #
def _z(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else x - x.mean()


def _zc(rows, key, by_combo=True):
    """z-score a field, within combo when there is more than one combo."""
    combos = {r["combo"] for r in rows}
    if not by_combo or len(combos) < 2:
        return _z([r[key] for r in rows])
    out = np.empty(len(rows))
    for c in combos:
        idx = [i for i, r in enumerate(rows) if r["combo"] == c]
        out[idx] = _z([rows[i][key] for i in idx])
    return out


def ols(y, X: dict):
    """OLS y ~ 1 + X. y, X columns are arrays. Returns ({name: beta}, r2)."""
    y = np.asarray(y, float)
    names = list(X)
    M = np.column_stack([np.ones(len(y))] + [np.asarray(X[n], float) for n in names])
    beta, *_ = np.linalg.lstsq(M, y, rcond=None)
    resid = y - M @ beta
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / ss_tot if ss_tot > 1e-12 else 0.0
    return {n: float(beta[i + 1]) for i, n in enumerate(names)}, float(r2)


def boot(rows, fit, B=400, seed=0):
    """Cluster bootstrap `fit(sample)->{name:coef}` over COMBOS (over rows if 1 combo).
    Returns {name: {est, lo, hi}} (2.5/97.5 percentiles)."""
    rng = np.random.default_rng(seed)
    combos = sorted({r["combo"] for r in rows})
    by = {c: [r for r in rows if r["combo"] == c] for c in combos}
    est = fit(rows)
    draws = defaultdict(list)
    for _ in range(B):
        if len(combos) >= 2:
            samp = [r for c in rng.choice(combos, len(combos)) for r in by[c]]
        else:
            samp = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        try:
            for k, v in fit(samp).items():
                draws[k].append(v)
        except Exception:
            continue
    out = {}
    for k, v in est.items():
        d = np.array(draws[k]) if draws[k] else np.array([v])
        out[k] = {"est": round(v, 4), "lo": round(float(np.percentile(d, 2.5)), 4),
                  "hi": round(float(np.percentile(d, 97.5)), 4)}
    return out


def _sig(ci):  # CI excludes 0?
    return "*" if (ci["lo"] > 0 or ci["hi"] < 0) else " "


# --------------------------------------------------------------------------- #
#  logistic separation machinery (AUC / IRLS) — for the discrimination test
# --------------------------------------------------------------------------- #
def _rankdata(a):
    """Average ranks (1-based), ties averaged — for a rank-based AUC."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    sa = a[order]
    i, n = 0, len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _auc(scores, y):
    """AUC = P(score | FP > score | harmless), tie-corrected. nan if one class."""
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = _rankdata(scores)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _logit_fit(X, y, ridge=0.1, iters=25):
    """L2-penalised logistic regression via IRLS (Newton). X includes an intercept
    column; predictors should be standardised. Ridge keeps it stable under quasi-
    separation. Returns the coefficient vector."""
    beta = np.zeros(X.shape[1])
    P = ridge * np.eye(X.shape[1])
    P[0, 0] = 0.0  # don't penalise the intercept
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = p * (1.0 - p) + 1e-9
        grad = X.T @ (y - p) - P @ beta
        H = X.T @ (X * W[:, None]) + P
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def _auc_of(sample, feats, y_fn):
    """Fit logistic FP ~ 1 + standardised feats on `sample`, return (AUC, coef vector).
    y_fn(sample) -> the 0/1 label array. nan AUC when a class is missing."""
    y = y_fn(sample)
    if y.sum() == 0 or y.sum() == len(y) or len(y) < 8:
        return float("nan"), None
    X = np.column_stack([np.ones(len(sample))] + [_zc(sample, f) for f in feats])
    beta = _logit_fit(X, y)
    return _auc(X @ beta, y), beta


# --------------------------------------------------------------------------- #
#  the models
# --------------------------------------------------------------------------- #
def m1_geometry_to_int(rows):
    pop = swap_cross(rows)
    if len(pop) < 8:
        return {"status": f"too few swap-cross rows ({len(pop)})"}

    def fit(s):
        b, _ = ols(_zc(s, "INT"), {"orth_base": _zc(s, "orth_base"),
                                   "along_target": _zc(s, "along_target")})
        return b
    _, r2 = ols(_zc(pop, "INT"), {"orth_base": _zc(pop, "orth_base"),
                                  "along_target": _zc(pop, "along_target")})
    return {"status": "ok", "n": len(pop), "r2": round(r2, 3), "coef": boot(pop, fit),
            "predict": "dormancy => orth_base > 0 dominant, along_target ~ 0"}


def m2_outcome(pop, label):
    """Generic M2: |NIE| ~ orth_base [+ INT] [+ in_target] [+ cross_task] on any FP
    population. Predictors that are degenerate for this strategy are dropped:
      - INT is omitted for self-push (INT ≡ 0 there);
      - in_target / cross_task are omitted when they don't vary (e.g. Gaussian push,
        which has no target and is not a directed cross-task swap).
    So the same routine adapts to STR swap, STR push, Gaussian push, and self-push."""
    if len(pop) < 8:
        return {"status": f"too few FP rows ({len(pop)})", "label": label, "n": len(pop)}
    int_ok = any(r.get("INT") is not None and abs(r["INT"]) > 1e-9 for r in pop)
    tgt_ok = len({bool(r.get("in_target_circuit")) for r in pop}) > 1
    cross_ok = len({r["base_task"] != r["target_task"] for r in pop}) > 1

    def preds(s):
        p = {"orth_base": _zc(s, "orth_base")}
        if int_ok:
            p["INT"] = _zc(s, "INT")
        if tgt_ok:
            p["in_target"] = np.array([1.0 if r.get("in_target_circuit") else 0.0 for r in s])
        if cross_ok:
            p["cross_task"] = np.array([1.0 if r["base_task"] != r["target_task"] else 0.0 for r in s])
        return p

    def fit(s):
        b, _ = ols(_zc(s, "abs_NIE"), preds(s))
        return b
    _, r2 = ols(_zc(pop, "abs_NIE"), preds(pop))
    return {"status": "ok", "label": label, "n": len(pop), "r2": round(r2, 3), "coef": boot(pop, fit),
            "predict": "off-manifold (orth_base) + INT drive the false attribution"}


def m2_int_to_fp(rows):
    pop = fp_all(rows)
    if len(pop) < 8:
        return {"status": f"too few FP-all rows ({len(pop)})"}

    def fit(s):
        b, _ = ols(_zc(s, "abs_NIE"),
                   {"INT": _zc(s, "INT"), "orth_base": _zc(s, "orth_base"),
                    "in_target": np.array([1.0 if r.get("in_target_circuit") else 0.0 for r in s]),
                    "cross_task": np.array([1.0 if r["base_task"] != r["target_task"] else 0.0 for r in s])})
        return b
    _, r2 = ols(_zc(pop, "abs_NIE"),
                {"INT": _zc(pop, "INT"), "orth_base": _zc(pop, "orth_base"),
                 "in_target": np.array([1.0 if r.get("in_target_circuit") else 0.0 for r in pop]),
                 "cross_task": np.array([1.0 if r["base_task"] != r["target_task"] else 0.0 for r in pop])})
    return {"status": "ok", "n": len(pop), "r2": round(r2, 3), "coef": boot(pop, fit),
            "predict": "dormancy => orth_base + INT drive FPs; cross_task tests if cross-task is worse than within-task"}


def m2b_borrowed(pop, label):
    """M2 on BORROWED FPs of one strategy: |NIE| ~ orth_base [+ INT] [+ along_target].
    along_target is included only when a target direction exists (STR swap / STR push);
    Gaussian push has no target, so it reduces to |NIE| ~ orth_base [+ INT] and just asks
    whether task-affiliated components get inflated attributions under random push.
    INT is dropped when degenerate (self-push)."""
    if len(pop) < 8:
        return {"status": f"too few borrowed-FP rows ({len(pop)})", "label": label, "n": len(pop)}
    has_along = all(r.get("along_target") is not None for r in pop)
    has_int = any(r.get("INT") is not None and abs(r["INT"]) > 1e-9 for r in pop)

    def preds(s):
        p = {"orth_base": _zc(s, "orth_base")}
        if has_int:
            p["INT"] = _zc(s, "INT")
        if has_along:
            p["along_target"] = _zc(s, "along_target")
        return p

    def fit(s):
        b, _ = ols(_zc(s, "abs_NIE"), preds(s))
        return b
    _, r2 = ols(_zc(pop, "abs_NIE"), preds(pop))
    return {"status": "ok", "label": label, "n": len(pop), "r2": round(r2, 3), "coef": boot(pop, fit),
            "predict": "borrowing => along_target > 0 drives borrowed FPs (when a target exists)"}


def mediation(rows):
    """orth_base -> INT -> |NIE| on the full FP population (within + cross task)."""
    pop = fp_all(rows)
    if len(pop) < 8:
        return {"status": f"too few FP-all rows ({len(pop)})"}

    def fit(s):
        ob = _zc(s, "orth_base")
        a, _ = ols(_zc(s, "INT"), {"ob": ob})
        bc, _ = ols(_zc(s, "abs_NIE"), {"INT": _zc(s, "INT"), "ob": ob})
        c, _ = ols(_zc(s, "abs_NIE"), {"ob": ob})
        indirect = a["ob"] * bc["INT"]
        prop = indirect / c["ob"] if abs(c["ob"]) > 1e-6 else float("nan")
        return {"a(orth->INT)": a["ob"], "b(INT->NIE)": bc["INT"],
                "c_total(orth->NIE)": c["ob"], "c'_direct": bc["ob"],
                "indirect(a*b)": indirect, "prop_mediated": prop}
    return {"status": "ok", "n": len(pop), "coef": boot(pop, fit),
            "predict": "dormancy => indirect (a*b) large, prop_mediated high (INT carries orth_base)"}


def m3_separation(rows):
    """Can INT separate pernicious (high |NIE|) from harmless (low |NIE|) among
    non-base-circuit components? Runs on nonbase_all (TN + FP together) regressing
    |NIE| ~ orth_base + INT + is_borrowed. Among components that SHOULDN'T matter,
    does INT predict which ones get falsely attributed? is_borrowed tests whether
    being in another task's circuit makes the FP worse."""
    pop = nonbase_all(rows)
    if len(pop) < 12:
        return {"status": f"too few nonbase rows ({len(pop)})"}
    n_borrowed = sum(1 for r in pop if r.get("in_target_circuit") or
                     (r.get("base_task") == r.get("target_task") and r.get("home_tasks")))

    def fit(s):
        borrowed = np.array([1.0 if (r.get("in_target_circuit") or
                    (r.get("base_task") == r.get("target_task") and r.get("home_tasks")))
                    else 0.0 for r in s])
        b, _ = ols(_zc(s, "abs_NIE"),
                   {"orth_base": _zc(s, "orth_base"), "INT": _zc(s, "INT"),
                    "is_borrowed": borrowed})
        return b
    borrowed_arr = np.array([1.0 if (r.get("in_target_circuit") or
                  (r.get("base_task") == r.get("target_task") and r.get("home_tasks")))
                  else 0.0 for r in pop])
    _, r2 = ols(_zc(pop, "abs_NIE"),
                {"orth_base": _zc(pop, "orth_base"), "INT": _zc(pop, "INT"),
                 "is_borrowed": borrowed_arr})
    return {"status": "ok", "n": len(pop), "n_borrowed": n_borrowed,
            "n_geometric": len(pop) - n_borrowed, "r2": round(r2, 3),
            "coef": boot(pop, fit),
            "predict": "separation => INT > 0 means INT predicts which non-circuit components get falsely attributed; "
                       "is_borrowed > 0 means task-affiliated components are worse"}


def m3l_separation_logit(base_pop, q=0.5, feats=("INT", "along_target"),
                         label_str="STR swap", B=250):
    """THE separation test, run on one strategy's rows (base_pop = both in- and off-
    circuit rows for that strategy). A false positive is an off-base-circuit component
    whose attributed |NIE| is as large as a genuine base-circuit member's (>= the
    q-quantile of in-circuit |NIE|, computed per combo). Everything else off-circuit is
    'harmless'.

    orth_base is the baseline discriminator; `feats` are the CANDIDATE extra separators
    (INT, along_target) tested for AUC gain beyond orth_base. Per feature f it reports:
      AUC_f          f alone
      gain_f         AUC(orth+f) - AUC(orth)   (added discrimination from f)
      AUC_f_hiorth   f alone WITHIN the high-orth half (the crux: among already-high-
                     divergence off-circuit comps, does f tell FP from harmless?)
      beta_f         f's standardised logistic coef in the full (orth + all feats) model
    plus AUC_orth and AUC_full. Features absent for this strategy (e.g. along_target on a
    directionless push) are dropped. Cluster-bootstrapped over combos."""
    have = lambda r, f: r.get(f) is not None and isinstance(r.get(f), (int, float)) and math.isfinite(r[f])
    pop0 = [r for r in base_pop if r.get("in_base_circuit") is False and _num(r, "orth_base", "NIE")]
    # KEEP a feature if it's present on >=80% of rows (a few missing gap rows from run_gap
    # skips must NOT drop the whole feature), then restrict the pop to rows complete on it.
    feats = [f for f in feats if pop0 and sum(have(r, f) for r in pop0) >= 0.8 * len(pop0)]
    pop = [r for r in pop0 if all(have(r, f) for f in feats)]
    ref = [r for r in base_pop if r.get("in_base_circuit") is True and _num(r, "NIE")]
    if len(pop) < 20 or len(ref) < 5 or not feats:
        return {"status": f"too few rows / no feats (off-circuit {len(pop)}, ref {len(ref)}, "
                          f"feats {feats}) [{label_str}]", "label": label_str}

    thr = {}                                     # per-combo FP threshold from in-circuit |NIE|
    for c in {r["combo"] for r in ref}:
        vals = [r["abs_NIE"] for r in ref if r["combo"] == c and r.get("abs_NIE") is not None]
        if vals:
            thr[c] = float(np.quantile(vals, q))

    def label(sample):
        return np.array([1.0 if (thr.get(r["combo"]) is not None and r["abs_NIE"] is not None
                                 and r["abs_NIE"] >= thr[r["combo"]]) else 0.0 for r in sample])
    y = label(pop)
    n_fp, n_tn = int(y.sum()), len(y) - int(y.sum())
    if n_fp < 5 or n_tn < 5:
        return {"status": f"degenerate FP labels (FP {n_fp}, harmless {n_tn}) at q={q} [{label_str}]",
                "label": label_str}

    def fit(s):
        out = {}
        a_o, _ = _auc_of(s, ["orth_base"], label)
        a_full, beta = _auc_of(s, ["orth_base"] + feats, label)
        out["AUC_orth"], out["AUC_full"], out["gain_full"] = a_o, a_full, (a_full - a_o)
        ob = np.array([r["orth_base"] for r in s])
        hi = [s[i] for i in range(len(s)) if ob[i] >= np.median(ob)]
        for j, f in enumerate(feats):
            a_f, _ = _auc_of(s, [f], label)
            a_of, _ = _auc_of(s, ["orth_base", f], label)
            a_hi, _ = _auc_of(hi, [f], label)
            out[f"AUC_{f}"] = a_f
            out[f"gain_{f}"] = a_of - a_o           # value of f over orth ALONE
            out[f"AUC_{f}_hiorth"] = a_hi
            out[f"beta_{f}"] = float(beta[2 + j]) if beta is not None else float("nan")
            # marginal of f in the FULL model = AUC(all) - AUC(all but f)  -> "with vs without f"
            a_drop, _ = _auc_of(s, ["orth_base"] + [g for g in feats if g != f], label)
            out[f"drop_{f}"] = a_full - a_drop
        return out
    return {"status": "ok", "n": len(pop), "n_fp": n_fp, "n_harmless": n_tn, "q_threshold": q,
            "label": label_str, "feats": feats, "coef": boot(pop, fit, B=B),
            "predict": "separation => gain_<f>>0 and AUC_<f>_hiorth>0.5: that feature distinguishes "
                       "pernicious (high-orth -> FP) from harmless (high-orth -> no FP) beyond divergence"}


def xint_separation(rows):
    """THE xINT payoff. Among interaction-carried candidates (rows that run_xint wrote an
    xint_off_frac onto), does the OFF-TASK xINT fraction separate ground-truth BORROWED
    (partners in another task's circuit) from healthy — better than |INT| alone? This is
    the 'which circuit does the interaction route into' test INT-scalar cannot do."""
    pop = [r for r in rows if r.get("xint_off_frac") is not None and _num(r, "xint_off_frac")
           and r.get("in_base_circuit") is False]
    if len(pop) < 12:
        return {"status": f"too few xINT candidates ({len(pop)}) — run run_xint.py first"}
    y = np.array([1.0 if _borrowed_row(r) else 0.0 for r in pop])
    n_b = int(y.sum())
    if n_b < 4 or n_b > len(y) - 4:
        return {"status": f"degenerate borrowed labels (borrowed {n_b}/{len(y)})"}
    have_int = all(r.get("INT") is not None for r in pop)

    def fit(s):
        ys = np.array([1.0 if _borrowed_row(r) else 0.0 for r in s])
        out = {"AUC_xint_off_frac": _auc(np.array([r["xint_off_frac"] for r in s]), ys)}
        if have_int:
            out["AUC_absINT"] = _auc(np.array([abs(r["INT"]) for r in s]), ys)
        return out
    return {"status": "ok", "n": len(pop), "n_borrowed": n_b, "coef": boot(pop, fit),
            "predict": "borrowing => AUC_xint_off_frac > 0.5 and > AUC_absINT: xINT partner "
                       "identity separates borrowed from healthy where |INT| magnitude cannot"}


def _xint_strat_bucket(r):
    push = r.get("push_site")
    if push == "input":
        return f"STR_swap_{'within' if r['base_task'] == r['target_task'] else 'cross'}"
    if push == "self":
        return f"{r['strategy']}_self"
    return f"{r['strategy']}_push"


def xint_separation_by_strategy(rows):
    """Per-strategy breakdown of the xINT borrowing separation — plus the DIRECTION check:
    mean xint_off_frac for borrowed vs harmless. If borrowed's mean is LOWER (AUC < 0.5), the
    off-task fraction is anti-predictive there (a real surprise, or a bucketing bug), not just
    weak. Split by strategy isolates whether it's uniform (points to a bug) or strategy-specific."""
    pop = [r for r in rows if r.get("xint_off_frac") is not None and _num(r, "xint_off_frac")
           and r.get("in_base_circuit") is False]
    if not pop:
        return {"status": "no xINT candidates — run run_xint.py (writeback)"}
    groups = defaultdict(list)
    for r in pop:
        groups[_xint_strat_bucket(r)].append(r)
    out = {}
    for bucket, items in sorted(groups.items()):
        y = np.array([1.0 if _borrowed_row(r) else 0.0 for r in items])
        nb = int(y.sum())
        rec = {"n": len(items), "n_borrowed": nb}
        if len(items) >= 8 and 3 <= nb <= len(items) - 3:
            off = np.array([r["xint_off_frac"] for r in items])
            rec["AUC_xint_off_frac"] = round(_auc(off, y), 3)
            if all(r.get("INT") is not None for r in items):
                rec["AUC_absINT"] = round(_auc(np.array([abs(r["INT"]) for r in items]), y), 3)
            rec["mean_offfrac_borrowed"] = round(float(off[y == 1].mean()), 3)
            rec["mean_offfrac_harmless"] = round(float(off[y == 0].mean()), 3)
        else:
            rec["note"] = "too few / degenerate labels"
        out[bucket] = rec
    return {"status": "ok", "by_strategy": out}


def xint_logistic_by_strategy(rows):
    """Per-strategy logistic P(borrowed) with a MAGNITUDE CONTROL. The raw xINT masses
    (off/shared/none) partly proxy total interaction magnitude, which itself correlates with
    borrowing — so we separate the two:
      xint_total  = |on|+|off|+|shared|+|none|      -> the magnitude covariate
      frac_*      = |bucket| / xint_total            -> magnitude-normalized STRUCTURE
    The decisive numbers: AUC_magnitude (xint_total alone), AUC_full (magnitude + structure
    fractions + coherence), and struct_gain = AUC_full - AUC_magnitude = how much the partner
    DISTRIBUTION adds BEYOND raw magnitude. drop_<frac> = each fraction's marginal in the full
    model. If struct_gain ~ 0, the earlier 'structure beats INT' was magnitude in disguise."""
    for r in rows:                                    # derive magnitude + fractions in place
        if r.get("xint_off_frac") is None:
            continue
        m = {k: (r.get(k) or 0.0) for k in ("xint_on_task", "xint_off_task", "xint_shared", "xint_none")}
        tot = float(sum(m.values()))
        r["xint_total"] = tot
        r["frac_off"] = (m["xint_off_task"] / tot) if tot > 1e-9 else 0.0
        r["frac_shared"] = (m["xint_shared"] / tot) if tot > 1e-9 else 0.0
        r["frac_none"] = (m["xint_none"] / tot) if tot > 1e-9 else 0.0

    MAG = "xint_total"
    STRUCT = ["xint_coherence", "frac_off", "frac_shared", "frac_none"]   # frac_on dropped (sum=1)
    hasf = lambda r, f: isinstance(r.get(f), (int, float)) and math.isfinite(r.get(f, float("nan")))
    pop0 = [r for r in rows if r.get("xint_off_frac") is not None and r.get("in_base_circuit") is False]
    if not pop0:
        return {"status": "no xINT candidates — run run_xint.py"}
    groups = defaultdict(list)
    for r in pop0:
        groups[_xint_strat_bucket(r)].append(r)

    def y_fn(s):
        return np.array([1.0 if _borrowed_row(r) else 0.0 for r in s])
    out = {}
    for bucket, items in sorted(groups.items()):
        struct = [f for f in STRUCT if sum(hasf(r, f) for r in items) >= 0.8 * len(items)]
        allf = [MAG] + struct
        pop = [r for r in items if all(hasf(r, f) for f in allf)]
        y = y_fn(pop)
        nb = int(y.sum())
        if len(pop) < 12 or nb < 4 or nb > len(pop) - 4 or not struct:
            out[bucket] = {"n": len(pop), "n_borrowed": nb, "note": "too few / degenerate"}
            continue
        a_mag, _ = _auc_of(pop, [MAG], y_fn)                 # magnitude ALONE
        a_full, _ = _auc_of(pop, allf, y_fn)                 # magnitude + structure
        rec = {"n": len(pop), "n_borrowed": nb, "struct": struct,
               "AUC_magnitude": round(a_mag, 3), "AUC_full": round(a_full, 3),
               "struct_gain": round(a_full - a_mag, 3)}      # <- structure BEYOND magnitude
        for f in struct:
            a_drop, _ = _auc_of(pop, [g for g in allf if g != f], y_fn)
            rec[f"drop_{f}"] = round(a_full - a_drop, 3)
        if all(r.get("INT") is not None for r in pop):
            rec["AUC_absINT"] = round(_auc(np.array([abs(r["INT"]) for r in pop]), y), 3)
        out[bucket] = rec
    return {"status": "ok", "by_strategy": out}


def xint_fp_prediction(rows):
    """Do the xINT indicators predict a large false positive (|NIE|), not just the borrowed
    label? Reuses M3L's FP definition (off-circuit |NIE| >= per-combo in-circuit median) but
    (a) restricts the off-circuit classification set to xINT CANDIDATES and (b) adds
    xint_total + structure fractions + gap_frac as features — so each gets an AUC / gain /
    drop for predicting the FP attribution. Per strategy. Bootstrapped over combos."""
    for r in rows:                                    # derive magnitude + fractions (idempotent)
        if r.get("xint_off_frac") is None:
            continue
        m = {k: (r.get(k) or 0.0) for k in ("xint_on_task", "xint_off_task", "xint_shared", "xint_none")}
        tot = float(sum(m.values()))
        r["xint_total"] = tot
        r["frac_off"] = (m["xint_off_task"] / tot) if tot > 1e-9 else 0.0
        r["frac_shared"] = (m["xint_shared"] / tot) if tot > 1e-9 else 0.0
        r["frac_none"] = (m["xint_none"] / tot) if tot > 1e-9 else 0.0

    feats = ("gap_frac", "xint_total", "xint_coherence", "frac_off", "frac_shared", "frac_none",
             "xint_partner_orth", "xint_partner_gap",         # interaction-with-off-manifold-partners
             "ripple_in_orth", "ripple_off_orth", "ripple_frac",   # c knocks the LIVE circuit off-manifold
             "ripple_xint_orth", "ripple_xint_in_orth")    # ... via the components it INTERACTS with
    strat = defaultdict(lambda: {"in": [], "cand": []})
    for r in rows:
        if "strategy" not in r:
            continue
        b = _xint_strat_bucket(r)
        if r.get("in_base_circuit") is True:
            strat[b]["in"].append(r)                  # in-circuit ref for the FP threshold
        elif r.get("in_base_circuit") is False and r.get("xint_off_frac") is not None:
            strat[b]["cand"].append(r)                # off-circuit candidates = classification set
    out = {}
    for b, d in sorted(strat.items()):
        out[b] = m3l_separation_logit(d["in"] + d["cand"], feats=feats, label_str=b, B=200)
    return {"status": "ok", "by_strategy": out}


def p5_push(rows, strategy="STR"):
    """Downstream INT ~ push geometry, for STR push (target-directed) or Gaussian push
    (no target -> falls back to push_orth_base, then orth_base). Population is the
    downstream components of the push site for that strategy."""
    pop = push_downstream(rows, strategy)
    if len(pop) < 8:
        return {"status": f"too few push-downstream rows ({len(pop)}) [{strategy}]", "n": len(pop)}

    def fit(s):
        preds = {}
        if all(r.get("push_along_target") is not None for r in s):
            preds["push_along_target"] = _zc(s, "push_along_target")
        if all(r.get("push_orth_base") is not None for r in s):
            preds["push_orth_base"] = _zc(s, "push_orth_base")
        if not preds:                       # Gaussian without push geo -> mediator's own off-manifoldness
            preds["orth_base"] = _zc(s, "orth_base")
        b, _ = ols(_zc(s, "INT"), preds)
        return b
    return {"status": "ok", "n": len(pop), "strategy": strategy, "coef": boot(pop, fit),
            "predict": "mediation => downstream INT scales with push_along_target (STR) / push_orth_base (Gaussian)"}


def dose_response(rows, pop_fn=swap_cross, fp_fn=fp_population, label="STR_swap"):
    """The recruitment curve for one strategy. pop_fn -> the effect population (mean INT
    per alpha); fp_fn -> the FP population (mean |NIE|_FP per alpha + the cross-alpha
    regression |NIE|_FP ~ alpha + orth_base). Defaults reproduce the STR-swap curve;
    pass (push_downstream-Gaussian, fp_gauss_push) for the Gaussian-push curve."""
    pop = pop_fn(rows)
    fp = fp_fn(rows)
    out = {}
    for a in sorted({r["alpha"] for r in pop}):
        pa = [r for r in pop if r["alpha"] == a]
        fpa = [r for r in fp if r["alpha"] == a]
        row = {"n": len(pa), "mean_INT": round(float(np.mean([r["INT"] for r in pa])), 4),
               "mean_absNIE_FP": (round(float(np.mean([r["abs_NIE"] for r in fpa])), 4) if fpa else None)}
        if len(pa) >= 8:
            preds = {"orth_base": _zc(pa, "orth_base")}
            if all(r.get("along_target") is not None for r in pa):
                preds["along_target"] = _zc(pa, "along_target")
            b, _ = ols(_zc(pa, "INT"), preds)
            row["m1_orth_base_beta"] = round(b["orth_base"], 4)
        out[f"a{a:g}"] = row
    cross_alpha = {}
    if len(fp) >= 8 and len({r["alpha"] for r in fp}) >= 2:
        def fit_dose(s):
            b, _ = ols(_zc(s, "abs_NIE"), {"alpha": _zc(s, "alpha"), "orth_base": _zc(s, "orth_base")})
            return b
        _, r2a = ols(_zc(fp, "abs_NIE"), {"alpha": _zc(fp, "alpha"), "orth_base": _zc(fp, "orth_base")})
        cross_alpha = {"r2": round(r2a, 3), "n": len(fp), "coef": boot(fp, fit_dose)}
    return {"status": "ok", "label": label, "by_alpha": out, "cross_alpha_regression": cross_alpha,
            "predict": "dormancy => mean_INT, |NIE|_FP and orth_base slope rise with alpha; "
                       "cross_alpha beta_alpha = dose-response strength"}


def null_direction(rows):
    """STR-push vs Gaussian-push INT, paired by (combo, seed, site, base, alpha)."""
    def key(r):
        return (r["combo"], r["seed"], r["site"], r["base_task"], r["alpha"])
    strp = {key(r): r["INT"] for r in rows if r["strategy"] == "STR"
            and r["push_site"] not in ("input", "self") and not r["degenerate"]}
    gau = {key(r): r["INT"] for r in rows if r["strategy"] == "Gaussian"
           and r["push_site"] not in ("input", "self") and not r["degenerate"]}
    diffs = [strp[k] - gau[k] for k in strp if k in gau]
    if not diffs:
        return {"status": "no paired STR/Gaussian push rows"}
    return {"status": "ok", "n_paired": len(diffs), "mean_STR_minus_Gauss": round(float(np.mean(diffs)), 4),
            "frac_positive": round(float(np.mean([d > 0 for d in diffs])), 3),
            "predict": "task-structured recruits more => STR-push INT > Gaussian-push INT"}


# --------------------------------------------------------------------------- #
def _write_csv(rows, path):
    cols = ["combo", "seed", "site", "ctype", "layer", "strategy", "push_site", "direction",
            "base_task", "target_task", "alpha", "mclass", "gt_label", "poly_membership",
            "home_tasks", "in_base_circuit", "in_target_circuit", "degenerate",
            "NIE", "PIE", "INT", "TE", "abs_NIE", "NIE_absmean",
            "atp_linear", "gap", "gap_frac", "int_linear", "gap_INT", "gap_INT_frac",
            "orth_base", "along_target", "along_nontarget_max", "along_nonbase_max",
            "push_along_target", "push_orth_base",
            "xint_off_frac", "xint_coherence", "xint_on_task", "xint_off_task", "xint_verdict",
            "xint_partner_orth", "xint_partner_gap",
            "ripple_in_orth", "ripple_off_orth", "ripple_frac",
            "ripple_xint_orth", "ripple_xint_in_orth"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _print_coef(name, block):
    print(f"\n== {name} ==")
    if block.get("status") != "ok":
        print(f"  {block.get('status')}")
        return
    extra = f" | R2 {block['r2']}" if "r2" in block else ""
    print(f"  n={block.get('n', block.get('n_paired'))}{extra}")
    for k, ci in block.get("coef", {}).items():
        print(f"    {k:<22} {ci['est']:+.3f}  [{ci['lo']:+.3f}, {ci['hi']:+.3f}] {_sig(ci)}")
    if "predict" in block:
        print(f"  [predict] {block['predict']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--combos", nargs="+", default=None, help="default: every poly_* found")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--out", default="interaction_analysis")
    args = ap.parse_args()

    rows = load_rows(args.results_root, args.combos, args.seeds)
    print(f"collected {len(rows)} directed rows | combos {sorted({r['combo'] for r in rows})} "
          f"| seeds {sorted({r['seed'] for r in rows})}")
    if not rows:
        raise SystemExit(f"no directed int.json under {args.results_root}/**/poly_*/int_seed*/interaction/")

    models = {
        "M1_geometry_to_INT": m1_geometry_to_int(rows),
        # --- M2 across strategies: same |NIE| outcome model on each corruption's FPs ---
        "M2_STR_swap": m2_int_to_fp(rows),                                # STR swap (input)
        "M2_STR_push_MLP": m2_outcome(fp_str_push(rows), "STR push @ MLP"),
        "M2_Gaussian_push": m2_outcome(fp_gauss_push(rows), "Gaussian push"),
        "M2_Gaussian_self": m2_outcome(fp_gauss_self(rows), "Gaussian self-push"),
        # --- M2b borrowed FPs, per strategy (along_target test where a target exists) ---
        "M2b_STR_swap": m2b_borrowed(fp_borrowed(rows), "STR swap borrowed"),
        "M2b_STR_push": m2b_borrowed(borrowed_of(fp_str_push(rows)), "STR push borrowed"),
        "M2b_Gaussian_push": m2b_borrowed(borrowed_of(fp_gauss_push(rows)), "Gaussian push borrowed"),
        "M3_separation": m3_separation(rows),
        # --- separation test, borrowing feature = along_nonbase_max (works for every
        #     strategy). SWAP split into within-task (realistic) vs cross-task (base case).
        "M3L_swap_within": m3l_separation_logit(
            swap_stratum(rows, within=True), feats=("INT", "gap_frac", "gap_INT_frac", "along_nonbase_max"),
            label_str="STR swap WITHIN-task (realistic)"),
        "M3L_swap_cross": m3l_separation_logit(
            swap_stratum(rows, within=False),
            feats=("INT", "gap_frac", "gap_INT_frac", "along_target", "along_nonbase_max"),
            label_str="STR swap CROSS-task (base case)"),
        "M3L_STR_push": m3l_separation_logit(
            push_downstream(rows, "STR"),
            feats=("INT", "gap_frac", "gap_INT_frac", "along_target", "along_nonbase_max"),
            label_str="STR push"),
        "M3L_Gaussian_push": m3l_separation_logit(
            push_downstream(rows, "Gaussian"), feats=("INT", "gap_frac", "gap_INT_frac", "along_nonbase_max"),
            label_str="Gaussian push"),
        "XINT_borrowing_separation": xint_separation(rows),
        "XINT_by_strategy": xint_separation_by_strategy(rows),
        "XINT_logistic_by_strategy": xint_logistic_by_strategy(rows),
        "XINT_fp_prediction": xint_fp_prediction(rows),
        "MED_orth_base_to_INT_to_NIE": mediation(rows),
        # --- P5 push mediation on both push strategies ---
        "P5_STR_push": p5_push(rows, "STR"),
        "P5_Gaussian_push": p5_push(rows, "Gaussian"),
        # --- dose-response on both STR swap and Gaussian push ---
        "DOSE_STR_swap": dose_response(rows),
        "DOSE_Gaussian_push": dose_response(rows, pop_fn=lambda r: push_downstream(r, "Gaussian"),
                                            fp_fn=fp_gauss_push, label="Gaussian_push"),
        "NULL_STR_vs_Gaussian_push": null_direction(rows),
    }
    n_fp_all = len(fp_all(rows))
    n_fp_borrow = len(fp_borrowed(rows))
    n_fp_geom = len(fp_geometric(rows))
    print(f"\n  FP tiers: fp_all={n_fp_all} | borrowed={n_fp_borrow} | geometric={n_fp_geom} "
          f"| nonbase_all={len(nonbase_all(rows))}")
    print(f"  strategy FP pops: STR_swap={len(fp_str_swap(rows))} | STR_push={len(fp_str_push(rows))} "
          f"| Gauss_push={len(fp_gauss_push(rows))} | Gauss_self={len(fp_gauss_self(rows))}")
    aud = audit(rows)
    os.makedirs(args.out, exist_ok=True)
    _write_csv(rows, os.path.join(args.out, "rows.csv"))
    json.dump({"audit": aud, **models}, open(os.path.join(args.out, "path_model.json"), "w"), indent=2)

    ncombos = len({r["combo"] for r in rows})
    print(f"\n== AUDIT: {len(aud)} combo-seed runs, {ncombos} combo(s) "
          f"({'cluster bootstrap over combos' if ncombos > 1 else 'row bootstrap — 1 combo, CIs optimistic'}) ==")
    _print_coef("M1  geometry -> interaction  (INT ~ orth_base + along_target) [STR swap]", models["M1_geometry_to_INT"])
    _print_coef("M2  STR swap    (|NIE| ~ INT + orth_base + in_target + cross_task)", models["M2_STR_swap"])
    _print_coef("M2  STR push@MLP  (|NIE| ~ orth_base [+ INT])", models["M2_STR_push_MLP"])
    _print_coef("M2  Gaussian push (|NIE| ~ orth_base [+ INT])", models["M2_Gaussian_push"])
    _print_coef("M2  Gaussian self (|NIE| ~ orth_base)  [INT degenerate]", models["M2_Gaussian_self"])
    _print_coef("M2b STR swap  borrowed (|NIE| ~ INT + orth_base + along_target)", models["M2b_STR_swap"])
    _print_coef("M2b STR push  borrowed (|NIE| ~ INT + orth_base + along_target)", models["M2b_STR_push"])
    _print_coef("M2b Gaussian push borrowed (|NIE| ~ INT + orth_base)  [no target]", models["M2b_Gaussian_push"])
    _print_coef("M3  separation (linear)  (|NIE| ~ orth_base + INT + is_borrowed) [nonbase]", models["M3_separation"])
    for m3l_key, m3l_name in (("M3L_swap_within", "STR swap WITHIN-task (realistic)"),
                              ("M3L_swap_cross", "STR swap CROSS-task (base case)"),
                              ("M3L_STR_push", "STR push"),
                              ("M3L_Gaussian_push", "Gaussian push")):
        m3l = models[m3l_key]
        print(f"\n== M3L  separation (LOGISTIC: P(FP) | off-circuit component) — {m3l_name} ==")
        if m3l.get("status") != "ok":
            print(f"  {m3l.get('status')}")
            continue
        print(f"  n={m3l['n']}  FP={m3l['n_fp']}  harmless={m3l['n_harmless']}  feats={m3l['feats']}  "
              f"(FP = off-circuit |NIE| >= per-combo median of in-circuit |NIE|)")
        for k, ci in m3l["coef"].items():
            base = 0.5 if k.startswith("AUC") else 0.0     # AUCs compare to 0.5, gains/betas to 0
            flag = "*" if (ci["lo"] > base or ci["hi"] < base) else " "
            print(f"    {k:<20} {ci['est']:+.3f}  [{ci['lo']:+.3f}, {ci['hi']:+.3f}] {flag}")
        print(f"  [predict] {m3l['predict']}")
    xs = models["XINT_borrowing_separation"]
    print("\n== XINT  borrowing separation (AUC of off-task xINT fraction, vs |INT|) ==")
    if xs.get("status") != "ok":
        print(f"  {xs.get('status')}")
    else:
        print(f"  n={xs['n']}  borrowed={xs['n_borrowed']}")
        for k, ci in xs["coef"].items():
            flag = "*" if (ci["lo"] > 0.5 or ci["hi"] < 0.5) else " "
            print(f"    {k:<20} {ci['est']:+.3f}  [{ci['lo']:+.3f}, {ci['hi']:+.3f}] {flag}")
        print(f"  [predict] {xs['predict']}")
    xbs = models["XINT_by_strategy"]
    if xbs.get("status") == "ok":
        print("\n== XINT  by strategy (AUC + direction check: mean off_frac borrowed vs harmless) ==")
        print(f"  {'strategy':<20} {'n':>4} {'n_borr':>7} {'AUC_off':>8} {'AUC_INT':>8} "
              f"{'off|borr':>9} {'off|harm':>9}")
        for name, r in xbs["by_strategy"].items():
            if "AUC_xint_off_frac" in r:
                print(f"  {name:<20} {r['n']:>4} {r['n_borrowed']:>7} "
                      f"{r['AUC_xint_off_frac']:>8} {r.get('AUC_absINT','  n/a'):>8} "
                      f"{r['mean_offfrac_borrowed']:>9} {r['mean_offfrac_harmless']:>9}")
            else:
                print(f"  {name:<20} {r['n']:>4} {r['n_borrowed']:>7}   {r.get('note','')}")
    xl = models["XINT_logistic_by_strategy"]
    if xl.get("status") == "ok":
        print("\n== XINT  logistic P(borrowed), MAGNITUDE-CONTROLLED per strategy ==")
        print("   struct_gain = AUC_full - AUC_magnitude = partner STRUCTURE beyond raw |xINT| magnitude")
        for name, r in xl["by_strategy"].items():
            if "AUC_full" not in r:
                print(f"  {name}: {r.get('note','')} (n={r.get('n')})")
                continue
            print(f"  {name}  n={r['n']} borrowed={r['n_borrowed']}  "
                  f"AUC_mag={r['AUC_magnitude']}  AUC_full={r['AUC_full']}  "
                  f"struct_gain={r['struct_gain']:+.3f}  AUC_|INT|={r.get('AUC_absINT','n/a')}")
            for f in r["struct"]:
                print(f"       {f:<16} drop={r[f'drop_{f}']:+.3f}")
    xfp = models["XINT_fp_prediction"]
    if xfp.get("status") == "ok":
        print("\n== XINT -> FP prediction (do xINT indicators predict large |NIE|, on candidates) ==")
        print("   AUC_full = orth+gap+xINT ; gain_full = over orth alone ; drop_f = marginal in full model")
        for name, r in xfp["by_strategy"].items():
            if r.get("status") != "ok":
                print(f"  {name}: {r.get('status')}")
                continue
            c = r["coef"]
            g = lambda k: c[k]["est"] if k in c else float("nan")
            print(f"  {name}  n={r['n']} FP={r['n_fp']}  AUC_orth={g('AUC_orth'):.3f}  "
                  f"AUC_full={g('AUC_full'):.3f}  gain_full={g('gain_full'):+.3f}")
            for f in r.get("feats", []):
                dk = f"drop_{f}"
                if dk in c:
                    print(f"       {f:<16} AUC_alone={g('AUC_'+f):.3f}  drop={g(dk):+.3f} {_sig(c[dk])}")
    _print_coef("MED  orth_base -> INT -> |NIE|  (mediation)", models["MED_orth_base_to_INT_to_NIE"])
    _print_coef("P5  STR push mediation  (INT ~ push_along_target) [downstream]", models["P5_STR_push"])
    _print_coef("P5  Gaussian push mediation  (INT ~ push_orth_base) [downstream]", models["P5_Gaussian_push"])
    for dose_key, dose_name in (("DOSE_STR_swap", "STR swap"), ("DOSE_Gaussian_push", "Gaussian push")):
        print(f"\n== DOSE response — {dose_name} (per alpha) ==")
        for a, d in models[dose_key].get("by_alpha", {}).items():
            print(f"  {a}: mean_INT {d['mean_INT']:+.3f} | |NIE|_FP {d.get('mean_absNIE_FP')} "
                  f"| orth_base beta {d.get('m1_orth_base_beta')}")
        ca = models[dose_key].get("cross_alpha_regression", {})
        if ca.get("coef"):
            print(f"  cross-alpha (|NIE|_FP ~ alpha + orth_base, n={ca['n']}, R2={ca['r2']}):")
            for k, ci in ca["coef"].items():
                print(f"    {k:<22} {ci['est']:+.3f}  [{ci['lo']:+.3f}, {ci['hi']:+.3f}] {_sig(ci)}")
    nd = models["NULL_STR_vs_Gaussian_push"]
    if nd.get("status") == "ok":
        print(f"\n== NULL: STR-push - Gaussian-push INT = {nd['mean_STR_minus_Gauss']:+.3f} "
              f"({nd['frac_positive']:.0%} positive, n={nd['n_paired']}) ==")
    print(f"\nwrote rows.csv + path_model.json under {args.out}/  ('*' = 95% CI excludes 0)")


if __name__ == "__main__":
    main()
