# FP Classification: Borrowed vs Epiotic — Full Summary

## Two definitions in the codebase

### Definition 1: Component-level (fig8, interaction data)

**Source:** `fig8_fp_anatomy_scatter.py`, line 69.

```python
fps["subset"] = fps["home"].apply(lambda h: "borrowed" if len(h) > 0 else "epiotic")
```

- **Borrowed:** `len(home_tasks) > 0` — the component belongs to at least one task's circuit (any task, not necessarily the base task being tested).
- **Epiotic:** `home_tasks` is empty — the component has no known circuit affiliation with any task.

This is a property of the **component itself**: if a component is in task 2's circuit, it's always "borrowed" regardless of what the perturbation does.

### Definition 2: Perturbation-level (endpoint data)

**Source:** `off_manifold/check_endpoint_density_v2.py`, line 237.

```python
K_task = int(max(abt, key=abt.get))  # argmax along across all tasks
home = {int(t) for t in (r.get("home_tasks") or [])}
subset = "borrowed" if K_task in home else "epotic"
```

- **K_task** = the task with the highest alignment (argmax `along_by_task`), i.e., the task the perturbation points toward most.
- **Borrowed:** `K_task in home_tasks` — the perturbation points toward a task this component genuinely belongs to.
- **Epiotic:** `K_task not in home_tasks` — the perturbation points toward a task this component has no affiliation with.

This is a property of the **specific perturbation record**: the same component can be borrowed under one regime (if K happens to be a home task) and epiotic under another (if K points elsewhere).

## Component-level vs row-level

A single component (identified by `seed, site`) can appear in many rows — one per `(combo, regime, push_site)` combination. So:

- **Component-level:** count each unique `(seed, site)` once. In push regimes, ~33–37% of unique FP components are borrowed (have any home task).
- **Row-level:** count every FP row. In push regimes, 62–87% of FP *rows* are borrowed.

The discrepancy: borrowed components in push generate more FP rows per component than epiotic ones. A component with real circuit wiring responds more strongly to directional push perturbations, so more of its rows exceed the NIE threshold. Swap replaces activations wholesale, so both borrowed and epiotic are disrupted equally — row-level proportions stay closer to component-level.

## Decomposition per strategy

### Definition 1 (component-level, fig8)

| Regime | Borrowed | Epiotic | Total | % Borrowed |
|---|---|---|---|---|
| swap (within) | 120 | 162 | 282 | 42.6% |
| swap (cross) | 695 | 1,409 | 2,104 | 33.0% |
| push (within) | 83 | 51 | 134 | 61.9% |
| push (cross) | 107 | 31 | 138 | 77.5% |
| Gaussian push | 96 | 15 | 111 | 86.5% |

Push has higher % borrowed because borrowed components are more responsive to directional perturbation (see above).

### Definition 2 (perturbation-level, endpoint data)

| Regime | Borrowed | Epiotic | Total | % Borrowed |
|---|---|---|---|---|
| swap (within) | 29 | 71 | 100 | 29.0% |
| swap (cross) | 251 | 1,006 | 1,257 | 20.0% |
| push (within) | 5 | 84 | 89 | 5.6% |
| push (cross) | 61 | 250 | 311 | 19.6% |

Push has lower % borrowed here because K (the argmax-along task) often doesn't coincide with a home task under push regimes — the perturbation direction is controlled by the push target, not by the component's circuit affiliation.

### Unique (seed, site) component counts

| Regime | Borrowed components | Total components | % Borrowed |
|---|---|---|---|
| swap (within) | 21 | 42 | 50.0% |
| swap (cross) | 21 | 42 | 50.0% |
| push (within) | 9 | 27 | 33.3% |
| push (cross) | 9 | 24 | 37.5% |
| Gaussian push | 9 | 19 | 47.4% |

At the component level, swap has the most balanced split (50/50). Push has fewer borrowed components (33–37%) — but those borrowed components produce disproportionately more FP rows.

## Endpoint distances per strategy (endpoint definition 2)

### d_base (distance to base task cloud)

| Regime | TP | FP borrowed | FP epiotic |
|---|---|---|---|
| swap (within) | 1.00 | 1.00 | 1.03 |
| swap (cross) | 4.09 | 4.40 | **16.81** |
| push (within) | 1.02 | 1.06 | 1.14 |
| push (cross) | 1.24 | 1.13 | 1.11 |

- Within-task regimes: d_base ≈ 1 for everyone (perturbation stays near base cloud).
- Swap cross: epiotic FPs fly far from base (16.81); TP and borrowed stay moderate (4.09, 4.40).
- Push cross: all groups stay near base (~1.1–1.2). Push perturbations are smaller and more directional.

### d_K (distance to K = argmax-along task cloud)

| Regime | TP | FP borrowed | FP epiotic |
|---|---|---|---|
| swap (within) | 3.42 | 28.91 | 9.89 |
| swap (cross) | 3.00 | 2.83 | **11.90** |
| push (within) | 3.90 | 2.91 | **10.54** |
| push (cross) | 3.72 | 3.05 | **11.01** |

- TP and FP borrowed have similar d_K (~3) across regimes — both land near a task cloud.
- FP epiotic consistently has d_K ≈ 10–12 — far from the task it nominally points toward.
- swap(w) borrowed d_K=28.91 is noisy (n=29 only).

### d_nearest (distance to nearest task cloud)

| Regime | TP | FP borrowed | FP epiotic |
|---|---|---|---|
| swap (within) | 1.00 | 1.00 | 1.03 |
| swap (cross) | 1.18 | 1.51 | 4.10 |
| push (within) | 1.01 | 1.06 | 1.14 |
| push (cross) | 1.07 | 1.13 | 1.11 |

- Everyone except swap(x) epiotic has d_nearest ≈ 1 (at or inside a cloud boundary).
- swap(x) epiotic: d_nearest=4.10 — not close to any task cloud. These endpoints are genuinely in no-man's-land.

## Which figures use which definition

| Figure | Definition | Data source |
|---|---|---|
| `fig8_fp_anatomy_scatter.py` | Component-level (`len(home_tasks) > 0`) | `interaction_analysis_*/rows.csv` |
| `fig_dist_by_role.py` | Perturbation-level (`K_task in home_tasks`) | `endpoint_density_v2.jsonl` `subset` field |
| `fig_along_by_role.py` | Perturbation-level (`K_task in home_tasks`) | `endpoint_density_v2.jsonl` `subset` field |
| `fig_fp_tp_instruments.py` | Component-level (from interaction data) | `interaction_analysis_ripplefull/rows.csv` |
