# Dormant Circuits

Off-manifold / dormant-circuit analysis for language models.

Built on the [causalab](https://github.com/goodfire-ai/causalab) toolkit (Goodfire AI),
written against commit `bf15b35`. This repository contains only the experiment code
under `off_manifold/`; it runs inside a causalab checkout and uses causalab's own
locked environment. (Deliberately no `pyproject.toml` here — one would shadow
causalab's project and make `uv run` resolve a fresh, unlocked environment.)

## Setup

```bash
git clone https://github.com/goodfire-ai/causalab.git
cd causalab && git checkout bf15b35
uv sync
git clone https://github.com/hallegirum/dormantcirucits.git dormant-circuit-analysis
cd dormant-circuit-analysis/off_manifold
uv run python activation_manifold.py --help   # runs in causalab's venv
```

## Layout

- `off_manifold/` — path-finding, activation-manifold fitting, and DAS/STR/Gaussian
  intervention experiments.

Generated outputs (`manifold_results/`, `path_finder_results/`, `das/`, `_out/`) are
gitignored and not tracked.
