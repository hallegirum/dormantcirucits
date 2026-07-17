# Dormant Circuits

Off-manifold / dormant-circuit analysis for language models.

Built on the [causalab](https://github.com/goodfire-ai/causalab) toolkit (Goodfire AI),
pinned to commit `bf15b35`. This repository contains only the experiment code under
`off_manifold/`; causalab is installed as a dependency (see `pyproject.toml`).

## Setup

```bash
uv sync
```

## Layout

- `off_manifold/` — path-finding, activation-manifold fitting, and DAS/STR/Gaussian
  intervention experiments.

Generated outputs (`manifold_results/`, `path_finder_results/`, `das/`, `_out/`) are
gitignored and not tracked.
