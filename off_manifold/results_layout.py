"""
Single source of truth for the results/ directory layout.

Imported by BOTH the writers (activation_manifold.py, collect_behaviour.py) and
the reader (off-distribution-analyses.py), so save-paths and load-paths are
computed by the same code and cannot drift.

Layout:

  results/
  ├── activation_manifolds/
  │   └── {site_id}__{context}/                 context = "global" | "(('entity', 'Monday'),)"
  │       ├── models/{layer}__{index}/          composed featurizer (load_manifold reads this)
  │       ├── manifold_spline/
  │       └── visualization/
  ├── intervened_reps/
  │   └── {label}__{category}/
  │       ├── {mode}_intervened_reps.safetensors   {site_id: (N, d_model)}
  │       ├── {method}_top_sites.json              site ranking (from find_path)
  │       └── das/{site_id}/                        DAS training artifacts (DAS mode only)
  └── output_manifolds/
      ├── behaviour_manifold/                    M_y — the output_manifold analysis root
      │   └── output_manifold/spline_s0.0/manifold_spline/   (load_output_manifold reads this)
      └── patched_output_distributions/
          └── {label}__{category}/
              └── {mode}_output_dists.safetensors  {site_id: (N, W)}

Naming note: manifolds are stored per (site_id, context) WITHOUT mode — a spline
is a PCA fit, independent of which intervention (STR/Gaussian/DAS) selected the
site; DAS sites are the STR∪Gaussian union, so their manifolds already exist.
Intervened reps and patched output distributions ARE per mode (they differ).
The "__" delimiter is safe: site_ids and context/label/category strings contain
single underscores at most, never "__".
"""

import os

# Root for all artifacts. Override per seed for a sweep so runs don't overwrite
# each other, e.g.  OFFDIST_RESULTS_DIR=results/seed_3 python activation_manifold.py --seed 3
# The writers and the reader both read this env var, so they stay in sync within
# a process.
RESULTS_DIR = os.environ.get("OFFDIST_RESULTS_DIR", "results")

ACT_MANIFOLDS = os.path.join(RESULTS_DIR, "activation_manifolds")
INTERVENED_REPS = os.path.join(RESULTS_DIR, "intervened_reps")
OUTPUT_MANIFOLDS = os.path.join(RESULTS_DIR, "output_manifolds")
BEHAVIOUR_MANIFOLD = os.path.join(OUTPUT_MANIFOLDS, "behaviour_manifold")
PATCHED_DISTS = os.path.join(OUTPUT_MANIFOLDS, "patched_output_distributions")

DELIM = "__"


def label_category_dir(root, label, category):
    """{root}/{label}__{category} — for intervened_reps and patched_output_distributions."""
    return os.path.join(root, f"{label}{DELIM}{category}")


def site_context_dir(site_id, context):
    """{activation_manifolds}/{site_id}__{context} — one fitted manifold."""
    return os.path.join(ACT_MANIFOLDS, f"{site_id}{DELIM}{context}")


def iter_site_contexts(site_id, manifold_root=ACT_MANIFOLDS):
    """Yield the context strings that have a fitted manifold for this site, by
    scanning activation_manifolds/ for entries named {site_id}__{context}."""
    prefix = f"{site_id}{DELIM}"
    if not os.path.isdir(manifold_root):
        return
    for entry in sorted(os.listdir(manifold_root)):
        if entry.startswith(prefix) and os.path.isdir(os.path.join(manifold_root, entry)):
            yield entry[len(prefix):]
