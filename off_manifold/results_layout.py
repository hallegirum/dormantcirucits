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
# Path-finding outputs ({method}_top_sites.json, site_provenance.json). Kept OUT
# of intervened_reps/ because path-finding runs before the data is split into
# contexts — mixing context-less ranking files into the context-suffixed reps
# dirs made the analysis's dir-scan mis-parse them.
SITE_RANKINGS = os.path.join(RESULTS_DIR, "site_rankings")

DELIM = "__"


def site_rankings_dir(label, category):
    """{site_rankings}/{label}__{category} — path-finding rankings (no context;
    site selection is per (label, category), shared across contexts/modes)."""
    return os.path.join(SITE_RANKINGS, f"{label}{DELIM}{category}")


def label_category_dir(root, label, category, context=None):
    """{root}/{label}__{category}[__{context}] — intervened_reps / patched dists.

    With `context`, the reps/dists for each manifold context (e.g. "wrap" /
    "no_wrap") live in their own dir, so each file holds ONLY that context's
    examples and can be compared against that context's own manifold
    (the "respective" comparison). Without it, the old flat layout.
    """
    name = f"{label}{DELIM}{category}"
    if context is not None:
        name += f"{DELIM}{context}"
    return os.path.join(root, name)


def iter_label_category_contexts(root):
    """Yield (label, category, context) parsed from dir names under `root`, but
    ONLY for dirs that actually hold data (a .safetensors file). This ignores any
    stray ranking/metadata-only dirs so they can't be mis-parsed as a context."""
    import glob
    if not os.path.isdir(root):
        return
    for entry in sorted(os.listdir(root)):
        d = os.path.join(root, entry)
        if not os.path.isdir(d) or not glob.glob(os.path.join(d, "*.safetensors")):
            continue
        parts = entry.split(DELIM)
        if len(parts) == 3:
            yield parts[0], parts[1], parts[2]
        elif len(parts) == 2:
            yield parts[0], parts[1], None


INTERACTION = os.path.join(RESULTS_DIR, "interaction")


def interaction_dir(component_id, strategy, push_site, direction="input", alpha=None):
    """{results}/interaction/{component_id}/{strategy}/{push_site}/{direction}[/a{alpha}]
    — one INT scan's row (int.json) for this component. Component-major so a
    component's strategies/directions list side by side (the borrowed-circuit /
    calibration view). ``direction`` = '{base}-{target}' for a directed scan (or
    'input'/'self' legacy); ``alpha`` in the path so the strength sweep does NOT
    overwrite itself, and ``direction`` so (base,target) cells don't collide — the
    two bugs that flattened the earlier runs. push_site: a site_id (mediation),
    'self' (self-push), or 'input' (swap). RESULTS_DIR is case-scoped."""
    safe = lambda s: str(s).replace(os.sep, "-")
    parts = [INTERACTION, str(component_id), str(strategy), safe(push_site), safe(direction)]
    if alpha is not None:
        parts.append(f"a{alpha:g}")
    return os.path.join(*parts)


def dormancy_dir():
    """{results}/dormancy — per-case circuit.json (true membership + top-k cutoff) and the
    dormancy_analysis output. RESULTS_DIR is per-case (OFFDIST_RESULTS_DIR=results/{case}),
    so this is case-scoped and won't collide across models."""
    return os.path.join(RESULTS_DIR, "dormancy")


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
