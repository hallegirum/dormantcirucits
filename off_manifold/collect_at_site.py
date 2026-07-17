"""Collect residual-stream activations at ONE (layer, token_position) site and
check whether causal *contexts* separate there.

This is the go/no-go check before fitting contextual manifolds: if the contexts
do not separate at the site that encodes ``result`` (the site locate flags), then
the context-conditional manifold M_C coincides with the global manifold there and
the off-contextual half of the hypothesis is empty at that site.

Pipeline:
  1. load task + model, build the (layer, position) unit  (identity featurizer -> raw activations)
  2. build contextual prompts (conditioning on e.g. entity), pooled per context
  3. collect_features -> activations (N, d_model)
  4. standardize + PCA, then quantify separation:
        - linear-probe (logistic-regression, CV) accuracy for CONTEXT vs chance
        - same for the TARGET (result) as a sanity check that we are at the right site
        - silhouette + between/within variance ratio for context
  5. save a 2D PCA scatter coloured by context and by result, print a verdict

Run:
    uv run python -m experiments.off_manifold.collect_at_site \
        --model gpt2 --layer 6 --position last_token \
        --domain weekdays --conditioning entity --n-per-context 30

NOTE on interpretation: at the result site, `result` should be highly decodable.
CONTEXT decodability above chance means context is *also* present -> contexts
separate -> M_C != M_global. If context sits at chance, context has collapsed at
this site (expected near the resolved-answer site); move earlier and re-run.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from causalab.tasks.loader import load_task
from causalab.tasks.natural_domains_arithmetic.config import NaturalDomainConfig
from causalab.io.pipelines import load_pipeline
from causalab.runner.helpers import build_targets_for_grid
from causalab.neural.activations.collect import collect_features

from experiments.off_manifold import context_datasets as C


TASK_NAME = "natural_domains_arithmetic"


def _model_task_accuracy(pipeline, dataset, *, batch_size=16, n_gen_tokens=12,
                         target_variable="result", n_show=8):
    """Robust task accuracy for instruct models: generate a short continuation and
    check whether the expected answer word appears (case-insensitive substring).

    This is robust to instruct-model preamble ("The answer is ...") and multi-token
    answers, unlike single-token exact match (which scores 0 on chat models even
    when they are right). Also returns a few example generations so you can SEE
    what the model actually produces.
    """
    prev = pipeline.max_new_tokens
    pipeline.max_new_tokens = n_gen_tokens
    correct = 0
    examples: list[tuple[str, str, str]] = []
    try:
        for start in range(0, len(dataset), batch_size):
            batch = dataset[start:start + batch_size]
            out = pipeline.generate([ex["input"] for ex in batch])
            strings = out["string"]
            if isinstance(strings, str):
                strings = [strings]
            for ex, gen in zip(batch, strings):
                expected = str(ex["input"][target_variable])
                if expected.lower() in str(gen).lower():
                    correct += 1
                if len(examples) < n_show:
                    examples.append(
                        (ex["input"]["raw_input"], expected, str(gen).strip())
                    )
    finally:
        pipeline.max_new_tokens = prev
    total = len(dataset)
    return {
        "accuracy": correct / total if total else float("nan"),
        "correct": correct,
        "total": total,
        "examples": examples,
    }


def collect_site_activations(
    *,
    model_name: str,
    layers: list[int],
    position: str,
    domain: str,
    conditioning_vars: list[str],
    n_per_context: int,
    seed: int,
    dtype: str | None,
    device: str | None,
    batch_size: int,
    check_accuracy: int = 0,
    use_chat_template: bool = False,
    group_by=None,
):
    """Collect activations at MANY layers (same position) in ONE forward pass.

    Returns (acts_by_layer {layer: (N, d)}, context_labels[N], result_labels[N], task).
    """
    task = load_task(TASK_NAME, task_cfg=NaturalDomainConfig(domain_type=domain))
    pipeline = load_pipeline(
        model_name=model_name,
        task=task,
        max_new_tokens=1,          # we only need the forward pass, not generation
        device=device,
        dtype=dtype,
    )
    # Wrap prompts in the model's chat template (user msg + generation prompt).
    # For instruct models this is often the difference between answering and not.
    # It also shifts `last_token` to the position right before the assistant reply,
    # which is the correct answer-readout slot for a chat model.
    if use_chat_template:
        if getattr(pipeline.tokenizer, "chat_template", None) is None:
            raise ValueError(
                f"--use-chat-template set but {model_name!r}'s tokenizer has no "
                "chat_template (base models don't). Use an -Instruct/-Chat model."
            )
        pipeline.use_chat_template = True

    # One residual-stream unit per layer (same position); identity featurizer =>
    # raw acts. Passing all units to collect_features collects every layer in a
    # SINGLE pass over the data (the hooks all fire on the same forward).
    targets, _positions = build_targets_for_grid(pipeline, task, layers, [position])
    units = [tgt.flatten()[0] for tgt in targets.values()]

    # Contextual prompts: {context_key: [traces]}, member-pooled per context.
    ctx_prompts = C.create_contextual_prompts(
        task.causal_model,
        conditioning_vars,
        n_per_context=n_per_context,
        seed=seed,
        group_by=group_by,
    )

    # Flatten into a collect dataset, tracking context + result label per row.
    dataset: list[dict] = []
    context_labels: list = []
    result_labels: list = []
    for ctx_key, traces in ctx_prompts.items():
        for t in traces:
            dataset.append({"input": t, "counterfactual_inputs": [t]})
            context_labels.append(ctx_key if len(ctx_key) > 1 else ctx_key[0])
            result_labels.append(t["result"])

    feats = collect_features(dataset, pipeline, units, batch_size=batch_size)
    acts_by_layer = {
        u.layer: feats[u.id].detach().float().cpu().numpy() for u in units
    }

    # Direct model TASK accuracy: does the model actually GENERATE the right
    # weekday? This is the honest "can the model do it" measure — distinct from,
    # and not implied by, linear-probe decodability.
    task_acc = None
    if check_accuracy > 0:
        task_acc = _model_task_accuracy(
            pipeline, dataset[:check_accuracy], batch_size=batch_size
        )

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        acts_by_layer,
        np.array(context_labels, dtype=object),
        np.array(result_labels, dtype=object),
        task,
        task_acc,
    )


def _probe_accuracy(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float]:
    """Cross-validated logistic-regression accuracy and its chance level.

    Chance = frequency of the majority class (a stricter baseline than 1/n_classes
    when classes are unbalanced).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    X = np.asarray(X, dtype=np.float64)  # float32 overflows on outlier-heavy models
    classes, counts = np.unique(y, return_counts=True)
    chance = counts.max() / counts.sum()
    if len(classes) < 2:
        return float("nan"), chance
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, multi_class="auto"),
    )
    # min class count bounds the number of CV folds.
    n_folds = int(min(5, counts.min()))
    if n_folds < 2:
        return float("nan"), chance
    # When a class is linearly separable (e.g. entity, which appears verbatim in
    # the prompt), logistic weights diverge and intermediate exp() overflows. This
    # is benign — the fitted classifier and its CV accuracy are still valid — so we
    # silence the resulting numerical warnings rather than distort the features.
    import warnings as _w

    with np.errstate(all="ignore"), _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        scores = cross_val_score(clf, X, y, cv=n_folds)
    return float(scores.mean()), float(chance)


def _between_within_ratio(X: np.ndarray, labels: np.ndarray) -> float:
    """Trace(between-class scatter) / Trace(within-class scatter) — Fisher-style.

    >1 means classes are pulled apart relative to their internal spread.
    """
    gmean = X.mean(axis=0)
    sb = sw = 0.0
    for c in np.unique(labels):
        Xc = X[labels == c]
        mu = Xc.mean(axis=0)
        sb += len(Xc) * float(((mu - gmean) ** 2).sum())
        sw += float(((Xc - mu) ** 2).sum())
    return sb / sw if sw > 0 else float("inf")


def _residualize_by(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Subtract the per-group mean, leaving only within-group variation.

    Residualizing by ``result`` removes everything the result axis explains, so a
    subsequent context-probe measures context structure *orthogonal to result* —
    i.e. whether M_C genuinely departs from the global (result) manifold rather
    than context and result being collinear.
    """
    Xr = X.copy()
    for c in np.unique(labels):
        m = labels == c
        Xr[m] = X[m] - X[m].mean(axis=0)
    return Xr


# --------------------------------------------------------------------------- #
# Permutation-based significance                                              #
# --------------------------------------------------------------------------- #
def _permute_labels(labels: np.ndarray, rng, strata: np.ndarray | None = None) -> np.ndarray:
    """Shuffle labels. If ``strata`` is given, shuffle only WITHIN each stratum.

    Stratified shuffling is used for the result-controlled context test: permuting
    context labels within each result class keeps the empirical context x result
    joint distribution intact, so the null tests "context structure beyond result"
    rather than the (trivial) context-result correlation.
    """
    out = np.array(labels, dtype=object).copy()
    if strata is None:
        return out[rng.permutation(len(out))]
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        out[idx] = out[idx][rng.permutation(len(idx))]
    return out


def _perm_test(stat_fn, X, labels, *, n_perm, rng, strata=None):
    """Permutation test for a separability statistic (higher = more separated).

    Returns dict {value, null_mean, effect, p}. ``effect`` = observed - null_mean
    is the calibrated margin (replaces the hand-picked "margin above chance");
    ``p`` = P(null >= observed) with +1 smoothing. The null distribution absorbs
    sample size, class count/balance, dimensionality and CV variance, so no
    arbitrary threshold is needed — significance is judged against this null.
    """
    observed = stat_fn(X, labels)
    if not np.isfinite(observed) or n_perm <= 0:
        return {"value": float(observed), "null_mean": float("nan"),
                "effect": float("nan"), "p": float("nan")}
    null = np.array([stat_fn(X, _permute_labels(labels, rng, strata)) for _ in range(n_perm)])
    null = null[np.isfinite(null)]
    null_mean = float(null.mean()) if null.size else float("nan")
    p = (float((null >= observed).sum()) + 1.0) / (null.size + 1.0) if null.size else float("nan")
    return {"value": float(observed), "null_mean": null_mean,
            "effect": float(observed - null_mean), "p": float(p)}


def analyze_and_plot(acts, context_labels, result_labels, *, out_dir, pca_dim, seed, title,
                     fig_name="pca_context_vs_result.png", n_probe=100, n_perm=200):
    import matplotlib
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    os.makedirs(out_dir, exist_ok=True)
    # float64 + finite-guard: some models have extreme outlier dims.
    acts = np.nan_to_num(np.asarray(acts, dtype=np.float64), posinf=0.0, neginf=0.0)
    # Drop dead / degenerate dimensions (near-zero variance) BEFORE standardizing:
    # dividing by a subnormal std is what reintroduces huge values and overflows
    # the downstream SVD / probe. These dims carry no class signal anyway.
    std = acts.std(axis=0)
    keep = std > 1e-8 * float(std.max())
    acts = acts[:, keep] if keep.any() else acts
    Xs = StandardScaler().fit_transform(acts)
    Xs = np.clip(Xs, -50.0, 50.0)

    # CRITICAL: probe on a PCA-reduced representation, not the raw d_model vector.
    # Raw activations are ill-conditioned (near-constant / outlier dims), which
    # makes the linear probe overflow and return a degenerate, layer-independent
    # accuracy (the "identical 0.514 across all layers" symptom). Reducing to
    # n_probe well-conditioned components fixes this and is standard for probing.
    n_comp = int(min(n_probe, Xs.shape[0] - 1, Xs.shape[1]))
    pca = PCA(n_components=n_comp).fit(Xs)
    Xp = pca.transform(Xs)                 # (N, n_comp)
    # Standardize + clip the probe space so the linear solver never sees extreme
    # values (low-variance PCA comps blow up under the probe's internal scaler on
    # pathological models). Bounded, well-conditioned features -> no solver overflow.
    Xp = np.clip(StandardScaler().fit_transform(Xp), -10.0, 10.0)
    Z = Xp[:, : min(pca_dim, Xp.shape[1])]  # first dims — plot / geometry space

    context_labels = np.asarray(context_labels, dtype=object)
    result_labels = np.asarray(result_labels, dtype=object)
    perm_rng = np.random.default_rng(seed)

    # Statistic functions (each returns a single scalar; higher = more separated).
    def _probe_stat(X, y):
        return _probe_accuracy(X, y, seed)[0]

    def _sil_stat(X, y):
        return silhouette_score(X, y) if len(np.unique(y)) > 1 else float("nan")

    # Result-controlled residual space (for the "context beyond result" test).
    Xr = _residualize_by(Xp, result_labels)

    # --- permutation-tested separability metrics ---
    # result probe: sanity that the variable is encoded (plain permutation).
    res = _perm_test(_probe_stat, Xp, result_labels, n_perm=n_perm, rng=perm_rng)
    # raw context probe: is context present at all (partly trivial; plain perm).
    ctx = _perm_test(_probe_stat, Xp, context_labels, n_perm=n_perm, rng=perm_rng)
    # context BEYOND result: the key test. Stratified permutation within result so
    # the null preserves the context x result joint (tests structure not collinearity).
    ctxb = _perm_test(_probe_stat, Xr, context_labels, n_perm=n_perm, rng=perm_rng,
                      strata=result_labels)
    # geometric separability of context clusters (plain permutation).
    sil = _perm_test(_sil_stat, Z, context_labels, n_perm=n_perm, rng=perm_rng)
    fisher = _perm_test(_between_within_ratio, Z, context_labels, n_perm=n_perm, rng=perm_rng)

    # --- 2D PCA scatter, coloured by context and by result ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, labels, name in (
        (axes[0], context_labels, "context"),
        (axes[1], result_labels, "result"),
    ):
        uniq = list(dict.fromkeys(labels.tolist()))
        cmap = matplotlib.colormaps.get_cmap(
            "tab10" if len(uniq) <= 10 else "tab20"
        ).resampled(len(uniq))
        for i, u in enumerate(uniq):
            m = labels == u
            ax.scatter(Z[m, 0], Z[m, 1], s=18, color=cmap(i), label=str(u), alpha=0.75)
        ax.set_title(f"coloured by {name}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(fontsize=7, markerscale=1.2, ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    fig_path = os.path.join(out_dir, fig_name)
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    metrics = {
        "n_examples": int(acts.shape[0]),
        "d_model": int(acts.shape[1]),
        "pca_var_explained": float(pca.explained_variance_ratio_[:pca_dim].sum()),
        # each of these is {value, null_mean, effect, p}
        "result_probe": res,
        "context_probe": ctx,
        "context_beyond_result": ctxb,
        "context_silhouette": sil,
        "context_fisher": fisher,
        "n_perm": int(n_perm),
        "alpha": 0.05,
        "figure": fig_path,
    }
    return metrics


def _sig(stat: dict, alpha: float = 0.05) -> bool:
    """Significant if the observed statistic exceeds its permutation null at alpha."""
    return np.isfinite(stat.get("p", np.nan)) and stat["p"] < alpha and stat.get("effect", 0) > 0


def _verdict(m: dict) -> str:
    a = m["alpha"]
    if not np.isfinite(m["context_probe"]["p"]):
        return "INCONCLUSIVE (permutation test not run / one class)"
    beyond_sig = _sig(m["context_beyond_result"], a)
    raw_sig = _sig(m["context_probe"], a)
    if beyond_sig:
        return (f"CONTEXTS SEPARATE BEYOND RESULT (p={m['context_beyond_result']['p']:.3f}, "
                f"effect={m['context_beyond_result']['effect']:+.3f}) — off-contextual testable & distinct")
    if raw_sig:
        return ("context significant but NOT beyond result — off-contextual axis "
                "likely collinear with result here")
    return "CONTEXTS COLLAPSED — no significant context structure; move earlier"


def _short_verdict(m: dict) -> str:
    """Compact per-layer flag for the sweep table."""
    if not np.isfinite(m["context_probe"]["p"]):
        return "n/a"
    if _sig(m["context_beyond_result"], m["alpha"]):
        return "SEP"        # separates beyond result (significant)
    if _sig(m["context_probe"], m["alpha"]):
        return "collinear"  # context significant but not beyond result
    return "COLLAPSE"


def plot_sweep(rows: list[tuple[int, dict]], out_dir: str, title: str) -> str:
    """Summary curve: result-probe and context-beyond-result probe vs layer.

    This is the site-selection read: pick a layer where the result curve is HIGH
    (variable encoded) AND the context-beyond-result curve is above chance
    (context still live) — that's where off-contextual is both meaningful and
    testable.
    """
    import matplotlib.pyplot as plt

    layers = [r[0] for r in rows]
    res = [r[1]["result_probe"]["value"] for r in rows]
    ctxb = [r[1]["context_beyond_result"]["value"] for r in rows]
    res_null = [r[1]["result_probe"]["null_mean"] for r in rows]
    ctxb_null = [r[1]["context_beyond_result"]["null_mean"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, res, "o-", color="C0", label="result probe")
    ax.plot(layers, ctxb, "s-", color="C1", label="context-beyond-result probe")
    ax.plot(layers, res_null, "--", color="C0", lw=1, alpha=0.5, label="result null")
    ax.plot(layers, ctxb_null, ":", color="C1", lw=1, alpha=0.5, label="context null")
    ax.set_xlabel("layer"); ax.set_ylabel("CV probe accuracy"); ax.set_ylim(0, 1.02)
    ax.set_title(title + "  (obs vs permutation null)"); ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "layer_sweep_summary.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B",
                    help="HF model id (gated Llama needs `huggingface-cli login`)")
    ap.add_argument("--layers", default="28",
                    help="comma-separated layers to sweep, e.g. 8,16,20,24,28")
    ap.add_argument("--position", default="last_token", choices=["last_token", "entity", "number"])
    ap.add_argument("--domain", default="weekdays")
    ap.add_argument("--conditioning", default="entity", help="comma-separated conditioning vars")
    ap.add_argument("--n-per-context", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--pca-dim", type=int, default=3)
    ap.add_argument("--check-accuracy", type=int, default=120,
                    help="measure model TASK accuracy on this many prompts (0 to skip)")
    ap.add_argument("--use-chat-template", action="store_true",
                    help="wrap prompts in the model's chat template (for -Instruct/-Chat models)")
    ap.add_argument("--n-perm", type=int, default=100,
                    help="permutation-null resamples for significance (0 to skip; "
                         "the probe tests are the cost driver — lower for speed, "
                         "raise for finer p-values, min p = 1/(n_perm+1))")
    ap.add_argument("--out-dir", default="experiments/off_manifold/_out/collect_site")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    conditioning_vars = [v.strip() for v in args.conditioning.split(",") if v.strip()]

    acts_by_layer, ctx, res, _task, task_acc = collect_site_activations(
        model_name=args.model,
        layers=layers,
        position=args.position,
        domain=args.domain,
        conditioning_vars=conditioning_vars,
        n_per_context=args.n_per_context,
        seed=args.seed,
        dtype=args.dtype,
        device=args.device,
        batch_size=args.batch_size,
        check_accuracy=args.check_accuracy,
        use_chat_template=args.use_chat_template,
    )

    # Silence benign numerical warnings (ill-conditioned intermediate matmuls in
    # PCA / the scaler on large-magnitude activations). The reported statistics are
    # computed on reduced, well-conditioned features and are unaffected.
    import warnings as _warnings

    with _warnings.catch_warnings(), np.errstate(all="ignore"):
        _warnings.simplefilter("ignore", RuntimeWarning)
        rows: list[tuple[int, dict]] = []
        for layer in layers:
            title = f"{args.model} L{layer}/{args.position} | {args.domain} | context={args.conditioning}"
            metrics = analyze_and_plot(
                acts_by_layer[layer], ctx, res,
                out_dir=args.out_dir,
                pca_dim=args.pca_dim,
                seed=args.seed,
                title=title,
                fig_name=f"pca_L{layer}.png",
                n_perm=args.n_perm,
            )
            rows.append((layer, metrics))

    # --- sweep table (observed value + permutation p-value per metric) ---
    print("\n" + "=" * 92)
    print(f"{args.model} | {args.domain} | position={args.position} | context={args.conditioning}"
          f"  (n_perm={args.n_perm}, sig@p<0.05)")
    print("=" * 92)
    hdr = (f"{'layer':>5} | {'result p':>13} | {'ctx>result eff (p)':>22} | "
           f"{'silh (p)':>13} | {'fisher (p)':>13} | verdict")
    print(hdr); print("-" * len(hdr))
    for layer, m in rows:
        r, cb, s, f = (m["result_probe"], m["context_beyond_result"],
                       m["context_silhouette"], m["context_fisher"])
        print(f"{layer:>5} | "
              f"{r['value']:.2f} (p={r['p']:.3f}) | "
              f"{cb['value']:.2f} {cb['effect']:+.2f} (p={cb['p']:.3f}) | "
              f"{s['value']:+.2f} (p={s['p']:.3f}) | "
              f"{f['value']:.2f} (p={f['p']:.3f}) | "
              f"{_short_verdict(m)}")

    if len(layers) > 1:
        sweep_title = f"{args.model} | {args.domain} | position={args.position}"
        path = plot_sweep(rows, args.out_dir, sweep_title)
        print(f"  sweep summary figure: {path}")

    # --- model TASK accuracy LAST, so it is never buried in warnings/progress ---
    print("\n" + "=" * 92)
    if task_acc is not None:
        print(f"  MODEL TASK ACCURACY: {task_acc['accuracy']:.3f} "
              f"({task_acc['correct']}/{task_acc['total']} generations contain the correct answer)")
        print("  (this — not the probe accuracy — is whether the model can DO the task)")
        if task_acc.get("examples"):
            print("\n  example generations  [prompt -> expected | model output]:")
            for raw, exp, gen in task_acc["examples"]:
                q = raw.replace("\n", " ")[:46]
                print(f"    {q!r:50s} -> {exp:10s} | {gen[:40]!r}")
    else:
        print("  MODEL TASK ACCURACY: not computed (re-run with --check-accuracy 120)")
    print("=" * 92 + "\n")


if __name__ == "__main__":
    main()
