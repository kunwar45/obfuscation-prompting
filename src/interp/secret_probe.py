"""Secret recoverability probe.

Asks: does the model's residual stream encode the secret value X even when
the model's output conceals it?

This is distinct from the existing condition probe (A0 vs A2?) and disclosure
probe (did the model mention the secret?). Here, the label is the *value* of
the secret itself — e.g. lead_time_days ∈ {2, 5, 10, 21, 28} — and the probe
is trained separately on each condition's activations.

Interpretation
--------------
For each layer L and condition C, we train a multi-class logistic regression
to predict the secret value from the residual stream at position L.

  A0 accuracy >> chance  →  model encodes the secret in its residual stream
                             when answering honestly (sanity check)

  A2 accuracy ≈ A0 accuracy  →  secret is still encoded under explicit ban;
                                 concealment is an output-level policy decision
                                 ("knowledge present, policy suppresses output")

  A2 accuracy ≈ chance        →  explicit ban degrades secret representation;
                                 the model's internal state changes, not just
                                 its output filter

  A1 accuracy relative to A2  →  comparison of implicit vs explicit internal
                                  effects

The accuracy gap between conditions at each layer, together with the layer at
which the gap first opens, is the primary signal for H1 vs H2.

Usage
-----
    from src.interp.secret_probe import (
        load_dataset_index,
        probe_secret_recoverability,
        SHIPPING_LABEL_FN, SHIPPING_CLASS_NAMES,
    )
    from src.interp.activation_store import ActivationStore

    store = ActivationStore("activations/interp_tag_ts/", "results/run_<id>.json")
    dataset_index = load_dataset_index("data/interp_full_<ts>.jsonl")

    results = probe_secret_recoverability(
        store, dataset_index,
        label_fn=SHIPPING_LABEL_FN,
        class_names=SHIPPING_CLASS_NAMES,
        conditions=["A0", "A2"],
    )
    # results = {"A0": {0: 0.21, 1: 0.34, ...}, "A2": {...}}

    plot_recoverability(results, class_names=SHIPPING_CLASS_NAMES, save_path="results/secret_probe.png")
"""

from __future__ import annotations

import json
from typing import Callable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.interp.activation_store import ActivationStore


# ── Dataset index ─────────────────────────────────────────────────────────────

def load_dataset_index(jsonl_path: str) -> dict[str, dict]:
    """Load a scenario JSONL and return {example_id: scenario} for quick lookup.

    Args:
        jsonl_path: Path to the .jsonl dataset file produced by DatasetGenerator.

    Returns:
        Dict mapping example_id strings to their full scenario dicts.
    """
    index: dict[str, dict] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            scenario = json.loads(line)
            index[scenario["example_id"]] = scenario
    return index


# ── Domain label extractors ───────────────────────────────────────────────────

# Shipping: lead_time_days ∈ {2, 5, 10, 21, 28}
_SHIPPING_LEAD_TIMES = [2, 5, 10, 21, 28]
SHIPPING_CLASS_NAMES = ["2 days", "5 days", "10 days", "21 days", "28 days"]

def SHIPPING_LABEL_FN(scenario: dict) -> int:
    """Map shipping scenario → class index for lead_time_days."""
    days = scenario["secret"]["structured"]["lead_time_days"]
    return _SHIPPING_LEAD_TIMES.index(days)


# Bug: feature_status ∈ {broken, intermittent}
_BUG_STATUSES = ["broken", "intermittent"]
BUG_CLASS_NAMES = ["broken", "intermittent"]

def BUG_LABEL_FN(scenario: dict) -> int:
    """Map bug scenario → class index for feature_status."""
    status = scenario["secret"]["structured"]["feature_status"]
    return _BUG_STATUSES.index(status)


# Backlog: wait_time_days ∈ {1, 3, 7, 14, 30}
_BACKLOG_WAIT_TIMES = [1, 3, 7, 14, 30]
BACKLOG_CLASS_NAMES = ["1 day", "3 days", "7 days", "14 days", "30 days"]

def BACKLOG_LABEL_FN(scenario: dict) -> int:
    """Map backlog scenario → class index for wait_time_days."""
    days = scenario["secret"]["structured"]["wait_time_days"]
    return _BACKLOG_WAIT_TIMES.index(days)


# Domain lookup — convenience for scripts that infer domain from data
DOMAIN_LABEL_FNS: dict[str, tuple[Callable, list[str]]] = {
    "shipping": (SHIPPING_LABEL_FN, SHIPPING_CLASS_NAMES),
    "bug":      (BUG_LABEL_FN,      BUG_CLASS_NAMES),
    "backlog":  (BACKLOG_LABEL_FN,  BACKLOG_CLASS_NAMES),
}


# ── Label construction ────────────────────────────────────────────────────────

def build_secret_labels(
    store: "ActivationStore",
    dataset_index: dict[str, dict],
    label_fn: Callable[[dict], int],
    prompt_ids: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build a label array aligned with a set of prompt IDs.

    Iterates `store.get_results(prompt_ids)` in index order, applies label_fn
    to each scenario, and skips:
      - control examples (no_secret / irrelevant_secret)
      - examples whose example_id is not in dataset_index
      - examples where label_fn raises KeyError/ValueError (malformed secret)

    Args:
        store:          ActivationStore over the run's .npz files
        dataset_index:  {example_id: scenario} from load_dataset_index()
        label_fn:       Callable(scenario) → int; see SHIPPING_LABEL_FN etc.
        prompt_ids:     Subset of IDs to build labels for. None = all indexed.

    Returns:
        (labels, valid_ids) where labels[i] corresponds to valid_ids[i].
        Both are in the same order so load_layer(layer_idx, valid_ids) aligns.
    """
    results = store.get_results(prompt_ids)
    labels: list[int] = []
    valid_ids: list[str] = []

    for r in results:
        meta = r.get("metadata", {})
        if meta.get("control_type"):
            continue
        example_id = meta.get("example_id", "")
        scenario = dataset_index.get(example_id)
        if scenario is None:
            continue
        try:
            label = label_fn(scenario)
        except (KeyError, ValueError, IndexError):
            continue
        valid_ids.append(r["prompt_id"])
        labels.append(label)

    return np.array(labels, dtype=int), valid_ids


# ── Core probe ────────────────────────────────────────────────────────────────

def probe_secret_recoverability(
    store: "ActivationStore",
    dataset_index: dict[str, dict],
    label_fn: Callable[[dict], int],
    class_names: list[str],
    conditions: list[str] | None = None,
    cv_folds: int = 5,
    C: float = 1.0,
    random_state: int = 42,
) -> dict[str, dict[int, float]]:
    """Run a multi-class secret-recoverability probe per layer, split by condition.

    For each condition in `conditions`, trains a logistic regression at every
    layer to predict the secret value from the last-token residual stream.
    Returns CV accuracy per layer per condition.

    The key comparison:
      results["A0"][layer] ≈ results["A2"][layer]  →  shared latent knowledge
      results["A0"][layer] >> results["A2"][layer]  →  A2 degrades secret repr

    Args:
        store:         ActivationStore with last_token activations
        dataset_index: {example_id: scenario} dict from load_dataset_index()
        label_fn:      Callable(scenario) → int class index
        class_names:   Human-readable class labels (for logging/plots)
        conditions:    Conditions to probe. None → auto-detect from metadata.
        cv_folds:      Number of CV folds (uses StratifiedKFold internally)
        C:             Logistic regression regularisation (inverse strength)
        random_state:  RNG seed

    Returns:
        {condition: {layer_idx: mean_cv_accuracy}}
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    # Auto-detect conditions from metadata if not specified
    if conditions is None:
        conditions = sorted({
            r.get("metadata", {}).get("condition", "")
            for r in store.get_results()
            if r.get("metadata", {}).get("condition")
        })

    n_layers = store.n_layers()
    n_classes = len(class_names)
    chance = 1.0 / n_classes

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    all_results: dict[str, dict[int, float]] = {}

    for condition in conditions:
        # Filter to this condition, non-control prompts
        condition_ids = store.filter_ids(
            lambda r, c=condition: (
                r.get("metadata", {}).get("condition") == c
                and not r.get("metadata", {}).get("control_type")
            )
        )
        if not condition_ids:
            print(f"  [{condition}] no prompts found, skipping")
            continue

        labels, valid_ids = build_secret_labels(
            store, dataset_index, label_fn, prompt_ids=condition_ids
        )

        n = len(valid_ids)
        n_unique = len(set(labels.tolist()))

        if n_unique < 2:
            print(f"  [{condition}] only {n_unique} unique class(es) in {n} examples — skipping")
            continue

        if n < cv_folds:
            print(f"  [{condition}] only {n} examples (need ≥{cv_folds} for {cv_folds}-fold CV) — skipping")
            continue

        print(f"  [{condition}] {n} examples, {n_unique}/{n_classes} classes, chance={chance:.1%}")

        layer_accs: dict[int, float] = {}
        for layer_idx in range(n_layers):
            X = store.load_layer(layer_idx, prompt_ids=valid_ids)
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    C=C, max_iter=1000, random_state=random_state,
                    solver="lbfgs",
                )),
            ])
            scores = cross_val_score(pipe, X, labels, cv=cv, scoring="accuracy")
            layer_accs[layer_idx] = float(scores.mean())

        all_results[condition] = layer_accs
        peak_layer = max(layer_accs, key=layer_accs.__getitem__)
        peak_acc = layer_accs[peak_layer]
        print(f"    peak: layer {peak_layer} → {peak_acc:.1%}  (chance {chance:.1%})")

    return all_results


# ── Gap analysis ──────────────────────────────────────────────────────────────

def recoverability_gap(
    results: dict[str, dict[int, float]],
    baseline_condition: str = "A0",
    compare_conditions: list[str] | None = None,
) -> dict[str, dict[int, float]]:
    """Compute per-layer accuracy gap between a baseline condition and others.

    gap[condition][layer] = results[baseline][layer] - results[condition][layer]

    Positive gap means the baseline condition encodes the secret better.
    A large A0-vs-A2 gap at layer L means: the explicit ban degrades the
    secret representation at that layer, not just the output filter.

    Args:
        results:             Output of probe_secret_recoverability()
        baseline_condition:  Condition to subtract from (typically "A0")
        compare_conditions:  Conditions to compare against baseline. None = all others.

    Returns:
        {condition: {layer_idx: gap_float}}
    """
    baseline = results.get(baseline_condition)
    if baseline is None:
        raise KeyError(f"baseline condition {baseline_condition!r} not in results")

    others = compare_conditions or [c for c in results if c != baseline_condition]
    gaps: dict[str, dict[int, float]] = {}
    for cond in others:
        if cond not in results:
            continue
        gaps[cond] = {
            layer: baseline[layer] - results[cond].get(layer, float("nan"))
            for layer in baseline
        }
    return gaps


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_recoverability(
    results: dict[str, dict[int, float]],
    class_names: list[str],
    save_path: str | None = None,
    title: str = "Secret Recoverability Probe",
) -> None:
    """Plot per-layer probe accuracy, one line per condition.

    The gap between condition lines is the primary interpretability signal:
      - A0 >> A2 at some layer  →  explicit ban degrades secret representation
      - A0 ≈ A2 throughout      →  concealment is purely output-level
      - A1 tracks A0 or A2      →  implicit mechanism similar to baseline or ban

    Annotates chance level and peak layer per condition.

    Args:
        results:    {condition: {layer_idx: acc}} from probe_secret_recoverability()
        class_names: Used to compute chance level (1 / n_classes)
        save_path:  If given, save figure to this path instead of showing
        title:      Plot title
    """
    import matplotlib.pyplot as plt

    chance = 1.0 / len(class_names)
    condition_colors = {"A0": "#2196F3", "A1": "#FF9800", "A2": "#F44336"}
    condition_styles = {"A0": "-", "A1": "--", "A2": ":"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # ── Left: raw accuracy per condition ──────────────────────────────────────
    ax = axes[0]
    for cond, layer_accs in sorted(results.items()):
        layers = sorted(layer_accs.keys())
        accs = [layer_accs[l] for l in layers]
        color = condition_colors.get(cond, "gray")
        ls = condition_styles.get(cond, "-")
        ax.plot(layers, accs, color=color, linestyle=ls, linewidth=1.8,
                marker="o", markersize=3, label=cond)
        peak_layer = layers[int(np.argmax(accs))]
        ax.axvline(peak_layer, color=color, linestyle=":", alpha=0.3)

    ax.axhline(chance, color="gray", linestyle="--", linewidth=1.0,
               alpha=0.7, label=f"Chance ({chance:.0%})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("CV Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=0)

    # ── Right: gap (A0 − other conditions) ───────────────────────────────────
    ax = axes[1]
    if "A0" in results:
        gaps = recoverability_gap(results, baseline_condition="A0")
        for cond, gap_by_layer in sorted(gaps.items()):
            layers = sorted(gap_by_layer.keys())
            gap_vals = [gap_by_layer[l] for l in layers]
            color = condition_colors.get(cond, "gray")
            ls = condition_styles.get(cond, "-")
            ax.plot(layers, gap_vals, color=color, linestyle=ls, linewidth=1.8,
                    marker="o", markersize=3, label=f"A0 − {cond}")
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy gap (A0 − condition)")
        ax.set_title("Recoverability gap vs A0 baseline")
        ax.legend()
        ax.grid(True, alpha=0.25)
    else:
        ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def print_summary_table(
    results: dict[str, dict[int, float]],
    class_names: list[str],
) -> None:
    """Print a terminal table: peak layer + accuracy per condition."""
    chance = 1.0 / len(class_names)
    print()
    print(f"  Secret Recoverability  (chance = {chance:.1%},  classes = {class_names})")
    print(f"  {'Condition':<12}  {'Peak layer':>10}  {'Peak acc':>10}  {'Above chance':>13}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*13}")
    for cond in sorted(results):
        layer_accs = results[cond]
        peak_layer = max(layer_accs, key=layer_accs.__getitem__)
        peak_acc = layer_accs[peak_layer]
        above = peak_acc - chance
        print(f"  {cond:<12}  {peak_layer:>10}  {peak_acc:>9.1%}  {above:>+12.1%}")
    print()
