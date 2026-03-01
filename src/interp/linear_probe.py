"""Linear probes for mechanistic interpretability.

Workflow:
1. store = ActivationStore(activations_dir, run_json)
2. labels = store.get_labels(lambda r: int(r["monitor_results"]["keyword"]["matched"]))
3. For each layer: X = store.load_layer(layer_idx)
4. Train sklearn LogisticRegression(max_iter=1000, C=1.0) on X, y with 5-fold CV
5. Plot layer-by-layer accuracy curve — identifies at which layer the model encodes "obfuscation"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.interp.activation_store import ActivationStore


def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
):
    """Train a logistic regression probe on a single layer's activations.

    Args:
        X: [n_prompts, d_model] activation matrix
        y: [n_prompts] integer labels
        C: Regularization strength (inverse; smaller = stronger regularization)
        max_iter: Maximum iterations for solver convergence
        random_state: RNG seed for reproducibility

    Returns:
        Fitted sklearn LogisticRegression model.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)),
    ])
    pipe.fit(X, y)
    return pipe


def probe_all_layers(
    store: "ActivationStore",
    labels: np.ndarray,
    n_layers: int | None = None,
    cv_folds: int = 5,
    C: float = 1.0,
    random_state: int = 42,
) -> dict[int, float]:
    """Train a probe on every layer and return CV accuracy per layer.

    Args:
        store: ActivationStore with loaded activations
        labels: Binary label array from store.get_labels(...)
        n_layers: Number of layers to probe (auto-detected from layer_0 presence if None)
        cv_folds: Number of cross-validation folds
        C: Logistic regression regularization parameter
        random_state: RNG seed

    Returns:
        dict mapping layer_idx -> mean CV accuracy
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score

    if n_layers is None:
        # Detect from first prompt's available keys
        sample = store.load(store.prompt_ids[0])
        layer_keys = [k for k in sample if k.startswith("layer_")]
        n_layers = len(layer_keys)

    layer_accs: dict[int, float] = {}
    for layer_idx in range(n_layers):
        X = store.load_layer(layer_idx)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=1000, random_state=random_state)),
        ])
        scores = cross_val_score(pipe, X, labels, cv=cv_folds, scoring="accuracy")
        layer_accs[layer_idx] = float(scores.mean())

    return layer_accs


def plot_layer_accuracy(layer_accs: dict[int, float]) -> None:
    """Plot layer-by-layer probe accuracy to identify when obfuscation is encoded.

    Requires matplotlib. Shows a line plot with layer index on x-axis and mean CV
    accuracy on y-axis. The peak layer is annotated.
    """
    import matplotlib.pyplot as plt

    layers = sorted(layer_accs.keys())
    accs = [layer_accs[l] for l in layers]
    peak_layer = layers[int(np.argmax(accs))]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, accs, marker="o", linewidth=1.5)
    ax.axvline(peak_layer, color="red", linestyle="--", alpha=0.6, label=f"Peak: layer {peak_layer}")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Chance (binary)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("CV Accuracy")
    ax.set_title("Linear Probe Accuracy by Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
