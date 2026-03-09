"""Interpretability plots for the concealment experiment.

Produces up to five figures saved to out_dir:
  1. probe_accuracy.png      — layer-by-layer CV accuracy (condition + disclosure probes)
  2. logit_lens.png          — P(secret_token | layer) for A0 vs A2 matched pairs
  3. activation_pca.png      — 2-D PCA of last-layer residual stream, coloured by condition
  4. probe_heatmap.png       — heatmap of probe accuracy across layers × probe types
  5. logit_lens_gap.png      — A0 minus A2 probability gap per layer

Usage
-----
  from scripts.plot_interp import plot_all_interp
  plot_all_interp(result_path, probe_results, logit_data, pca_data, out_dir)
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np


# ── Figure 1: Probe accuracy by layer ────────────────────────────────────────

def fig_probe_accuracy(
    probe_results: dict[str, dict[int, float]],
    out_dir: str,
) -> None:
    import matplotlib.pyplot as plt

    if not probe_results:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = {"condition": "#2196F3", "disclosure": "#FF9800"}
    styles = {"condition": "-o", "disclosure": "--s"}

    for probe_name, layer_accs in probe_results.items():
        layers = sorted(layer_accs.keys())
        accs   = [layer_accs[l] for l in layers]
        peak   = layers[int(np.argmax(accs))]
        ax.plot(layers, accs,
                styles.get(probe_name, "-"),
                color=colors.get(probe_name, "gray"),
                label=f"{probe_name} probe (peak L{peak}={max(accs):.1%})",
                linewidth=2, markersize=5)
        ax.axvline(peak, color=colors.get(probe_name, "gray"),
                   linestyle=":", alpha=0.4)

    ax.axhline(0.5, color="black", linestyle="--", alpha=0.3, label="Chance (50%)")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean CV accuracy")
    ax.set_title("Linear Probe Accuracy by Layer\n"
                 "(where does the model encode the concealment decision?)")
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    fig.tight_layout()

    path = os.path.join(out_dir, "probe_accuracy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 2: Logit lens ──────────────────────────────────────────────────────

def fig_logit_lens(logit_data: list[dict], out_dir: str) -> None:
    import matplotlib.pyplot as plt

    if not logit_data:
        return

    n_pairs = len(logit_data)
    fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 4), squeeze=False)

    for i, entry in enumerate(logit_data):
        ax = axes[0][i]
        a0 = np.array(entry["a0_probs"])
        a2 = np.array(entry["a2_probs"])
        layers = np.arange(len(a0))

        ax.plot(layers, a0, "-o", color="#2196F3", markersize=4,
                linewidth=1.5, label="A0 (transparent)")
        ax.plot(layers, a2, "--s", color="#F44336", markersize=4,
                linewidth=1.5, label="A2 (suppress)")
        ax.fill_between(layers, a0, a2,
                        where=(a0 >= a2), alpha=0.10, color="#2196F3",
                        label="A0 > A2")

        ax.set_title(f"{entry['example_id']}\nsecret='{entry['secret']}' "
                     f"tok='{entry.get('target_token_str', '')}'"
                     , fontsize=8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("P(secret token)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Logit Lens: P(secret token) across layers\n"
                 "A0 vs A2 for matched prompt pairs", y=1.02)
    fig.tight_layout()

    path = os.path.join(out_dir, "logit_lens.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 3: Activation PCA ──────────────────────────────────────────────────

def fig_activation_pca(pca_data: dict, out_dir: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if not pca_data or len(pca_data.get("X_2d", [])) < 3:
        return

    X_2d = np.array(pca_data["X_2d"])
    colors = pca_data["colors"]
    conditions = pca_data["conditions"]
    layer = pca_data.get("layer", "?")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.75, s=60, edgecolors="white", linewidths=0.5)

    cond_set = sorted(set(conditions))
    color_map = {"A0": "#2196F3", "A1": "#FF9800", "A2": "#F44336"}
    legend_elements = [
        Patch(facecolor=color_map.get(c, "#9E9E9E"), label=c) for c in cond_set
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"Activation PCA — Layer {layer} (last-token residual stream)\n"
                 "Coloured by condition (A0=transparent, A2=suppress)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    path = os.path.join(out_dir, "activation_pca.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 4: Probe heatmap ───────────────────────────────────────────────────

def fig_probe_heatmap(probe_results: dict[str, dict[int, float]], out_dir: str) -> None:
    import matplotlib.pyplot as plt

    if len(probe_results) < 2:
        return  # Only interesting with multiple probes

    probe_names = list(probe_results.keys())
    all_layers = sorted({l for accs in probe_results.values() for l in accs})
    matrix = np.array([
        [probe_results[name].get(l, 0.0) for l in all_layers]
        for name in probe_names
    ])  # [n_probes, n_layers]

    fig, ax = plt.subplots(figsize=(max(8, len(all_layers) * 0.4), 2.5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.02, label="CV accuracy")

    ax.set_yticks(range(len(probe_names)))
    ax.set_yticklabels(probe_names)
    ax.set_xlabel("Layer index")
    ax.set_title("Probe Accuracy Heatmap")

    # Annotate peak per row
    for row_i, name in enumerate(probe_names):
        accs = [probe_results[name].get(l, 0) for l in all_layers]
        peak_col = int(np.argmax(accs))
        ax.annotate("★", (peak_col, row_i), ha="center", va="center",
                    fontsize=10, color="black")

    fig.tight_layout()
    path = os.path.join(out_dir, "probe_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 5: Logit-lens gap (A0 − A2) ────────────────────────────────────────

def fig_logit_lens_gap(logit_data: list[dict], out_dir: str) -> None:
    import matplotlib.pyplot as plt

    if not logit_data:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    palette = ["#2196F3", "#4CAF50", "#9C27B0"]

    for i, entry in enumerate(logit_data):
        a0 = np.array(entry["a0_probs"])
        a2 = np.array(entry["a2_probs"])
        gap = a0 - a2
        layers = np.arange(len(gap))
        color = palette[i % len(palette)]
        ax.plot(layers, gap, "-o", markersize=4, linewidth=1.5,
                color=color, label=f"{entry['example_id']} ('{entry['secret']}')")
        ax.fill_between(layers, 0, gap, where=(gap > 0), alpha=0.08, color=color)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("P(secret | A0) − P(secret | A2)")
    ax.set_title("Logit-Lens Gap: A0 − A2\n"
                 "Positive = A0 more likely to output secret at this layer")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    path = os.path.join(out_dir, "logit_lens_gap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def plot_all_interp(
    result_path: str,
    probe_results: dict[str, dict[int, float]],
    logit_data: list[dict],
    pca_data: dict,
    out_dir: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("  [ERROR] matplotlib not installed. Run: pip install matplotlib")
        return

    fig_probe_accuracy(probe_results, out_dir)
    fig_probe_heatmap(probe_results, out_dir)
    fig_logit_lens(logit_data, out_dir)
    fig_logit_lens_gap(logit_data, out_dir)
    fig_activation_pca(pca_data, out_dir)
