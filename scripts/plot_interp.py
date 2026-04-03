"""Interpretability plots for the concealment experiment.

Produces up to seven figures saved to out_dir:
  1. probe_accuracy.png      — layer-by-layer CV accuracy (all probes)
  2. logit_lens.png          — P(secret_token | layer) for A0/A1/A2 matched groups
  3. activation_pca.png      — 2-D PCA of last-layer residual stream, coloured by condition
  4. probe_heatmap.png       — heatmap of probe accuracy across layers x probe types
  5. logit_lens_gap.png      — probability gaps per layer
  6. a1_projection.png       — A1-concealing projection onto A0-A2 axis per layer
  7. cosine_similarity.png   — pairwise cosine similarity of behavioral subgroups per layer

Usage
-----
  from scripts.plot_interp import plot_all_interp
  plot_all_interp(result_path, probe_results, logit_data, pca_data, out_dir,
                  projection_data=..., cosine_data=...)
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np


# ── Probe colors & styles ───────────────────────────────────────────────────

PROBE_COLORS = {
    "condition": "#2196F3",
    "disclosure": "#FF9800",
    "a1c_vs_a2c": "#9C27B0",
    "a1c_vs_a0d": "#4CAF50",
    "a1d_vs_a0d": "#9E9E9E",
}

PROBE_STYLES = {
    "condition": "-o",
    "disclosure": "--s",
    "a1c_vs_a2c": "-^",
    "a1c_vs_a0d": "-.v",
    "a1d_vs_a0d": ":D",
}

PROBE_LABELS = {
    "condition": "A0 vs A2 (condition)",
    "disclosure": "Disclosure (regex)",
    "a1c_vs_a2c": "A1-conceal vs A2-conceal",
    "a1c_vs_a0d": "A1-conceal vs A0-disclose",
    "a1d_vs_a0d": "A1-disclose vs A0-disclose",
}


# ── Figure 1: Probe accuracy by layer ────────────────────────────────────────

def fig_probe_accuracy(
    probe_results: dict[str, dict[int, float]],
    out_dir: str,
) -> None:
    import matplotlib.pyplot as plt

    if not probe_results:
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    for probe_name, layer_accs in probe_results.items():
        layers = sorted(layer_accs.keys())
        accs   = [layer_accs[l] for l in layers]
        peak   = layers[int(np.argmax(accs))]
        style = PROBE_STYLES.get(probe_name, "-")
        color = PROBE_COLORS.get(probe_name, "gray")
        label = PROBE_LABELS.get(probe_name, probe_name)
        ax.plot(layers, accs, style,
                color=color,
                label=f"{label} (peak L{peak}={max(accs):.1%})",
                linewidth=2, markersize=5)
        ax.axvline(peak, color=color, linestyle=":", alpha=0.4)

    ax.axhline(0.5, color="black", linestyle="--", alpha=0.3, label="Chance (50%)")
    # Add 33% line if any 3-class probe is present
    if any("3class" in k for k in probe_results):
        ax.axhline(1/3, color="black", linestyle=":", alpha=0.3, label="Chance (33%)")

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean CV accuracy")
    ax.set_title("Linear Probe Accuracy by Layer\n"
                 "(where does the model encode the concealment decision?)")
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    fig.tight_layout()

    path = os.path.join(out_dir, "probe_accuracy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 2: Logit lens ────────────────────────────────────────────────────

def fig_logit_lens(logit_data: list[dict], out_dir: str) -> None:
    import matplotlib.pyplot as plt

    if not logit_data:
        return

    n_entries = len(logit_data)
    fig, axes = plt.subplots(1, n_entries, figsize=(5 * n_entries, 4), squeeze=False)

    for i, entry in enumerate(logit_data):
        ax = axes[0][i]
        a0 = np.array(entry["a0_probs"])
        a2 = np.array(entry["a2_probs"])
        layers = np.arange(len(a0))

        ax.plot(layers, a0, "-o", color="#2196F3", markersize=4,
                linewidth=1.5, label="A0 (transparent)")
        ax.plot(layers, a2, "--s", color="#F44336", markersize=4,
                linewidth=1.5, label="A2 (suppress)")

        # A1 line if available
        if "a1_probs" in entry:
            a1 = np.array(entry["a1_probs"])
            a1_tag = "disclosed" if entry.get("a1_disclosed") else "concealed"
            ax.plot(layers, a1, "-.^", color="#FF9800", markersize=4,
                    linewidth=1.5, label=f"A1 ({a1_tag})")

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
                 "Matched prompt groups by condition", y=1.02)
    fig.tight_layout()

    path = os.path.join(out_dir, "logit_lens.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 3: Activation PCA ────────────────────────────────────────────────

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
                 "Coloured by condition (A0=transparent, A1=pressure, A2=suppress)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    path = os.path.join(out_dir, "activation_pca.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 4: Probe heatmap ─────────────────────────────────────────────────

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

    display_names = [PROBE_LABELS.get(n, n) for n in probe_names]

    fig, ax = plt.subplots(figsize=(max(8, len(all_layers) * 0.4), max(2.5, len(probe_names) * 0.6)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.02, label="CV accuracy")

    ax.set_yticks(range(len(probe_names)))
    ax.set_yticklabels(display_names, fontsize=8)
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


# ── Figure 5: Logit-lens gap (A0 − A2, A0 − A1, A1 − A2) ──────────────────

def fig_logit_lens_gap(logit_data: list[dict], out_dir: str) -> None:
    import matplotlib.pyplot as plt

    if not logit_data:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    palette = ["#2196F3", "#4CAF50", "#9C27B0"]

    for i, entry in enumerate(logit_data):
        a0 = np.array(entry["a0_probs"])
        a2 = np.array(entry["a2_probs"])
        gap_a0_a2 = a0 - a2
        layers = np.arange(len(gap_a0_a2))
        color = palette[i % len(palette)]
        eid = entry["example_id"]

        ax.plot(layers, gap_a0_a2, "-o", markersize=4, linewidth=1.5,
                color=color, label=f"{eid} A0−A2")
        ax.fill_between(layers, 0, gap_a0_a2, where=(gap_a0_a2 > 0), alpha=0.08, color=color)

        # A0 − A1 and A1 − A2 gaps if A1 data available
        if "a1_probs" in entry:
            a1 = np.array(entry["a1_probs"])
            gap_a0_a1 = a0 - a1
            gap_a1_a2 = a1 - a2
            ax.plot(layers, gap_a0_a1, "--^", markersize=3, linewidth=1.0,
                    color=color, alpha=0.6, label=f"{eid} A0−A1")
            ax.plot(layers, gap_a1_a2, ":v", markersize=3, linewidth=1.0,
                    color=color, alpha=0.6, label=f"{eid} A1−A2")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("P(secret) gap")
    ax.set_title("Logit-Lens Gap\n"
                 "Positive = higher P(secret) in left condition")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    path = os.path.join(out_dir, "logit_lens_gap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 6: A1 projection onto A0↔A2 axis ─────────────────────────────────

def fig_a1_projection(projection_data: dict[int, dict[str, float]], out_dir: str) -> None:
    import matplotlib.pyplot as plt

    if not projection_data:
        return

    layers = sorted(projection_data.keys())
    a0_means = [projection_data[l]["a0_mean"] for l in layers]
    a0_stds  = [projection_data[l]["a0_std"] for l in layers]
    a1c_means = [projection_data[l]["a1c_mean"] for l in layers]
    a1c_stds  = [projection_data[l]["a1c_std"] for l in layers]
    a2_means = [projection_data[l]["a2_mean"] for l in layers]
    a2_stds  = [projection_data[l]["a2_std"] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 5))

    # A0 band
    ax.fill_between(layers,
                    [m - s for m, s in zip(a0_means, a0_stds)],
                    [m + s for m, s in zip(a0_means, a0_stds)],
                    alpha=0.15, color="#2196F3")
    ax.plot(layers, a0_means, "-o", color="#2196F3", linewidth=2,
            markersize=5, label="A0 (transparent)")

    # A1-concealing band
    ax.fill_between(layers,
                    [m - s for m, s in zip(a1c_means, a1c_stds)],
                    [m + s for m, s in zip(a1c_means, a1c_stds)],
                    alpha=0.15, color="#FF9800")
    ax.plot(layers, a1c_means, "-^", color="#FF9800", linewidth=2,
            markersize=5, label="A1-concealing")

    # A2 band
    ax.fill_between(layers,
                    [m - s for m, s in zip(a2_means, a2_stds)],
                    [m + s for m, s in zip(a2_means, a2_stds)],
                    alpha=0.15, color="#F44336")
    ax.plot(layers, a2_means, "-s", color="#F44336", linewidth=2,
            markersize=5, label="A2 (suppress)")

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Projection onto concealment direction")
    ax.set_title("A1 Projection onto A0↔A2 Separating Hyperplane\n"
                 "(if A1-concealing tracks A2 → same suppression mechanism)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    path = os.path.join(out_dir, "a1_projection.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 7: Cosine similarity across layers ───────────────────────────────

def fig_cosine_similarity(cosine_data: dict[int, dict[str, float]], out_dir: str) -> None:
    import matplotlib.pyplot as plt

    if not cosine_data:
        return

    layers = sorted(cosine_data.keys())

    fig, ax = plt.subplots(figsize=(10, 4))

    cos_a1c_a2c = [cosine_data[l].get("cos_a1c_a2c", np.nan) for l in layers]
    cos_a1c_a0d = [cosine_data[l].get("cos_a1c_a0d", np.nan) for l in layers]

    ax.plot(layers, cos_a1c_a2c, "-^", color="#9C27B0", linewidth=2,
            markersize=5, label="cos(A1-conceal, A2-conceal)")
    ax.plot(layers, cos_a1c_a0d, "-o", color="#4CAF50", linewidth=2,
            markersize=5, label="cos(A1-conceal, A0-disclose)")

    # A1-disclosing vs A0-disclosing if available
    if "cos_a1d_a0d" in cosine_data.get(layers[0], {}):
        cos_a1d_a0d = [cosine_data[l].get("cos_a1d_a0d", np.nan) for l in layers]
        ax.plot(layers, cos_a1d_a0d, ":D", color="#9E9E9E", linewidth=1.5,
                markersize=4, label="cos(A1-disclose, A0-disclose)")

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Cosine similarity (mean activations)")
    ax.set_title("Cosine Similarity of Behavioral Subgroups\n"
                 "(higher = more similar representations)")
    ax.set_ylim(None, 1.02)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    path = os.path.join(out_dir, "cosine_similarity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Entry point ──────────────────────────────────────────────────────────────

def plot_all_interp(
    result_path: str,
    probe_results: dict[str, dict[int, float]],
    logit_data: list[dict],
    pca_data: dict,
    out_dir: str,
    projection_data: dict | None = None,
    cosine_data: dict | None = None,
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

    if projection_data:
        fig_a1_projection(projection_data, out_dir)
    if cosine_data:
        fig_cosine_similarity(cosine_data, out_dir)
