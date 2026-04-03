"""Publication-quality plots for the controlled framing experiment.

Produces five figures saved as both .pdf and .png (300 dpi) alongside
the results JSON:

  1. framing_main.{pdf,png}             — disclosure rate per condition (Wilson CI)
                                          with Cohen's h effect-size annotations
  2. framing_by_dimension.{pdf,png}     — mean disclosure rate per dimension
  3. framing_heatmap.{pdf,png}          — condition × secret-value heatmap
  4. framing_monitor_agreement.{pdf,png}— regex vs LLM scatter by dimension
  5. framing_spectrum.{pdf,png}         — empirical explicitness spectrum (line plot)

Usage
-----
  python scripts/plot_framing.py results/run_<id>.json
  # or programmatically:
  from scripts.plot_framing import plot_all_framing
  plot_all_framing("results/run_<id>.json")
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from typing import Any


# ── Data helpers ───────────────────────────────────────────────────────────────

def _load(path: str) -> tuple[dict, list[dict]]:
    with open(path) as f:
        payload = json.load(f)
    results = payload.get("results", payload) if isinstance(payload, dict) else payload
    return payload, results


def _subtitle(payload: dict, results: list[dict]) -> str:
    """Build a subtitle string from run metadata for plot titles."""
    model = payload.get("config", {}).get("base_model", "unknown model")
    model_short = model.rsplit("/", 1)[-1] if "/" in model else model
    n_regular = sum(1 for r in results if _control_type(r) is None)
    meta = payload.get("run_metadata", {})
    n_scenarios = meta.get("n_scenarios", "?")
    env = meta.get("environment", "?")
    return f"({model_short}, {env}, {n_scenarios} scenarios, n={n_regular} regular prompts)"


def _regex_disclosed(r: dict) -> bool:
    regex = r.get("monitor_results", {}).get("regex", {})
    return bool(regex.get("contains_secret_exact") or regex.get("contains_secret_partial"))


def _llm_disclosed(r: dict) -> bool | None:
    llm = r.get("monitor_results", {}).get("llm", {})
    v = llm.get("mentioned_secret")
    return bool(v) if v is not None else None


def _control_type(r: dict) -> str | None:
    return r.get("metadata", {}).get("control_type")


def _framing_key(r: dict) -> str:
    return r.get("metadata", {}).get("framing_condition", r.get("metadata", {}).get("condition", "unknown"))


def _framing_dim(r: dict) -> str:
    return r.get("metadata", {}).get("framing_dimension", "unknown")


def _secret_label(r: dict) -> str:
    """Return canonical secret text (first keyword hint or 'unknown')."""
    hints = r.get("metadata", {}).get("keyword_hints", [])
    if hints:
        return hints[0]
    return "unknown"


# ── Statistics ─────────────────────────────────────────────────────────────────

def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval."""
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap 95% CI."""
    import random
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(rng.choices(values, k=n)) / n
        for _ in range(n_boot)
    )
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def cohen_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions.

    h = 2·arcsin(√p1) − 2·arcsin(√p2)
    Positive h means p1 > p2 (p1 = base, p2 = condition).
    Convention here: h > 0 means condition suppresses relative to base.
    """
    phi1 = 2 * math.asin(math.sqrt(max(0.0, min(1.0, p1))))
    phi2 = 2 * math.asin(math.sqrt(max(0.0, min(1.0, p2))))
    return phi1 - phi2


def two_prop_z_test(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-proportion z-test p-value (two-tailed).

    Returns p-value; uses pooled proportion under H0: p1 == p2.
    """
    if n1 == 0 or n2 == 0:
        return 1.0
    p1 = k1 / n1
    p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    # Normal CDF approximation via math.erfc
    p_val = math.erfc(abs(z) / math.sqrt(2))
    return p_val


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ── Condition ordering ─────────────────────────────────────────────────────────

def _get_order() -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Return (ordered_keys, dim_of_key, label_of_key).

    Falls back gracefully if src.framing is not importable (e.g. plotting
    standalone from a different machine).
    """
    try:
        from src.framing.conditions import CONDITION_ORDER, REGISTRY
        order = CONDITION_ORDER
        dim_of = {k: REGISTRY[k].dimension for k in order}
        label_of = {k: REGISTRY[k].label for k in order}
    except ImportError:
        order = []
        dim_of = {}
        label_of = {}
    return order, dim_of, label_of


def _dim_color() -> dict[str, str]:
    return {
        "baseline":  "#607D8B",
        "motivation": "#2196F3",
        "incentive":  "#FF9800",
        "audience":   "#4CAF50",
        "control":    "#F44336",
    }


def _savefig(fig: Any, out_dir: str, stem: str) -> None:
    """Save figure as both PDF and PNG at 300 dpi."""
    import matplotlib.pyplot as plt
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ── Figure 1: Disclosure by condition (main result) ───────────────────────────

def fig_framing_main(results: list[dict], out_dir: str, subtitle: str = "") -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    regular = [r for r in results if _control_type(r) is None]

    order, dim_of, label_of = _get_order()
    seen_keys = {_framing_key(r) for r in regular}

    # Use CONDITION_ORDER if available, else sort seen keys
    keys = [k for k in order if k in seen_keys] or sorted(seen_keys)

    rates, lo_errs, hi_errs, colors = [], [], [], []
    counts = []
    dim_colors = _dim_color()
    base_rate = None
    base_n = 0

    for k in keys:
        subset = [r for r in regular if _framing_key(r) == k]
        n = len(subset)
        p = sum(_regex_disclosed(r) for r in subset) / n if n else 0.0
        lo, hi = wilson_ci(p, n)
        rates.append(p)
        lo_errs.append(p - lo)
        hi_errs.append(hi - p)
        counts.append(n)
        dim = dim_of.get(k, "unknown")
        colors.append(dim_colors.get(dim, "#9E9E9E"))
        if k == "BASE":
            base_rate = p
            base_n = n

    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(max(10, len(keys) * 0.9), 5.5))

    bars = ax.bar(
        x, rates,
        color=colors,
        alpha=0.85,
        yerr=[lo_errs, hi_errs],
        error_kw={"elinewidth": 1.2, "capsize": 3, "ecolor": "black"},
        zorder=3,
    )

    # Percentage labels + Cohen's h annotations
    if base_rate is not None:
        for xi, (k, p, n, hi_e) in enumerate(zip(keys, rates, counts, hi_errs)):
            ax.text(xi, p + hi_e + 0.02, f"{p:.0%}", ha="center", va="bottom", fontsize=7)

            if k != "BASE":
                h = cohen_h(base_rate, p)
                k_disclosed = int(p * n)
                base_k = int(base_rate * base_n)
                pval = two_prop_z_test(base_k, base_n, k_disclosed, n)
                stars = _sig_stars(pval)
                h_str = f"h={h:+.2f}{stars}" if stars else f"h={h:+.2f}"
                ax.text(xi, p + hi_e + 0.07, h_str, ha="center", va="bottom",
                        fontsize=6, color="dimgray", style="italic")
        # BASE label (no effect size vs self)
        base_idx = keys.index("BASE")
        p_b, hi_b = rates[base_idx], hi_errs[base_idx]
        ax.text(base_idx, p_b + hi_b + 0.02, f"{p_b:.0%}", ha="center", va="bottom", fontsize=7)
    else:
        for xi, (p, hi_e) in enumerate(zip(rates, hi_errs)):
            ax.text(xi, p + hi_e + 0.02, f"{p:.0%}", ha="center", va="bottom", fontsize=7)

    # n= labels below x-axis
    for xi, n in enumerate(counts):
        ax.text(xi, -0.10, f"n={n}", ha="center", va="top", fontsize=6, color="gray",
                transform=ax.get_xaxis_transform())

    # Baseline reference line
    if base_rate is not None:
        ax.axhline(base_rate, color="black", linestyle="--", linewidth=0.8,
                   label=f"BASE ({base_rate:.0%})", zorder=2)

    # Dimension dividers
    prev_dim = None
    for i, k in enumerate(keys):
        dim = dim_of.get(k, "")
        if prev_dim is not None and dim != prev_dim:
            ax.axvline(i - 0.5, color="gray", linewidth=0.6, linestyle=":", zorder=1)
        prev_dim = dim

    # Legend for dimensions
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=dim_colors[d], alpha=0.85, label=d.capitalize())
        for d in ["baseline", "motivation", "incentive", "audience", "control"]
        if d in dim_colors
    ]
    ax.legend(handles=legend_elements + [
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=0.8, label="BASE rate")
    ], loc="upper right", fontsize=8)

    tick_labels = [label_of.get(k, k) for k in keys]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Secret disclosure rate (regex)")
    title = "Disclosure Rate by Framing Condition (Wilson 95% CI, Cohen's h vs BASE)"
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title)
    ax.set_ylim(-0.05, 1.22)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()

    _savefig(fig, out_dir, "framing_main")


# ── Figure 2: Disclosure by dimension ─────────────────────────────────────────

def fig_framing_by_dimension(results: list[dict], out_dir: str, subtitle: str = "") -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    regular = [r for r in results if _control_type(r) is None]
    dim_colors = _dim_color()

    # Dimensions (exclude baseline; it's the reference)
    dims = ["motivation", "incentive", "audience", "control"]

    regex_rates, regex_lo, regex_hi = [], [], []
    llm_rates, llm_lo, llm_hi = [], [], []

    # BASE reference
    base_sub = [r for r in regular if _framing_key(r) == "BASE"]
    base_regex = sum(_regex_disclosed(r) for r in base_sub) / len(base_sub) if base_sub else 0.0
    base_llm_vals = [_llm_disclosed(r) for r in base_sub if _llm_disclosed(r) is not None]
    base_llm = sum(base_llm_vals) / len(base_llm_vals) if base_llm_vals else 0.0

    for dim in dims:
        subset = [r for r in regular if _framing_dim(r) == dim]

        # Regex
        n = len(subset)
        p = sum(_regex_disclosed(r) for r in subset) / n if n else 0.0
        lo, hi = wilson_ci(p, n)
        regex_rates.append(p)
        regex_lo.append(p - lo)
        regex_hi.append(hi - p)

        # LLM
        llm_vals = [_llm_disclosed(r) for r in subset if _llm_disclosed(r) is not None]
        lp = sum(llm_vals) / len(llm_vals) if llm_vals else 0.0
        llo, lhi = bootstrap_ci([float(v) for v in llm_vals])
        llm_rates.append(lp)
        llm_lo.append(lp - llo)
        llm_hi.append(lhi - lp)

    x = np.arange(len(dims))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x - width / 2, regex_rates, width,
        label="Regex monitor",
        color=[dim_colors.get(d, "#9E9E9E") for d in dims],
        alpha=0.85,
        yerr=[regex_lo, regex_hi],
        error_kw={"elinewidth": 1.2, "capsize": 3, "ecolor": "black"},
    )
    ax.bar(
        x + width / 2, llm_rates, width,
        label="LLM monitor",
        color=[dim_colors.get(d, "#9E9E9E") for d in dims],
        alpha=0.45,
        yerr=[llm_lo, llm_hi],
        error_kw={"elinewidth": 1.2, "capsize": 3, "ecolor": "black"},
    )

    # Labels
    for i, (rr, lr) in enumerate(zip(regex_rates, llm_rates)):
        ax.text(x[i] - width / 2, rr + 0.03, f"{rr:.0%}", ha="center", va="bottom", fontsize=8)
        ax.text(x[i] + width / 2, lr + 0.03, f"{lr:.0%}", ha="center", va="bottom", fontsize=8)

    ax.axhline(base_regex, color="#607D8B", linestyle="--", linewidth=1.0,
               label=f"BASE regex ({base_regex:.0%})")
    ax.axhline(base_llm, color="#607D8B", linestyle=":", linewidth=1.0,
               label=f"BASE LLM ({base_llm:.0%})")

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in dims])
    ax.set_ylabel("Mean secret disclosure rate")
    ax.set_title(f"Disclosure Rate by Framing Dimension\n{subtitle}" if subtitle else
                 "Disclosure Rate by Framing Dimension")
    ax.set_ylim(-0.05, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()

    _savefig(fig, out_dir, "framing_by_dimension")


# ── Figure 3: Heatmap — condition × secret value ──────────────────────────────

def fig_framing_heatmap(results: list[dict], out_dir: str, subtitle: str = "") -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    regular = [r for r in results if _control_type(r) is None]
    if not regular:
        return

    order, _, label_of = _get_order()
    seen_keys = {_framing_key(r) for r in regular}
    keys = [k for k in order if k in seen_keys] or sorted(seen_keys)

    # Secret values: sort numerically where possible
    def _sort_key(s: str) -> float:
        import re
        m = re.match(r"(\d+)", s)
        return float(m.group(1)) if m else float("inf")

    secrets = sorted({_secret_label(r) for r in regular}, key=_sort_key)

    # Build matrix
    matrix = np.full((len(keys), len(secrets)), float("nan"))
    for i, k in enumerate(keys):
        for j, s in enumerate(secrets):
            sub = [r for r in regular if _framing_key(r) == k and _secret_label(r) == s]
            if sub:
                matrix[i, j] = sum(_regex_disclosed(r) for r in sub) / len(sub)

    fig, ax = plt.subplots(figsize=(max(6, len(secrets) * 1.4), max(5, len(keys) * 0.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(secrets)))
    ax.set_xticklabels(secrets, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels([label_of.get(k, k) for k in keys], fontsize=8)
    ax.set_xlabel("Secret value")
    ax.set_ylabel("Framing condition")
    ax.set_title(f"Regex Disclosure Rate: Condition × Secret Value\n{subtitle}" if subtitle else
                 "Regex Disclosure Rate: Condition × Secret Value")

    # Annotate cells
    for i in range(len(keys)):
        for j in range(len(secrets)):
            v = matrix[i, j]
            if not math.isnan(v):
                text_color = "black" if 0.3 < v < 0.7 else "white"
                ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                        fontsize=7, color=text_color)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Disclosure rate")
    fig.tight_layout()

    _savefig(fig, out_dir, "framing_heatmap")


# ── Figure 4: Monitor agreement scatter per dimension ─────────────────────────

def fig_framing_monitor_agreement(results: list[dict], out_dir: str, subtitle: str = "") -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    regular = [r for r in results if _control_type(r) is None]
    if not regular:
        return

    dims = ["motivation", "incentive", "audience", "control"]
    _, dim_of, _ = _get_order()
    dim_colors = _dim_color()

    # Build per-dim color maps (one color per condition key within dim)
    _PALETTE = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]

    rng = np.random.default_rng(seed=0)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes_flat = axes.flatten()

    for ax_idx, dim in enumerate(dims):
        ax = axes_flat[ax_idx]
        dim_subset = [r for r in regular if _framing_dim(r) == dim]
        if not dim_subset:
            ax.set_visible(False)
            continue

        # Group by condition key for colors
        keys_in_dim = sorted({_framing_key(r) for r in dim_subset})
        key_color = {k: _PALETTE[i % len(_PALETTE)] for i, k in enumerate(keys_in_dim)}

        for k in keys_in_dim:
            sub = [r for r in dim_subset if _framing_key(r) == k]
            xs = [float(_regex_disclosed(r)) for r in sub]
            ys_raw = [_llm_disclosed(r) for r in sub]
            ys = [float(y) if y is not None else 0.5 for y in ys_raw]

            # Add jitter
            jitter = rng.uniform(-0.08, 0.08, size=(len(xs), 2))
            xs_j = [x + j for x, j in zip(xs, jitter[:, 0])]
            ys_j = [y + j for y, j in zip(ys, jitter[:, 1])]

            ax.scatter(xs_j, ys_j, label=k, color=key_color[k], alpha=0.6, s=30)

        ax.set_xlim(-0.25, 1.25)
        ax.set_ylim(-0.25, 1.25)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Not disclosed", "Disclosed"], fontsize=8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Not disclosed", "Disclosed"], fontsize=8)
        ax.set_xlabel("Regex monitor", fontsize=9)
        ax.set_ylabel("LLM monitor", fontsize=9)
        ax.set_title(dim.capitalize(), fontsize=10, color=dim_colors.get(dim, "black"))
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(linewidth=0.3, alpha=0.5)

    fig.suptitle("Regex vs LLM Monitor Agreement by Dimension", fontsize=12)
    fig.tight_layout()

    _savefig(fig, out_dir, "framing_monitor_agreement")


# ── Figure 5: Empirical explicitness spectrum ──────────────────────────────────

def fig_framing_spectrum(results: list[dict], out_dir: str) -> None:
    """Line plot of conditions ordered empirically by mean disclosure rate (low→high).

    Conditions are placed on the x-axis in ascending disclosure-rate order
    (most suppressive on the left), making the discovered spectrum visible
    without imposing the a-priori theoretical ordering.  Error bars show
    Wilson 95% CI; points are coloured by framing dimension.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    regular = [r for r in results if _control_type(r) is None]
    if not regular:
        return

    order, dim_of, label_of = _get_order()
    seen_keys = {_framing_key(r) for r in regular}
    all_keys = [k for k in order if k in seen_keys] or sorted(seen_keys)

    # Compute per-condition stats
    stats = {}
    for k in all_keys:
        subset = [r for r in regular if _framing_key(r) == k]
        n = len(subset)
        p = sum(_regex_disclosed(r) for r in subset) / n if n else 0.0
        lo, hi = wilson_ci(p, n)
        stats[k] = {"p": p, "n": n, "lo": lo, "hi": hi}

    # Sort empirically by disclosure rate (ascending = most suppressive first)
    sorted_keys = sorted(all_keys, key=lambda k: stats[k]["p"])

    dim_colors = _dim_color()
    xs = list(range(len(sorted_keys)))
    ys = [stats[k]["p"] for k in sorted_keys]
    lo_errs = [stats[k]["p"] - stats[k]["lo"] for k in sorted_keys]
    hi_errs = [stats[k]["hi"] - stats[k]["p"] for k in sorted_keys]
    point_colors = [dim_colors.get(dim_of.get(k, ""), "#9E9E9E") for k in sorted_keys]

    fig, ax = plt.subplots(figsize=(max(10, len(sorted_keys) * 0.9), 5))

    # Line connecting points
    ax.plot(xs, ys, color="gray", linewidth=0.8, zorder=1)

    # Error bars + scatter coloured by dimension
    ax.errorbar(
        xs, ys,
        yerr=[lo_errs, hi_errs],
        fmt="none",
        elinewidth=1.0, capsize=3, ecolor="black", zorder=2,
    )

    # Scatter points coloured by dimension
    seen_dims = set()
    for xi, (k, y, c) in enumerate(zip(sorted_keys, ys, point_colors)):
        dim = dim_of.get(k, "unknown")
        lbl = dim.capitalize() if dim not in seen_dims else None
        ax.scatter(xi, y, color=c, s=60, zorder=3, label=lbl)
        seen_dims.add(dim)

    # Annotate points with condition key
    for xi, (k, y, hi_e) in enumerate(zip(sorted_keys, ys, hi_errs)):
        ax.text(xi, y + hi_e + 0.03, k, ha="center", va="bottom",
                fontsize=6, rotation=55, color="dimgray")

    ax.set_xticks(xs)
    ax.set_xticklabels([label_of.get(k, k) for k in sorted_keys],
                       rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Secret disclosure rate (regex)")
    ax.set_title("Empirical Framing Spectrum: Conditions Ordered by Disclosure Rate (Wilson 95% CI)")
    ax.set_ylim(-0.05, 1.20)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    # Legend (dimension colours)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", fontsize=8, title="Dimension")

    fig.tight_layout()
    _savefig(fig, out_dir, "framing_spectrum")


# ── Effect size table ──────────────────────────────────────────────────────────

def print_effect_sizes(results: list[dict]) -> None:
    """Print a terminal table of Cohen's h effect sizes and p-values vs BASE."""
    regular = [r for r in results if _control_type(r) is None]

    order, dim_of, label_of = _get_order()
    seen_keys = {_framing_key(r) for r in regular}
    keys = [k for k in order if k in seen_keys] or sorted(seen_keys)

    # BASE stats
    base_sub = [r for r in regular if _framing_key(r) == "BASE"]
    base_n = len(base_sub)
    base_k = sum(_regex_disclosed(r) for r in base_sub)
    base_p = base_k / base_n if base_n else 0.0
    base_lo, base_hi = wilson_ci(base_p, base_n)

    print()
    print(f"  Effect sizes relative to BASE ({base_p:.0%}, n={base_n})")
    print(f"  {'─' * 72}")
    print(f"  {'COND':<12} {'DIM':<12} {'RATE':>6}  {'Cohen h':>8}  {'95% CI':>14}  {'p-value':>10}  SIG")
    print(f"  {'─' * 72}")

    for k in keys:
        if k == "BASE":
            continue
        subset = [r for r in regular if _framing_key(r) == k]
        n = len(subset)
        k_count = sum(_regex_disclosed(r) for r in subset)
        p = k_count / n if n else 0.0
        h = cohen_h(base_p, p)
        pval = two_prop_z_test(base_k, base_n, k_count, n)
        stars = _sig_stars(pval)

        # Wilson CI on h (approximate: propagate via delta method)
        p_lo, p_hi = wilson_ci(p, n)
        h_lo = cohen_h(base_p, p_hi)   # note: higher p → smaller h (less suppression)
        h_hi = cohen_h(base_p, p_lo)
        ci_str = f"[{h_lo:+.2f}, {h_hi:+.2f}]"

        pval_str = f"p < 0.001" if pval < 0.001 else f"p = {pval:.3f}"
        dim = dim_of.get(k, "")
        print(f"  {k:<12} {dim:<12} {p:>6.1%}  {h:>+8.2f}  {ci_str:>14}  {pval_str:>10}  {stars}")

    print(f"  {'─' * 72}")
    print(f"  Cohen's h: small=0.20, medium=0.50, large=0.80")
    print(f"  * p<0.05  ** p<0.01  *** p<0.001")
    print()


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary_table(results: list[dict]) -> None:
    regular = [r for r in results if _control_type(r) is None]
    controls = [r for r in results if _control_type(r) is not None]

    order, dim_of, label_of = _get_order()
    seen_keys = {_framing_key(r) for r in regular}
    keys = [k for k in order if k in seen_keys] or sorted(seen_keys)

    W_key = 12
    W_dim = 12
    W_label = 26
    print()
    print(f"  {'─' * 82}")
    print(f"  {'COND':<{W_key}} {'DIM':<{W_dim}} {'LABEL':<{W_label}} "
          f"{'N':>5}  {'REGEX%':>7}  {'LLM%':>7}")
    print(f"  {'─' * 82}")

    for k in keys:
        subset = [r for r in regular if _framing_key(r) == k]
        n = len(subset)
        regex_r = sum(_regex_disclosed(r) for r in subset) / n if n else 0.0
        llm_vals = [_llm_disclosed(r) for r in subset if _llm_disclosed(r) is not None]
        llm_r = sum(llm_vals) / len(llm_vals) if llm_vals else float("nan")
        dim = dim_of.get(k, "")
        label = label_of.get(k, "")
        print(f"  {k:<{W_key}} {dim:<{W_dim}} {label:<{W_label}} "
              f"{n:>5}  {regex_r:>7.1%}  {llm_r:>7.1%}")

    if controls:
        print(f"  {'─' * 82}")
        for ctrl_type in ("no_secret", "irrelevant_secret"):
            subset = [r for r in controls if _control_type(r) == ctrl_type]
            if not subset:
                continue
            n = len(subset)
            regex_r = sum(_regex_disclosed(r) for r in subset) / n if n else 0.0
            llm_vals = [_llm_disclosed(r) for r in subset if _llm_disclosed(r) is not None]
            llm_r = sum(llm_vals) / len(llm_vals) if llm_vals else float("nan")
            print(f"  {ctrl_type:<{W_key}} {'control':<{W_dim}} {'':<{W_label}} "
                  f"{n:>5}  {regex_r:>7.1%}  {llm_r:>7.1%}")

    print(f"  {'─' * 82}")
    print(f"  Total: {len(results)} prompts ({len(regular)} regular, {len(controls)} controls)")
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def plot_all_framing(results_path: str) -> None:
    """Generate all framing plots for a results JSON file."""
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("  [ERROR] matplotlib not installed. Run: pip install matplotlib")
        return

    results = _load(results_path)
    out_dir = os.path.dirname(results_path) or "."

    print_summary_table(results)
    print_effect_sizes(results)
    fig_framing_main(results, out_dir)
    fig_framing_by_dimension(results, out_dir)
    fig_framing_heatmap(results, out_dir)
    fig_framing_monitor_agreement(results, out_dir)
    fig_framing_spectrum(results, out_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <results_json_path>")
        sys.exit(1)
    plot_all_framing(sys.argv[1])
    print("Done.")
