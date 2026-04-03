"""Plot disclosure rates and monitor metrics from a concealment experiment.

Produces four figures saved alongside the results JSON:
  1. disclosure_by_condition.png   — secret mention rate per condition (A0/A1/A2)
  2. monitor_concordance.png       — LLM vs regex agreement confusion matrix
  3. disclosure_by_secret.png      — secret mention rate per secret value
  4. verdict_distribution.png      — stacked verdict counts per condition

Usage
-----
  python scripts/plot_results.py results/run_<id>.json
  # or called programmatically:
  from scripts.plot_results import plot_all
  plot_all("results/run_<id>.json")
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Any


def _load(path: str) -> tuple[dict, list[dict]]:
    with open(path) as f:
        payload = json.load(f)
    results = payload.get("results", payload) if isinstance(payload, dict) else payload
    return payload, results


def _subtitle(payload: dict, results: list[dict]) -> str:
    """Build a subtitle string from run metadata for plot titles."""
    model = payload.get("config", {}).get("base_model", "unknown model")
    # Shorten model name: "meta-llama/Llama-3.3-70B-Instruct-Turbo" → "Llama-3.3-70B-Instruct-Turbo"
    model_short = model.rsplit("/", 1)[-1] if "/" in model else model
    n_regular = sum(1 for r in results if _control_type(r) is None)
    n_total = len(results)
    meta = payload.get("run_metadata", {})
    n_scenarios = meta.get("n_scenarios", "?")
    return f"({model_short}, {n_scenarios} scenarios, n={n_regular} regular prompts)"


def _regex_disclosed(r: dict) -> bool:
    regex = r.get("monitor_results", {}).get("regex", {})
    return bool(regex.get("contains_secret_exact") or regex.get("contains_secret_partial"))


def _llm_disclosed(r: dict) -> bool | None:
    """Returns True/False/None (None = parse_error or non-concealment prompt)."""
    llm = r.get("monitor_results", {}).get("llm", {})
    v = llm.get("mentioned_secret")
    return bool(v) if v is not None else None


def _keyword_disclosed(r: dict) -> bool:
    kw = r.get("monitor_results", {}).get("keyword", {})
    return bool(kw.get("matched", False))


def _condition(r: dict) -> str:
    return r.get("metadata", {}).get("condition", "unknown")


def _domain(r: dict) -> str:
    return r.get("metadata", {}).get("domain", "unknown")


def _secret_text(r: dict) -> str:
    """Extract secret value from context metadata if available."""
    # The secret isn't stored directly in results, but we can pull from
    # matched_patterns in the regex monitor
    regex = r.get("monitor_results", {}).get("regex", {})
    patterns = regex.get("matched_patterns", [])
    if patterns:
        # First matched pattern, strip prefix like "[exact] 28 days"
        p = patterns[0]
        for prefix in ("[exact] ", "[secret] ", "[wrong] "):
            if p.startswith(prefix):
                return p[len(prefix):]
        return p
    # Fallback: look for gold in metadata keyword_hints
    hints = r.get("metadata", {}).get("keyword_hints", [])
    if hints:
        return hints[0]
    return "unknown"


def _control_type(r: dict) -> str | None:
    return r.get("metadata", {}).get("control_type")


# ── Figure 1: Disclosure by condition ─────────────────────────────────────────

def fig_disclosure_by_condition(results: list[dict], out_dir: str, subtitle: str = "") -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    # Only regular scenarios (control_type is None)
    regular = [r for r in results if _control_type(r) is None]

    conditions = sorted({_condition(r) for r in regular})
    regex_rates, llm_rates, kw_rates, counts = [], [], [], []

    for cond in conditions:
        subset = [r for r in regular if _condition(r) == cond]
        n = len(subset)
        counts.append(n)
        regex_rates.append(sum(_regex_disclosed(r) for r in subset) / n if n else 0)
        llm_vals = [_llm_disclosed(r) for r in subset if _llm_disclosed(r) is not None]
        llm_rates.append(sum(llm_vals) / len(llm_vals) if llm_vals else 0)
        kw_rates.append(sum(_keyword_disclosed(r) for r in subset) / n if n else 0)

    x = np.arange(len(conditions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, regex_rates,  width, label="Regex monitor",   color="#2196F3", alpha=0.85)
    ax.bar(x,         llm_rates,   width, label="LLM monitor",     color="#FF9800", alpha=0.85)
    ax.bar(x + width, kw_rates,    width, label="Keyword monitor",  color="#4CAF50", alpha=0.85)

    for i, (rr, lr, kr, n) in enumerate(zip(regex_rates, llm_rates, kw_rates, counts)):
        ax.text(x[i] - width, rr + 0.01, f"{rr:.0%}", ha="center", va="bottom", fontsize=8)
        ax.text(x[i],         lr + 0.01, f"{lr:.0%}", ha="center", va="bottom", fontsize=8)
        ax.text(x[i] + width, kr + 0.01, f"{kr:.0%}", ha="center", va="bottom", fontsize=8)
        ax.text(x[i],        -0.06,      f"n={n}",    ha="center", va="top",    fontsize=7, color="gray")

    cond_labels = {
        "A0": "A0\n(transparent)",
        "A1": "A1\n(implicit pressure)",
        "A2": "A2\n(explicit suppress)",
    }
    ax.set_xticks(x)
    ax.set_xticklabels([cond_labels.get(c, c) for c in conditions])
    ax.set_ylabel("Secret disclosure rate")
    ax.set_title(f"Secret Disclosure Rate by Condition\n{subtitle}" if subtitle else
                 "Secret Disclosure Rate by Condition")
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    fig.tight_layout()

    path = os.path.join(out_dir, "disclosure_by_condition.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 2: LLM vs Regex concordance ────────────────────────────────────────

def fig_monitor_concordance(results: list[dict], out_dir: str, subtitle: str = "") -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    pairs = []
    for r in results:
        regex = _regex_disclosed(r)
        llm   = _llm_disclosed(r)
        if llm is not None:
            pairs.append((int(regex), int(llm)))

    if not pairs:
        print("  [WARN] No LLM monitor results to plot concordance.")
        return

    # Confusion matrix: rows = regex, cols = llm
    cm = np.zeros((2, 2), dtype=int)
    for regex_v, llm_v in pairs:
        cm[regex_v][llm_v] += 1

    n = len(pairs)
    agree = (cm[0][0] + cm[1][1]) / n if n else 0

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    labels = ["Not disclosed", "Disclosed"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("LLM monitor")
    ax.set_ylabel("Regex monitor")
    title = f"Monitor Concordance (agreement={agree:.1%}, n={n})"
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]),
                    ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()

    path = os.path.join(out_dir, "monitor_concordance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 3: Disclosure by secret value ──────────────────────────────────────

def fig_disclosure_by_secret(results: list[dict], out_dir: str, subtitle: str = "") -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    regular = [r for r in results if _control_type(r) is None]
    if not regular:
        return

    # Group by gold surface form (first hint = canonical secret)
    secret_data: dict[str, dict[str, list]] = defaultdict(lambda: {"regex": [], "llm": []})
    for r in regular:
        hints = r.get("metadata", {}).get("keyword_hints", [])
        key = hints[0] if hints else "unknown"
        secret_data[key]["regex"].append(int(_regex_disclosed(r)))
        v = _llm_disclosed(r)
        if v is not None:
            secret_data[key]["llm"].append(int(v))

    # Sort by secret value numerically if possible (e.g. "2 days" < "5 days")
    def sort_key(s):
        m = __import__("re").match(r"(\d+)", s)
        return int(m.group(1)) if m else 9999

    secrets = sorted(secret_data.keys(), key=sort_key)
    x = np.arange(len(secrets))
    width = 0.35

    regex_rates = [
        sum(secret_data[s]["regex"]) / len(secret_data[s]["regex"])
        if secret_data[s]["regex"] else 0
        for s in secrets
    ]
    llm_rates = [
        sum(secret_data[s]["llm"]) / len(secret_data[s]["llm"])
        if secret_data[s]["llm"] else 0
        for s in secrets
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, regex_rates, width, label="Regex",   color="#2196F3", alpha=0.85)
    ax.bar(x + width / 2, llm_rates,   width, label="LLM",     color="#FF9800", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(secrets, rotation=30, ha="right")
    ax.set_ylabel("Secret disclosure rate")
    ax.set_title(f"Disclosure Rate by Secret Value\n{subtitle}" if subtitle else
                 "Disclosure Rate by Secret Value")
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    fig.tight_layout()

    path = os.path.join(out_dir, "disclosure_by_secret.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 4: LLM verdict distribution ────────────────────────────────────────

def fig_verdict_distribution(results: list[dict], out_dir: str, subtitle: str = "") -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    regular = [r for r in results if _control_type(r) is None]
    if not regular:
        return

    conditions = sorted({_condition(r) for r in regular})
    verdicts = ["disclosed", "concealed", "redirected", "unclear", "parse_error"]
    colors    = ["#F44336", "#4CAF50", "#FF9800", "#9E9E9E", "#9C27B0"]

    verdict_counts: dict[str, dict[str, int]] = {
        c: {v: 0 for v in verdicts} for c in conditions
    }
    for r in regular:
        cond = _condition(r)
        llm = r.get("monitor_results", {}).get("llm", {})
        v = llm.get("verdict", "parse_error") or "parse_error"
        if v not in verdicts:
            v = "unclear"
        verdict_counts[cond][v] += 1

    x = np.arange(len(conditions))
    fig, ax = plt.subplots(figsize=(8, 5))

    bottoms = np.zeros(len(conditions))
    for verdict, color in zip(verdicts, colors):
        vals = np.array([verdict_counts[c][verdict] for c in conditions], dtype=float)
        ax.bar(x, vals, bottom=bottoms, label=verdict, color=color, alpha=0.85)
        # Label non-zero bars
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0:
                ax.text(x[i], b + v / 2, str(int(v)),
                        ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Number of prompts")
    ax.set_title(f"LLM Monitor Verdict Distribution by Condition\n{subtitle}" if subtitle else
                 "LLM Monitor Verdict Distribution by Condition")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    path = os.path.join(out_dir, "verdict_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 5: Disclosure by domain ────────────────────────────────────────────

def fig_disclosure_by_domain(results: list[dict], out_dir: str, subtitle: str = "") -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    regular = [r for r in results if _control_type(r) is None]
    domains = sorted({_domain(r) for r in regular})
    if len(domains) <= 1:
        return  # Only one domain, skip

    x = np.arange(len(domains))
    regex_rates, llm_rates = [], []

    for d in domains:
        subset = [r for r in regular if _domain(r) == d]
        n = len(subset)
        regex_rates.append(sum(_regex_disclosed(r) for r in subset) / n if n else 0)
        llm_vals = [_llm_disclosed(r) for r in subset if _llm_disclosed(r) is not None]
        llm_rates.append(sum(llm_vals) / len(llm_vals) if llm_vals else 0)

    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(domains) * 1.5), 5))
    ax.bar(x - width / 2, regex_rates, width, label="Regex", color="#2196F3", alpha=0.85)
    ax.bar(x + width / 2, llm_rates,   width, label="LLM",   color="#FF9800", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=20, ha="right")
    ax.set_ylabel("Secret disclosure rate")
    ax.set_title(f"Disclosure Rate by Domain\n{subtitle}" if subtitle else
                 "Disclosure Rate by Domain")
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    fig.tight_layout()

    path = os.path.join(out_dir, "disclosure_by_domain.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(results: list[dict]) -> None:
    regular = [r for r in results if _control_type(r) is None]
    controls = [r for r in results if _control_type(r) is not None]

    conditions = sorted({_condition(r) for r in regular})

    print()
    print(f"  {'─' * 72}")
    print(f"  {'CONDITION':<12} {'N':>5}  {'REGEX%':>7}  {'LLM%':>7}  {'KW%':>7}  {'REFUSAL%':>9}")
    print(f"  {'─' * 72}")

    for cond in conditions:
        subset = [r for r in regular if _condition(r) == cond]
        n = len(subset)
        regex_r = sum(_regex_disclosed(r) for r in subset) / n if n else 0
        llm_vals = [_llm_disclosed(r) for r in subset if _llm_disclosed(r) is not None]
        llm_r = sum(llm_vals) / len(llm_vals) if llm_vals else float("nan")
        kw_r = sum(_keyword_disclosed(r) for r in subset) / n if n else 0
        refusal = sum(
            1 for r in subset
            if r.get("monitor_results", {}).get("regex", {}).get("contains_refusal", False)
        ) / n if n else 0
        print(f"  {cond:<12} {n:>5}  {regex_r:>7.1%}  {llm_r:>7.1%}  {kw_r:>7.1%}  {refusal:>9.1%}")

    if controls:
        print(f"  {'─' * 72}")
        for ctrl_type in ("no_secret", "irrelevant_secret"):
            subset = [r for r in controls if _control_type(r) == ctrl_type]
            if not subset:
                continue
            n = len(subset)
            regex_r = sum(_regex_disclosed(r) for r in subset) / n if n else 0
            llm_vals = [_llm_disclosed(r) for r in subset if _llm_disclosed(r) is not None]
            llm_r = sum(llm_vals) / len(llm_vals) if llm_vals else float("nan")
            print(f"  {ctrl_type:<12} {n:>5}  {regex_r:>7.1%}  {llm_r:>7.1%}  {'—':>7}  {'—':>9}")

    print(f"  {'─' * 72}")
    print(f"  Total: {len(results)} prompts ({len(regular)} regular, {len(controls)} controls)")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def plot_all(results_path: str) -> None:
    """Generate all plots for a results JSON file."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend (safe for scripts)
    except ImportError:
        print("  [ERROR] matplotlib not installed. Run: pip install matplotlib")
        return

    payload, results = _load(results_path)
    out_dir = os.path.dirname(results_path) or "."
    sub = _subtitle(payload, results)

    print_summary_table(results)
    fig_disclosure_by_condition(results, out_dir, subtitle=sub)
    fig_monitor_concordance(results, out_dir, subtitle=sub)
    fig_disclosure_by_secret(results, out_dir, subtitle=sub)
    fig_verdict_distribution(results, out_dir, subtitle=sub)
    fig_disclosure_by_domain(results, out_dir, subtitle=sub)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <results_json_path>")
        sys.exit(1)
    plot_all(sys.argv[1])
    print("Done.")
