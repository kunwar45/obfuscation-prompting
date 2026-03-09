"""ActivationStore: load and index .npz activation files for downstream analysis."""

from __future__ import annotations

import json
import os
from typing import Callable

import numpy as np


class ActivationStore:
    """Index activations saved by HFClient.save_activations().

    Handles both bare list JSON and the wrapped format saved by ResultStorage:
        {"run_id": ..., "results": [...]}

    Usage:
        store = ActivationStore("activations/<run_id>", "results/<run>.json")
        X = store.load_layer(0)          # [n_prompts, d_model]
        labels = store.get_labels(lambda r: int(r["monitor_results"]["keyword"]["matched"]))
    """

    def __init__(self, activations_dir: str, run_results_path: str):
        self.activations_dir = activations_dir

        with open(run_results_path) as f:
            raw = json.load(f)

        # ResultStorage wraps results: {"run_id": ..., "results": [...]}
        if isinstance(raw, dict):
            self._results: list[dict] = raw.get("results", [])
            self.run_metadata: dict = {k: v for k, v in raw.items() if k != "results"}
        else:
            self._results = raw
            self.run_metadata = {}

        # Index: prompt_id -> .npz path
        self._index: dict[str, str] = {}
        for result in self._results:
            pid = result.get("prompt_id", "")
            path = result.get("activation_path", "")
            if path and os.path.exists(path):
                self._index[pid] = path

        self._prompt_ids = [r["prompt_id"] for r in self._results if r["prompt_id"] in self._index]

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def load(self, prompt_id: str) -> dict[str, np.ndarray]:
        """Load all arrays for a single prompt. Keys: layer_0…layer_N, token_ids."""
        path = self._index[prompt_id]
        with np.load(path) as f:
            return dict(f)

    def load_layer(
        self,
        layer_idx: int,
        prompt_ids: list[str] | None = None,
    ) -> np.ndarray:
        """Return [n_prompts, d_model] matrix of last-token activations for one layer.

        Args:
            layer_idx: Layer index.
            prompt_ids: Optional subset of prompt IDs to load. Defaults to all indexed prompts.

        Assumes capture_mode="last_token". For full_sequence, takes the last position.
        """
        ids = prompt_ids if prompt_ids is not None else self._prompt_ids
        key = f"layer_{layer_idx}"
        rows = []
        for pid in ids:
            data = self.load(pid)
            arr = data[key]
            if arr.ndim == 2:
                arr = arr[-1]
            rows.append(arr)
        return np.stack(rows, axis=0)

    def get_labels(
        self,
        label_fn: Callable[[dict], int],
        prompt_ids: list[str] | None = None,
    ) -> np.ndarray:
        """Apply label_fn to each result dict to build a label array.

        Args:
            label_fn: Callable that maps a result dict to an integer label.
            prompt_ids: Optional subset to restrict labels to. Defaults to all indexed prompts.

        Example:
            labels = store.get_labels(
                lambda r: int(r["monitor_results"]["keyword"]["matched"])
            )
        """
        ids_set = set(prompt_ids) if prompt_ids is not None else None
        labels = []
        for result in self._results:
            pid = result["prompt_id"]
            if pid not in self._index:
                continue
            if ids_set is not None and pid not in ids_set:
                continue
            labels.append(label_fn(result))
        return np.array(labels, dtype=int)

    def filter_ids(self, predicate: Callable[[dict], bool]) -> list[str]:
        """Return prompt_ids where predicate(result_dict) is True.

        Example — only A0 and A2:
            ids = store.filter_ids(
                lambda r: r["metadata"]["condition"] in ("A0", "A2")
            )
        """
        return [
            r["prompt_id"]
            for r in self._results
            if r["prompt_id"] in self._index and predicate(r)
        ]

    def get_results(
        self,
        prompt_ids: list[str] | None = None,
    ) -> list[dict]:
        """Return result dicts for indexed prompts (optionally filtered)."""
        ids_set = set(prompt_ids) if prompt_ids is not None else None
        return [
            r for r in self._results
            if r["prompt_id"] in self._index
            and (ids_set is None or r["prompt_id"] in ids_set)
        ]

    def n_layers(self) -> int:
        """Number of hidden layers (inferred from first saved .npz)."""
        if not self._prompt_ids:
            return 0
        sample = self.load(self._prompt_ids[0])
        return len([k for k in sample if k.startswith("layer_")])

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def prompt_ids(self) -> list[str]:
        return list(self._prompt_ids)

    def __len__(self) -> int:
        return len(self._prompt_ids)
