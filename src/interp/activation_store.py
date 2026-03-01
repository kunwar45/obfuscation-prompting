"""ActivationStore: load and index .npz activation files for downstream analysis."""

from __future__ import annotations

import json
import os
from typing import Callable

import numpy as np


class ActivationStore:
    """Index activations saved by HFClient.save_activations().

    Usage:
        store = ActivationStore("activations/<run_id>", "results/<run>.json")
        X = store.load_layer(0)          # [n_prompts, d_model]
        labels = store.get_labels(lambda r: int(r["monitor_results"]["keyword"]["matched"]))
    """

    def __init__(self, activations_dir: str, run_results_path: str):
        self.activations_dir = activations_dir

        with open(run_results_path) as f:
            self._results: list[dict] = json.load(f)

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

    def load_layer(self, layer_idx: int) -> np.ndarray:
        """Return [n_prompts, d_model] matrix of last-token activations for one layer.

        Assumes activations were captured with capture_mode="last_token".
        For full_sequence mode each row is the last token position.
        """
        key = f"layer_{layer_idx}"
        rows = []
        for pid in self._prompt_ids:
            data = self.load(pid)
            arr = data[key]
            # If full_sequence: arr is [seq_len, d_model]; take last token
            if arr.ndim == 2:
                arr = arr[-1]
            rows.append(arr)
        return np.stack(rows, axis=0)

    def get_labels(self, label_fn: Callable[[dict], int]) -> np.ndarray:
        """Apply label_fn to each result dict to build a label array.

        Example:
            labels = store.get_labels(
                lambda r: int(r["monitor_results"]["keyword"]["matched"])
            )
        """
        labels = []
        for result in self._results:
            if result["prompt_id"] in self._index:
                labels.append(label_fn(result))
        return np.array(labels, dtype=int)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def prompt_ids(self) -> list[str]:
        return list(self._prompt_ids)

    def __len__(self) -> int:
        return len(self._prompt_ids)
