"""Tests for ActivationStore using synthetic .npz fixtures."""

import json
import os
import tempfile

import numpy as np
import pytest

from src.interp.activation_store import ActivationStore


def _make_fixture(tmpdir: str, n_prompts: int = 3, n_layers: int = 4, d_model: int = 16):
    """Write synthetic results JSON and .npz files, return (activations_dir, results_path)."""
    act_dir = os.path.join(tmpdir, "activations")
    os.makedirs(act_dir, exist_ok=True)

    results = []
    for i in range(n_prompts):
        pid = f"prompt_{i:03d}"
        npz_path = os.path.join(act_dir, f"{pid}.npz")

        arrays = {"token_ids": np.array([1, 2, 3, 4])}
        for layer in range(n_layers + 1):
            arrays[f"layer_{layer}"] = np.random.randn(d_model).astype(np.float32)
        np.savez_compressed(npz_path, **arrays)

        results.append({
            "prompt_id": pid,
            "prompt": f"Question {i}",
            "timestamp": "2026-01-01T00:00:00",
            "activation_path": npz_path,
            "monitor_results": {"keyword": {"matched": i % 2}},
            "metadata": {},
        })

    results_path = os.path.join(tmpdir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f)

    return act_dir, results_path, n_prompts, n_layers, d_model


def test_load_returns_dict():
    with tempfile.TemporaryDirectory() as tmpdir:
        act_dir, results_path, n, nl, d = _make_fixture(tmpdir)
        store = ActivationStore(act_dir, results_path)
        data = store.load("prompt_000")
        assert isinstance(data, dict)
        assert "token_ids" in data
        assert "layer_0" in data


def test_load_layer_shape():
    with tempfile.TemporaryDirectory() as tmpdir:
        act_dir, results_path, n, nl, d = _make_fixture(tmpdir)
        store = ActivationStore(act_dir, results_path)
        X = store.load_layer(0)
        assert X.shape == (n, d), f"Expected ({n}, {d}), got {X.shape}"


def test_load_layer_all_layers():
    with tempfile.TemporaryDirectory() as tmpdir:
        act_dir, results_path, n, nl, d = _make_fixture(tmpdir)
        store = ActivationStore(act_dir, results_path)
        for layer_idx in range(nl + 1):
            X = store.load_layer(layer_idx)
            assert X.shape == (n, d)


def test_get_labels():
    with tempfile.TemporaryDirectory() as tmpdir:
        act_dir, results_path, n, nl, d = _make_fixture(tmpdir)
        store = ActivationStore(act_dir, results_path)
        labels = store.get_labels(lambda r: int(r["monitor_results"]["keyword"]["matched"]))
        assert labels.shape == (n,)
        assert set(labels.tolist()).issubset({0, 1})


def test_len():
    with tempfile.TemporaryDirectory() as tmpdir:
        act_dir, results_path, n, nl, d = _make_fixture(tmpdir)
        store = ActivationStore(act_dir, results_path)
        assert len(store) == n


def test_prompt_ids():
    with tempfile.TemporaryDirectory() as tmpdir:
        act_dir, results_path, n, nl, d = _make_fixture(tmpdir)
        store = ActivationStore(act_dir, results_path)
        assert store.prompt_ids == [f"prompt_{i:03d}" for i in range(n)]


def test_missing_npz_not_indexed():
    """Prompts whose activation_path doesn't exist should be silently excluded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        act_dir, results_path, n, nl, d = _make_fixture(tmpdir)

        # Corrupt one result: point to a non-existent file
        with open(results_path) as f:
            results = json.load(f)
        results[0]["activation_path"] = "/nonexistent/path.npz"
        with open(results_path, "w") as f:
            json.dump(results, f)

        store = ActivationStore(act_dir, results_path)
        assert len(store) == n - 1
        assert "prompt_000" not in store.prompt_ids
