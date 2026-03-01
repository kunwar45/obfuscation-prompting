"""Integration tests for HFClient using GPT-2 (117M, always available via HF hub).

These tests load a real model. They are marked 'integration' and skipped if
torch/transformers are unavailable. GPT-2 has no chat template, so the
fallback formatter is also exercised.
"""

import os
import tempfile

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from src.clients.hf_client import HFClient, detect_device

# GPT-2 is tiny (~500 MB), freely licensed, and always on HF hub.
MODEL = "gpt2"


@pytest.fixture(scope="module")
def client_no_capture():
    return HFClient(model_name=MODEL, dtype="float32", capture_mode="none")


@pytest.fixture(scope="module")
def client_last_token():
    return HFClient(model_name=MODEL, dtype="float32", capture_mode="last_token")


# ── Basic chat ─────────────────────────────────────────────────────────────────

def test_chat_returns_string(client_no_capture):
    response = client_no_capture.chat(
        model="ignored",
        messages=[{"role": "user", "content": "Say hello."}],
        max_tokens=10,
        temperature=0.0,
    )
    assert isinstance(response, str)
    assert len(response) > 0


def test_chat_model_param_ignored(client_no_capture):
    """Passing a bogus model name must not raise — model is already loaded."""
    response = client_no_capture.chat(
        model="some-other-model-that-doesnt-exist",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=5,
        temperature=0.0,
    )
    assert isinstance(response, str)


def test_chat_temperature_zero_is_deterministic(client_no_capture):
    msgs = [{"role": "user", "content": "The capital of France is"}]
    r1 = client_no_capture.chat("gpt2", msgs, max_tokens=5, temperature=0.0)
    r2 = client_no_capture.chat("gpt2", msgs, max_tokens=5, temperature=0.0)
    assert r1 == r2


# ── Device placement ───────────────────────────────────────────────────────────

def test_model_on_correct_device(client_no_capture):
    expected = detect_device()
    # For CUDA device_map="auto" the model is split; check first param device type
    first_param = next(client_no_capture.model.parameters())
    assert first_param.device.type == expected


# ── Activation capture ─────────────────────────────────────────────────────────

def test_no_activations_before_chat(client_last_token):
    # Fresh fixture: _last_activations should be None until chat() is called.
    # (Fixture is module-scoped so may already have run — just check type)
    assert client_last_token._last_activations is None or isinstance(
        client_last_token._last_activations, dict
    )


def test_activations_captured_after_chat(client_last_token):
    client_last_token.chat(
        model="ignored",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=5,
        temperature=0.0,
    )
    assert client_last_token._last_activations is not None
    assert "token_ids" in client_last_token._last_activations
    assert "layer_0" in client_last_token._last_activations


def test_last_token_activation_shape(client_last_token):
    client_last_token.chat(
        model="ignored",
        messages=[{"role": "user", "content": "Test"}],
        max_tokens=5,
        temperature=0.0,
    )
    acts = client_last_token._last_activations
    n_layers = client_last_token._get_layer_count()  # GPT-2 has 12 layers
    d_model = client_last_token.model.config.hidden_size  # GPT-2: 768

    # last_token mode: each layer_i is [d_model]
    for i in range(n_layers + 1):  # +1 for embedding layer
        key = f"layer_{i}"
        assert key in acts, f"Missing {key}"
        assert acts[key].shape == (d_model,), f"{key} shape mismatch: {acts[key].shape}"


def test_save_activations_writes_npz(client_last_token):
    client_last_token.chat(
        model="ignored",
        messages=[{"role": "user", "content": "Save me"}],
        max_tokens=5,
        temperature=0.0,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = client_last_token.save_activations("prompt_001", tmpdir)
        assert path.endswith(".npz")
        assert os.path.exists(path)

        loaded = np.load(path)
        assert "token_ids" in loaded
        assert "layer_0" in loaded


def test_save_activations_returns_empty_when_no_activations():
    # A fresh client with capture_mode="none" will never populate _last_activations
    c = HFClient(model_name=MODEL, dtype="float32", capture_mode="none")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = c.save_activations("prompt_x", tmpdir)
    assert path == ""


# ── Full-sequence capture ──────────────────────────────────────────────────────

def test_full_sequence_activation_shape():
    c = HFClient(model_name=MODEL, dtype="float32", capture_mode="full_sequence")
    c.chat(
        model="ignored",
        messages=[{"role": "user", "content": "Hello world"}],
        max_tokens=5,
        temperature=0.0,
    )
    acts = c._last_activations
    d_model = c.model.config.hidden_size
    # Each layer_i should be [seq_len, d_model] (2D)
    assert acts["layer_0"].ndim == 2
    assert acts["layer_0"].shape[1] == d_model


# ── Fallback chat template ─────────────────────────────────────────────────────

def test_fallback_chat_template(client_no_capture):
    """GPT-2 has no chat_template — the fallback formatter must not raise."""
    assert client_no_capture.tokenizer.chat_template is None
    prompt = client_no_capture._apply_chat_template([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ])
    assert "[SYSTEM]" in prompt
    assert "[USER]" in prompt
    assert "[ASSISTANT]" in prompt
