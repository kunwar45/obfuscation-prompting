"""Tests for device detection logic."""

import torch
from src.clients.hf_client import detect_device


def test_detect_device_returns_valid_string():
    device = detect_device()
    assert device in ("cuda", "mps", "cpu")


def test_detect_device_priority():
    """If CUDA is available it should win; else MPS; else CPU."""
    device = detect_device()
    if torch.cuda.is_available():
        assert device == "cuda"
    elif torch.backends.mps.is_available():
        assert device == "mps"
    else:
        assert device == "cpu"


def test_detect_device_is_usable():
    """The returned device string should be a valid torch device."""
    device = detect_device()
    t = torch.zeros(2, device=device)
    assert t.device.type == device
