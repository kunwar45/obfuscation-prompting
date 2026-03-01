"""Tests for ChatClient Protocol and client interchangeability."""

import pytest
from src.clients import ChatClient


class _FakeClient:
    """Minimal object that satisfies ChatClient Protocol without importing torch."""
    def chat(self, model: str, messages: list[dict], max_tokens: int, temperature: float) -> str:
        return "hello"


def test_fake_client_satisfies_protocol():
    assert isinstance(_FakeClient(), ChatClient)


def test_together_client_satisfies_protocol():
    """TogetherClient must satisfy ChatClient without instantiating (just check class)."""
    pytest.importorskip("together", reason="together package not installed in this env")
    from src.clients.together_client import TogetherClient
    import inspect
    sig = inspect.signature(TogetherClient.chat)
    params = list(sig.parameters.keys())
    # self, model, messages, max_tokens, temperature
    assert "model" in params
    assert "messages" in params
    assert "max_tokens" in params
    assert "temperature" in params


def test_hf_client_class_satisfies_protocol():
    """HFClient.chat must have the same signature (without instantiating the model)."""
    from src.clients.hf_client import HFClient
    import inspect
    sig = inspect.signature(HFClient.chat)
    params = list(sig.parameters.keys())
    assert "model" in params
    assert "messages" in params
    assert "max_tokens" in params
    assert "temperature" in params
