from __future__ import annotations

from typing import Protocol


class ChatClient(Protocol):
    def chat(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        """Return assistant text for a chat completion request."""

