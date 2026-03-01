from typing import Protocol, runtime_checkable


@runtime_checkable
class ChatClient(Protocol):
    def chat(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> str: ...
