from __future__ import annotations

import re


class MockClient:
    """Deterministic no-network client for end-to-end pipeline smoke tests."""

    def chat(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        _ = (model, max_tokens, temperature)
        system = ""
        user = ""
        for message in messages:
            if message.get("role") == "system":
                system = message.get("content", "")
            elif message.get("role") == "user":
                user = message.get("content", "")

        answer = self._infer_answer(system=system, user=user)
        return (
            "<reasoning>\n"
            f"I extracted the most likely direct answer from prompt/context: {answer}.\n"
            "</reasoning>\n"
            "<answer>\n"
            f"{answer}\n"
            "</answer>"
        )

    @staticmethod
    def _infer_answer(system: str, user: str) -> str:
        text = f"{system}\n{user}"
        days = re.search(r"\b(\d+)\s*days?\b", text, flags=re.IGNORECASE)
        weeks = re.search(r"\b(\d+)\s*weeks?\b", text, flags=re.IGNORECASE)
        if days:
            return f"{days.group(1)} days"
        if weeks:
            return f"{weeks.group(1)} weeks"
        return "mock-answer"

