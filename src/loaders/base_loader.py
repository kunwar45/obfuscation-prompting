from abc import ABC, abstractmethod


class BasePromptLoader(ABC):
    @abstractmethod
    def load(self) -> list[dict]:
        """Return a list of prompt dicts with schema: {"id": str, "text": str}"""
