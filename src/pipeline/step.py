from abc import ABC, abstractmethod

from .result import PromptResult


class PipelineStep(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def run(self, result: PromptResult) -> PromptResult:
        pass
