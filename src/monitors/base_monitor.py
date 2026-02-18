from abc import ABC, abstractmethod


class BaseMonitor(ABC):
    """A monitor receives a reasoning trace and returns a dict of findings."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier used as the key in monitor_results."""

    @abstractmethod
    def run(self, trace: str) -> dict:
        """Analyse the trace and return a serialisable dict."""
