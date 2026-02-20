from src.monitors.base_monitor import BaseMonitor
from src.pipeline.result import PromptResult
from src.pipeline.step import PipelineStep


class MonitorStep(PipelineStep):
    """Pipeline step that fans out to one or more monitors.

    Each monitor receives the full PromptResult and may inspect the reasoning
    trace, final answer, and any loader-supplied metadata (e.g. keyword_hints).
    """

    def __init__(self, monitors: list[BaseMonitor]):
        self._monitors = monitors

    @property
    def name(self) -> str:
        return "monitor"

    def run(self, result: PromptResult) -> PromptResult:
        for monitor in self._monitors:
            result.monitor_results[monitor.name] = monitor.run(result)
        return result
