from abc import ABC, abstractmethod


class BasePromptLoader(ABC):
    """Abstract base class for prompt loaders.

    Subclasses must implement ``load()`` and return a list of prompt dicts.

    Prompt dict schema
    ------------------
    Required keys:
        ``id``   : str  — unique identifier for the prompt
        ``text`` : str  — the full prompt text sent to the base model

    Optional keys:
        ``metadata`` : dict — arbitrary per-prompt context forwarded to
                       PromptResult.metadata and available to all monitors.

    Metadata conventions
    --------------------
    Loaders may populate ``metadata`` with keys that downstream monitors
    recognise. The following keys are part of the established convention:

    ``keyword_hints`` : list[str]
        Additional terms the KeywordMonitor should search for in the
        reasoning trace for this specific prompt. Use this when the
        terms to watch are only known at load time.

        Example (MedQA obfuscation): the real medication names that the
        model should never use — ["SERTRALINE", "FLUOXETINE", "PAROXETINE"].

    Any other keys are passed through untouched and appear verbatim in the
    output JSON, making it easy to carry dataset-specific fields
    (correct answer, domain, mapping, etc.) alongside the result.
    """

    @abstractmethod
    def load(self) -> list[dict]:
        """Return a list of prompt dicts."""
