from .base_monitor import BaseMonitor


class KeywordMonitor(BaseMonitor):
    """Heuristic monitor that checks for the presence of keywords or phrases in the trace.

    Args:
        keywords: Words or phrases to search for.
        case_sensitive: If False (default), matching ignores case.
    """

    def __init__(self, keywords: list[str], case_sensitive: bool = False):
        self.keywords = keywords
        self.case_sensitive = case_sensitive

    @property
    def name(self) -> str:
        return "keyword"

    def run(self, trace: str) -> dict:
        haystack = trace if self.case_sensitive else trace.lower()
        found = [
            kw for kw in self.keywords
            if (kw if self.case_sensitive else kw.lower()) in haystack
        ]
        return {
            "matched": len(found) > 0,
            "found": found,
            "keywords_checked": self.keywords,
            "case_sensitive": self.case_sensitive,
        }
