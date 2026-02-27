from dataclasses import dataclass, field


@dataclass
class Secret:
    """A secret value embedded in a scenario context."""
    text: str                          # primary canonical form (e.g. "28 days")
    surface_forms: list[str]           # all valid surface forms (e.g. ["28 days", "4 weeks"])
    structured: dict = field(default_factory=dict)  # domain-specific structured data


@dataclass
class Scenario:
    """One generated scenario (pre-serialisation)."""
    example_id: str
    domain: str
    control_type: str | None           # None | "no_secret" | "irrelevant_secret"
    secret: Secret
    context: str
    conditions: dict                   # {"A0": str, "A1": str, "A2": str}
    queries: dict                      # {"B0": str, "B1": str, "B1_paraphrases": [...], "followups": [...]}
    gold: list[str]
    regex_monitor: dict

    def to_dict(self) -> dict:
        return {
            "example_id": self.example_id,
            "domain": self.domain,
            "control_type": self.control_type,
            "secret": {
                "text": self.secret.text,
                "surface_forms": self.secret.surface_forms,
                "structured": self.secret.structured,
            },
            "context": self.context,
            "conditions": self.conditions,
            "queries": self.queries,
            "gold": self.gold,
            "regex_monitor": self.regex_monitor,
        }
