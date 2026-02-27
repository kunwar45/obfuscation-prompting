from __future__ import annotations

import json

from .base_loader import BasePromptLoader


class ConcealmentLoader(BasePromptLoader):
    """Load and expand a concealment dataset JSONL into pipeline prompt dicts.

    Each JSONL line is a scenario. This loader expands each scenario into one
    prompt dict per (condition, query_type) pair selected by configuration.

    Prompt dict schema
    ------------------
    ``id``       : ``{example_id}_{condition}_{query_type}``
                   e.g. ``shipping_0042_A1_B1``
    ``text``     : ``{context}\\n\\n{query}``
    ``metadata`` : {
        ``system_prompt``  : the A-condition string (used by BaseModelStep),
        ``keyword_hints``  : gold surface forms (for KeywordMonitor),
        ``regex_monitor``  : precomputed pattern dict (for RegexMonitor),
        ``example_id``     : original scenario ID,
        ``domain``         : domain name,
        ``condition``      : "A0" | "A1" | "A2",
        ``query_type``     : "B0" | "B1" | "B1_p0" | ... | "followup_0" | ...,
        ``control_type``   : null | "no_secret" | "irrelevant_secret",
    }

    Control scenarios
    -----------------
    - ``no_secret`` controls: only condition A0, only query_type B1
    - ``irrelevant_secret`` controls: only condition A0, only query_type B0

    Args:
        jsonl_path:   Path to the generated .jsonl file.
        conditions:   Comma-separated conditions to expand, e.g. "A0,A1,A2".
        query_types:  Comma-separated query types to expand, e.g. "B1".
                      Supported values: B0, B1, B1_p0..B1_p2, followup_0.
    """

    def __init__(
        self,
        jsonl_path: str,
        conditions: str = "A0,A1,A2",
        query_types: str = "B1",
    ):
        self.jsonl_path = jsonl_path
        self.conditions = [c.strip() for c in conditions.split(",") if c.strip()]
        self.query_types = [q.strip() for q in query_types.split(",") if q.strip()]

    def load(self) -> list[dict]:
        scenarios = self._read_jsonl()
        prompts = []
        for scenario in scenarios:
            prompts.extend(self._expand_scenario(scenario))
        return prompts

    def _read_jsonl(self) -> list[dict]:
        scenarios = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    scenarios.append(json.loads(line))
        return scenarios

    def _expand_scenario(self, scenario: dict) -> list[dict]:
        control_type = scenario.get("control_type")
        example_id = scenario["example_id"]
        domain = scenario["domain"]
        context = scenario["context"]
        conditions_map = scenario["conditions"]
        queries_map = scenario["queries"]
        gold = scenario.get("gold", [])
        regex_monitor = scenario.get("regex_monitor", {})

        # Determine which (condition, query_type) pairs apply
        if control_type == "no_secret":
            pairs = [("A0", "B1")]
        elif control_type == "irrelevant_secret":
            pairs = [("A0", "B0")]
        else:
            pairs = [
                (cond, qt)
                for cond in self.conditions
                for qt in self.query_types
                if cond in conditions_map
            ]

        prompts = []
        for condition, query_type in pairs:
            query = self._resolve_query(queries_map, query_type)
            if query is None:
                continue
            system_prompt = conditions_map.get(condition, "")
            prompt_id = f"{example_id}_{condition}_{query_type}"
            prompts.append({
                "id": prompt_id,
                "text": f"{context}\n\n{query}",
                "metadata": {
                    "system_prompt": system_prompt,
                    "keyword_hints": gold,
                    "regex_monitor": regex_monitor,
                    "example_id": example_id,
                    "domain": domain,
                    "condition": condition,
                    "query_type": query_type,
                    "control_type": control_type,
                },
            })
        return prompts

    @staticmethod
    def _resolve_query(queries_map: dict, query_type: str) -> str | None:
        """Resolve a query_type string to the actual query text.

        Supported query_type values:
          B0              → queries["B0"]
          B1              → queries["B1"]
          B1_p0..B1_p2   → queries["B1_paraphrases"][0..2]
          followup_0      → queries["followups"][0]
        """
        if query_type in ("B0", "B1"):
            return queries_map.get(query_type)

        if query_type.startswith("B1_p"):
            try:
                idx = int(query_type[4:])
                paraphrases = queries_map.get("B1_paraphrases", [])
                if idx < len(paraphrases):
                    return paraphrases[idx]
            except (ValueError, IndexError):
                pass
            return None

        if query_type.startswith("followup_"):
            try:
                idx = int(query_type[9:])
                followups = queries_map.get("followups", [])
                if idx < len(followups):
                    return followups[idx]
            except (ValueError, IndexError):
                pass
            return None

        return None
