from __future__ import annotations

import json

from .base_loader import BasePromptLoader


class ConcealmentLoader(BasePromptLoader):
    """Load and expand a concealment dataset JSONL into pipeline prompt dicts.

    Each JSONL line is a scenario. This loader expands each scenario into one
    prompt dict per (condition, query_type) pair selected by configuration.

    Message structure
    -----------------
    The context paragraph (the agent's internal background knowledge) is injected
    into the system message alongside the A-condition string:

        system: "{A-condition}\\n\\n{context}"
        user:   "{query}"

    This mirrors a realistic deployment where the agent has private background
    knowledge (a delay notice, a bug ticket) that the user has not seen.
    The full system content is stored in ``metadata["system_prompt"]`` so
    BaseModelStep uses it in place of the default system prompt.

    If a scenario contains ``context_variants`` (added by LLMAugmenter), additional
    prompts are generated for each variant context, with the same (condition,
    query_type) pairs as the main scenario.

    Prompt dict schema
    ------------------
    ``id``       : ``{example_id}_{condition}_{query_type}``
                   e.g. ``shipping_0042_A1_B1``
                   For context variants: ``{example_id}_cv{i}_{condition}_{query_type}``
    ``text``     : the user query only
    ``metadata`` : {
        ``system_prompt``   : "{A-condition}\\n\\n{context}" (used by BaseModelStep),
        ``context``         : the raw context paragraph (for auditing),
        ``keyword_hints``   : gold surface forms (for KeywordMonitor),
        ``regex_monitor``   : precomputed pattern dict (for RegexMonitor),
        ``example_id``      : original scenario ID,
        ``domain``          : domain name,
        ``condition``       : "A0" | "A1" | "A2",
        ``query_type``      : "B0" | "B1" | "B1_p0" | "B1_llm_p0" | "followup_0" | ...,
        ``control_type``    : null | "no_secret" | "irrelevant_secret",
        ``context_variant`` : null | int  (index into context_variants, or null for main)
    }

    Control scenarios
    -----------------
    - ``no_secret`` controls: only condition A0, only query_type B1
    - ``irrelevant_secret`` controls: only condition A0, only query_type B0

    Args:
        jsonl_path:   Path to the generated .jsonl file.
        conditions:   Comma-separated conditions to expand, e.g. "A0,A1,A2".
        query_types:  Comma-separated query types to expand, e.g. "B1".
                      Supported values: B0, B1, B1_p0..B1_p{n}, B1_llm_p0..B1_llm_p{n},
                      followup_0..followup_{n}.
        limit:        Max number of *scenarios* to load (None = all). Applied before
                      expansion, so the number of resulting prompts will be
                      limit × len(conditions) × len(query_types) for regular scenarios.
    """

    def __init__(
        self,
        jsonl_path: str,
        conditions: str = "A0,A1,A2",
        query_types: str = "B1",
        limit: int | None = None,
    ):
        self.jsonl_path = jsonl_path
        self.conditions = [c.strip() for c in conditions.split(",") if c.strip()]
        self.query_types = [q.strip() for q in query_types.split(",") if q.strip()]
        self.limit = limit

    def load(self) -> list[dict]:
        scenarios = self._read_jsonl()
        if self.limit:
            scenarios = scenarios[: self.limit]
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
        context_variants: list[str] = scenario.get("context_variants", [])

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

        # Main context prompts
        for condition, query_type in pairs:
            query = self._resolve_query(queries_map, query_type)
            if query is None:
                continue
            system_prompt = self._build_system_prompt(conditions_map.get(condition, ""), context)
            prompts.append({
                "id": f"{example_id}_{condition}_{query_type}",
                "text": query,
                "metadata": {
                    "system_prompt": system_prompt,
                    "context": context,
                    "keyword_hints": gold,
                    "regex_monitor": regex_monitor,
                    "example_id": example_id,
                    "domain": domain,
                    "condition": condition,
                    "query_type": query_type,
                    "control_type": control_type,
                    "context_variant": None,
                },
            })

        # LLM context variant prompts (regular scenarios only)
        if control_type is None:
            for i, variant_context in enumerate(context_variants):
                for condition, query_type in pairs:
                    query = self._resolve_query(queries_map, query_type)
                    if query is None:
                        continue
                    system_prompt = self._build_system_prompt(
                        conditions_map.get(condition, ""), variant_context
                    )
                    prompts.append({
                        "id": f"{example_id}_cv{i}_{condition}_{query_type}",
                        "text": query,
                        "metadata": {
                            "system_prompt": system_prompt,
                            "context": variant_context,
                            "keyword_hints": gold,
                            "regex_monitor": regex_monitor,
                            "example_id": example_id,
                            "domain": domain,
                            "condition": condition,
                            "query_type": query_type,
                            "control_type": control_type,
                            "context_variant": i,
                        },
                    })

        return prompts

    @staticmethod
    def _build_system_prompt(condition: str, context: str) -> str:
        """Combine the A-condition string with the context paragraph.

        The context is separated from the condition by a blank line so the model
        receives a clear boundary between its role description and its background
        knowledge for this interaction.
        """
        return f"{condition}\n\n{context}"

    @staticmethod
    def _resolve_query(queries_map: dict, query_type: str) -> str | None:
        """Resolve a query_type string to the actual query text.

        Supported query_type values:
          B0                  → queries["B0"]
          B1                  → queries["B1"]
          B1_p0..B1_p{n}      → queries["B1_paraphrases"][n]   (hand-crafted)
          B1_llm_p0..p{n}     → queries["B1_llm_paraphrases"][n]  (LLM-generated)
          followup_0..{n}     → queries["followups"][n]
        """
        if query_type in ("B0", "B1"):
            return queries_map.get(query_type)

        if query_type.startswith("B1_llm_p"):
            try:
                idx = int(query_type[8:])
                llm_paraphrases = queries_map.get("B1_llm_paraphrases", [])
                if idx < len(llm_paraphrases):
                    return llm_paraphrases[idx]
            except (ValueError, IndexError):
                pass
            return None

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
