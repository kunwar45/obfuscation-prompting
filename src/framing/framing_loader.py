"""FramingLoader — load a concealment JSONL using explicit framing conditions.

Mirrors ``ConcealmentLoader`` exactly, but replaces A0/A1/A2 conditions with
the 13 framing conditions defined in ``src.framing.conditions``.

Prompt dict schema
------------------
``id``       : ``{example_id}_{framing_key}_{query_type}``
               e.g. ``shipping_0003_M_inst_B1``
``text``     : the user query only
``metadata`` : {
    ``system_prompt``      : "{framing_template}\\n\\n{context}",
    ``context``            : raw context paragraph (for auditing),
    ``keyword_hints``      : gold surface forms (for KeywordMonitor),
    ``regex_monitor``      : precomputed pattern dict (for RegexMonitor),
    ``example_id``         : original scenario ID,
    ``domain``             : domain name,
    ``condition``          : framing_key  (backward-compat with plot_results.py),
    ``query_type``         : "B1" | "B0" | ...,
    ``control_type``       : null | "no_secret" | "irrelevant_secret",
    ``context_variant``    : null | int,
    ``framing_condition``  : framing_key (same as ``condition``),
    ``framing_dimension``  : dimension string (e.g. "motivation"),
    ``framing_label``      : human-readable label,
    ``framing_template``   : the raw template string,
    ``secret_structured``  : scenario["secret"]["structured"] dict,
}

Control scenarios
-----------------
- ``no_secret``         : BASE framing only, query_type B1
- ``irrelevant_secret`` : BASE framing only, query_type B0
"""

from __future__ import annotations

import json

from .conditions import CONDITION_ORDER, REGISTRY, render
from src.loaders.base_loader import BasePromptLoader


class FramingLoader(BasePromptLoader):
    """Load and expand a concealment JSONL using explicit framing conditions.

    Args:
        jsonl_path:    Path to the generated .jsonl file.
        role:          Role string substituted into framing templates
                       (e.g. "customer service assistant").
        framing_keys:  List of condition keys to expand.  None = all 13.
        query_types:   Comma-separated query types, e.g. "B1".
        limit:         Max number of *scenarios* to load (None = all).
                       Applied before expansion.
    """

    def __init__(
        self,
        jsonl_path: str,
        role: str = "customer service assistant",
        framing_keys: list[str] | None = None,
        query_types: str = "B1",
        limit: int | None = None,
    ):
        self.jsonl_path = jsonl_path
        self.role = role
        self.framing_keys = framing_keys if framing_keys is not None else list(CONDITION_ORDER)
        self.query_types = [q.strip() for q in query_types.split(",") if q.strip()]
        self.limit = limit

        # Validate keys
        unknown = [k for k in self.framing_keys if k not in REGISTRY]
        if unknown:
            raise ValueError(f"Unknown framing keys: {unknown!r}")

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
        queries_map = scenario["queries"]
        gold = scenario.get("gold", [])
        regex_monitor = scenario.get("regex_monitor", {})
        secret_structured = scenario.get("secret", {}).get("structured", {})
        context_variants: list[str] = scenario.get("context_variants", [])

        # Determine which (framing_key, query_type) pairs apply.
        # Controls always use BASE framing with a fixed query type.
        if control_type == "no_secret":
            pairs = [("BASE", "B1")]
        elif control_type == "irrelevant_secret":
            pairs = [("BASE", "B0")]
        else:
            pairs = [
                (fk, qt)
                for fk in self.framing_keys
                for qt in self.query_types
            ]

        prompts = []

        # Main context prompts
        for framing_key, query_type in pairs:
            query = self._resolve_query(queries_map, query_type)
            if query is None:
                continue
            system_prompt = self._build_system_prompt(framing_key, self.role, context)
            cond = REGISTRY[framing_key]
            prompts.append({
                "id": f"{example_id}_{framing_key}_{query_type}",
                "text": query,
                "metadata": {
                    "system_prompt": system_prompt,
                    "context": context,
                    "keyword_hints": gold,
                    "regex_monitor": regex_monitor,
                    "example_id": example_id,
                    "domain": domain,
                    "condition": framing_key,          # backward compat
                    "query_type": query_type,
                    "control_type": control_type,
                    "context_variant": None,
                    "framing_condition": framing_key,
                    "framing_dimension": cond.dimension,
                    "framing_label": cond.label,
                    "framing_template": cond.template,
                    "secret_structured": secret_structured,
                },
            })

        # LLM context variant prompts (regular scenarios only)
        if control_type is None:
            for i, variant_context in enumerate(context_variants):
                for framing_key, query_type in pairs:
                    query = self._resolve_query(queries_map, query_type)
                    if query is None:
                        continue
                    system_prompt = self._build_system_prompt(framing_key, self.role, variant_context)
                    cond = REGISTRY[framing_key]
                    prompts.append({
                        "id": f"{example_id}_cv{i}_{framing_key}_{query_type}",
                        "text": query,
                        "metadata": {
                            "system_prompt": system_prompt,
                            "context": variant_context,
                            "keyword_hints": gold,
                            "regex_monitor": regex_monitor,
                            "example_id": example_id,
                            "domain": domain,
                            "condition": framing_key,
                            "query_type": query_type,
                            "control_type": control_type,
                            "context_variant": i,
                            "framing_condition": framing_key,
                            "framing_dimension": cond.dimension,
                            "framing_label": cond.label,
                            "framing_template": cond.template,
                            "secret_structured": secret_structured,
                        },
                    })

        return prompts

    @staticmethod
    def _build_system_prompt(framing_key: str, role: str, context: str) -> str:
        """Combine the rendered framing template with the scenario context."""
        condition_string = render(framing_key, role)
        return f"{condition_string}\n\n{context}"

    @staticmethod
    def _resolve_query(queries_map: dict, query_type: str) -> str | None:
        """Resolve query_type string to actual query text (mirrors ConcealmentLoader)."""
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
