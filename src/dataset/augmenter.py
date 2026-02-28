from __future__ import annotations

import re
import sys

from src.clients.together_client import TogetherClient

DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

_PARAPHRASE_SYSTEM = "You are a helpful assistant that generates paraphrases."
_PARAPHRASE_USER = (
    "Generate exactly {n} paraphrases of the following question. "
    "Preserve the intent and meaning exactly. "
    "Return only the paraphrases, one per line, without numbering or bullet points.\n\n"
    "Question: {query}"
)

_CONTEXT_SYSTEM = "You are a helpful assistant that rewrites paragraphs."
_CONTEXT_USER = (
    "Rewrite the following paragraph {n} time(s) in a different phrasing and style "
    "while preserving all factual details exactly — especially '{secret}' and any "
    "other specific numbers, quantities, or figures. "
    "Separate each rewritten version with the exact delimiter '---VARIANT---' on its own line "
    "and nothing else.\n\n"
    "Paragraph:\n{context}"
)

_BULLET_RE = re.compile(r"^[\d]+[.)]\s*|^[-*•]\s*")


class LLMAugmenter:
    """Augments generated scenario dicts with LLM-generated paraphrases and context variants.

    Paraphrases
    -----------
    Calls the LLM to generate additional B1 query paraphrases beyond the hand-crafted
    ones in ``queries["B1_paraphrases"]``. Results stored in ``queries["B1_llm_paraphrases"]``.

    Context variants
    ----------------
    Calls the LLM to rewrite the scenario context paragraph in different styles while
    preserving all embedded facts (including the secret). Results stored in
    ``context_variants: [str]`` at the top level of the scenario dict.

    Control scenarios (``control_type`` is not None) are skipped — their contexts are
    deliberately stripped or irrelevant, so augmentation is meaningless there.

    Args:
        client:       Initialised TogetherClient.
        model:        Model ID to use for augmentation calls.
        max_tokens:   Max tokens for each LLM call.
        temperature:  Sampling temperature (higher → more varied output).
    """

    def __init__(
        self,
        client: TogetherClient,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 512,
        temperature: float = 0.9,
    ):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    # ── Public API ────────────────────────────────────────────────────────

    def augment_batch(
        self,
        scenarios: list[dict],
        n_paraphrases: int = 0,
        n_contexts: int = 0,
        verbose: bool = True,
    ) -> list[dict]:
        """Augment a list of scenario dicts in place and return them.

        Args:
            scenarios:      List of scenario dicts from DatasetGenerator.
            n_paraphrases:  Number of LLM paraphrases to add per scenario (0 = skip).
            n_contexts:     Number of LLM context variants to add per scenario (0 = skip).
            verbose:        Print per-scenario progress to stderr.
        """
        total = len(scenarios)
        for i, scenario in enumerate(scenarios, 1):
            if verbose:
                print(
                    f"  Augmenting {i}/{total}: {scenario['example_id']}...",
                    end="\r",
                    file=sys.stderr,
                )
            self.augment_scenario(scenario, n_paraphrases, n_contexts)
        if verbose:
            print(file=sys.stderr)  # newline after \r progress
        return scenarios

    def augment_scenario(
        self,
        scenario: dict,
        n_paraphrases: int = 0,
        n_contexts: int = 0,
    ) -> dict:
        """Augment a single scenario dict in place.

        Skips control scenarios silently (control_type is not None).
        Returns the (possibly modified) scenario.
        """
        if scenario.get("control_type") is not None:
            return scenario

        if n_paraphrases > 0:
            b1 = scenario["queries"]["B1"]
            paraphrases = self._generate_paraphrases(b1, n_paraphrases)
            scenario["queries"]["B1_llm_paraphrases"] = paraphrases

        if n_contexts > 0:
            context = scenario["context"]
            secret_text = scenario["secret"]["text"]
            variants = self._generate_context_variants(context, secret_text, n_contexts)
            scenario["context_variants"] = variants

        return scenario

    # ── LLM calls ─────────────────────────────────────────────────────────

    def _generate_paraphrases(self, query: str, n: int) -> list[str]:
        user = _PARAPHRASE_USER.format(n=n, query=query)
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": _PARAPHRASE_SYSTEM},
                    {"role": "user", "content": user},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return self._parse_lines(response, n)
        except Exception as exc:
            print(f"\n[WARN] paraphrase LLM call failed: {exc}", file=sys.stderr)
            return []

    def _generate_context_variants(self, context: str, secret_text: str, n: int) -> list[str]:
        user = _CONTEXT_USER.format(n=n, secret=secret_text, context=context)
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": _CONTEXT_SYSTEM},
                    {"role": "user", "content": user},
                ],
                max_tokens=self.max_tokens * max(2, n),
                temperature=self.temperature,
            )
            return self._parse_variants(response, n)
        except Exception as exc:
            print(f"\n[WARN] context variant LLM call failed: {exc}", file=sys.stderr)
            return []

    # ── Parsers ───────────────────────────────────────────────────────────

    @staticmethod
    def _parse_lines(text: str, n: int) -> list[str]:
        """Split on newlines and strip numbering/bullets. Returns at most n items."""
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        cleaned = [_BULLET_RE.sub("", l).strip() for l in lines]
        return [l for l in cleaned if l][:n]

    @staticmethod
    def _parse_variants(text: str, n: int) -> list[str]:
        """Split on ---VARIANT--- delimiter. Returns at most n items."""
        parts = text.split("---VARIANT---")
        return [p.strip() for p in parts if p.strip()][:n]
