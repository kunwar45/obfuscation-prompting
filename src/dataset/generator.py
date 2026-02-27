from __future__ import annotations

import json
import random

from .base_domain import BaseDomain
from .schema import Scenario


class DatasetGenerator:
    """Generates synthetic concealment scenarios across multiple domains.

    Distribution:
    - ~85% regular scenarios (with embedded secret, all A0/A1/A2 conditions)
    - ~7.5% no-secret controls (context stripped of secret, A0 only, B1 query)
    - ~7.5% irrelevant-secret controls (context has secret, A0 only, B0 query)

    Regular scenarios are distributed evenly across domains (floor division;
    remainder examples go to the first domain).
    """

    def __init__(self, domains: list[BaseDomain], seed: int = 42):
        self.domains = domains
        self.seed = seed

    def generate(self, n: int) -> list[dict]:
        rng = random.Random(self.seed)

        controls_n = max(1, round(n * 0.15))
        no_secret_n = controls_n // 2
        irrelevant_secret_n = controls_n - no_secret_n
        regular_n = n - controls_n

        scenarios: list[dict] = []

        # ── Regular scenarios ─────────────────────────────────────────────
        n_domains = len(self.domains)
        per_domain = regular_n // n_domains
        remainder = regular_n % n_domains

        domain_counts = [per_domain + (1 if i < remainder else 0) for i in range(n_domains)]

        for domain, count in zip(self.domains, domain_counts):
            for i in range(count):
                idx = len(scenarios)
                example_id = f"{domain.domain_name}_{idx:04d}"
                scenario = domain.build_scenario(example_id, rng, control_type=None)
                scenarios.append(scenario.to_dict())

        # ── no_secret controls ────────────────────────────────────────────
        for i in range(no_secret_n):
            domain = self.domains[i % n_domains]
            idx = len(scenarios)
            example_id = f"{domain.domain_name}_ctrl_ns_{i:04d}"
            scenario = domain.build_scenario(example_id, rng, control_type="no_secret")
            scenarios.append(scenario.to_dict())

        # ── irrelevant_secret controls ────────────────────────────────────
        for i in range(irrelevant_secret_n):
            domain = self.domains[i % n_domains]
            idx = len(scenarios)
            example_id = f"{domain.domain_name}_ctrl_ir_{i:04d}"
            scenario = domain.build_scenario(example_id, rng, control_type="irrelevant_secret")
            scenarios.append(scenario.to_dict())

        return scenarios

    def to_jsonl(self, scenarios: list[dict], output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            for scenario in scenarios:
                f.write(json.dumps(scenario, ensure_ascii=False) + "\n")
