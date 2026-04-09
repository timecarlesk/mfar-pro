"""
Step 3: HarnessConfig — config-driven parameterization of rerank pipeline.

Instead of hardcoded format_doc/prompt/alpha, the pipeline reads a JSON config.
The proposer outputs config changes, not code.

Usage:
    from meta_harness.harness_config import HarnessConfig, load_config, save_config
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


# Default prompt (current hardcoded version from rerank.py)
DEFAULT_RERANK_PROMPT = """\
Query: "{query}"
Document: {doc_text}

This query contains negation or constraints. Score how well this document satisfies the query INCLUDING any negation constraints (e.g., "not indicated" means the document should have contraindication data, not indication data).

Score 0-10 where 0=completely irrelevant, 10=perfect match.
Output ONLY the number."""


@dataclass
class HarnessConfig:
    """Configuration for the rerank pipeline, tunable by the proposer."""

    # ── format_doc settings ─────────────────────────────────────────────
    # negation_pattern → ordered list of fields to show with content
    # If a query's pattern isn't in this dict, falls back to "default"
    field_priority: dict = field(default_factory=lambda: {
        "default": [],  # empty = use Stage 2 boost_fields as-is
    })

    # Max items to show per field (e.g., "indication: drug1, drug2, drug3")
    max_items_per_field: int = 5

    # Max total chars for document string
    max_chars: int = 800

    # How to show non-boosted fields: "[has data]" or "" (hide entirely)
    show_suppressed_as: str = "[has data]"

    # ── prompt settings ─────────────────────────────────────────────────
    # Base prompt template (must contain {query} and {doc_text} placeholders)
    prompt_template: str = DEFAULT_RERANK_PROMPT

    # Extra instruction appended after the base prompt
    prompt_suffix: str = ""

    # ── score merging settings ──────────────────────────────────────────
    # answer_type → alpha value. Falls back to "default".
    alpha_by_type: dict = field(default_factory=lambda: {
        "default": 0.7,
    })

    # Number of top candidates to rerank
    top_k: int = 50

    # ── metadata ────────────────────────────────────────────────────────
    round: int = 0
    rationale: str = "baseline config (matches current hardcoded behavior)"

    def get_alpha(self, answer_type: Optional[str] = None) -> float:
        """Get alpha for a given answer_type, with fallback to default."""
        if answer_type and answer_type in self.alpha_by_type:
            return self.alpha_by_type[answer_type]
        return self.alpha_by_type.get("default", 0.7)

    def get_field_priority(self, negation_pattern: Optional[str] = None) -> list:
        """Get field priority for a negation pattern, with fallback to default."""
        if negation_pattern and negation_pattern in self.field_priority:
            return self.field_priority[negation_pattern]
        return self.field_priority.get("default", [])

    def get_full_prompt(self) -> str:
        """Get the complete prompt template (base + suffix)."""
        prompt = self.prompt_template
        if self.prompt_suffix:
            prompt = prompt.rstrip() + "\n" + self.prompt_suffix
        return prompt


def save_config(config: HarnessConfig, path: str):
    """Save config as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"  Config saved → {path}")


def load_config(path: str) -> HarnessConfig:
    """Load config from JSON."""
    with open(path) as f:
        data = json.load(f)
    return HarnessConfig(**data)


def baseline_config() -> HarnessConfig:
    """Return the baseline config matching current hardcoded behavior."""
    return HarnessConfig()


if __name__ == "__main__":
    # Generate baseline config
    config_dir = os.path.join(
        os.path.dirname(__file__), "configs")
    os.makedirs(config_dir, exist_ok=True)

    config = baseline_config()
    path = os.path.join(config_dir, "round_0_baseline.json")
    save_config(config, path)
    print(f"\n  Baseline config:")
    print(json.dumps(asdict(config), indent=2))
