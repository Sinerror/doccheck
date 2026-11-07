"""Runtime configuration. Everything comes from env / .env (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


# USD per 1M tokens: (input, output, cached_input).
# Source: https://router.cheap/api/pricing, snapshot 2026-10-19 (see DECISIONS.md).
# deepseek-v4.1-flash has peak/off-peak tiers; we take the peak tier (upper bound).
# gpt-6-luna has billing_multiplier=4, already folded in.
# TODO: проверить актуальные цены — роутер меняет тарифы, это оценка, а не счёт.
PRICES_PER_MTOK: dict[str, tuple[float, float, float]] = {
    "deepseek-v4.1-flash": (0.30, 1.20, 0.006),
    "gpt-6.1-sol": (2.00, 10.00, 0.10),
    "gpt-6-luna": (0.40, 2.00, 0.04),
    "glm-5.3": (1.40, 4.40, 0.26),
}


def price_usd(model: str, tokens_in: int, tokens_out: int, cached_in: int = 0) -> float | None:
    p = PRICES_PER_MTOK.get(model)
    if p is None:
        return None
    fresh = max(tokens_in - cached_in, 0)
    return (fresh * p[0] + tokens_out * p[1] + cached_in * p[2]) / 1_000_000


@dataclass
class Settings:
    llm_base_url: str = field(default_factory=lambda: _env("DOCCHECK_LLM_BASE_URL", "https://router.cheap/v1"))
    llm_api_key: str = field(default_factory=lambda: _env("DOCCHECK_LLM_API_KEY"))
    scout_model: str = field(default_factory=lambda: _env("DOCCHECK_SCOUT_MODEL", "gpt-6-luna"))
    analyst_model: str = field(default_factory=lambda: _env("DOCCHECK_ANALYST_MODEL", "deepseek-v4.1-flash"))
    critic_model: str = field(default_factory=lambda: _env("DOCCHECK_CRITIC_MODEL", "gpt-6.1-sol"))
    # used once when a model returns schema-invalid structured output twice in a row
    fallback_model: str = field(default_factory=lambda: _env("DOCCHECK_FALLBACK_MODEL", "deepseek-v4.1-flash"))
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL", "http://127.0.0.1:6333"))
    qdrant_api_key: str | None = field(default_factory=lambda: _env("QDRANT_API_KEY") or None)
    collection: str = field(default_factory=lambda: _env("DOCCHECK_COLLECTION", "doccheck_corpus"))
    openalex_mailto: str = field(default_factory=lambda: _env("OPENALEX_MAILTO"))
    openalex_api_key: str = field(default_factory=lambda: _env("OPENALEX_API_KEY"))
    runs_dir: Path = field(default_factory=lambda: Path(_env("DOCCHECK_RUNS_DIR", str(ROOT / "runs"))))
    llm_concurrency: int = field(default_factory=lambda: int(_env("DOCCHECK_LLM_CONCURRENCY", "4")))
    # hard cap per call (reasoning + answer): a degenerate generation once ran to 128k tokens
    max_output_tokens: int = field(default_factory=lambda: int(_env("DOCCHECK_MAX_OUTPUT_TOKENS", "12000")))
    # client-side runaway guard (the router ignores max_tokens / max_completion_tokens for gpt-6.1-sol)
    max_stream_chars: int = field(default_factory=lambda: int(_env("DOCCHECK_MAX_STREAM_CHARS", "48000")))
    llm_call_max_s: float = field(default_factory=lambda: float(_env("DOCCHECK_LLM_CALL_MAX_S", "360")))
    llm_timeout_s: float = field(default_factory=lambda: float(_env("DOCCHECK_LLM_TIMEOUT_S", "180")))
    record: bool = field(default_factory=lambda: _env("DOCCHECK_RECORD") == "1")

    @property
    def has_llm_key(self) -> bool:
        return bool(self.llm_api_key)


def get_settings() -> Settings:
    return Settings()


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt(name: str, **values: object) -> tuple[str, str]:
    """Return (system, user) for prompts/{name}.md. Parts are split by a `---USER---` line;
    `$placeholders` are filled with string.Template.safe_substitute."""
    from string import Template

    raw = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    system, _, user = raw.partition("\n---USER---\n")
    vals = {k: str(v) for k, v in values.items()}
    return Template(system).safe_substitute(vals).strip(), Template(user).safe_substitute(vals).strip()
