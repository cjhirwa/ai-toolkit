"""Token pricing, in USD per 1M tokens as (input, output) or (input, output, cached input).

The defaults are standard paid-tier text prices taken on 2026-10-08 from Google's and
OpenAI's pricing pages (short-context prices where they depend on prompt length); older
OpenAI entries no longer on OpenAI's page were not re-checked. Prices change, so put the
models you actually use in a JSON file and point LLMGW_PRICING at it:

    {"gemini-3.7-flash": [0.75, 3.75, 0.075], "gpt-5": [1.25, 10.0]}

Matching is exact first, then the longest key that prefixes the model name. Unknown
models report ``cost_usd = None`` rather than a guess. Cached input tokens are charged at
the third price when one is given, otherwise at the full input price.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from .types import Usage

DEFAULT_PRICES: dict[str, tuple[float, ...]] = {
    # OpenAI, standard short-context text prices from OpenAI's pricing page on 2026-10-08.
    # Long context and cache writes are priced separately and not modelled here.
    "gpt-6-astra": (10.00, 50.00, 1.00),
    "gpt-6.1-sol": (2.00, 10.00, 0.10),
    "gpt-6-luna": (0.10, 0.50, 0.01),
    "gpt-5.6-sol": (4.00, 20.00, 0.40),
    "gpt-5.3-codex": (1.75, 14.00, 0.175),
    "chat-latest": (5.00, 30.00, 0.50),
    # Older OpenAI models no longer listed on that page; prices not re-checked.
    "gpt-4o-mini": (0.15, 0.60, 0.075),
    "gpt-4o": (2.50, 10.00, 1.25),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    # Gemini, text input. Longest matching prefix wins, so "-lite" variants get their own price.
    # 3.6-3.8 Flash are at an introductory price until 2026-12-31; it doubles on 2027-01-01.
    "gemini-3.8-flash": (0.75, 3.75, 0.075),
    "gemini-3.7-flash": (0.75, 3.75, 0.075),
    "gemini-3.6-flash": (0.75, 3.75, 0.075),
    "gemini-3.5-flash": (1.50, 9.00, 0.15),
    "gemini-3.5-flash-lite": (0.30, 2.50, 0.03),
    "gemini-3.1-pro-preview": (2.00, 12.00, 0.20),  # > 200k-token prompts: 4.00 / 18.00 / 0.40
    "gemini-3.1-flash-lite": (0.25, 1.50, 0.025),
    "gemini-3-flash-preview": (0.50, 3.00, 0.05),
    "gemini-2.5-pro": (1.25, 10.00, 0.125),  # > 200k-token prompts: 2.50 / 15.00 / 0.25
    "gemini-2.5-flash": (0.30, 2.50, 0.03),
    "gemini-2.5-flash-lite": (0.10, 0.40, 0.01),
    "gemini-embedding-2": (0.20, 0.0),  # text; also matches gemini-embedding-2-preview
}


@lru_cache(maxsize=1)
def prices() -> dict[str, tuple[float, ...]]:
    table = dict(DEFAULT_PRICES)
    path = os.getenv("LLMGW_PRICING")
    if path and Path(path).exists():
        table.update({k: tuple(v) for k, v in json.loads(Path(path).read_text()).items()})
    return table


def price_for(model: str) -> tuple[float, ...] | None:
    table = prices()
    if model in table:
        return table[model]
    matches = [k for k in table if model.startswith(k)]
    return table[max(matches, key=len)] if matches else None


def cost(model: str, usage: Usage) -> float | None:
    p = price_for(model)
    if p is None:
        return None
    cached = min(usage.cached_tokens, usage.input_tokens)  # input_tokens includes cached ones
    cached_rate = p[2] if len(p) > 2 else p[0]
    return ((usage.input_tokens - cached) * p[0] + cached * cached_rate + usage.output_tokens * p[1]) / 1_000_000
