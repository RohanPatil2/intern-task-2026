"""Application settings loaded from environment variables.

Using a plain dataclass-style class (rather than pydantic-settings) keeps
the dependency list minimal while still giving us typed, centralised config.
All values are read once at import time so every module gets the same object.
"""

import os

from dotenv import load_dotenv

# Load .env before reading os.getenv so local dev works without manual export.
load_dotenv()


class Settings:
    """Typed container for all runtime configuration.

    Attributes:
        anthropic_api_key: Secret key for the Anthropic API.
        model: Anthropic model ID.  claude-3-5-haiku is the sweet spot of
            speed, cost, and accuracy for this task (~200 tok/s, $0.80/1M in).
        cache_ttl_seconds: How long a cached response stays valid.
        cache_max_size: Maximum number of entries in the in-memory cache.
        request_timeout: Per-request Anthropic API timeout in seconds.
            Kept at 25 s to comfortably fit inside the 30 s endpoint limit.
    """

    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    # claude-haiku-4-5 is the fastest current model ($1/M in, $5/M out).
    # Use LLM_MODEL=claude-sonnet-4-6 for higher accuracy at higher cost.
    model: str = os.getenv("LLM_MODEL", "claude-haiku-4-5")
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    cache_max_size: int = int(os.getenv("CACHE_MAX_SIZE", "1000"))
    request_timeout: float = float(os.getenv("REQUEST_TIMEOUT", "25.0"))


# Singleton — import this everywhere instead of constructing a new instance.
settings = Settings()
