"""FastAPI dependency factories.

Using functools.lru_cache ensures the LLMService is constructed only once
(singleton pattern), so the Anthropic async client and its connection pool
are shared across every request rather than rebuilt per-call.
"""

from functools import lru_cache

from app.services.llm import LLMService


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    """Return the application-wide LLMService singleton.

    FastAPI will call this for every request that declares a
    ``LLMService = Depends(get_llm_service)`` parameter, but
    ``lru_cache`` ensures the constructor runs exactly once.

    Returns:
        The shared LLMService instance.
    """
    return LLMService()
