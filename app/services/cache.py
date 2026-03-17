"""In-memory TTL cache for FeedbackResponse objects.

Architecture note:
  Using cachetools.TTLCache (in-process memory) rather than Redis keeps the
  Docker Compose stack to a single container and removes the latency of a
  network round-trip.  The trade-off is that cache entries are lost on restart
  and are not shared across multiple worker processes.  For a horizontally
  scaled production deployment, swap the TTLCache backend for an async Redis
  client (e.g., redis.asyncio) using the same make_cache_key / get_cached /
  set_cached interface — no other code changes required.

Cache key design:
  SHA-256 of "{sentence}|{target_language}|{native_language}" (all normalised
  to lower-case with leading/trailing whitespace stripped) gives a fixed-size,
  collision-resistant key that is safe to use as a dict key regardless of the
  input language or script.
"""

import hashlib
from threading import Lock

from cachetools import TTLCache

from app.core.config import settings
from app.models.schemas import FeedbackResponse

# Module-level singleton — shared across all requests in the same process.
_cache: TTLCache = TTLCache(
    maxsize=settings.cache_max_size,
    ttl=settings.cache_ttl_seconds,
)

# TTLCache is not thread-safe, so protect every read/write with a lock.
# Since LLM calls are async and we release the GIL during I/O, a threading
# Lock is sufficient here; no asyncio.Lock needed.
_lock = Lock()


def make_cache_key(sentence: str, target_language: str, native_language: str) -> str:
    """Build a deterministic, fixed-length cache key from the three request fields.

    Args:
        sentence: The learner's sentence (whitespace-normalised).
        target_language: The language being practised.
        native_language: The learner's first language.

    Returns:
        64-character lower-hex SHA-256 digest.
    """
    # Normalise to avoid cache misses from irrelevant formatting differences
    # such as trailing spaces or inconsistent capitalisation ("Spanish" vs "spanish").
    payload = (
        f"{sentence.strip()}"
        f"|{target_language.lower().strip()}"
        f"|{native_language.lower().strip()}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_cached(key: str) -> FeedbackResponse | None:
    """Return a previously cached response, or None if absent / expired.

    Args:
        key: Cache key produced by :func:`make_cache_key`.

    Returns:
        The cached FeedbackResponse, or None on a cache miss.
    """
    with _lock:
        return _cache.get(key)


def set_cached(key: str, value: FeedbackResponse) -> None:
    """Store a response in the cache under the given key.

    Args:
        key: Cache key produced by :func:`make_cache_key`.
        value: The FeedbackResponse to store.
    """
    with _lock:
        _cache[key] = value
