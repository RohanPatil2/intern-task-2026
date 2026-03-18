"""In-memory TTL cache for FeedbackResponse objects with hit/miss telemetry.

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
  collision-resistant key safe for any script or language.

Telemetry:
  _hits / _misses counters are exposed via get_stats() which the GET /stats
  endpoint reads to give operators real-time visibility into cache efficiency.
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

# TTLCache is not thread-safe, so every read/write is guarded by this lock.
_lock = Lock()

# Simple hit/miss counters for the /stats endpoint.
# Using plain ints (not atomics) is safe because the GIL serialises increments.
_hits: int = 0
_misses: int = 0


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
    # such as trailing spaces or inconsistent capitalisation.
    payload = (
        f"{sentence.strip()}"
        f"|{target_language.lower().strip()}"
        f"|{native_language.lower().strip()}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_cached(key: str) -> FeedbackResponse | None:
    """Return a previously cached response, or None if absent / expired.

    Increments the hit or miss counter for telemetry.

    Args:
        key: Cache key produced by :func:`make_cache_key`.

    Returns:
        The cached FeedbackResponse, or None on a cache miss.
    """
    global _hits, _misses
    with _lock:
        result = _cache.get(key)
        if result is not None:
            _hits += 1
        else:
            _misses += 1
        return result


def set_cached(key: str, value: FeedbackResponse) -> None:
    """Store a response in the cache under the given key.

    Args:
        key: Cache key produced by :func:`make_cache_key`.
        value: The FeedbackResponse to store.
    """
    with _lock:
        _cache[key] = value


def get_stats() -> dict:
    """Return a snapshot of cache metrics for the /stats endpoint.

    Returns:
        Dict with keys: cache_size, cache_maxsize, hits, misses, hit_rate.
    """
    global _hits, _misses
    with _lock:
        size = len(_cache)
        hits = _hits
        misses = _misses

    total = hits + misses
    return {
        "cache_size": size,
        "cache_maxsize": settings.cache_max_size,
        "cache_ttl_seconds": settings.cache_ttl_seconds,
        "hits": hits,
        "misses": misses,
        # hit_rate is 0.0 until at least one request has been processed.
        "hit_rate": round(hits / total, 4) if total > 0 else 0.0,
    }
