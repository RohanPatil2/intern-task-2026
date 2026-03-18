"""Shared pytest fixtures for the language feedback test suite.

This file is loaded automatically by pytest before any test module runs.
It provides:
  - Cache isolation: clears the TTL cache and hit/miss counters before
    every test so no test can influence another through stale cached data.
  - In-flight map cleanup: resets the asyncio Future map so concurrent-
    dedup tests start from a known empty state.

Without this, tests run in sequence could share cache state and produce
misleading results (e.g. a unit test appearing to "pass" because an
integration test already seeded the cache with the same sentence).
"""

import pytest


@pytest.fixture(autouse=True)
def isolate_cache_and_inflight():
    """Reset all shared service state before every test.

    Marked ``autouse=True`` so it applies to every test in every file
    without needing to be explicitly requested — zero chance of a
    developer forgetting to include it on a new test.

    Yields:
        None — setup only, no teardown needed (state is reset at the
        start of the *next* test anyway).
    """
    # Clear the persistent TTL cache and reset hit/miss counters.
    import app.services.cache as cache_module

    with cache_module._lock:
        cache_module._cache.clear()
        cache_module._hits = 0
        cache_module._misses = 0

    # Clear the asyncio in-flight Future map.
    # This prevents a stale Future from a previous test from blocking
    # a subsequent test that uses the same cache key.
    from app.services.llm import _in_flight

    _in_flight.clear()

    yield
