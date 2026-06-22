from __future__ import annotations

import os
import time
from pathlib import Path

import joblib

CACHE_DIR = Path(os.environ.get("ARIN_CACHE_DIR", "/tmp/arin_dss_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Joblib memory cache for preprocessed feature arrays.
# Keyed by all arguments (use ignore=["self"] when decorating instance methods).
feature_cache = joblib.Memory(location=str(CACHE_DIR / "features"), verbose=0)


class SimpleCache:
    """In-process result cache. Replaced by Redis at the API layer."""

    def __init__(self) -> None:
        self._store: dict[str, tuple] = {}

    def get(self, key: str):
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value, expire: int = 3600) -> None:
        self._store[key] = (value, time.monotonic() + expire)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)


analysis_cache = SimpleCache()
