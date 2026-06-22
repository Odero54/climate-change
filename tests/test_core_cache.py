"""Tests for core/cache.py — SimpleCache."""
import time

import pytest

from core.cache import SimpleCache


class TestSimpleCache:
    def setup_method(self):
        self.cache = SimpleCache()

    def test_set_and_get_returns_value(self):
        self.cache.set("k", "hello", expire=60)
        assert self.cache.get("k") == "hello"

    def test_get_missing_key_returns_none(self):
        assert self.cache.get("nonexistent") is None

    def test_expired_entry_returns_none(self):
        self.cache.set("k", "value", expire=0)
        # expire=0 means it expires immediately (monotonic already past)
        time.sleep(0.01)
        assert self.cache.get("k") is None

    def test_delete_removes_entry(self):
        self.cache.set("k", 42, expire=60)
        self.cache.delete("k")
        assert self.cache.get("k") is None

    def test_delete_nonexistent_key_is_safe(self):
        self.cache.delete("ghost")  # must not raise

    def test_overwrite_entry(self):
        self.cache.set("k", "first", expire=60)
        self.cache.set("k", "second", expire=60)
        assert self.cache.get("k") == "second"

    def test_stores_arbitrary_objects(self):
        obj = {"nested": [1, 2, 3]}
        self.cache.set("obj", obj, expire=60)
        assert self.cache.get("obj") == obj

    def test_multiple_keys_independent(self):
        self.cache.set("a", 1, expire=60)
        self.cache.set("b", 2, expire=60)
        assert self.cache.get("a") == 1
        assert self.cache.get("b") == 2

    def test_expired_entry_is_evicted_on_get(self):
        self.cache.set("k", "v", expire=0)
        time.sleep(0.01)
        self.cache.get("k")
        assert "k" not in self.cache._store
