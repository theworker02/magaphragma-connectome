"""Bounded source-region cache — tile values unchanged, I/O/prep reused."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Hashable


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    bytes_logical: int = 0
    bytes_from_storage: int = 0
    bytes_avoided: int = 0

    def as_dict(self) -> dict:
        req = max(1, self.hits + self.misses)
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / req,
            "bytes_logical": self.bytes_logical,
            "bytes_from_storage": self.bytes_from_storage,
            "bytes_avoided": self.bytes_avoided,
            "io_avoided_frac": self.bytes_avoided / max(1, self.bytes_logical),
        }


class SourceRegionCache:
    """LRU cache of raw volumes keyed by bounds or chunk id."""

    def __init__(self, max_entries: int = 4):
        self.max_entries = max(1, int(max_entries))
        self._store: OrderedDict[Hashable, Any] = OrderedDict()
        self.stats = CacheStats()

    def get_or_fetch(self, key: Hashable, nbytes: int, fetch_fn: Callable[[], Any]) -> Any:
        self.stats.bytes_logical += int(nbytes)
        if key in self._store:
            self.stats.hits += 1
            self.stats.bytes_avoided += int(nbytes)
            self._store.move_to_end(key)
            return self._store[key]
        self.stats.misses += 1
        value = fetch_fn()
        self.stats.bytes_from_storage += int(nbytes)
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)
        return value

    def clear(self) -> None:
        self._store.clear()
