"""Match a parsed food name to a verified-DB key — three strategies, one API.

The grounding step has a little retrieval problem hiding inside it: the LLM
says "grilled chicken breast", the DB has "chicken breast". How you bridge that
gap decides the no-hit rate. We compare three matchers and pick by measured
hit-rate (see eval_match.py):

  substring  what we started with — key must be a literal substring. Brittle.
  bm25       token-based ranked match. Handles word order / partial overlap.
  fuzzy      edit-distance (rapidfuzz). Also handles typos & spelling variants.

Each matcher: match(name, keys) -> matched key or None.
"""
from __future__ import annotations

import re

from . import config

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


# --------------------------------------------------------------------------- #
# 1. substring (the baseline)
# --------------------------------------------------------------------------- #
def substring_match(name: str, keys: list[str]) -> str | None:
    n = name.lower()
    for key in sorted(keys, key=len, reverse=True):  # longest first
        if key in n:
            return key
    return None


# --------------------------------------------------------------------------- #
# 2. BM25 (lexical, token-ranked)
# --------------------------------------------------------------------------- #
class _BM25:
    """Built once per key-set and cached (BM25 indexes the corpus up front)."""

    def __init__(self, keys: list[str]):
        from rank_bm25 import BM25Okapi

        self.keys = keys
        self._bm25 = BM25Okapi([_tokens(k) for k in keys])

    def match(self, name: str, threshold: float = 0.1) -> str | None:
        q = _tokens(name)
        if not q:
            return None
        scores = self._bm25.get_scores(q)
        best = max(range(len(scores)), key=lambda i: scores[i])
        return self.keys[best] if scores[best] > threshold else None


_bm25_cache: dict[tuple[str, ...], _BM25] = {}


def bm25_match(name: str, keys: list[str]) -> str | None:
    cache_key = tuple(keys)
    if cache_key not in _bm25_cache:
        _bm25_cache[cache_key] = _BM25(keys)
    return _bm25_cache[cache_key].match(name)


# --------------------------------------------------------------------------- #
# 3. fuzzy (edit-distance)
# --------------------------------------------------------------------------- #
def fuzzy_match(name: str, keys: list[str], threshold: float | None = None) -> str | None:
    if threshold is None:
        threshold = config.FUZZY_THRESHOLD
    from rapidfuzz import fuzz, process

    res = process.extractOne(name.lower(), keys, scorer=fuzz.WRatio)
    if res and res[1] >= threshold:
        return res[0]
    return None


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
MATCHERS = {
    "substring": substring_match,
    "bm25": bm25_match,
    "fuzzy": fuzzy_match,
}


def get(name: str):
    if name not in MATCHERS:
        raise ValueError(f"unknown matcher {name!r}; expected one of {sorted(MATCHERS)}")
    return MATCHERS[name]
