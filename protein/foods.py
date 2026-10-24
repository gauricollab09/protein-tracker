"""Grounding: protein numbers come from here, never from the LLM.

Source of truth, in order:
  1. a small curated VERIFIED DB (USDA per-100 g values) for common staples, and
  2. Open Food Facts live CGI search (cached on disk) as a fallback,
  3. otherwise no_hit -> flag for manual confirm, never guess.

Why curated-first: OFF is a *packaged-food* DB — for generic raw ingredients it
returns noisy branded matches (a salmon-flavoured snack at 7.8 g) or null
protein, and it rate-limits. Measured, not assumed. Unit -> grams conversion is
plain Python and is, on purpose, where the dominant residual error (portion
estimation) lives.

    ground_item(item) -> GroundResult(grams, protein_g, protein_per_100g, source, no_hit)
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass

import httpx

from . import config, matchers
from .schema import FoodItem

log = logging.getLogger("protein.foods")
_verified_lock = threading.Lock()
_cache_lock = threading.Lock()

# Sentinel: an OFF lookup that failed due to a network/parse error (not a real
# no-hit). Never written to the on-disk cache, so the next call retries.
_FETCH_FAILED = object()

# --- portion model (the deliberate source of estimation error) -------------
# grams per 1 "count" of a whole item.
COUNT_GRAMS = {
    "egg": 50.0,
    "slice": 28.0,           # generic slice (bread)
    "bread": 28.0,
    "banana": 118.0,
    "apple": 182.0,
    "tortilla": 49.0,
    "scoop": 30.0,           # protein powder scoop
}
# mass units -> grams (ml treated as ~1 g/ml for liquids).
UNIT_GRAMS = {"g": 1.0, "gram": 1.0, "grams": 1.0, "oz": 28.35, "ounce": 28.35,
              "ml": 1.0, "milliliter": 1.0}
# rough volume defaults (food-agnostic = error on purpose).
VOLUME_GRAMS = {"cup": 200.0, "tbsp": 15.0, "tsp": 5.0, "bowl": 240.0, "katori": 150.0}
# container defaults.
CONTAINER_GRAMS = {"can": 140.0, "bar": 60.0, "serving": 150.0}


@dataclass
class GroundResult:
    name: str
    grams: float
    protein_per_100g: float | None
    protein_g: float
    source: str  # "verified" | "off" | "none"
    no_hit: bool


# --------------------------------------------------------------------------- #
# unit -> grams
# --------------------------------------------------------------------------- #
def to_grams(item: FoodItem) -> float:
    unit = (item.unit or "").strip().lower()
    # the parser sometimes emits plural units ("slices", "scoops"): singularize
    # when the singular form is a unit we know, so it doesn't fall through to
    # the number-is-grams last resort.
    if unit.endswith("s") and unit[:-1] in {**UNIT_GRAMS, **VOLUME_GRAMS, **CONTAINER_GRAMS, **COUNT_GRAMS}:
        unit = unit[:-1]
    qty = float(item.quantity or 0)
    if unit in UNIT_GRAMS:
        return qty * UNIT_GRAMS[unit]
    if unit in VOLUME_GRAMS:
        return qty * VOLUME_GRAMS[unit]
    if unit in CONTAINER_GRAMS:
        return qty * CONTAINER_GRAMS[unit]
    if unit in COUNT_GRAMS:  # countable unit named directly: "slice", "scoop"
        return qty * COUNT_GRAMS[unit]
    if unit in ("count", "", "whole", "piece", "pieces", "unit", "units"):
        per = _count_weight(item.name)
        return qty * per
    # unknown unit -> treat the number as grams (last-resort, logged via no_hit upstream)
    return qty


def _count_weight(name: str) -> float:
    n = name.lower()
    for key, grams in COUNT_GRAMS.items():
        if key in n:
            return grams
    return 100.0  # unknown countable -> assume ~100 g (estimation error, by design)


# --------------------------------------------------------------------------- #
# verified DB (curated USDA values), checked before OFF
# --------------------------------------------------------------------------- #
_verified: dict[str, float] | None = None


def _load_verified() -> dict[str, float]:
    global _verified
    with _verified_lock:
        if _verified is None:
            raw = json.loads((config.DATA / "verified_foods.json").read_text())
            _verified = {k.lower(): float(v) for k, v in raw.items() if not k.startswith("_")}
            log.info("loaded %d verified foods", len(_verified))
        return _verified


def _verified_lookup(name: str) -> float | None:
    table = _load_verified()
    key = matchers.get(config.MATCHER)(name, list(table))
    return table[key] if key is not None else None


# --------------------------------------------------------------------------- #
# protein per 100 g, from Open Food Facts (cached)
# --------------------------------------------------------------------------- #
_cache: dict[str, float | None] | None = None


def _load_cache() -> dict[str, float | None]:
    global _cache
    if _cache is None:
        if config.OFF_CACHE.exists():
            _cache = json.loads(config.OFF_CACHE.read_text())
        else:
            _cache = {}
    return _cache


def _save_cache() -> None:
    if _cache is None:
        return
    config.OFF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=config.OFF_CACHE.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(_cache, f, indent=2, sort_keys=True)
        os.replace(tmp, config.OFF_CACHE)  # atomic: readers never see a torn file
    except BaseException:
        os.unlink(tmp)
        raise


def lookup(name: str) -> tuple[float | None, str]:
    """Return (protein g per 100 g, source). source in {verified, off, none}."""
    # 1. curated verified DB
    val = _verified_lookup(name)
    if val is not None:
        log.debug("%r grounded from verified DB: %.1f g/100g", name, val)
        return val, "verified"

    # 2. Open Food Facts live fallback (cached)
    key = name.strip().lower()
    with _cache_lock:
        cache = _load_cache()
        if key in cache:
            cached = cache[key]
            log.debug("%r served from OFF cache", name)
            return (cached, "off") if cached is not None else (None, "none")
    value = _off_lookup(key)  # network call outside the lock
    if value is _FETCH_FAILED:
        return None, "none"
    with _cache_lock:
        _load_cache()[key] = value
        _save_cache()
    if value is None:
        log.info("%r not found anywhere -> no_hit", name)
    return (value, "off") if value is not None else (None, "none")


def protein_per_100g(name: str) -> float | None:
    """Convenience: just the value (used by smoke tests)."""
    return lookup(name)[0]


def _off_lookup(name: str) -> float | None:
    params = {
        "search_terms": name,
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page_size": 8,
        "fields": "product_name,nutriments",
    }
    try:
        r = httpx.get(
            config.OFF_SEARCH_URL,
            params=params,
            headers={"User-Agent": config.OFF_USER_AGENT},
            timeout=config.OFF_TIMEOUT_S,
        )
        r.raise_for_status()
        products = r.json().get("products", [])
    except Exception as exc:
        log.warning("OFF lookup failed for %r: %s", name, exc)
        return _FETCH_FAILED

    # First product with a sane protein value wins (deterministic).
    for p in products:
        val = (p.get("nutriments") or {}).get("proteins_100g")
        if isinstance(val, (int, float)) and 0 < float(val) <= 100:
            return round(float(val), 2)
    return None


# --------------------------------------------------------------------------- #
# ground one item end to end
# --------------------------------------------------------------------------- #
def ground_item(item: FoodItem) -> GroundResult:
    grams = to_grams(item)
    per100, source = lookup(item.name)
    if per100 is None:
        return GroundResult(item.name, grams, None, 0.0, source="none", no_hit=True)
    protein = round(per100 * grams / 100.0, 2)
    return GroundResult(item.name, grams, per100, protein, source=source, no_hit=False)
