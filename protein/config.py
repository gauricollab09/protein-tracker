"""Central knobs. Everything reads from here so decisions live in one place."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# --- LLM provider selection ------------------------------------------------
# "auto" -> Claude if a key is present, else Ollama. Force with "claude"/"ollama".
_PROVIDER = os.getenv("PROTEIN_LLM", "auto").lower()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-8")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def provider() -> str:
    """Resolve the effective provider name: 'claude' or 'ollama'."""
    if _PROVIDER == "claude":
        return "claude"
    if _PROVIDER == "ollama":
        return "ollama"
    # auto
    return "claude" if ANTHROPIC_API_KEY else "ollama"


# --- Domain knobs ----------------------------------------------------------
PROTEIN_GOAL_G = float(os.getenv("PROTEIN_GOAL_G", "70"))

# Eval tolerance band: "within +/- N grams of truth".
WITHIN_G = 5.0

# Open Food Facts live lookup.
OFF_SEARCH_URL = "https://world.openfoodfacts.org/cgi/search.pl"
OFF_USER_AGENT = "ProteinTracker/0.1 (2-day-workshop demo; gaurimae@gmail.com)"
OFF_CACHE = DATA / "off_cache.json"
OFF_TIMEOUT_S = 12.0
