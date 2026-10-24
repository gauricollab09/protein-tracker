"""Shim kept so `uv run eval_match.py` still works; real code: protein/eval_matchers.py"""
from protein.eval_matchers import main

if __name__ == "__main__":
    main()
