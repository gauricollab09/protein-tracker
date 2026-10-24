"""Shim kept so `uv run eval.py` still works; real code: protein/eval_pipeline.py"""
from protein.eval_pipeline import main

if __name__ == "__main__":
    main()
