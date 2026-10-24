"""Protein Tracker — the LLM is a parser, not a nutrition oracle.

v0  naive   : ask the model for the protein number directly (loses the A/B)
v2  grounded: model parses items only -> verified DB grounds -> Python computes
eval        : run both over a labeled set -> MAE before/after (the hero artifact)
"""
