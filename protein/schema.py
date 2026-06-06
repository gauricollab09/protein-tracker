"""The contract between the LLM and the rest of the pipeline.

The LLM is only ever allowed to produce a `Meal` (a list of items it parsed out
of free text). It never produces protein numbers — those come from the DB.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class FoodItem(BaseModel):
    name: str = Field(description="canonical food name, singular, no brand, e.g. 'chicken breast'")
    quantity: float = Field(description="numeric amount; 1 if unstated", ge=0)
    unit: str = Field(description="unit of the quantity: g, oz, cup, slice, or 'count' for whole items")


class Meal(BaseModel):
    items: list[FoodItem] = Field(default_factory=list)


# JSON Schema handed to the model for forced structured output (Claude tool / Ollama format).
MEAL_JSON_SCHEMA = Meal.model_json_schema()
