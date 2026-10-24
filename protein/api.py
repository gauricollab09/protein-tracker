"""HTTP surface. Thin: validation/errors live in the library, this maps them.

    uv run protein-api            # or: uvicorn protein.api:app
    POST /analyze      {"text": "150g chicken and 2 eggs"}
    POST /v0/estimate  {"text": "..."}   (naive baseline, for the live A/B)
    GET  /health
"""
from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from . import config, foods, v0, v2
from .demo_page import DEMO_HTML
from .errors import InputError, LLMError

log = logging.getLogger("protein.api")

app = FastAPI(
    title="Protein Tracker",
    version="1.0.0",
    description="The LLM is a parser, not a nutrition oracle: items are parsed "
    "by an LLM, protein numbers come from a verified DB, totals are computed.",
)


class AnalyzeRequest(BaseModel):
    text: str = Field(examples=["150g grilled chicken breast and 2 eggs"])


class ItemOut(BaseModel):
    name: str
    grams: float
    protein_per_100g: float | None
    protein_g: float
    source: str
    no_hit: bool


class AnalyzeResponse(BaseModel):
    text: str
    provider: str
    items: list[ItemOut]
    total_protein_g: float
    goal_g: float
    meets_goal: bool
    no_hits: list[str] = Field(description="foods not found in any source; flagged, never guessed")


class EstimateResponse(BaseModel):
    text: str
    provider: str
    total_protein_g: float


@app.exception_handler(InputError)
async def _input_error(request: Request, exc: InputError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(LLMError)
async def _llm_error(request: Request, exc: LLMError):
    return JSONResponse(status_code=503, content={"detail": f"LLM provider failed: {exc}"})


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    log.info("%s %s -> %d in %.0f ms", request.method, request.url.path,
             response.status_code, 1000 * (time.perf_counter() - t0))
    return response


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def demo() -> str:
    return DEMO_HTML


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "provider": config.provider(),
        "matcher": config.MATCHER,
        "verified_foods": len(foods._load_verified()),
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    res = v2.analyze(req.text)
    return AnalyzeResponse(
        text=res.text,
        provider=res.provider,
        items=[ItemOut(**vars(i)) for i in res.items],
        total_protein_g=res.total_protein_g,
        goal_g=config.PROTEIN_GOAL_G,
        meets_goal=res.meets_goal,
        no_hits=res.no_hits,
    )


@app.post("/v0/estimate", response_model=EstimateResponse)
def estimate(req: AnalyzeRequest) -> EstimateResponse:
    return EstimateResponse(
        text=req.text, provider=config.provider(), total_protein_g=v0.total_protein(req.text)
    )


def main() -> None:
    import os

    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
