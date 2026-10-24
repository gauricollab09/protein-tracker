import pytest
from fastapi.testclient import TestClient

from protein import api, v2
from protein.errors import LLMError
from protein.schema import FoodItem, Meal

client = TestClient(app=api.app, raise_server_exceptions=False)


def test_demo_page():
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "Protein Tracker" in r.text


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["verified_foods"] > 0
    assert body["provider"] in ("claude", "ollama")


def test_analyze_ok(monkeypatch):
    meal = Meal(items=[FoodItem(name="chicken breast", quantity=150, unit="g")])
    monkeypatch.setattr(v2.llm, "parse_meal", lambda text: meal)
    r = client.post("/analyze", json={"text": "150g grilled chicken breast"})
    assert r.status_code == 200
    body = r.json()
    assert body["total_protein_g"] == pytest.approx(46.5)
    assert body["items"][0]["source"] == "verified"
    assert body["meets_goal"] is False


def test_analyze_empty_is_422():
    r = client.post("/analyze", json={"text": "   "})
    assert r.status_code == 422


def test_analyze_llm_down_is_503(monkeypatch):
    def boom(text):
        raise LLMError("provider down")

    monkeypatch.setattr(v2.llm, "parse_meal", boom)
    r = client.post("/analyze", json={"text": "2 eggs"})
    assert r.status_code == 503
    assert "provider down" in r.json()["detail"]


def test_v0_estimate(monkeypatch):
    from protein import v0

    monkeypatch.setattr(v0.llm, "ask_total", lambda text: 42.0)
    r = client.post("/v0/estimate", json={"text": "2 eggs"})
    assert r.status_code == 200
    assert r.json()["total_protein_g"] == 42.0
