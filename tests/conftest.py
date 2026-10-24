import pytest


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    from protein import llm

    monkeypatch.setattr(llm, "_sleep", lambda s: None)
