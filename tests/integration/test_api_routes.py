"""
tests/integration/test_api_routes.py
========================================
Checkpoint 13 — FastAPI Route Integration Tests
Checkpoint 19 — Auth + Rate Limiting

Tests that the API routes respond with correct status codes and shapes.
Uses starlette TestClient without touching ChromaDB or real LLMs.
"""
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.middleware.auth import verify_api_key
from api.middleware.rate_limiter import enforce_rate_limit
from db.models import ApiKeyRow


def _fake_api_key() -> ApiKeyRow:
    return ApiKeyRow(id=1, key_hash="test-hash", key_prefix="dmars_test", name="test-key", is_active=True)


@pytest.fixture()
def client():
    """Return a TestClient with DB creation mocked out and auth bypassed.

    Auth (verify_api_key) and rate limiting (enforce_rate_limit) are
    overridden via FastAPI's dependency_overrides — the standard pattern
    for testing protected routes — rather than requiring a real Postgres +
    Redis connection just to exercise route-level validation logic.
    tests/integration/test_task_queue.py separately exercises the *real*
    auth/rate-limit path end-to-end against a live server.

    VectorStore is instantiated inside task_queue/tasks.py (RQ worker context),
    not in api.routes.query — it never runs synchronously during these route
    tests, so it doesn't need to be mocked here.
    """
    app.dependency_overrides[verify_api_key] = _fake_api_key
    app.dependency_overrides[enforce_rate_limit] = lambda: None
    with patch("api.main.create_all_tables", new_callable=AsyncMock):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def unauthenticated_client():
    """A client with no auth override — for testing the auth gate itself."""
    with patch("api.main.create_all_tables", new_callable=AsyncMock):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c



# =============================================================================
# 1. GET /health
# =============================================================================
def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# =============================================================================
# 2. GET /agents/performance
# =============================================================================
def test_agents_performance(client):
    response = client.get("/agents/performance")
    assert response.status_code == 200
    data = response.json()
    assert "neutral_analyst" in data
    assert "data_first" in data
    assert "skeptic" in data


# =============================================================================
# 3. POST /query — validation errors (422)
# =============================================================================
def test_query_empty_question_returns_422(client):
    response = client.post("/query", json={"question": "", "fact_set": ["a fact"]})
    assert response.status_code == 422


def test_query_no_facts_returns_422(client):
    response = client.post("/query", json={"question": "Why did BTC spike?", "fact_set": []})
    assert response.status_code == 422


def test_query_no_agents_returns_422(client):
    response = client.post("/query", json={
        "question": "Why did BTC spike?",
        "fact_set": ["volume up"],
        "selected_agents": [],
    })
    assert response.status_code == 422


def test_query_unknown_agent_returns_422(client):
    response = client.post("/query", json={
        "question": "Why did BTC spike?",
        "fact_set": ["volume up"],
        "selected_agents": ["not_a_real_agent"],
    })
    assert response.status_code == 422


# =============================================================================
# 4. GET /history
# =============================================================================
def test_history_returns_list(client):
    with patch("memory.history.HistoryStore.get_history", new_callable=AsyncMock, return_value=[]):
        response = client.get("/history")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


# =============================================================================
# 5. GET /history/{id} — not found
# =============================================================================
def test_history_detail_not_found(client):
    with patch("memory.history.HistoryStore.get_query_detail", new_callable=AsyncMock, return_value=None):
        response = client.get("/history/99999")
        assert response.status_code == 404


# =============================================================================
# 6. Checkpoint 19 — auth gate (real DB lookup, no override)
# =============================================================================
def test_query_without_api_key_returns_401(unauthenticated_client):
    response = unauthenticated_client.post("/query", json={
        "question": "Why did BTC spike?",
        "fact_set": ["volume up"],
        "selected_agents": ["data_first"],
    })
    assert response.status_code == 401


def test_query_with_invalid_api_key_returns_401(unauthenticated_client):
    response = unauthenticated_client.post(
        "/query",
        json={"question": "Why did BTC spike?", "fact_set": ["volume up"], "selected_agents": ["data_first"]},
        headers={"X-API-Key": "dmars_this-is-not-a-real-key"},
    )
    assert response.status_code == 401


def test_history_without_api_key_returns_401(unauthenticated_client):
    response = unauthenticated_client.get("/history")
    assert response.status_code == 401


def test_health_never_requires_api_key(unauthenticated_client):
    response = unauthenticated_client.get("/health")
    assert response.status_code == 200
