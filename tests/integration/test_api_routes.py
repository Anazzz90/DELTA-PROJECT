"""
tests/integration/test_api_routes.py
========================================
Checkpoint 13 — FastAPI Route Integration Tests

Tests that the API routes respond with correct status codes and shapes.
Uses starlette TestClient without touching ChromaDB or real LLMs.
"""
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture()
def client():
    """Return a TestClient with DB creation mocked out.

    VectorStore is instantiated inside task_queue/tasks.py (RQ worker context),
    not in api.routes.query — it never runs synchronously during these route
    tests, so it doesn't need to be mocked here.
    """
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
