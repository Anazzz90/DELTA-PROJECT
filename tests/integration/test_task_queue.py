# tests/integration/test_task_queue.py
# Checkpoint 14 — Integration tests for the Redis + RQ task queue.
# Checkpoint 19 — Integration tests for Auth + Rate Limiting.
#
# Prerequisites:
#   - Redis running: docker run -d -p 6379:6379 redis
#   - PostgreSQL running with the api_keys table migrated (alembic upgrade head)
#   - RQ worker running: python task_queue/worker.py
#   - FastAPI running: poetry run python run_server.py (Windows+Postgres) or
#     uvicorn api.main:app --reload (SQLite / Linux / macOS)
#   - ADMIN_SECRET set in .env (required to mint the test API key below)
#
# Run:
#   pytest tests/integration/test_task_queue.py -v

import os
import time
import pytest
import httpx

BASE_URL = os.getenv("DMARS_API_URL", "http://127.0.0.1:8000")
POLL_TIMEOUT_SECONDS = 120   # 2 minutes max for a full pipeline run
POLL_INTERVAL_SECONDS = 3


VALID_PAYLOAD = {
    "question": "Why did BTC spike 8% in the last hour?",
    "fact_set": [
        "BTC price moved from $60,000 to $64,800 in 55 minutes.",
        "Open interest in BTC futures increased by 12% simultaneously.",
        "No major news events detected during the move.",
    ],
    "domain_profile": "intraday_trading",
    "selected_agents": ["neutral_analyst", "skeptic", "contrarian"],
    "meta_ai_enabled": False,
}


@pytest.fixture(scope="module")
def api_key() -> str:
    """
    Mint one real API key for this test module via POST /auth/create-key,
    using ADMIN_SECRET from the environment. Skips the module if
    ADMIN_SECRET isn't configured (key creation is intentionally disabled
    without it — see api/middleware/auth.py:verify_admin_secret).
    """
    admin_secret = os.getenv("ADMIN_SECRET", "")
    if not admin_secret:
        pytest.skip("ADMIN_SECRET not set — cannot mint a test API key")

    with httpx.Client() as client:
        resp = client.post(
            f"{BASE_URL}/auth/create-key",
            json={"name": "test_task_queue.py"},
            headers={"X-Admin-Secret": admin_secret},
        )
    if resp.status_code != 200:
        pytest.skip(f"Could not create test API key ({resp.status_code}): {resp.text}")
    return resp.json()["api_key"]


@pytest.fixture()
def auth_client(api_key):
    """An httpx.Client pre-configured with a valid X-API-Key header."""
    with httpx.Client(headers={"X-API-Key": api_key}) as client:
        yield client


def poll_until_done(job_id: str, client: httpx.Client) -> dict:
    """Poll GET /query/{job_id}/status until finished or failed, or timeout."""
    deadline = time.time() + POLL_TIMEOUT_SECONDS
    while time.time() < deadline:
        resp = client.get(f"{BASE_URL}/query/{job_id}/status")
        assert resp.status_code == 200, f"Status check failed: {resp.text}"
        data = resp.json()
        if data["status"] in ("finished", "failed", "not_found"):
            return data
        print(f"  [poll] job {job_id} → {data['status']}")
        time.sleep(POLL_INTERVAL_SECONDS)
    pytest.fail(f"Job {job_id} did not complete within {POLL_TIMEOUT_SECONDS}s")


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestTaskQueueEnqueue:
    """POST /query should enqueue and return a job_id immediately."""

    def test_post_query_returns_job_id(self, auth_client):
        resp = auth_client.post(f"{BASE_URL}/query", json=VALID_PAYLOAD)
        assert resp.status_code == 202, resp.text
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        assert data["poll_url"] == f"/query/{data['job_id']}/status"

    def test_post_query_empty_question_rejected(self, auth_client):
        bad = {**VALID_PAYLOAD, "question": ""}
        resp = auth_client.post(f"{BASE_URL}/query", json=bad)
        assert resp.status_code == 422

    def test_post_query_empty_facts_rejected(self, auth_client):
        bad = {**VALID_PAYLOAD, "fact_set": []}
        resp = auth_client.post(f"{BASE_URL}/query", json=bad)
        assert resp.status_code == 422

    def test_post_query_no_agents_rejected(self, auth_client):
        bad = {**VALID_PAYLOAD, "selected_agents": []}
        resp = auth_client.post(f"{BASE_URL}/query", json=bad)
        assert resp.status_code == 422


class TestTaskQueueStatusPolling:
    """GET /query/{job_id}/status reflects correct job lifecycle."""

    def test_status_not_found_for_fake_job(self, auth_client):
        resp = auth_client.get(f"{BASE_URL}/query/not-a-real-job-id/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "not_found"

    def test_queued_status_immediately_after_enqueue(self, auth_client):
        enqueue_resp = auth_client.post(f"{BASE_URL}/query", json=VALID_PAYLOAD)
        assert enqueue_resp.status_code == 202
        job_id = enqueue_resp.json()["job_id"]

        # Check immediately — should be queued or started
        status_resp = auth_client.get(f"{BASE_URL}/query/{job_id}/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] in ("queued", "started", "finished")


class TestTaskQueueCompletion:
    """End-to-end: job should finish with a valid QueryResponse."""

    @pytest.mark.slow
    def test_full_pipeline_completes_successfully(self, auth_client):
        enqueue_resp = auth_client.post(f"{BASE_URL}/query", json=VALID_PAYLOAD)
        assert enqueue_resp.status_code == 202
        job_id = enqueue_resp.json()["job_id"]

        result = poll_until_done(job_id, auth_client)

        assert result["status"] == "finished", f"Job failed: {result.get('error')}"
        assert result["result"] is not None
        qr = result["result"]
        assert "query_id" in qr
        assert "system_confidence_score" in qr
        assert isinstance(qr["agent_results"], list)
        assert len(qr["agent_results"]) > 0

    @pytest.mark.slow
    def test_three_concurrent_queries_all_complete(self, api_key):
        """Submit 3 simultaneous queries — all should finish without crashing."""
        import concurrent.futures

        def submit_and_poll():
            with httpx.Client(headers={"X-API-Key": api_key}) as client:
                enqueue_resp = client.post(f"{BASE_URL}/query", json=VALID_PAYLOAD)
                assert enqueue_resp.status_code == 202
                job_id = enqueue_resp.json()["job_id"]
                return poll_until_done(job_id, client)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(submit_and_poll) for _ in range(3)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 3
        for r in results:
            assert r["status"] == "finished", f"A concurrent job failed: {r.get('error')}"


class TestRedisUnavailable:
    """FastAPI should return 503 cleanly if Redis is down (not crash)."""

    def test_health_still_works_independently(self):
        """The /health endpoint must not depend on Redis, and needs no API key."""
        with httpx.Client() as client:
            resp = client.get(f"{BASE_URL}/health")
        assert resp.status_code == 200


class TestAuthAndRateLimiting:
    """Checkpoint 19 — live end-to-end auth + rate limiting against the real server."""

    def test_post_query_without_api_key_returns_401(self):
        with httpx.Client() as client:
            resp = client.post(f"{BASE_URL}/query", json=VALID_PAYLOAD)
        assert resp.status_code == 401

    def test_post_query_with_invalid_api_key_returns_401(self):
        with httpx.Client(headers={"X-API-Key": "dmars_not-a-real-key"}) as client:
            resp = client.post(f"{BASE_URL}/query", json=VALID_PAYLOAD)
        assert resp.status_code == 401

    def test_post_query_with_valid_api_key_processes_normally(self, auth_client):
        resp = auth_client.post(f"{BASE_URL}/query", json=VALID_PAYLOAD)
        assert resp.status_code == 202

    @pytest.mark.slow
    def test_spamming_requests_triggers_rate_limit(self, api_key):
        """Checkpoint 19 criterion: N+ requests in a minute -> 429."""
        from config.settings import settings

        with httpx.Client(headers={"X-API-Key": api_key}) as client:
            statuses = [
                client.post(f"{BASE_URL}/query", json=VALID_PAYLOAD).status_code
                for _ in range(settings.rate_limit_per_minute + 10)
            ]

        assert 429 in statuses, f"Expected a 429 among {statuses}"

    @pytest.mark.slow
    def test_different_api_keys_have_separate_rate_limit_buckets(self):
        """A second, freshly-created key must not inherit the first key's usage count."""
        admin_secret = os.getenv("ADMIN_SECRET", "")
        if not admin_secret:
            pytest.skip("ADMIN_SECRET not set — cannot mint a second test API key")

        with httpx.Client() as client:
            resp = client.post(
                f"{BASE_URL}/auth/create-key",
                json={"name": "test_task_queue.py-second-key"},
                headers={"X-Admin-Secret": admin_secret},
            )
        second_key = resp.json()["api_key"]

        with httpx.Client(headers={"X-API-Key": second_key}) as client:
            resp = client.post(f"{BASE_URL}/query", json=VALID_PAYLOAD)

        assert resp.status_code == 202, "A brand-new key should not be pre-rate-limited"
