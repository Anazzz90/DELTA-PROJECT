# tests/integration/test_task_queue.py
# Checkpoint 14 — Integration tests for the Redis + RQ task queue.
#
# Prerequisites:
#   - Redis running: docker run -d -p 6379:6379 redis
#   - RQ worker running: python task_queue/worker.py
#   - FastAPI running: uvicorn api.main:app --reload
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

    def test_post_query_returns_job_id(self):
        with httpx.Client() as client:
            resp = client.post(f"{BASE_URL}/query", json=VALID_PAYLOAD)
        assert resp.status_code == 202, resp.text
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        assert data["poll_url"] == f"/query/{data['job_id']}/status"

    def test_post_query_empty_question_rejected(self):
        bad = {**VALID_PAYLOAD, "question": ""}
        with httpx.Client() as client:
            resp = client.post(f"{BASE_URL}/query", json=bad)
        assert resp.status_code == 422

    def test_post_query_empty_facts_rejected(self):
        bad = {**VALID_PAYLOAD, "fact_set": []}
        with httpx.Client() as client:
            resp = client.post(f"{BASE_URL}/query", json=bad)
        assert resp.status_code == 422

    def test_post_query_no_agents_rejected(self):
        bad = {**VALID_PAYLOAD, "selected_agents": []}
        with httpx.Client() as client:
            resp = client.post(f"{BASE_URL}/query", json=bad)
        assert resp.status_code == 422


class TestTaskQueueStatusPolling:
    """GET /query/{job_id}/status reflects correct job lifecycle."""

    def test_status_not_found_for_fake_job(self):
        with httpx.Client() as client:
            resp = client.get(f"{BASE_URL}/query/not-a-real-job-id/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "not_found"

    def test_queued_status_immediately_after_enqueue(self):
        with httpx.Client() as client:
            enqueue_resp = client.post(f"{BASE_URL}/query", json=VALID_PAYLOAD)
            assert enqueue_resp.status_code == 202
            job_id = enqueue_resp.json()["job_id"]

            # Check immediately — should be queued or started
            status_resp = client.get(f"{BASE_URL}/query/{job_id}/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] in ("queued", "started", "finished")


class TestTaskQueueCompletion:
    """End-to-end: job should finish with a valid QueryResponse."""

    @pytest.mark.slow
    def test_full_pipeline_completes_successfully(self):
        with httpx.Client() as client:
            enqueue_resp = client.post(f"{BASE_URL}/query", json=VALID_PAYLOAD)
            assert enqueue_resp.status_code == 202
            job_id = enqueue_resp.json()["job_id"]

            result = poll_until_done(job_id, client)

        assert result["status"] == "finished", f"Job failed: {result.get('error')}"
        assert result["result"] is not None
        qr = result["result"]
        assert "query_id" in qr
        assert "system_confidence_score" in qr
        assert isinstance(qr["agent_results"], list)
        assert len(qr["agent_results"]) > 0

    @pytest.mark.slow
    def test_three_concurrent_queries_all_complete(self):
        """Submit 3 simultaneous queries — all should finish without crashing."""
        import concurrent.futures

        def submit_and_poll():
            with httpx.Client() as client:
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
        """The /health endpoint must not depend on Redis."""
        with httpx.Client() as client:
            resp = client.get(f"{BASE_URL}/health")
        assert resp.status_code == 200
