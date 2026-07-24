"""
tests/fixtures/test_llm_call.py
=================================
Checkpoint 4 -- Manual LLM Router Test Script

This is a MANUAL test script (not pytest) that makes real API calls
to verify the LLM router works end-to-end with your actual API keys.

Run from the dmars/ directory:
    poetry run python tests/fixtures/test_llm_call.py

Tests:
    1. Groq llama-3.1-8b-instant  -> fast, free tier
    2. Groq llama-3.3-70b-versatile -> larger Groq model
    3. Ollama mistral:7b           -> requires Ollama running locally (SKIPPED if not running)
    4. Retry simulation            -> uses a bad model to trigger retry behavior
    5. Cost + token data           -> verifies metadata is returned
"""

import sys
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from llm.router import LLMRouter, LLMResponse

# =============================================================================
# Shared test data
# =============================================================================

SYSTEM_PROMPT = """You are a concise reasoning assistant.
Answer in maximum 2 sentences. Return only plain text."""

USER_PROMPT = """Question: Why did BTC spike 8% in the last hour?
Facts:
- BTC volume up 3x in 60 minutes
- Large derivatives positions liquidated
- No major news reported

Give a brief explanation."""

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


def print_result(test_num, name, status, detail=""):
    icon = {"PASS": "OK  ", "FAIL": "FAIL", "SKIP": "SKIP"}[status]
    print(f"\n  [{icon}] Test {test_num}: {name}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"       {line}")


def print_response(resp):
    if resp.success:
        print(f"       Model    : {resp.model}")
        print(f"       Tokens   : {resp.prompt_tokens} prompt + {resp.completion_tokens} completion = {resp.total_tokens} total")
        print(f"       Cost     : ${resp.cost_usd:.6f} USD")
        print(f"       Latency  : {resp.latency_ms} ms")
        preview = resp.content[:200].replace("\n", " ")
        print(f"       Response : {preview}{'...' if len(resp.content) > 200 else ''}")
    else:
        print(f"       Error    : {resp.error}")


# =============================================================================
# Tests
# =============================================================================

def test_1_groq_fast(router):
    """Test 1: Groq llama-3.1-8b-instant (fast, free tier)."""
    print("\n" + "=" * 60)
    print("TEST 1 -- Groq llama-3.1-8b-instant")
    print("=" * 60)

    resp = router.call_model_direct(
        model="groq/llama-3.1-8b-instant",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
    )

    if resp.success and resp.content:
        print_result(1, "Groq llama-3.1-8b-instant", PASS)
        print_response(resp)
        assert resp.total_tokens > 0, "Token count should be > 0"
        assert resp.latency_ms > 0, "Latency should be > 0"
        return PASS
    else:
        print_result(1, "Groq llama-3.1-8b-instant", FAIL, resp.error or "Empty response")
        return FAIL


def test_2_groq_large(router):
    """Test 2: Groq llama-3.3-70b-versatile (larger model)."""
    print("\n" + "=" * 60)
    print("TEST 2 -- Groq llama-3.3-70b-versatile")
    print("=" * 60)

    resp = router.call_model_direct(
        model="groq/llama-3.3-70b-versatile",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
    )

    if resp.success and resp.content:
        print_result(2, "Groq llama-3.3-70b-versatile", PASS)
        print_response(resp)
        assert resp.total_tokens > 0, "Token count should be > 0"
        return PASS
    else:
        print_result(2, "Groq llama-3.3-70b-versatile", FAIL, resp.error or "Empty response")
        return FAIL


def test_3_ollama(router):
    """Test 3: Ollama mistral:7b (local). Skipped if Ollama not running."""
    print("\n" + "=" * 60)
    print("TEST 3 -- Ollama (mistral:7b) -- requires local Ollama")
    print("=" * 60)

    import httpx
    try:
        r = httpx.get("http://localhost:11434", timeout=2)
        if r.status_code != 200:
            raise ConnectionError()
    except Exception:
        print_result(3, "Ollama mistral:7b", SKIP,
                     "Ollama not running locally. Start with: ollama serve")
        return SKIP

    resp = router.call_model_direct(
        model="ollama/mistral:7b",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
        timeout=60,
    )

    if resp.success and resp.content:
        print_result(3, "Ollama mistral:7b", PASS)
        print_response(resp)
        return PASS
    else:
        print_result(3, "Ollama mistral:7b", FAIL, resp.error or "Empty response")
        return FAIL


def test_4_retry_simulation(router):
    """
    Test 4: Retry simulation.
    Uses a fake model name to trigger repeated failures.
    Tenacity should attempt 3 times then give up.
    Watch for 'WARNING ... Retrying' messages -- this is expected.
    """
    print("\n" + "=" * 60)
    print("TEST 4 -- Retry simulation (bad model -> 3 retries -> fail)")
    print("          You should see WARNING messages below. This is expected.")
    print("=" * 60)

    logging.getLogger("llm.resilience").setLevel(logging.WARNING)

    start = time.perf_counter()
    resp = router.call_model_direct(
        model="groq/FAKE-MODEL-DOES-NOT-EXIST",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
        timeout=5,
    )
    elapsed = time.perf_counter() - start

    if not resp.success:
        print_result(
            4, "Retry simulation", PASS,
            f"Correctly failed after retries.\n"
            f"Error type : {resp.error[:120] if resp.error else 'unknown'}\n"
            f"Total time : {elapsed:.1f}s"
        )
        return PASS
    else:
        print_result(4, "Retry simulation", FAIL,
                     "Call should have failed on a non-existent model")
        return FAIL


def test_5_cost_and_tokens(router):
    """
    Test 5: Verify cost + token metadata returned with every response.
    Uses Groq fast tier (free).
    """
    print("\n" + "=" * 60)
    print("TEST 5 -- Cost + Token metadata verification")
    print("=" * 60)

    resp = router.call_model_direct(
        model="groq/llama-3.1-8b-instant",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
    )

    checks = {
        "success is True":       resp.success,
        "total_tokens > 0":      resp.total_tokens > 0,
        "prompt_tokens > 0":     resp.prompt_tokens > 0,
        "completion_tokens > 0": resp.completion_tokens > 0,
        "cost_usd >= 0":         resp.cost_usd >= 0,
        "latency_ms > 0":        resp.latency_ms > 0,
        "model field populated": bool(resp.model),
        "agent_name field set":  bool(resp.agent_name),
    }

    all_passed = all(checks.values())
    for check, result in checks.items():
        icon = "OK   " if result else "FAIL "
        print(f"       [{icon}] {check}")

    if all_passed:
        print_result(5, "Cost + Token metadata", PASS)
        return PASS
    else:
        failed = [k for k, v in checks.items() if not v]
        print_result(5, "Cost + Token metadata", FAIL,
                     f"Failed checks: {failed}")
        return FAIL


# =============================================================================
# Runner
# =============================================================================

def main():
    print("\n" + "#" * 60)
    print("# DMARS -- Checkpoint 4: LLM Router Manual Test")
    print("#" * 60)

    router = LLMRouter()
    results = {}

    results["Groq Fast"]     = test_1_groq_fast(router)
    results["Groq Large"]    = test_2_groq_large(router)
    results["Ollama"]        = test_3_ollama(router)
    results["Retry"]         = test_4_retry_simulation(router)
    results["Cost+Tokens"]   = test_5_cost_and_tokens(router)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test, status in results.items():
        icon = {"PASS": "OK  ", "FAIL": "FAIL", "SKIP": "SKIP"}[status]
        print(f"  [{icon}] {test}")

    passed  = sum(1 for s in results.values() if s == PASS)
    skipped = sum(1 for s in results.values() if s == SKIP)
    failed  = sum(1 for s in results.values() if s == FAIL)
    print(f"\n  {passed} passed | {skipped} skipped | {failed} failed")

    if failed > 0:
        print("\nCheckpoint 4: INCOMPLETE -- fix failing tests above.")
        sys.exit(1)
    else:
        print("\nCheckpoint 4: ALL TESTS PASSED (or skipped)")
        sys.exit(0)


if __name__ == "__main__":
    main()
