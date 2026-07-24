"""
llm/resilience.py
==================
Checkpoint 4 (retry) / Checkpoint 16 (full resilience layer)

Wraps raw LiteLLM calls with:
  - 3 retries with exponential backoff (2s → 4s → 8s), 4 attempts total
  - Retry only on transient errors (rate limits, timeouts, connection issues)
  - No retry on permanent errors (auth failures, bad requests)
  - Structured logging before each retry attempt
  - Circuit breaker: after all retries are exhausted, the agent is marked
    FAILED (llm/router.py converts the raised exception into an
    LLMResponse with `.error` set) and the pipeline continues without it
    (core/pipeline.py never lets one agent's failure abort the others).

Usage (internal — called by llm/router.py):
    from llm.resilience import call_with_retry
    raw = call_with_retry(model, messages, temperature, timeout)
"""

import logging

import litellm
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Retryable vs Non-Retryable exceptions
# =============================================================================

# These are transient — worth retrying (server-side or network issues)
RETRYABLE_EXCEPTIONS = (
    litellm.exceptions.RateLimitError,         # 429 — slow down
    litellm.exceptions.ServiceUnavailableError, # 503 — provider down
    litellm.exceptions.Timeout,                 # Request timed out
    litellm.exceptions.APIConnectionError,      # Network connectivity
)

# These are permanent — retrying won't help
NON_RETRYABLE_EXCEPTIONS = (
    litellm.exceptions.AuthenticationError,     # Wrong API key
    litellm.exceptions.BadRequestError,         # Malformed request
    litellm.exceptions.NotFoundError,           # Model not found
)


# =============================================================================
# Core retry-wrapped LiteLLM call
# =============================================================================

@retry(
    reraise=True,                                       # Raise original exception after all retries
    stop=stop_after_attempt(4),                         # 1 initial attempt + 3 retries
    wait=wait_exponential(multiplier=2, min=2, max=8),  # 2s → 4s → 8s backoff
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def call_with_retry(
    model: str,
    messages: list[dict],
    temperature: float,
    timeout: int,
    **kwargs,
) -> object:
    """
    Make a LiteLLM completion call with automatic retry on transient failures.

    Retry schedule:
      Attempt 1: immediate
      Attempt 2: wait ~2s
      Attempt 3: wait ~4s
      Attempt 4: wait ~8s
      After the 4th failure (3 retries exhausted): raises the original
      exception — the circuit breaker trips, and the caller (llm/router.py)
      converts this into a failed LLMResponse rather than crashing.

    Args:
        model:       LiteLLM model string (e.g. "gpt-4o-mini", "groq/mixtral-8x7b-32768")
        messages:    OpenAI-format message list [{"role": ..., "content": ...}]
        temperature: Sampling temperature
        timeout:     Request timeout in seconds
        **kwargs:    Extra arguments passed to litellm.completion (e.g. api_base, api_key)

    Returns:
        Raw LiteLLM ModelResponse object

    Raises:
        litellm.exceptions.RateLimitError: After all retries exhausted
        litellm.exceptions.Timeout:        After all retries exhausted
        litellm.exceptions.AuthenticationError: Immediately (not retried)
    """
    return litellm.completion(
        model=model,
        messages=messages,
        temperature=temperature,
        timeout=timeout,
        **kwargs,
    )
