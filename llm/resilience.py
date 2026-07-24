"""
llm/resilience.py
==================
Checkpoint 4 — Tenacity Retry + Circuit Breaker

Wraps raw LiteLLM calls with:
  - 3 retry attempts with exponential backoff (2s → 4s → 8s)
  - Retry only on transient errors (rate limits, timeouts, connection issues)
  - No retry on permanent errors (auth failures, bad requests)
  - Structured logging before each retry attempt

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
    stop=stop_after_attempt(3),                         # Max 3 attempts
    wait=wait_exponential(multiplier=1, min=2, max=8),  # 2s → 4s → 8s backoff
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
      After 3rd failure: raises the original exception

    Args:
        model:       LiteLLM model string (e.g. "gpt-4o-mini", "groq/mixtral-8x7b-32768")
        messages:    OpenAI-format message list [{"role": ..., "content": ...}]
        temperature: Sampling temperature
        timeout:     Request timeout in seconds
        **kwargs:    Extra arguments passed to litellm.completion (e.g. api_base, api_key)

    Returns:
        Raw LiteLLM ModelResponse object

    Raises:
        litellm.exceptions.RateLimitError: After 3 failed rate-limited attempts
        litellm.exceptions.Timeout:        After 3 timed-out attempts
        litellm.exceptions.AuthenticationError: Immediately (not retried)
    """
    return litellm.completion(
        model=model,
        messages=messages,
        temperature=temperature,
        timeout=timeout,
        **kwargs,
    )
