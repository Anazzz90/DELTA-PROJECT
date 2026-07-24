"""
llm/router.py
==============
Checkpoint 4 — LLM Router (LiteLLM)

Maps each agent to its designated model and makes the actual LLM call
via LiteLLM. Returns a structured LLMResponse with content, token
usage, cost, and latency — ready for the scoring engine and aggregator.

Agent → Model assignments (Phase 1):
    neutral_analyst  → gpt-4o-mini       (OpenAI — balanced reasoning)
    data_first       → ollama/mistral:7b   (Local — fact-bound, cheap)
    skeptic          → groq/llama-3.1-8b-instant (Groq — fast adversarial)

Phase 2 additions (inactive until CP11):
    contrarian       → groq/llama3-70b-8192
    intuition        → ollama/llama3:8b
    meta_ai          → gpt-4o

Usage:
    from llm.router import LLMRouter

    router = LLMRouter()
    response = router.call(
        agent_name="neutral_analyst",
        system_prompt=prompt.system,
        user_prompt=prompt.user,
    )
    print(response.content)
    print(f"Tokens: {response.total_tokens} | Cost: ${response.cost_usd:.6f} | Latency: {response.latency_ms}ms")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import litellm

from config.settings import settings
from llm.resilience import call_with_retry
from observability.tracer import configure_langfuse, log_generation

logger = logging.getLogger(__name__)

# Suppress LiteLLM's noisy debug output
litellm.suppress_debug_info = True
litellm.set_verbose = False

# =============================================================================
# Agent → Model Map
# =============================================================================

AGENT_MODEL_MAP: dict[str, str] = {
    # Phase 1 -- 3 MVP agents (All switched to Chinese models via SiliconFlow)
    "neutral_analyst": "deepseek-ai/DeepSeek-V3",             # DeepSeek -- balanced reasoning
    "data_first":      "groq/llama-3.1-8b-instant",          # Groq     -- extremely fast, fact-bound
    "skeptic":         "groq/llama-3.3-70b-versatile",       # Groq     -- fast adversarial reasoning
    # Phase 2 -- 2 additional agents (Checkpoint 11)
    "contrarian":      "zai-org/GLM-4.5-Air",                # Zhipu AI -- different architecture
    "intuition":       "Qwen/Qwen2.5-72B-Instruct",          # Alibaba  -- powerful independent reasoning
    # Phase 2 -- Meta-AI synthesizer (Checkpoint 12)
    "meta_ai":         "deepseek-ai/DeepSeek-V3",             # DeepSeek -- final synthesis
}

# Per-agent request timeout (seconds)
# Ollama agents get more time since local inference is slower
AGENT_TIMEOUTS: dict[str, int] = {
    "neutral_analyst": 60,
    "data_first":      30,   # Groq is very fast
    "skeptic":         30,   # Groq is very fast
    "contrarian":      60,
    "intuition":       60,
    "meta_ai":         90,
}
DEFAULT_TIMEOUT = 30


# =============================================================================
# Response Type
# =============================================================================

@dataclass
class LLMResponse:
    """
    Structured result from a single agent LLM call.
    Includes the response content plus all observability metadata.
    """
    agent_name:        str
    model:             str
    content:           str           # Raw text response from the LLM
    prompt_tokens:     int   = 0
    completion_tokens: int   = 0
    total_tokens:      int   = 0
    cost_usd:          float = 0.0   # Estimated USD cost for this call
    latency_ms:        float = 0.0   # End-to-end time in milliseconds
    cached:            bool  = False  # True if result came from GPTCache (CP18)
    error:             Optional[str] = None  # Set when call fails

    @property
    def success(self) -> bool:
        """True if the call completed without error."""
        return self.error is None

    def cost_summary(self) -> str:
        return (
            f"agent={self.agent_name} model={self.model} "
            f"tokens={self.total_tokens} cost=${self.cost_usd:.6f} "
            f"latency={self.latency_ms}ms"
        )


# =============================================================================
# Router
# =============================================================================

class LLMRouter:
    """
    Routes each DMARS agent to its designated LLM via LiteLLM.

    Responsibilities:
    - Resolve agent name → model string
    - Inject API credentials
    - Apply retry logic (via llm/resilience.py)
    - Track token usage and estimated cost
    - Measure end-to-end latency
    """

    def __init__(self) -> None:
        self._configure_api_keys()
        configure_langfuse()

    def _configure_api_keys(self) -> None:
        """Inject API keys from settings into LiteLLM."""
        if settings.openai_api_key:
            litellm.openai_key = settings.openai_api_key
        if settings.groq_api_key:
            litellm.groq_key = settings.groq_api_key
        if settings.google_api_key:
            litellm.google_key = settings.google_api_key
        # SiliconFlow handled per-call in self.call()
        litellm.drop_params = True  # Ignore unsupported params per provider

    # =========================================================================
    # Public API
    # =========================================================================

    def call(
        self,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        model_override: Optional[str] = None,
        temperature: float = 0.3,
        trace_id: Optional[str] = None,
    ) -> LLMResponse:
        """
        Make a single LLM call for the given agent.

        Args:
            agent_name:     Agent name (used for model resolution + timeout)
            system_prompt:  Rendered system prompt from delta_protocol.py
            user_prompt:    Rendered user prompt from delta_protocol.py
            model_override: Override the default model (useful for testing)
            temperature:    Sampling temperature (0.3 = consistent, analytical)
            trace_id:       Optional id grouping this call with sibling agent
                             calls from the same pipeline run into one
                             LangFuse trace (Checkpoint 17). No-op if
                             tracing is disabled.

        Returns:
            LLMResponse with content, usage, cost, and latency.
            On failure: LLMResponse with error set, content = "".
        """
        model   = model_override or AGENT_MODEL_MAP.get(agent_name, "gpt-4o-mini")
        timeout = AGENT_TIMEOUTS.get(agent_name, DEFAULT_TIMEOUT)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        # Inject SiliconFlow credentials for Chinese models
        kwargs = {}
        # Identify models hosted on SiliconFlow (lowercase brands for matching)
        sf_models = ["deepseek", "qwen", "glm", "zai-org", "qwq"]
        if any(brand in model.lower() for brand in sf_models):
            if settings.siliconflow_api_key:
                kwargs["api_base"] = "https://api.siliconflow.com/v1"
                kwargs["api_key"] = settings.siliconflow_api_key
                # Force LiteLLM to use OpenAI-compatible provider for SiliconFlow
                if not model.startswith("openai/"):
                    model = f"openai/{model}"

        logger.info(f"[{agent_name}] Calling {model} (timeout={timeout}s)")
        start = time.perf_counter()

        try:
            raw  = call_with_retry(model, messages, temperature, timeout, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000

            content           = raw.choices[0].message.content or ""
            usage             = getattr(raw, "usage", None)
            prompt_tokens     = getattr(usage, "prompt_tokens", 0)     or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            total_tokens      = getattr(usage, "total_tokens", 0)      or 0

            try:
                cost_usd = litellm.completion_cost(completion_response=raw)
            except Exception:
                cost_usd = 0.0

            resp = LLMResponse(
                agent_name=agent_name,
                model=model,
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=round(cost_usd, 8),
                latency_ms=round(elapsed_ms, 2),
            )
            logger.info(f"[{agent_name}] OK — {resp.cost_summary()}")
            log_generation(
                trace_id, agent_name, model, messages, content,
                prompt_tokens, completion_tokens, total_tokens,
                resp.cost_usd, resp.latency_ms,
            )
            return resp

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(
                f"Agent {agent_name} failed after 3 retries — skipping "
                f"({type(e).__name__}: {e})"
            )
            log_generation(
                trace_id, agent_name, model, messages, "",
                latency_ms=round(elapsed_ms, 2),
                error=f"{type(e).__name__}: {e}",
            )
            return LLMResponse(
                agent_name=agent_name,
                model=model,
                content="",
                latency_ms=round(elapsed_ms, 2),
                error=f"{type(e).__name__}: {e}",
            )

    def call_model_direct(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> LLMResponse:
        """
        Call a specific model directly by name (bypasses agent mapping).
        Useful for testing individual models.
        """
        return self.call(
            agent_name=f"direct:{model}",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_override=model,
            temperature=temperature,
        )

