"""
core/delta_protocol.py
=======================
Checkpoint 3 — Prompt Template Renderer

Loads agent prompt YAML files and renders them via Jinja2,
injecting the question, fact set, and optional domain profile.

This is the bridge between:
  - prompts/v1/*.yaml  (the Delta-First Protocol templates)
  - agents/*.py        (which call render() to get their final prompt)

Usage:
    from core.delta_protocol import DeltaProtocol

    protocol = DeltaProtocol()
    prompt = protocol.render(
        agent_name="neutral_analyst",
        question="Why did BTC spike 8%?",
        fact_set=["Volume up 3x", "Derivatives liquidated", "No major news"],
        domain_profile="intraday_trading",
    )
    print(prompt.system)  # rendered system prompt
    print(prompt.user)    # rendered user prompt

Can also be run directly for a quick sanity check:
    poetry run python core/delta_protocol.py
"""

from __future__ import annotations

import logging
import yaml
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, StrictUndefined, TemplateError

from config.settings import settings

logger = logging.getLogger(__name__)

_FALLBACK_VERSION = "v1"


# =============================================================================
# Data Types
# =============================================================================

@dataclass
class RenderedPrompt:
    """A fully rendered, injection-ready prompt for a single agent."""
    agent_name:     str
    model:          str          # e.g. "gpt-4o-mini", "ollama/mistral:7b"
    version:        str          # prompt version used, e.g. "v1"
    system:         str          # rendered system prompt (Delta-First Protocol)
    user:           str          # rendered user prompt (question + facts)
    description:    str          # agent's cognitive role description


# =============================================================================
# Exceptions
# =============================================================================

class PromptNotFoundError(FileNotFoundError):
    """Raised when a prompt YAML file cannot be found for a given agent."""
    pass


class PromptRenderError(ValueError):
    """Raised when Jinja2 fails to render a prompt (e.g. missing variable)."""
    pass


# =============================================================================
# Renderer
# =============================================================================

class DeltaProtocol:
    """
    Loads and renders Delta-First Protocol prompt templates for any agent.

    Templates are stored as YAML files under:
        prompts/{active_version}/{agent_name}.yaml

    The active version is controlled by settings.active_prompt_version
    (set via ACTIVE_PROMPT_VERSION env var, default: v1).

    This class is stateless — it reads from disk on every render() call,
    so YAML changes are reflected immediately without restarting.
    """

    def __init__(self) -> None:
        self._env = Environment(
            undefined=StrictUndefined,    # Raise on missing variables, no silent empties
            trim_blocks=True,             # Remove newline after {% %} blocks
            lstrip_blocks=True,           # Strip leading whitespace before {% %} blocks
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def render(
        self,
        agent_name: str,
        question: str,
        fact_set: list[str],
        domain_profile: str | None = None,
    ) -> RenderedPrompt:
        """
        Load the YAML for agent_name and render both system and user prompts.

        Args:
            agent_name:     Name of the agent (e.g. "neutral_analyst").
            question:       The reasoning question submitted by the user.
            fact_set:       List of verified fact strings.
            domain_profile: Optional domain context (e.g. "intraday_trading").

        Returns:
            RenderedPrompt with fully rendered system + user strings.

        Raises:
            PromptNotFoundError: If the YAML file does not exist.
            PromptRenderError:   If Jinja2 rendering fails.
        """
        raw = self._load_yaml(agent_name)
        context = {
            "question":       question,
            "fact_set":       fact_set,
            "domain_profile": domain_profile or "",
        }
        system = self._render_string(raw["system"], context, agent_name, "system")
        user   = self._render_string(raw["user"],   context, agent_name, "user")

        return RenderedPrompt(
            agent_name=agent_name,
            model=raw.get("model", "unknown"),
            version=raw.get("version", settings.active_prompt_version),
            system=system.strip(),
            user=user.strip(),
            description=raw.get("description", ""),
        )

    def render_all(
        self,
        agent_names: list[str],
        question: str,
        fact_set: list[str],
        domain_profile: str | None = None,
    ) -> dict[str, RenderedPrompt]:
        """
        Render prompts for multiple agents at once.

        Returns:
            dict mapping agent_name → RenderedPrompt
        """
        return {
            name: self.render(name, question, fact_set, domain_profile)
            for name in agent_names
        }

    def get_prompt_path(self, agent_name: str, version: str | None = None) -> Path:
        """Returns the filesystem path for an agent's YAML prompt file."""
        v = version or settings.active_prompt_version
        return settings.prompts_dir.parent / v / f"{agent_name}.yaml"

    def list_available_agents(self) -> list[str]:
        """Returns a list of available agent names based on YAML files present."""
        return [p.stem for p in settings.prompts_dir.glob("*.yaml")]

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _load_yaml(self, agent_name: str) -> dict:
        """
        Load and parse the YAML file for agent_name.

        Checkpoint 20 — if the active prompt version is missing the file
        (e.g. an incomplete or deleted version folder), falls back to
        _FALLBACK_VERSION ("v1") with a logged warning rather than
        crashing the whole pipeline over one missing prompt variant.
        """
        active_version = settings.active_prompt_version
        path = self.get_prompt_path(agent_name, active_version)

        if not path.exists() and active_version != _FALLBACK_VERSION:
            fallback_path = self.get_prompt_path(agent_name, _FALLBACK_VERSION)
            if fallback_path.exists():
                logger.warning(
                    f"Prompt version '{active_version}' has no file for agent "
                    f"'{agent_name}' (expected {path}) — falling back to "
                    f"'{_FALLBACK_VERSION}': {fallback_path}"
                )
                path = fallback_path

        if not path.exists():
            raise PromptNotFoundError(
                f"Prompt file not found for agent '{agent_name}'.\n"
                f"Expected path: {path}\n"
                f"Active prompt version: {settings.active_prompt_version}\n"
                f"Available agents: {self.list_available_agents()}"
            )
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _render_string(
        self, template_str: str, context: dict, agent_name: str, field: str
    ) -> str:
        """Render a single Jinja2 template string with the given context."""
        try:
            template = self._env.from_string(template_str)
            return template.render(**context)
        except TemplateError as e:
            raise PromptRenderError(
                f"Failed to render '{field}' prompt for agent '{agent_name}': {e}"
            ) from e


# =============================================================================
# Quick test — run directly to verify rendering works
# poetry run python core/delta_protocol.py
# =============================================================================

if __name__ == "__main__":
    import json

    TEST_QUESTION = "Why did BTC spike 8% in the last hour?"
    TEST_FACTS = [
        "BTC volume up 3x in 60 minutes",
        "Large derivatives positions liquidated",
        "No major news reported in that window",
    ]
    TEST_DOMAIN = "intraday_trading"

    protocol = DeltaProtocol()
    agents = ["neutral_analyst", "data_first", "skeptic"]

    print("=" * 70)
    print("DMARS -- Delta-First Protocol Renderer -- Quick Test")
    print("=" * 70)
    print(f"Question    : {TEST_QUESTION}")
    print(f"Facts       : {json.dumps(TEST_FACTS, indent=2)}")
    print(f"Domain      : {TEST_DOMAIN}")
    print(f"Version     : {settings.active_prompt_version}")
    print(f"Prompts dir : {settings.prompts_dir}")
    print()

    for agent_name in agents:
        print(f"{'-' * 70}")
        print(f"AGENT: {agent_name.upper()}")
        print(f"{'-' * 70}")
        try:
            prompt = protocol.render(
                agent_name=agent_name,
                question=TEST_QUESTION,
                fact_set=TEST_FACTS,
                domain_profile=TEST_DOMAIN,
            )
            print(f"Model      : {prompt.model}")
            print(f"Description: {prompt.description}")
            print()
            print("[SYSTEM PROMPT]")
            print(prompt.system[:500] + "..." if len(prompt.system) > 500 else prompt.system)
            print()
            print("[USER PROMPT]")
            print(prompt.user)
            print()

            # Verify all 6 steps are present
            steps_present = all(
                f"STEP {i}" in prompt.system for i in range(1, 7)
            )
            print(f"✅ All 6 Delta-First steps present: {steps_present}")
            placeholders_gone = (
                "{{ question }}" not in prompt.user and
                "{{ fact_set }}" not in prompt.user
            )
            print(f"✅ Jinja2 placeholders resolved: {placeholders_gone}")

        except (PromptNotFoundError, PromptRenderError) as e:
            print(f"❌ ERROR: {e}")

    print(f"\n{'=' * 70}")
    print("Available agents:", protocol.list_available_agents())
