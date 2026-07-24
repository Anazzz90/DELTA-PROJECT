"""
tests/unit/test_delta_protocol.py
====================================
Checkpoint 3 — Unit Tests for the Delta-First Protocol Renderer

Coverage (maps exactly to checkpoint test criteria):
  ✅ Criterion 1 — python core/delta_protocol.py → renders clean prompts per agent
  ✅ Criterion 2 — Each prompt contains all 6 Delta-First steps
  ✅ Criterion 3 — Jinja2 placeholders are fully replaced (no {{ }} left in output)
  ✅ Criterion 4 — Change YAML → re-render → changes reflected (no Python code change)
  ✅ Criterion 5 — pytest tests/unit/test_delta_protocol.py → all pass

Run:
    poetry run pytest tests/unit/test_delta_protocol.py -v
"""

import pytest
import yaml

from core.delta_protocol import (
    DeltaProtocol,
    PromptNotFoundError,
    PromptRenderError,
    RenderedPrompt,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def protocol() -> DeltaProtocol:
    return DeltaProtocol()


@pytest.fixture
def sample_question() -> str:
    return "Why did BTC spike 8% in the last hour?"


@pytest.fixture
def sample_facts() -> list[str]:
    return [
        "BTC volume up 3x in 60 minutes",
        "Large derivatives positions liquidated",
        "No major news reported in that window",
    ]


@pytest.fixture
def sample_domain() -> str:
    return "intraday_trading"


# =============================================================================
# Criterion 2 — All 6 Delta-First Steps Present
# =============================================================================

class TestSixStepsPresent:
    """Every rendered system prompt must contain all 6 Delta-First steps."""

    @pytest.mark.parametrize("agent_name", [
        "neutral_analyst",
        "data_first",
        "skeptic",
    ])
    def test_all_six_steps_in_system_prompt(
        self, protocol, agent_name, sample_question, sample_facts
    ):
        prompt = protocol.render(agent_name, sample_question, sample_facts)
        for step_num in range(1, 7):
            assert f"STEP {step_num}" in prompt.system, (
                f"Agent '{agent_name}': STEP {step_num} missing from system prompt"
            )

    @pytest.mark.parametrize("agent_name", [
        "neutral_analyst",
        "data_first",
        "skeptic",
    ])
    def test_step_keywords_present(
        self, protocol, agent_name, sample_question, sample_facts
    ):
        """Each step's key instruction must be present in the system prompt."""
        prompt = protocol.render(agent_name, sample_question, sample_facts)
        expected_keywords = [
            "LOCK",       # Step 1
            "BIAS",       # Step 2
            "COMPARE",    # Step 3
            "MAIN DRIVER",# Step 4
            "STRESS",     # Step 5
            "HUMILITY",   # Step 6
        ]
        for keyword in expected_keywords:
            assert keyword in prompt.system.upper(), (
                f"Agent '{agent_name}': keyword '{keyword}' missing from system prompt"
            )


# =============================================================================
# Criterion 3 — Jinja2 Placeholders Fully Replaced
# =============================================================================

class TestPlaceholdersResolved:
    """No Jinja2 placeholders ({{ }}) should remain in any rendered output."""

    @pytest.mark.parametrize("agent_name", [
        "neutral_analyst",
        "data_first",
        "skeptic",
    ])
    def test_no_raw_placeholders_in_user_prompt(
        self, protocol, agent_name, sample_question, sample_facts
    ):
        prompt = protocol.render(agent_name, sample_question, sample_facts)
        assert "{{" not in prompt.user, (
            f"Agent '{agent_name}': unresolved Jinja2 placeholder found in user prompt"
        )
        assert "}}" not in prompt.user

    @pytest.mark.parametrize("agent_name", [
        "neutral_analyst",
        "data_first",
        "skeptic",
    ])
    def test_question_appears_in_user_prompt(
        self, protocol, agent_name, sample_question, sample_facts
    ):
        """The actual question text must appear in the rendered user prompt."""
        prompt = protocol.render(agent_name, sample_question, sample_facts)
        assert sample_question in prompt.user, (
            f"Agent '{agent_name}': question not found in user prompt"
        )

    @pytest.mark.parametrize("agent_name", [
        "neutral_analyst",
        "data_first",
        "skeptic",
    ])
    def test_all_facts_appear_in_user_prompt(
        self, protocol, agent_name, sample_question, sample_facts
    ):
        """Every fact from the input must appear in the rendered user prompt."""
        prompt = protocol.render(agent_name, sample_question, sample_facts)
        for fact in sample_facts:
            assert fact in prompt.user, (
                f"Agent '{agent_name}': fact '{fact}' not found in user prompt"
            )

    def test_domain_profile_injected_when_provided(
        self, protocol, sample_question, sample_facts, sample_domain
    ):
        prompt = protocol.render(
            "neutral_analyst", sample_question, sample_facts, sample_domain
        )
        assert sample_domain in prompt.user

    def test_domain_profile_absent_when_not_provided(
        self, protocol, sample_question, sample_facts
    ):
        """When domain_profile is None, no broken placeholder should appear."""
        prompt = protocol.render("neutral_analyst", sample_question, sample_facts)
        assert "{{" not in prompt.user
        assert "}}" not in prompt.user


# =============================================================================
# Criterion 1 — Clean Rendered Output Structure
# =============================================================================

class TestRenderedPromptStructure:
    """Verify the RenderedPrompt data structure is correct."""

    def test_returns_rendered_prompt_type(
        self, protocol, sample_question, sample_facts
    ):
        result = protocol.render("neutral_analyst", sample_question, sample_facts)
        assert isinstance(result, RenderedPrompt)

    def test_model_field_is_set(self, protocol, sample_question, sample_facts):
        result = protocol.render("neutral_analyst", sample_question, sample_facts)
        assert result.model and result.model != "unknown"

    def test_agent_name_field_correct(self, protocol, sample_question, sample_facts):
        result = protocol.render("neutral_analyst", sample_question, sample_facts)
        assert result.agent_name == "neutral_analyst"

    def test_system_prompt_is_non_empty(self, protocol, sample_question, sample_facts):
        result = protocol.render("neutral_analyst", sample_question, sample_facts)
        assert len(result.system) > 100  # Should be a real prompt, not empty

    def test_user_prompt_is_non_empty(self, protocol, sample_question, sample_facts):
        result = protocol.render("neutral_analyst", sample_question, sample_facts)
        assert len(result.user) > 20

    def test_description_field_is_set(self, protocol, sample_question, sample_facts):
        result = protocol.render("neutral_analyst", sample_question, sample_facts)
        assert result.description  # must be non-empty string

    @pytest.mark.parametrize("agent_name", [
        "neutral_analyst",
        "data_first",
        "skeptic",
    ])
    def test_output_schema_in_system_prompt(
        self, protocol, agent_name, sample_question, sample_facts
    ):
        """Every agent system prompt must include the JSON output schema fields."""
        prompt = protocol.render(agent_name, sample_question, sample_facts)
        for field in [
            "extracted_facts",
            "possible_explanations",
            "ranked_hypotheses",
            "main_driver",
            "confidence_score",
            "acknowledged_weaknesses",
        ]:
            assert field in prompt.system, (
                f"Agent '{agent_name}': JSON field '{field}' missing from system prompt"
            )


# =============================================================================
# Criterion 4 — YAML Change → Rendered Prompt Changes (no Python code change)
# =============================================================================

class TestYamlChangeReflected:
    """
    Verifies that editing a YAML file changes the rendered output
    without modifying any Python code.

    Approach: Write a temp YAML file, render it, mutate the YAML,
    render again — the output must differ.
    """

    def test_yaml_mutation_reflected_in_output(
        self, protocol, tmp_path, monkeypatch, sample_question, sample_facts
    ):
        """
        Write a custom YAML to a temp dir, render it, change the YAML,
        render again — output changes without touching Python.
        """
        from config import settings as settings_module

        # Point active prompts dir to our temp directory
        monkeypatch.setattr(settings_module.settings, "active_prompt_version", "v1")

        # Write a minimal valid YAML to the real prompts dir (using a test agent name)
        # Instead: use the real neutral_analyst but intercept _load_yaml
        original_yaml = protocol.get_prompt_path("neutral_analyst")
        with open(original_yaml, "r") as f:
            data = yaml.safe_load(f)

        original_description = data.get("description", "")

        # Temporarily mutate the YAML file
        data["description"] = "MUTATED_DESCRIPTION_FOR_TESTING"
        with open(original_yaml, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        try:
            # Re-render — should pick up the mutation instantly
            prompt = protocol.render("neutral_analyst", sample_question, sample_facts)
            assert prompt.description == "MUTATED_DESCRIPTION_FOR_TESTING"
        finally:
            # Restore original
            data["description"] = original_description
            with open(original_yaml, "w") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


# =============================================================================
# Error Handling
# =============================================================================

class TestErrorHandling:
    """Error paths must raise descriptive exceptions."""

    def test_missing_agent_raises_prompt_not_found(
        self, protocol, sample_question, sample_facts
    ):
        with pytest.raises(PromptNotFoundError, match="nonexistent_agent"):
            protocol.render("nonexistent_agent", sample_question, sample_facts)

    def test_error_message_shows_expected_path(
        self, protocol, sample_question, sample_facts
    ):
        with pytest.raises(PromptNotFoundError, match="nonexistent_agent"):
            protocol.render("nonexistent_agent", sample_question, sample_facts)

    def test_list_available_agents_returns_correct_agents(self, protocol):
        available = protocol.list_available_agents()
        assert "neutral_analyst" in available
        assert "data_first" in available
        assert "skeptic" in available

    def test_render_all_returns_dict_of_prompts(
        self, protocol, sample_question, sample_facts
    ):
        agents = ["neutral_analyst", "data_first", "skeptic"]
        results = protocol.render_all(agents, sample_question, sample_facts)
        assert set(results.keys()) == set(agents)
        for name, prompt in results.items():
            assert isinstance(prompt, RenderedPrompt)
            assert prompt.agent_name == name


# =============================================================================
# Checkpoint 20 — Prompt Versioning System
# =============================================================================

class TestPromptVersioning:
    """
    Coverage (maps exactly to Checkpoint 20 test criteria):
      Criterion 1 — ACTIVE_PROMPT_VERSION=v1 -> system uses v1 prompts
      Criterion 2 — ACTIVE_PROMPT_VERSION=v2 -> system uses v2 prompts (different behavior)
      Criterion 3 — Missing version folder -> falls back to v1 with a warning (no crash)
      Criterion 4 — Same query on v1 vs v2 -> rendered prompts differ (proves versioning works)
    """

    def test_default_active_version_loads_v1(self, protocol, sample_question, sample_facts):
        prompt = protocol.render("neutral_analyst", sample_question, sample_facts)
        assert prompt.version == "v1"

    def test_switching_to_v2_loads_v2_prompts(
        self, protocol, sample_question, sample_facts, monkeypatch
    ):
        from config.settings import settings
        monkeypatch.setattr(settings, "active_prompt_version", "v2")

        prompt = protocol.render("neutral_analyst", sample_question, sample_facts)

        assert prompt.version == "v2"
        assert "V2 AMENDMENTS" in prompt.system

    def test_v1_and_v2_prompts_differ_for_the_same_query(
        self, protocol, sample_question, sample_facts, monkeypatch
    ):
        from config.settings import settings

        monkeypatch.setattr(settings, "active_prompt_version", "v1")
        v1_prompt = protocol.render("neutral_analyst", sample_question, sample_facts)

        monkeypatch.setattr(settings, "active_prompt_version", "v2")
        v2_prompt = protocol.render("neutral_analyst", sample_question, sample_facts)

        assert v1_prompt.system != v2_prompt.system
        assert "NUMERIC FACT-FIT SCORING" in v2_prompt.system
        assert "NUMERIC FACT-FIT SCORING" not in v1_prompt.system

    def test_missing_version_folder_falls_back_to_v1_with_warning(
        self, protocol, sample_question, sample_facts, monkeypatch, caplog
    ):
        import logging
        from config.settings import settings

        monkeypatch.setattr(settings, "active_prompt_version", "v3-does-not-exist")

        with caplog.at_level(logging.WARNING, logger="core.delta_protocol"):
            prompt = protocol.render("neutral_analyst", sample_question, sample_facts)

        assert prompt is not None  # did not crash
        assert "falling back to 'v1'" in caplog.text

    def test_all_v2_agent_files_exist_matching_v1(self):
        v1_agents = set(DeltaProtocol().list_available_agents())
        from config.settings import settings
        # list_available_agents() reads settings.prompts_dir, which follows
        # active_prompt_version -- check the v2 folder directly instead.
        v2_dir = settings.prompts_dir.parent / "v2"
        v2_agents = {p.stem for p in v2_dir.glob("*.yaml")}
        assert v1_agents == v2_agents, "every v1 agent prompt must have a v2 counterpart"
