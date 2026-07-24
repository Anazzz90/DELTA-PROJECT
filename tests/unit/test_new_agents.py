"""
tests/unit/test_new_agents.py
================================
Checkpoint 11 -- Unit Tests for Contrarian + Intuition Agents

Coverage (maps exactly to checkpoint test criteria):
  Criterion 1 -- All 5 agents exist and have correct names + descriptions
  Criterion 2 -- Contrarian prompt instructs challenging the majority view
  Criterion 3 -- Intuition prompt instructs fast pattern-based reasoning
  Criterion 4 -- All 5 agents use different models (cognitive diversity)
  Criterion 5 -- Pipeline accepts all 5 agents without error

No API calls. Pure Python.

Run:
    poetry run pytest tests/unit/test_new_agents.py -v
"""

import pytest

from agents.base_agent import BaseAgent
from agents.contrarian import ContrarianAgent
from agents.data_first import DataFirstAgent
from agents.intuition import IntuitionAgent
from agents.neutral_analyst import NeutralAnalyst
from agents.skeptic import SkepticAgent
from core.pipeline import Pipeline
from llm.router import AGENT_MODEL_MAP


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def all_five_agents() -> list[BaseAgent]:
    return [
        NeutralAnalyst(),
        DataFirstAgent(),
        SkepticAgent(),
        ContrarianAgent(),
        IntuitionAgent(),
    ]


# =============================================================================
# Criterion 1 -- All 5 agents exist with correct names and descriptions
# =============================================================================

class TestAgentIdentity:

    def test_contrarian_name(self):
        assert ContrarianAgent().name == "contrarian"

    def test_intuition_name(self):
        assert IntuitionAgent().name == "intuition"

    def test_contrarian_description_is_non_empty(self):
        agent = ContrarianAgent()
        assert len(agent.description) > 10

    def test_intuition_description_is_non_empty(self):
        agent = IntuitionAgent()
        assert len(agent.description) > 10

    def test_contrarian_description_mentions_counter_narrative(self):
        desc = ContrarianAgent().description.lower()
        assert any(word in desc for word in ["challenge", "counter", "consensus", "argument"])

    def test_intuition_description_mentions_pattern(self):
        desc = IntuitionAgent().description.lower()
        assert any(word in desc for word in ["pattern", "heuristic", "fast", "gut", "recognition"])

    def test_all_five_agents_are_base_agent_subclasses(self, all_five_agents):
        for agent in all_five_agents:
            assert isinstance(agent, BaseAgent), f"{agent.name} is not a BaseAgent subclass"

    def test_all_five_agent_names_are_unique(self, all_five_agents):
        names = [a.name for a in all_five_agents]
        assert len(names) == len(set(names)), "Duplicate agent names detected"


# =============================================================================
# Criterion 2 -- Contrarian prompt instructs challenging the majority view
# =============================================================================

class TestContrarianPrompt:

    def test_contrarian_yaml_exists(self):
        from pathlib import Path
        yaml_path = Path("prompts/v1/contrarian.yaml")
        assert yaml_path.exists(), "prompts/v1/contrarian.yaml not found"

    def test_contrarian_yaml_has_correct_model(self):
        import yaml
        from pathlib import Path
        data = yaml.safe_load(Path("prompts/v1/contrarian.yaml").read_text(encoding="utf-8"))
        assert "glm" in data["model"].lower() or "thudm" in data["model"].lower()

    def test_contrarian_yaml_agent_name_matches(self):
        import yaml
        from pathlib import Path
        data = yaml.safe_load(Path("prompts/v1/contrarian.yaml").read_text(encoding="utf-8"))
        assert data["agent"] == "contrarian"

    def test_contrarian_system_prompt_instructs_to_challenge(self):
        import yaml
        from pathlib import Path
        data = yaml.safe_load(Path("prompts/v1/contrarian.yaml").read_text(encoding="utf-8"))
        system = data["system"].lower()
        assert any(word in system for word in ["challenge", "challeng", "counter", "never agree", "wrong"])

    def test_contrarian_system_prompt_instructs_never_agree(self):
        import yaml
        from pathlib import Path
        data = yaml.safe_load(Path("prompts/v1/contrarian.yaml").read_text(encoding="utf-8"))
        system = data["system"].lower()
        assert "never" in system or "not" in system or "challeng" in system

    def test_contrarian_user_prompt_has_template_variables(self):
        import yaml
        from pathlib import Path
        data = yaml.safe_load(Path("prompts/v1/contrarian.yaml").read_text(encoding="utf-8"))
        template = data["user"]
        assert "{{ question }}" in template
        assert "{{ fact_set }}" in template or "{% for fact" in template


# =============================================================================
# Criterion 3 -- Intuition prompt instructs fast pattern-based reasoning
# =============================================================================

class TestIntuitionPrompt:

    def test_intuition_yaml_exists(self):
        from pathlib import Path
        yaml_path = Path("prompts/v1/intuition.yaml")
        assert yaml_path.exists(), "prompts/v1/intuition.yaml not found"

    def test_intuition_yaml_has_correct_model(self):
        import yaml
        from pathlib import Path
        data = yaml.safe_load(Path("prompts/v1/intuition.yaml").read_text(encoding="utf-8"))
        assert "qwen" in data["model"].lower() or "alibaba" in data["model"].lower()

    def test_intuition_yaml_agent_name_matches(self):
        import yaml
        from pathlib import Path
        data = yaml.safe_load(Path("prompts/v1/intuition.yaml").read_text(encoding="utf-8"))
        assert data["agent"] == "intuition"

    def test_intuition_system_prompt_instructs_fast_pattern(self):
        import yaml
        from pathlib import Path
        data = yaml.safe_load(Path("prompts/v1/intuition.yaml").read_text(encoding="utf-8"))
        system = data["system"].lower()
        assert any(word in system for word in ["pattern", "fast", "heuristic", "gut", "quick"])

    def test_intuition_system_prompt_instructs_historical_analogy(self):
        import yaml
        from pathlib import Path
        data = yaml.safe_load(Path("prompts/v1/intuition.yaml").read_text(encoding="utf-8"))
        system = data["system"].lower()
        assert any(word in system for word in ["histor", "analogy", "seen", "before", "similar"])

    def test_intuition_user_prompt_has_template_variables(self):
        import yaml
        from pathlib import Path
        data = yaml.safe_load(Path("prompts/v1/intuition.yaml").read_text(encoding="utf-8"))
        template = data["user"]
        assert "{{ question }}" in template
        assert "{{ fact_set }}" in template or "{% for fact" in template


# =============================================================================
# Criterion 4 -- All 5 agents use different models (cognitive diversity)
# =============================================================================

class TestCognitiveDiversity:

    def test_all_five_models_are_in_router(self):
        required = ["neutral_analyst", "data_first", "skeptic", "contrarian", "intuition"]
        for name in required:
            assert name in AGENT_MODEL_MAP, f"'{name}' missing from AGENT_MODEL_MAP"

    def test_contrarian_uses_zhipu_model(self):
        model = AGENT_MODEL_MAP["contrarian"]
        assert "glm" in model.lower() or "thudm" in model.lower(), \
            f"Contrarian should use Zhipu/GLM model, got: {model}"

    def test_intuition_uses_alibaba_model(self):
        model = AGENT_MODEL_MAP["intuition"]
        assert "qwen" in model.lower() or "alibaba" in model.lower(), \
            f"Intuition should use Alibaba/Qwen model, got: {model}"

    def test_neutral_analyst_uses_deepseek_model(self):
        model = AGENT_MODEL_MAP["neutral_analyst"]
        assert "deepseek" in model.lower(), \
            f"Neutral Analyst should use DeepSeek model, got: {model}"

    def test_all_five_models_are_distinct(self):
        models = [AGENT_MODEL_MAP[n] for n in
                  ["neutral_analyst", "data_first", "skeptic", "contrarian", "intuition"]]
        assert len(models) == len(set(models)), \
            f"Duplicate model assignments detected: {models}"

    def test_three_different_ai_architectures_present(self):
        """
        DeepSeek, Qwen (Alibaba), and GLM (Zhipu AI) = 3 distinct Chinese providers.
        This test verifies we have at least 3 unique provider families.
        """
        models = list(AGENT_MODEL_MAP.values())
        providers = set()
        for m in models:
            if "deepseek" in m.lower(): providers.add("deepseek")
            if "qwen" in m.lower(): providers.add("alibaba")
            if "glm" in m.lower() or "thudm" in m.lower(): providers.add("zhipu")
        assert len(providers) >= 3, f"Expected >=3 AI providers, got: {providers}"


# =============================================================================
# Criterion 5 -- Pipeline accepts all 5 agents without error
# =============================================================================

class TestFiveAgentPipeline:

    def test_pipeline_accepts_all_five_agents(self, all_five_agents):
        pipeline = Pipeline(agents=all_five_agents)
        assert pipeline is not None

    def test_pipeline_agent_count_is_five(self, all_five_agents):
        pipeline = Pipeline(agents=all_five_agents)
        assert len(pipeline.agents) == 5

    def test_pipeline_contains_contrarian(self, all_five_agents):
        pipeline = Pipeline(agents=all_five_agents)
        names = [a.name for a in pipeline.agents]
        assert "contrarian" in names

    def test_pipeline_contains_intuition(self, all_five_agents):
        pipeline = Pipeline(agents=all_five_agents)
        names = [a.name for a in pipeline.agents]
        assert "intuition" in names

    def test_pipeline_contains_all_original_agents(self, all_five_agents):
        pipeline = Pipeline(agents=all_five_agents)
        names = [a.name for a in pipeline.agents]
        for expected in ["neutral_analyst", "data_first", "skeptic"]:
            assert expected in names, f"'{expected}' missing from 5-agent pipeline"

    def test_dashboard_agent_config_has_all_five(self):
        """
        Verify the Streamlit dashboard AGENT_CONFIG was updated to include
        the 2 new agents.
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        # Read the file and check that contrarian and intuition are present
        src = Path("dashboard/streamlit_app.py").read_text(encoding="utf-8")
        assert "contrarian" in src, "Dashboard AGENT_CONFIG missing 'contrarian'"
        assert "intuition"  in src, "Dashboard AGENT_CONFIG missing 'intuition'"
        assert "ContrarianAgent" in src, "Dashboard not importing ContrarianAgent"
        assert "IntuitionAgent"  in src, "Dashboard not importing IntuitionAgent"
