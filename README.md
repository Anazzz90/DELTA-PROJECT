# DMARS — Delta-First Multi-AI Reasoning System

> Version: 0.1.0 | Phase: 1 MVP | Status: In Development

A modular, multi-agent AI reasoning system for high-stakes decision-making. Combines the **Delta-First Protocol v4.4** (6-step structured reasoning) with a 5-agent consensus engine to eliminate bias, overconfidence, and shallow conclusions.

---

## Quick Start

### 1. Prerequisites
- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation) 1.8+
- [Ollama](https://ollama.com) (for local LLMs — Data-First + Intuition agents)

### 2. Install Dependencies
```bash
poetry install
```

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

### 4. Run the MVP Dashboard (Phase 1)
```bash
poetry run streamlit run dashboard/streamlit_app.py
```

---

## Development Phases

| Phase | Status | Target |
|---|---|---|
| Phase 1 — MVP | 🔄 In Progress | 3 agents + Streamlit dashboard |
| Phase 2 — Full Backend | ⏳ Pending | FastAPI + Redis + LangFuse |
| Phase 3 — Production | ⏳ Pending | Docker Compose + Grafana |

---

## Project Structure

```
dmars/
├── config/          # Settings, domain profiles, scoring weights
├── prompts/         # Versioned Jinja2 prompt templates (YAML)
├── agents/          # Individual agent implementations
├── core/            # Delta-First protocol, pipeline, scoring, conflict detection
├── llm/             # LiteLLM router, cache, resilience layer
├── memory/          # Vector store (ChromaDB/Qdrant) + history archive
├── api/             # FastAPI backend (Phase 2)
├── queue/           # Redis + RQ task queue (Phase 2)
├── observability/   # LangFuse tracing (Phase 2)
├── dashboard/       # Streamlit MVP + Grafana production dashboard
├── db/              # SQLAlchemy models + Alembic migrations
└── tests/           # Unit + integration tests
```

---

## Agents

| Agent | Model | Cost | Role |
|---|---|---|---|
| Neutral Analyst | GPT-4o-mini | ~$0.01/q | Balanced, objective reasoning |
| Contrarian | Groq LLaMA3-70B | Free | Challenges dominant narratives |
| Data-First | Ollama Mistral-7B | $0 | Strictly fact-based reasoning |
| Skeptic | Groq Mixtral-8x7B | Free | Actively breaks conclusions |
| Intuition | Ollama LLaMA3-8B | $0 | Fast heuristic judgment |
| Meta-AI | GPT-4o (optional) | ~$0.02/q | Final synthesis |

**Target cost: < $0.05 per query**

---

## Running Tests

```bash
# All tests
poetry run pytest

# Unit tests only
poetry run pytest tests/unit/

# Integration tests only
poetry run pytest tests/integration/
```

---

## Environment Variables

See `.env.example` for all required variables with descriptions.
