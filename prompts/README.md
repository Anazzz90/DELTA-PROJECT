# Prompt Versioning Guide
## DMARS — Prompt Templates

> Version: 1.0 | Managed by: `core/delta_protocol.py` | Active version: controlled via `ACTIVE_PROMPT_VERSION` env var

---

## How It Works

All agent prompts live in versioned YAML files under `prompts/`.
The Python code **never** hardcodes prompt text — it only renders whatever YAML is active.

```
prompts/
├── v1/                          ← Active version (ACTIVE_PROMPT_VERSION=v1)
│   ├── neutral_analyst.yaml
│   ├── data_first.yaml
│   ├── skeptic.yaml
│   ├── contrarian.yaml          (added in Checkpoint 11)
│   ├── intuition.yaml           (added in Checkpoint 11)
│   └── meta_ai.yaml             (added in Checkpoint 12)
├── v2/                          ← A/B test version (added in Checkpoint 20)
│   └── ...
└── README.md                    ← This file
```

---

## Switching Versions

To switch which prompt version all agents use, change one environment variable:

```bash
# In your .env file:
ACTIVE_PROMPT_VERSION=v1    # default
ACTIVE_PROMPT_VERSION=v2    # switch to v2 (A/B test)
```

No code changes needed. The renderer picks up the new version on the next call.

---

## YAML File Structure

Every agent YAML file must follow this structure exactly:

```yaml
agent: <agent_name>           # Must match filename without .yaml
version: v1                   # Prompt version tag
model: <litellm_model_string> # e.g. gpt-4o-mini, ollama/mistral:7b
description: "<agent role>"   # Short description of the agent's cognitive role

system: |
  <Full system prompt with all 6 Delta-First Protocol steps embedded>
  Must include the JSON output schema.

user: |
  <User prompt template — uses Jinja2 variables>
  {{ question }}       ← injected by delta_protocol.py
  {{ fact_set }}       ← list of verified facts
  {{ domain_profile }} ← optional domain context string
```

---

## Jinja2 Variables

| Variable | Type | Required | Description |
|---|---|---|---|
| `{{ question }}` | `str` | ✅ Yes | The reasoning question |
| `{{ fact_set }}` | `list[str]` | ✅ Yes | Verified fact list (use `{% for fact in fact_set %}`) |
| `{{ domain_profile }}` | `str` | ❌ Optional | Domain context (e.g. `intraday_trading`) |

---

## Delta-First Protocol — 6 Steps (Must appear in every system prompt)

All 6 steps must be present in every agent's system prompt:

| Step | Label | Purpose |
|---|---|---|
| STEP 1 | LOCK VERIFIED FACTS | Use only provided facts — no outside knowledge |
| STEP 2 | BIAS CHECK | Identify and override any cognitive bias |
| STEP 3 | GENERATE AND COMPARE EXPLANATIONS | All plausible hypotheses, ranked by fact fit |
| STEP 4 | IDENTIFY THE MAIN DRIVER | Single most-likely root cause |
| STEP 5 | STRESS TEST | Would the event occur without the main driver? |
| STEP 6 | HUMILITY NOTE | Honest statement of uncertainty and missing data |

---

## Adding a New Prompt Version (A/B Testing)

1. Create `prompts/v2/` folder
2. Copy files from `v1/` into `v2/`
3. Edit the v2 copies (change tone, add steps, modify structure)
4. Set `ACTIVE_PROMPT_VERSION=v2` in `.env`
5. Run a query — system uses v2 automatically
6. Compare outputs. If v2 is worse, set back to `v1` — zero code change.

---

## Required JSON Output Schema

Every agent system prompt must instruct the LLM to return this exact JSON schema:

```json
{
  "extracted_facts": [],
  "possible_explanations": [],
  "ranked_hypotheses": [],
  "main_driver": "",
  "confidence_score": 0.0,
  "acknowledged_weaknesses": []
}
```
