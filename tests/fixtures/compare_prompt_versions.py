"""
tests/fixtures/compare_prompt_versions.py
============================================
Checkpoint 20 — Prompt Version Comparison Script

Manual script (not pytest) that runs the SAME question through an agent
twice — once on prompt version v1, once on v2 — using real LLM calls, and
prints both outputs side by side so you can visually confirm versioning
actually changes behavior (Checkpoint 20 criterion #4).

Run from the dmars/ directory:
    poetry run python tests/fixtures/compare_prompt_versions.py
    poetry run python tests/fixtures/compare_prompt_versions.py --agent skeptic
"""

__test__ = False  # not a pytest suite — see tests/fixtures/test_llm_call.py for why

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import settings
from core.delta_protocol import DeltaProtocol
from llm.router import LLMRouter

QUESTION = "Why did BTC spike 8% in the last hour?"
FACTS = [
    "BTC volume up 3x in 60 minutes",
    "Large derivatives positions liquidated",
    "No major news reported in that window",
]
DOMAIN = "intraday_trading"


def run_version(agent_name: str, version: str, router: LLMRouter) -> dict:
    protocol = DeltaProtocol()
    original_version = settings.active_prompt_version
    settings.active_prompt_version = version
    try:
        prompt = protocol.render(agent_name, QUESTION, FACTS, DOMAIN)
    finally:
        settings.active_prompt_version = original_version

    response = router.call(
        agent_name=agent_name,
        system_prompt=prompt.system,
        user_prompt=prompt.user,
    )
    return {
        "version": version,
        "model": prompt.model,
        "success": response.success,
        "content": response.content,
        "error": response.error,
        "latency_ms": response.latency_ms,
        "total_tokens": response.total_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="neutral_analyst", help="Agent name to compare (default: neutral_analyst)")
    args = parser.parse_args()

    print("#" * 70)
    print(f"# DMARS -- Prompt Version Comparison: {args.agent}")
    print("#" * 70)
    print(f"Question: {QUESTION}")
    print(f"Facts:    {FACTS}")
    print()

    router = LLMRouter()
    v1_result = run_version(args.agent, "v1", router)
    v2_result = run_version(args.agent, "v2", router)

    for result in (v1_result, v2_result):
        print("-" * 70)
        print(f"VERSION: {result['version']}  |  model={result['model']}  |  "
              f"success={result['success']}  |  latency={result['latency_ms']}ms")
        print("-" * 70)
        if result["success"]:
            try:
                parsed = json.loads(result["content"])
                print(json.dumps(parsed, indent=2))
            except json.JSONDecodeError:
                print(result["content"])
        else:
            print(f"ERROR: {result['error']}")
        print()

    if v1_result["success"] and v2_result["success"]:
        identical = v1_result["content"].strip() == v2_result["content"].strip()
        print("=" * 70)
        print(f"Outputs identical: {identical}")
        if identical:
            print("WARNING: v1 and v2 produced byte-identical output — versioning may not be working.")
        else:
            print("Confirmed: v1 and v2 produced different output for the same input.")
        print("=" * 70)


if __name__ == "__main__":
    main()
