from fastapi import APIRouter, Depends

from api.middleware.auth import verify_api_key

router = APIRouter(tags=["Agents"], dependencies=[Depends(verify_api_key)])

# Basic static response for now until performance metrics are fully implemented
@router.get("/agents/performance")
async def get_agent_performance():
    return {
        "neutral_analyst": {"model": "deepseek-ai/DeepSeek-V3", "role": "Balanced reasoning"},
        "data_first": {"model": "llama-3.1-8b-instant", "role": "Strictly fact-bound"},
        "skeptic": {"model": "llama-3.3-70b-versatile", "role": "Adversarial reasoning"},
        "contrarian": {"model": "THUDM/glm-4-9b-chat", "role": "Independent logic"},
        "intuition": {"model": "Qwen/Qwen2.5-32B-Instruct", "role": "Pattern recognition"},
    }
