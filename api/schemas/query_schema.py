from typing import Optional

from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(..., description="The query to analyze")
    fact_set: list[str] = Field(default_factory=list, description="Verified facts for grounding")
    domain_profile: Optional[str] = Field("general", description="Domain context for the agents")
    selected_agents: Optional[list[str]] = Field(
        ["neutral_analyst", "data_first", "skeptic", "contrarian", "intuition"],
        description="List of agent keys to run"
    )
    meta_ai_enabled: bool = Field(True, description="Whether to run the final Meta-AI synthesis")
