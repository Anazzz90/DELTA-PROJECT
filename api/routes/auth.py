"""
api/routes/auth.py
=====================
Checkpoint 19 — Admin route to mint API keys.

Gated by X-Admin-Secret (api.middleware.auth.verify_admin_secret), not a
regular API key — see that function's docstring for why.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.middleware.auth import generate_key, hash_key, verify_admin_secret
from db.models import ApiKeyRow
from db.session import get_session

router = APIRouter(prefix="/auth", tags=["Auth"])


class CreateKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Human-readable label for this key")


class CreateKeyResponse(BaseModel):
    api_key: str = Field(..., description="The plaintext key — shown once, never retrievable again")
    key_prefix: str
    name: str


@router.post("/create-key", response_model=CreateKeyResponse, dependencies=[Depends(verify_admin_secret)])
async def create_key(request: CreateKeyRequest) -> CreateKeyResponse:
    plaintext = generate_key()

    async with get_session() as session:
        row = ApiKeyRow(
            key_hash=hash_key(plaintext),
            key_prefix=plaintext[:14],
            name=request.name,
            is_active=True,
        )
        session.add(row)
        await session.commit()

    return CreateKeyResponse(api_key=plaintext, key_prefix=plaintext[:14], name=request.name)
