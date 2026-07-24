from typing import Any

from fastapi import APIRouter

from memory.history import HistoryStore

router = APIRouter(tags=["History"])

@router.get("/history", response_model=list[dict[str, Any]])
async def get_history(limit: int = 20):
    store = HistoryStore()
    return await store.get_history(limit=limit)

@router.get("/history/{query_id}", response_model=dict[str, Any])
async def get_query_detail(query_id: int):
    store = HistoryStore()
    detail = await store.get_query_detail(query_id)
    if not detail:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Query not found")
    return detail
