from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health, query, history, agents, research
from db.session import create_all_tables


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run DB table creation on startup."""
    await create_all_tables()
    yield


app = FastAPI(
    title="DMARS API",
    description="Delta-First Multi-AI Reasoning System REST API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(query.router)
app.include_router(history.router)
app.include_router(agents.router)
app.include_router(research.router)
