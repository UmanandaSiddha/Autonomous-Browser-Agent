from fastapi import FastAPI

from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.db.database import init_db

from backend.api.routes.auth import (
    router as auth_router,
)

from backend.api.routes.automations import (
    router as automation_router,
)

from backend.api.routes.platform_auth import (
    router as platform_auth_router
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[DB] Initializing database...")
    init_db()
    print("[DB] Database ready")

    yield

app = FastAPI(
    title="Browser Automation Agent",
    description=(
        "AI-powered browser automation "
        "using Camoufox, LangGraph and Ollama."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "browser-automation-agent",
    }


app.include_router(auth_router)

app.include_router(automation_router)

app.include_router(platform_auth_router)