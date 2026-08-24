from fastapi import FastAPI

from backend.api.routes.auth import (
    router as auth_router,
)

from backend.api.routes.automations import (
    router as automation_router,
)


app = FastAPI(
    title="Browser Automation Agent",
    description=(
        "AI-powered browser automation "
        "using Camoufox, LangGraph and Ollama."
    ),
    version="0.1.0",
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "browser-automation-agent",
    }


app.include_router(auth_router)
app.include_router(automation_router)