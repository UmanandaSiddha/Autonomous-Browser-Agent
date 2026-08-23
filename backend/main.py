from fastapi import FastAPI

app = FastAPI(
    title="Browser Automation Agent",
    description="Autonomous browser automation agent with local LLMs",
    version="0.1.0",
)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "browser-automation-agent",
    }