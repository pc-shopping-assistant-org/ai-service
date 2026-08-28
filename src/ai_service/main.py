import uvicorn
from fastapi import FastAPI

from ai_service.api.router import api_router

app = FastAPI(
    title="PC Shopping Assistant AI Service",
    description="AI service for PC component recommendations and compatibility checks.",
    version="0.1.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "health",
            "description": "Service health and readiness endpoints.",
        },
    ],
)

app.include_router(api_router, prefix="/api/v1")


def run() -> None:
    uvicorn.run("ai_service.main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
