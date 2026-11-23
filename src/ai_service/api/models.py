"""Public model catalog endpoint exposing enabled models and thinking modes."""

from fastapi import APIRouter, HTTPException, Request

from ai_service.config.models import PublicModelsCatalog, load_model_catalog
from ai_service.config.settings import get_settings
from ai_service.schemas.response import ApiResponse, ResponseMessage

router = APIRouter(tags=["assistant"])


@router.get("/models", response_model=ApiResponse[PublicModelsCatalog])
async def list_models(request: Request) -> ApiResponse[PublicModelsCatalog]:
    stateful = getattr(request.app.state, "stateful", None)
    catalog = getattr(stateful, "models", None) if stateful is not None else None
    if catalog is None:
        settings = get_settings()
        if not settings.models_config_path:
            raise HTTPException(
                status_code=503,
                detail="Model configuration is not initialized",
            )
        try:
            catalog = load_model_catalog(settings.models_config_path)
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None
    return ApiResponse(
        data=catalog.public_catalog(),
        message=ResponseMessage.AI_MODELS_LISTED,
    )
