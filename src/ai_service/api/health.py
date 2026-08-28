from fastapi import APIRouter

from ai_service.schemas.health import HealthData
from ai_service.schemas.response import ApiResponse

router = APIRouter()


@router.get("/health", response_model=ApiResponse[HealthData])
async def health() -> ApiResponse[HealthData]:
    return ApiResponse(data=HealthData(status="ok"), message="health.ok")
