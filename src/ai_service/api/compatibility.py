import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from ai_service.api.dependencies import get_build_compatibility
from ai_service.application.errors import BackendUnavailableError
from ai_service.capabilities.pc_builder.application.build_check import (
    BuildCheck,
    BuildCompatibilityService,
    CandidateCheckRequest,
    CandidateVerdicts,
    CompatibilityCheckRequest,
    ComponentNotFoundError,
)
from ai_service.schemas.response import ApiResponse, ErrorDetail, ResponseMessage

router = APIRouter()


def _unavailable() -> ApiResponse[Any]:
    return ApiResponse(
        data=None,
        message=ResponseMessage.AI_BACKEND_UNAVAILABLE,
        errors=[ErrorDetail(code=ResponseMessage.AI_BACKEND_UNAVAILABLE.value, message="Catalog service is unavailable")],
    )


def _not_found(slot: str) -> ApiResponse[Any]:
    return ApiResponse(
        data=None,
        message=ResponseMessage.AI_PRODUCT_NOT_FOUND,
        errors=[ErrorDetail(code=ResponseMessage.AI_PRODUCT_NOT_FOUND.value, message="Part was not found", field=f"slots.{slot}")],
    )


@router.post("/compatibility/check", response_model=ApiResponse[BuildCheck | None], tags=["compatibility"])
async def check_compatibility(
    request: CompatibilityCheckRequest,
    service: Annotated[BuildCompatibilityService, Depends(get_build_compatibility)],
) -> ApiResponse[BuildCheck | None]:
    """Checks the parts chosen for a build against each other. Needs no sign-in, like the stateless search."""
    try:
        return ApiResponse(data=await service.check(request))
    except BackendUnavailableError:
        return _unavailable()
    except ComponentNotFoundError as missing:
        return _not_found(missing.slot)


@router.post("/compatibility/candidates", response_model=ApiResponse[CandidateVerdicts | None], tags=["compatibility"])
async def check_candidates(
    request: CandidateCheckRequest,
    service: Annotated[BuildCompatibilityService, Depends(get_build_compatibility)],
) -> ApiResponse[CandidateVerdicts | None]:
    """Tells, for each variant on offer for one slot, whether it fits the parts already chosen for the others."""
    try:
        return ApiResponse(data=await service.candidates(request))
    except BackendUnavailableError:
        return _unavailable()
    except ComponentNotFoundError as missing:
        return _not_found(missing.slot)


@lru_cache(maxsize=1)
def _load_prebuilt_builds() -> list[dict[str, Any]]:
    candidates = (
        Path("/app/data/prebuilt_pc_builds.json"),
        Path.cwd() / "data" / "prebuilt_pc_builds.json",
        Path(__file__).resolve().parents[4] / "data" / "prebuilt_pc_builds.json",
        Path(__file__).resolve().parents[3] / "data" / "prebuilt_pc_builds.json",
    )
    data_file = next((p for p in candidates if p.exists()), None)
    if not data_file:
        return []
    try:
        with data_file.open(encoding="utf-8") as f:
            builds = json.load(f)
        return builds if isinstance(builds, list) else []
    except (OSError, json.JSONDecodeError):
        return []


@router.get("/compatibility/prebuilt-builds", response_model=ApiResponse[list[dict[str, Any]]], tags=["compatibility"])
def get_prebuilt_builds() -> ApiResponse[list[dict[str, Any]]]:
    """Returns prebuilt conflict-free PC configurations generated from catalog_db."""
    return ApiResponse(data=_load_prebuilt_builds())

