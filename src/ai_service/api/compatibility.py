import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, ValidationError

from ai_service.api.dependencies import get_build_compatibility, get_container
from ai_service.application.cpu_work import run_cpu
from ai_service.application.errors import BackendUnavailableError
from ai_service.application.ports.hardware import ComponentCategory, ComponentSpec
from ai_service.capabilities.assistant.limits import (
    HARDWARE_CATALOG_PAGE_SIZE,
    MAX_HARDWARE_CATALOG_PAGES,
)
from ai_service.capabilities.pc_builder.application.build_check import (
    SLOT_CATEGORIES,
    BuildCheck,
    BuildCompatibilityService,
    CandidateCheckRequest,
    CandidateVerdicts,
    CompatibilityCheckRequest,
    ComponentNotFoundError,
)
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.capabilities.pc_builder.errors import NoFeasibleBuildError
from ai_service.capabilities.pc_builder.schemas import (
    BuildObjective,
    ConstraintSource,
    ConstraintValue,
    PCBuildConstraints,
    UseCaseProfile,
)
from ai_service.infrastructure.commerce.hardware_catalog import (
    SPEC_FIELDS,
    STORAGE_MASTER_TYPES,
    HardwarePage,
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


class OptimizeBuildRequest(BaseModel):
    budget_vnd: int = Field(ge=5_000_000, le=500_000_000)
    use_case: UseCaseProfile = UseCaseProfile.GAMING_1440P
    secondary_use_case: UseCaseProfile | None = None
    secondary_use_case_weight: float = Field(default=0.30, ge=0.0, le=0.5)
    objective: BuildObjective = BuildObjective.BALANCED
    pinned_slots: dict[str, UUID] = Field(default_factory=dict, max_length=8)


class OptimizeBuildResponse(BaseModel):
    selection: dict[str, UUID]
    total_price: int
    score: float
    use_case: UseCaseProfile
    objective: BuildObjective
    compatibility_status: str


_CATEGORY_TO_SLOT: dict[ComponentCategory, str] = {
    ComponentCategory.CPU: "CPU",
    ComponentCategory.MAINBOARD: "MAINBOARD",
    ComponentCategory.RAM: "RAM",
    ComponentCategory.GPU: "GPU",
    ComponentCategory.STORAGE: "SSD",
    ComponentCategory.PSU: "PSU",
    ComponentCategory.CASE: "CASE",
    ComponentCategory.COOLER: "COOLER",
}


async def _load_live_candidates() -> list[ComponentSpec]:
    settings = get_container().settings
    base_url = settings.backend_api_url.rstrip("/")
    catalog: list[ComponentSpec] = []
    cursor: UUID | None = None
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            for _ in range(MAX_HARDWARE_CATALOG_PAGES):
                params: dict[str, str | int] = {"limit": HARDWARE_CATALOG_PAGE_SIZE}
                if cursor is not None:
                    params["cursor"] = str(cursor)
                from ai_service.infrastructure.observability import trace_headers

                headers = trace_headers()
                response = (
                    await client.get(
                        f"{base_url}/products/snapshot", params=params, headers=headers
                    )
                    if headers
                    else await client.get(f"{base_url}/products/snapshot", params=params)
                )
                response.raise_for_status()
                body = response.json()
                page = HardwarePage.model_validate(body.get("data") if isinstance(body, dict) else body)
                for detail in page.items:
                    if detail.status != "ACTIVE" or detail.hardware_type is None:
                        continue
                    try:
                        category = (
                            ComponentCategory.STORAGE
                            if detail.hardware_type in STORAGE_MASTER_TYPES
                            else ComponentCategory(detail.hardware_type)
                        )
                    except ValueError:
                        continue
                    fields: dict[str, Any] = {
                        field: detail.specifications[field]
                        for field in SPEC_FIELDS
                        if field in detail.specifications
                    }
                    if detail.hardware_type in STORAGE_MASTER_TYPES:
                        fields["storage_media"] = detail.hardware_type
                    if detail.brand_name is not None:
                        fields["brand"] = detail.brand_name
                    for variant in detail.variants:
                        if variant.status != "ACTIVE":
                            continue
                        raw: dict[str, Any] = {
                            **fields,
                            "id": variant.id,
                            "name": detail.name,
                            "category": category,
                            "price": variant.price,
                        }
                        try:
                            catalog.append(ComponentSpec.model_validate(raw))
                        except ValidationError:
                            continue
                if not page.has_next or page.next_cursor is None or page.next_cursor == cursor:
                    break
                cursor = page.next_cursor
    except (httpx.HTTPError, ValueError) as exc:
        raise BackendUnavailableError("Catalog service is unavailable") from exc
    return catalog


@router.post("/compatibility/optimize", response_model=ApiResponse[OptimizeBuildResponse | None], tags=["compatibility"])
async def optimize_build(
    request: OptimizeBuildRequest,
) -> ApiResponse[OptimizeBuildResponse | None]:
    """Runs the deterministic PC optimizer live against the active catalog, honoring pinned slots."""
    try:
        candidates = await _load_live_candidates()
    except BackendUnavailableError:
        return _unavailable()

    pinned_parts: dict[ComponentCategory, UUID] = {}
    for slot_name, variant_id in request.pinned_slots.items():
        category = SLOT_CATEGORIES.get(slot_name.upper())
        if category is not None:
            pinned_parts[category] = variant_id

    constraints = PCBuildConstraints(
        target_budget_vnd=ConstraintValue[int](
            value=request.budget_vnd, source=ConstraintSource.USER, locked=True
        ),
        use_case=ConstraintValue[UseCaseProfile](
            value=request.use_case, source=ConstraintSource.USER, locked=True
        ),
        secondary_use_case=request.secondary_use_case,
        secondary_use_case_weight=request.secondary_use_case_weight,
        pinned_parts=pinned_parts,
    )
    try:
        app_result = await run_cpu(
            lambda: PCBuildApplicationService().build_pc(
                constraints, candidates, request.objective
            )
        )
    except NoFeasibleBuildError as exc:
        return ApiResponse(
            data=None,
            message=ResponseMessage.VALIDATION_ERROR,
            errors=[ErrorDetail(code="NO_FEASIBLE_BUILD", message=str(exc))],
        )

    chosen = app_result.optimization.builds.get(request.objective)
    if chosen is None and app_result.recommendation.objective is not None:
        chosen = app_result.optimization.builds.get(app_result.recommendation.objective)
    if chosen is None:
        return ApiResponse(
            data=None,
            message=ResponseMessage.VALIDATION_ERROR,
            errors=[ErrorDetail(code="NO_FEASIBLE_BUILD", message="No feasible PC build found for the given constraints.")],
        )

    selection: dict[str, UUID] = {}
    for category, part in chosen.parts.items():
        slot_key = _CATEGORY_TO_SLOT.get(category)
        if slot_key and part.id is not None:
            selection[slot_key] = part.id

    return ApiResponse(
        data=OptimizeBuildResponse(
            selection=selection,
            total_price=chosen.total_price,
            score=round(float(chosen.objective_score or 0.0), 2),
            use_case=request.use_case,
            objective=chosen.objective or request.objective,
            compatibility_status=chosen.compatibility_status.value,
        )
    )

