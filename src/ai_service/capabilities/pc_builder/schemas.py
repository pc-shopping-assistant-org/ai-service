"""Pydantic schemas for PC Builder capability tools and core domain models."""

from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai_service.application.ports.hardware import (
    ComponentCategory,
    ComponentSpec,
    ResolutionTier,
)

# Boundary configuration for external agent tools (flexible)
TOOL_CONFIG = ConfigDict(extra="ignore")

# Domain configuration for core deterministic models (strictly forbidden extra fields)
CORE_CONFIG = ConfigDict(extra="forbid")


class CheckCompatibilityArgs(BaseModel):
    """Parameters for checking compatibility among chosen components."""

    model_config = TOOL_CONFIG

    components: list[ComponentSpec] = Field(
        description="List of selected PC components with their category, socket, form factor, and dimensions.",
        min_length=1,
    )


class CalculateWattageArgs(BaseModel):
    """Parameters for calculating power consumption and PSU recommendation."""

    model_config = TOOL_CONFIG

    components: list[ComponentSpec] = Field(
        description="List of selected components to calculate power requirements.",
        min_length=1,
    )
    selected_psu_watts: int | None = Field(
        default=None,
        description="Wattage of the power supply currently chosen by the customer (e.g. 650).",
    )


class UseCaseProfile(StrEnum):
    """Calibrated use-case profiles for hardware weighting and envelope calculation."""

    GAMING_1080P = "GAMING_1080P"
    GAMING_1440P = "GAMING_1440P"
    GAMING_4K = "GAMING_4K"
    CONTENT_CREATION_3D = "CONTENT_CREATION_3D"
    AI_DATA_SCIENCE = "AI_DATA_SCIENCE"
    OFFICE_BUDGET = "OFFICE_BUDGET"


class CompatibilityStatus(StrEnum):
    """Feasibility outcome for physical and electrical compatibility checks."""

    COMPATIBLE = "COMPATIBLE"
    INCOMPATIBLE = "INCOMPATIBLE"
    UNKNOWN = "UNKNOWN"


class BuildObjective(StrEnum):
    """Specific optimization objectives yielding distinct build variations."""

    PERFORMANCE = "PERFORMANCE"
    BALANCED = "BALANCED"
    UPGRADE_FRIENDLY = "UPGRADE_FRIENDLY"


class RecommendBuildArgs(BaseModel):
    """Parameters for synthesizing an optimal PC build based on budget and use case."""

    model_config = TOOL_CONFIG

    budget_vnd: int = Field(
        description="Target customer budget in VND (e.g. 15000000 for 15 million VND).",
        ge=5_000_000,
    )
    use_case: UseCaseProfile = Field(
        default=UseCaseProfile.GAMING_1440P,
        description="Primary use-case profile (GAMING_1080P, GAMING_1440P, GAMING_4K, CONTENT_CREATION_3D, AI_DATA_SCIENCE, OFFICE_BUDGET).",
    )
    resolution: ResolutionTier | None = Field(
        default=None,
        description="Target gaming or work resolution (1080P, 1440P, 4K).",
    )
    purpose: str | None = Field(
        default=None,
        description="[Deprecated: use use_case] Legacy build purpose alias.",
    )

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_purpose(cls, data: Any) -> Any:
        if isinstance(data, dict) and "purpose" in data and "use_case" not in data:
            p = str(data.get("purpose"))
            if "." in p:
                p = p.split(".")[-1]
            mapping = {
                "GAMING_ESPORTS": UseCaseProfile.GAMING_1080P,
                "GAMING_AAA": UseCaseProfile.GAMING_1440P,
                "GRAPHIC_DESIGN": UseCaseProfile.CONTENT_CREATION_3D,
                "AI_DEEP_LEARNING": UseCaseProfile.AI_DATA_SCIENCE,
                "OFFICE": UseCaseProfile.OFFICE_BUDGET,
            }
            if p in mapping:
                data["use_case"] = mapping[p]
        return data


class FindAlternativesArgs(BaseModel):
    """Parameters for finding alternative compatible components when a part is out of stock or too expensive."""

    model_config = TOOL_CONFIG

    slot: ComponentCategory = Field(description="Component slot to replace (e.g. MAINBOARD, GPU, RAM, COOLER).")
    required_socket: str | None = Field(default=None, description="CPU/Mainboard socket to match (e.g. LGA1700, AM5).")
    required_ram_type: str | None = Field(default=None, description="RAM standard to match (e.g. DDR4, DDR5).")
    max_price: int | None = Field(default=None, description="Upper price ceiling in VND.")
    limit: int = Field(default=5, ge=1, le=10)


class AlternativeComponentView(BaseModel):
    model_config = TOOL_CONFIG

    id: UUID | None = None
    name: str
    price: int
    specs_summary: str
    in_stock: bool = True


class FindAlternativesOutput(BaseModel):
    model_config = TOOL_CONFIG

    slot: ComponentCategory
    requested_socket: str | None = None
    requested_ram_type: str | None = None
    alternatives: list[AlternativeComponentView]
    message: str


class AnalyzeBottleneckArgs(BaseModel):
    """Parameters for analyzing bottleneck and performance balance between CPU and GPU."""

    model_config = TOOL_CONFIG

    cpu_name: str = Field(description="CPU model name (e.g. 'Core i5-12400F', 'Ryzen 7 7800X3D', 'i3-12100F').")
    gpu_name: str = Field(description="GPU model name (e.g. 'GeForce RTX 4070 Super', 'RTX 4060', 'RX 7800 XT').")
    resolution: ResolutionTier = Field(
        default=ResolutionTier.RES_1440P,
        description="Target gaming or work resolution (1080P, 1440P, 4K).",
    )


class AssessUpgradePathArgs(BaseModel):
    """Parameters for assessing component upgrade paths and motherboard longevity."""

    model_config = TOOL_CONFIG

    socket: str = Field(default="AM5", description="Current motherboard socket (e.g. AM4, AM5, LGA1700, LGA1200).")
    psu_wattage: int = Field(default=650, description="Current PSU wattage.")
    ram_type: str = Field(default="DDR5", description="Current RAM standard (DDR4, DDR5).")
    current_gpu: str | None = Field(default=None, description="Current GPU model if any.")


class RecommendPeripheralsArgs(BaseModel):
    """Parameters for recommending matching peripherals for a completed PC build."""

    model_config = TOOL_CONFIG

    gpu_name: str = Field(description="GPU model name (e.g. 'GeForce RTX 4080 SUPER', 'RTX 4070 SUPER', 'RX 7800 XT').")
    target_use_case: str = Field(
        default="Gaming",
        description="Primary target use case (e.g. 'Gaming AAA Đồ Họa Đẹp', 'Esports', 'Video Editing').",
    )
    target_resolution: ResolutionTier = Field(
        default=ResolutionTier.RES_1440P,
        description="Gaming or display resolution tier.",
    )
    budget_vnd: int | None = Field(
        default=None,
        description="Remaining or dedicated budget for peripherals in VND.",
    )


class ConstraintSource(StrEnum):
    USER = "USER"          # User nói trực tiếp ("Tôi cần NVIDIA", "Ngân sách 25tr")
    INFERRED = "INFERRED"  # LLM/rule suy ra từ ngữ cảnh ("Thích chơi AAA 1440p")
    DEFAULT = "DEFAULT"    # Fallback ban đầu
    SYSTEM = "SYSTEM"      # Policy/business rule cưỡng chế ("Bảo đảm an toàn điện PSU")


class ConstraintValue[T](BaseModel):
    """Wrapper holding a constraint value along with provenance, confidence, and lock state."""

    model_config = CORE_CONFIG

    value: T
    source: ConstraintSource = ConstraintSource.DEFAULT
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    locked: bool = False

    @model_validator(mode="after")
    def _verify_lock_provenance(self) -> ConstraintValue[T]:
        if self.locked and self.source not in {ConstraintSource.USER, ConstraintSource.SYSTEM}:
            raise ValueError(
                f"Constraint with source '{self.source}' cannot be locked. Only USER or SYSTEM constraints can be locked."
            )
        return self


class OwnedComponent(BaseModel):
    """Component owned by user, specifying category and optional specific spec/model."""

    model_config = CORE_CONFIG

    category: ComponentCategory
    component: ComponentSpec | None = None
    exclude_from_budget: bool = True


class PowerEstimate(BaseModel):
    """Realistic sustained peak power calculation and commercial PSU guidance."""

    model_config = CORE_CONFIG

    cpu_estimated_peak_watts: float
    gpu_estimated_peak_watts: float
    platform_estimated_watts: float
    estimated_sustained_peak_watts: float
    transient_allowance_watts: float
    minimum_capacity_watts: int
    recommended_capacity_watts: int
    sizing_rationale: str

    @property
    def minimum_psu_watts(self) -> int:
        return self.minimum_capacity_watts

    @property
    def recommended_psu_watts(self) -> int:
        return self.recommended_capacity_watts


class ScoreDimension(BaseModel):
    """Individual dimension evaluated during multi-attribute utility scoring."""

    model_config = CORE_CONFIG

    name: str
    raw_value: float | int | str
    normalized_score: float
    weight: float
    contribution: float


class ScoreBreakdown(BaseModel):
    """Complete transparent mathematical breakdown of build objective score."""

    model_config = CORE_CONFIG

    total_score: float
    dimensions: list[ScoreDimension]

    def __float__(self) -> float:
        return float(self.total_score)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (int, float)):
            return bool(self.total_score == other)
        if isinstance(other, ScoreBreakdown):
            return bool(self.total_score == other.total_score and self.dimensions == other.dimensions)
        return super().__eq__(other)

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, (int, float)):
            return bool(self.total_score < other)
        if isinstance(other, ScoreBreakdown):
            return bool(self.total_score < other.total_score)
        return NotImplemented

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, (int, float)):
            return bool(self.total_score > other)
        if isinstance(other, ScoreBreakdown):
            return bool(self.total_score > other.total_score)
        return NotImplemented

    def __le__(self, other: Any) -> bool:
        if isinstance(other, (int, float)):
            return bool(self.total_score <= other)
        if isinstance(other, ScoreBreakdown):
            return bool(self.total_score <= other.total_score)
        return NotImplemented

    def __ge__(self, other: Any) -> bool:
        if isinstance(other, (int, float)):
            return bool(self.total_score >= other)
        if isinstance(other, ScoreBreakdown):
            return bool(self.total_score >= other.total_score)
        return NotImplemented

    def __abs__(self) -> float:
        return abs(self.total_score)

    def __sub__(self, other: Any) -> float:
        if isinstance(other, (int, float)):
            return self.total_score - float(other)
        if isinstance(other, ScoreBreakdown):
            return self.total_score - other.total_score
        return NotImplemented

    def __rsub__(self, other: Any) -> float:
        if isinstance(other, (int, float)):
            return float(other) - self.total_score
        return NotImplemented


class MetricEvidence(BaseModel):
    """Factual metric evidence computed directly by the deterministic engine."""

    model_config = CORE_CONFIG

    metric: str
    raw_value: float | int | str
    normalized_score: float | None = None
    weight: float | None = None
    contribution: float | None = None
    unit: str | None = None
    source: str | None = None
    label: str = ""


class RankedBuild(BaseModel):
    """A fully verified, compatible build optimized for a specific objective.

    Pure mathematical/hardware evidence; free of ungrounded natural language claims.
    """

    model_config = CORE_CONFIG

    objective: BuildObjective
    objective_score: float
    total_price: int
    parts: dict[ComponentCategory, ComponentSpec]
    power_estimate: PowerEstimate
    evidence: list[MetricEvidence] = Field(default_factory=list)
    compatibility_status: CompatibilityStatus
    owned_categories: list[ComponentCategory] = Field(default_factory=list)
    spending_prices: dict[ComponentCategory, int] = Field(default_factory=dict)


class PCBuildConstraints(BaseModel):
    """Structured constraints governing the combinatorial search space."""

    model_config = CORE_CONFIG

    target_budget_vnd: ConstraintValue[int] = Field(
        default_factory=lambda: ConstraintValue[int](value=20_000_000, source=ConstraintSource.DEFAULT, confidence=1.0)
    )
    use_case: ConstraintValue[UseCaseProfile] = Field(
        default_factory=lambda: ConstraintValue[UseCaseProfile](value=UseCaseProfile.GAMING_1080P, source=ConstraintSource.DEFAULT, confidence=1.0)
    )
    target_resolution: ConstraintValue[ResolutionTier | None] = Field(
        default_factory=lambda: ConstraintValue[ResolutionTier | None](value=None, source=ConstraintSource.DEFAULT)
    )
    target_fps: ConstraintValue[int | None] = Field(
        default_factory=lambda: ConstraintValue[int | None](value=None, source=ConstraintSource.DEFAULT)
    )
    preferred_gpu_brand: ConstraintValue[str | None] = Field(
        default_factory=lambda: ConstraintValue[str | None](value=None, source=ConstraintSource.DEFAULT)
    )
    preferred_cpu_brand: ConstraintValue[str | None] = Field(
        default_factory=lambda: ConstraintValue[str | None](value=None, source=ConstraintSource.DEFAULT)
    )
    form_factor_preference: ConstraintValue[str | None] = Field(
        default_factory=lambda: ConstraintValue[str | None](value=None, source=ConstraintSource.DEFAULT)
    )
    owned_parts: list[OwnedComponent] = Field(default_factory=list)

    @field_validator("target_fps")
    @classmethod
    def _validate_target_fps(cls, v: ConstraintValue[int | None]) -> ConstraintValue[int | None]:
        if v.value is not None and not (30 <= v.value <= 1000):
            raise ValueError(f"target_fps must be between 30 and 1000 FPS, got {v.value}")
        return v

    @property
    def owned_categories(self) -> list[ComponentCategory]:
        return [p.category for p in self.owned_parts]


class OptimizationResult(BaseModel):
    """Output from the deterministic multi-objective PC optimization engine."""

    model_config = CORE_CONFIG

    target_budget_vnd: int
    use_case: UseCaseProfile
    builds: dict[BuildObjective, RankedBuild]
    candidates_evaluated: int
    pruned_count: int
    rejected_candidates: list[str] = Field(default_factory=list)


class PublicStatePatch(BaseModel):
    """Sanitized state patch sent to client UI without exposing internal session internals."""

    model_config = TOOL_CONFIG

    target_budget_vnd: int | None = None
    use_case: str | None = None
    owned_categories: list[str] = Field(default_factory=list)
    locked_preferences: dict[str, str] = Field(default_factory=dict)


__all__ = [
    "CORE_CONFIG",
    "TOOL_CONFIG",
    "AlternativeComponentView",
    "AnalyzeBottleneckArgs",
    "AssessUpgradePathArgs",
    "BuildObjective",
    "CalculateWattageArgs",
    "CheckCompatibilityArgs",
    "CompatibilityStatus",
    "ConstraintSource",
    "ConstraintValue",
    "FindAlternativesArgs",
    "FindAlternativesOutput",
    "MetricEvidence",
    "OptimizationResult",
    "OwnedComponent",
    "PCBuildConstraints",
    "PowerEstimate",
    "PublicStatePatch",
    "RankedBuild",
    "RecommendBuildArgs",
    "RecommendPeripheralsArgs",
    "ScoreBreakdown",
    "ScoreDimension",
    "UseCaseProfile",
]
