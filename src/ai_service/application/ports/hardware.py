"""Outbound port for PC hardware compatibility rules and power calculation."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

EXTRA_HARDWARE_SPEC_FIELDS = frozenset(
    {"radiator_mm", "max_radiator_mm", "m2_slots", "interface", "cooler_type", "kit_modules", "core_count"}
)


def _optional_int(val: Any) -> int | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


class ComponentCategory(StrEnum):
    CPU = "CPU"
    MAINBOARD = "MAINBOARD"
    RAM = "RAM"
    GPU = "GPU"
    PSU = "PSU"
    CASE = "CASE"
    COOLER = "COOLER"
    STORAGE = "STORAGE"


class ComponentSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: UUID | None = None
    name: str
    category: ComponentCategory
    price: int = 0
    socket: str | None = None
    ram_type: str | None = None  # e.g., "DDR4", "DDR5"
    ram_slots: int | None = None
    form_factor: str | None = None  # e.g., "ATX", "Micro-ATX", "Mini-ITX"
    supported_form_factors: list[str] = Field(default_factory=list)
    tdp_watts: int | None = None
    wattage: int | None = None  # For PSU rated wattage
    gpu_length_mm: int | float | None = None
    max_gpu_length_mm: int | float | None = None
    cooler_height_mm: int | float | None = None
    max_cooler_height_mm: int | float | None = None
    performance_score: float | None = None
    vram_gb: int | None = None
    capacity_gb: int | None = None
    psu_tier: str | None = None
    efficiency_rating: str | None = None
    brand: str | None = None
    has_integrated_graphics: bool = False
    integrated_graphics_score: float | None = None
    integrated_graphics_power_watts: int | None = None
    includes_stock_cooler: bool = False
    stock_cooler_height_mm: int | None = None
    stock_cooler_tdp_watts: int | None = None
    stock_cooler_score: float | None = None
    is_integrated: bool = False
    supported_sockets: list[str] = Field(default_factory=list)
    extra_specs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _lift_extra_hardware_specs(cls, data: Any) -> Any:
        if isinstance(data, dict):
            extra = dict(data.get("extra_specs") or {})
            for key in EXTRA_HARDWARE_SPEC_FIELDS:
                if key in data and data[key] is not None:
                    extra.setdefault(key, data[key])
            if extra:
                data = dict(data)
                data["extra_specs"] = extra
        return data

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> ComponentSpec:
        if update:
            lifted = EXTRA_HARDWARE_SPEC_FIELDS.intersection(update.keys())
            if lifted:
                update = dict(update)
                extra = dict(update.get("extra_specs") or self.extra_specs)
                for key in lifted:
                    value = update.pop(key)
                    if value is None:
                        extra.pop(key, None)
                    else:
                        extra[key] = value
                update["extra_specs"] = extra
        return super().model_copy(update=update, deep=deep)

    @property
    def kit_modules(self) -> int | None:
        return _optional_int(self.extra_specs.get("kit_modules"))

    @property
    def radiator_mm(self) -> int | None:
        return _optional_int(self.extra_specs.get("radiator_mm"))

    @property
    def max_radiator_mm(self) -> int | None:
        return _optional_int(self.extra_specs.get("max_radiator_mm"))

    @property
    def m2_slots(self) -> int | None:
        return _optional_int(self.extra_specs.get("m2_slots"))

    @property
    def interface(self) -> str | None:
        val = self.extra_specs.get("interface")
        return str(val) if val is not None else None


class CompatibilityIssue(BaseModel):
    level: Literal["ERROR", "WARNING", "INFO"]
    category: str
    components_involved: list[str]
    message: str
    suggestion: str | None = None


class CompatibilityReport(BaseModel):
    is_compatible: bool
    errors_count: int
    warnings_count: int
    issues: list[CompatibilityIssue] = Field(default_factory=list)


class WattageReport(BaseModel):
    estimated_tdp_watts: int
    recommended_psu_watts: int
    selected_psu_watts: int | None = None
    is_sufficient: bool | None = None
    headroom_watts: int | None = None
    efficiency_rating_recommended: str = "80 Plus Bronze / Gold"
    note: str


class BuildPurpose(StrEnum):
    GAMING_ESPORTS = "GAMING_ESPORTS"
    GAMING_AAA = "GAMING_AAA"
    GRAPHIC_DESIGN = "GRAPHIC_DESIGN"
    OFFICE = "OFFICE"
    AI_DEEP_LEARNING = "AI_DEEP_LEARNING"


class RecommendedPart(BaseModel):
    slot: ComponentCategory
    name: str
    estimated_price: int
    key_specs: str


class RecommendedBuild(BaseModel):
    purpose: BuildPurpose
    target_budget: int
    total_estimated_price: int
    parts: list[RecommendedPart]
    summary: str
    compatibility_guaranteed: bool = True


class ResolutionTier(StrEnum):
    RES_1080P = "1080P"
    RES_1440P = "1440P"
    RES_4K = "4K"


class BottleneckRating(StrEnum):
    BALANCED = "BALANCED"
    CPU_BOTTLENECK = "CPU_BOTTLENECK"
    GPU_BOTTLENECK = "GPU_BOTTLENECK"


class BottleneckReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cpu_name: str
    gpu_name: str
    resolution: ResolutionTier
    bottleneck_percentage: float
    status: BottleneckRating
    explanation: str
    recommendation: str


class UpgradePathReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    platform_socket: str
    platform_lifecycle: str
    cpu_upgrade_options: list[str]
    gpu_upgrade_headroom: str
    ram_upgradeability: str
    verdict: str


class PeripheralRecommendation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    gpu_model: str
    target_use_case: str
    recommended_display_resolution: str
    recommended_refresh_rate: str
    recommended_panel: str
    display_examples: list[str]
    recommended_peripherals: list[str]
    advice: str


class HardwareRuleEngine(Protocol):
    """Domain rule engine interface for PC component compatibility & sizing."""

    def check_compatibility(self, components: list[ComponentSpec]) -> CompatibilityReport:
        """Validate electrical and physical constraints between components."""

    def calculate_wattage(
        self,
        components: list[ComponentSpec],
        selected_psu_watts: int | None = None,
    ) -> WattageReport:
        """Calculate TDP sum and recommend minimum PSU wattage."""

    def recommend_build(
        self,
        budget: int,
        purpose: BuildPurpose = BuildPurpose.GAMING_AAA,
    ) -> RecommendedBuild:
        """Synthesize an optimal balanced PC configuration within budget."""

    def analyze_bottleneck(
        self,
        cpu_name: str,
        gpu_name: str,
        resolution: ResolutionTier = ResolutionTier.RES_1440P,
    ) -> BottleneckReport:
        """Analyze performance balance and bottleneck percentage between CPU and GPU."""

    def assess_upgrade_path(
        self,
        socket: str,
        psu_wattage: int,
        ram_type: str,
        current_gpu: str | None = None,
    ) -> UpgradePathReport:
        """Evaluate platform lifecycle and future component upgrade potential."""

    def recommend_peripherals(
        self,
        gpu_name: str,
        target_use_case: str = "Gaming",
    ) -> PeripheralRecommendation:
        """Recommend matching monitor specifications and peripherals."""


class ComponentSource(Protocol):
    """Where the specs of a chosen part come from: the catalog that owns the product."""

    async def component(self, variant_id: UUID, category: ComponentCategory) -> ComponentSpec | None:
        """The spec of one variant read as the given category; None when the catalog has no such variant."""
