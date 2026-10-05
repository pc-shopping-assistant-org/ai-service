"""Pydantic schemas for PC Builder capability tools."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ai_service.application.ports.hardware import (
    BuildPurpose,
    ComponentCategory,
    ComponentSpec,
)


class CheckCompatibilityArgs(BaseModel):
    """Parameters for checking compatibility among chosen components."""

    model_config = ConfigDict(extra="ignore")

    components: list[ComponentSpec] = Field(
        description="List of selected PC components with their category, socket, form factor, and dimensions.",
        min_length=1,
    )


class CalculateWattageArgs(BaseModel):
    """Parameters for calculating power consumption and PSU recommendation."""

    model_config = ConfigDict(extra="ignore")

    components: list[ComponentSpec] = Field(
        description="List of selected components to calculate power requirements.",
        min_length=1,
    )
    selected_psu_watts: int | None = Field(
        default=None,
        description="Wattage of the power supply currently chosen by the customer (e.g. 650).",
    )


class RecommendBuildArgs(BaseModel):
    """Parameters for synthesizing an optimal PC build based on budget and use case."""

    model_config = ConfigDict(extra="ignore")

    budget_vnd: int = Field(
        description="Target customer budget in VND (e.g. 15000000 for 15 million VND).",
        ge=5_000_000,
    )
    purpose: BuildPurpose = Field(
        default=BuildPurpose.GAMING_AAA,
        description="Primary purpose of the PC (GAMING_ESPORTS, GAMING_AAA, GRAPHIC_DESIGN, OFFICE, AI_DEEP_LEARNING).",
    )


class FindAlternativesArgs(BaseModel):
    """Parameters for finding alternative compatible components when a part is out of stock or too expensive."""

    model_config = ConfigDict(extra="ignore")

    slot: ComponentCategory = Field(description="Component slot to replace (e.g. MAINBOARD, GPU, RAM, COOLER).")
    required_socket: str | None = Field(default=None, description="CPU/Mainboard socket to match (e.g. LGA1700, AM5).")
    required_ram_type: str | None = Field(default=None, description="RAM standard to match (e.g. DDR4, DDR5).")
    max_price: int | None = Field(default=None, description="Upper price ceiling in VND.")
    limit: int = Field(default=5, ge=1, le=10)


class AlternativeComponentView(BaseModel):
    id: UUID | None = None
    name: str
    price: int
    specs_summary: str
    in_stock: bool = True


class FindAlternativesOutput(BaseModel):
    slot: ComponentCategory
    requested_socket: str | None = None
    requested_ram_type: str | None = None
    alternatives: list[AlternativeComponentView]
    message: str


__all__ = [
    "AlternativeComponentView",
    "CalculateWattageArgs",
    "CheckCompatibilityArgs",
    "FindAlternativesArgs",
    "FindAlternativesOutput",
    "RecommendBuildArgs",
]
