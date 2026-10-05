"""Pydantic schemas for the Web Search capability."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HardwareCompatibilitySearchArgs(BaseModel):
    """Parameters for looking up real-world compatibility between two specific components."""

    model_config = ConfigDict(extra="ignore")

    part_a: str = Field(description="First component name (e.g. 'DeepCool AK620', 'i5-13600K').")
    part_b: str = Field(description="Second component name (e.g. 'Corsair Vengeance RGB DDR5', 'MSI B760M').")
    specific_concern: str | None = Field(
        default=None,
        description="Specific concern if any (e.g. 'RAM clearance', 'BIOS update required', 'case fit').",
    )


class ProductReviewSearchArgs(BaseModel):
    """Parameters for searching real-world benchmarks, temperatures, and reviews."""

    model_config = ConfigDict(extra="ignore")

    product_name: str = Field(description="Primary hardware product to research (e.g. 'RTX 4070 Super', 'Ryzen 7 7800X3D').")
    compare_with: str | None = Field(default=None, description="Optional rival product to compare (e.g. 'RX 7900 GRE').")
    focus_aspect: Literal["GAMING_FPS", "THERMAL_NOISE", "PRODUCTIVITY", "GENERAL"] = Field(
        default="GENERAL",
        description="Specific aspect to focus review search on.",
    )


class BuildGuideSearchArgs(BaseModel):
    """Parameters for searching community-tested PC build configurations."""

    model_config = ConfigDict(extra="ignore")

    budget_vnd: int | None = Field(default=None, description="Budget in VND (e.g. 25000000).")
    target_use_case: str = Field(
        default="Gaming AAA",
        description="Target use case or game (e.g. 'Black Myth Wukong 2K', 'Render 3D Blender', 'Esports 240Hz').",
    )


class GameRequirementSearchArgs(BaseModel):
    """Parameters for looking up official hardware requirements of a game or software."""

    model_config = ConfigDict(extra="ignore")

    game_title: str = Field(description="Name of the game or software (e.g. 'Black Myth Wukong', 'AutoCAD 2026', 'Cyberpunk 2077').")
    target_resolution: Literal["1080P", "1440P", "4K"] = Field(
        default="1080P",
        description="Desired display resolution.",
    )


class HardwareIssueSearchArgs(BaseModel):
    """Parameters for researching known defects, stability recalls, or thermal advisories."""

    model_config = ConfigDict(extra="ignore")

    product_name: str = Field(description="Hardware product to inspect (e.g. 'Intel Core i9-14900K', 'RTX 4090 12VHPWR').")
    concern: str | None = Field(
        default=None,
        description="Known suspected issue (e.g. 'instability crash Vmin shift', 'melting cable', 'overheating').",
    )


class PsuTierSearchArgs(BaseModel):
    """Parameters for checking the safety grade and reliability tier of a power supply model."""

    model_config = ConfigDict(extra="ignore")

    psu_model_name: str = Field(description="Full PSU model name and brand (e.g. 'MSI MAG A650BN', 'Corsair RM850x', 'Seasonic Focus GX').")


class LiveSearchResultView(BaseModel):
    title: str
    url: str
    snippet: str


class LiveSearchResponse(BaseModel):
    search_topic: str
    category: str
    results_found: int
    results: list[LiveSearchResultView]
    analysis_prompt: str


__all__ = [
    "BuildGuideSearchArgs",
    "GameRequirementSearchArgs",
    "HardwareCompatibilitySearchArgs",
    "HardwareIssueSearchArgs",
    "LiveSearchResponse",
    "LiveSearchResultView",
    "ProductReviewSearchArgs",
    "PsuTierSearchArgs",
]
