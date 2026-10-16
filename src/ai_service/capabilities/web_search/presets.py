"""Domain query presets for hardware research, specifications, and reviews."""

from __future__ import annotations

from ai_service.application.ports.web_search import WebSearchCategory
from ai_service.capabilities.web_search.research_contracts import (
    HardwareResearchFocus,
    SoftwareType,
)

HARDWARE_SEARCH_FOCUS: dict[HardwareResearchFocus, tuple[WebSearchCategory, str]] = {
    HardwareResearchFocus.GENERAL: (
        WebSearchCategory.REVIEWS,
        "independent review strengths weaknesses",
    ),
    HardwareResearchFocus.PERFORMANCE: (
        WebSearchCategory.REVIEWS,
        "gaming performance benchmark frame time power review",
    ),
    HardwareResearchFocus.PRODUCTIVITY: (
        WebSearchCategory.REVIEWS,
        "application productivity rendering benchmark review",
    ),
    HardwareResearchFocus.THERMAL_NOISE: (
        WebSearchCategory.REVIEWS,
        "thermal temperature noise dBA cooling test review",
    ),
    HardwareResearchFocus.MOTHERBOARD: (
        WebSearchCategory.REVIEWS,
        "VRM thermal BIOS CPU support connectivity review",
    ),
    HardwareResearchFocus.MEMORY: (
        WebSearchCategory.REVIEWS,
        "memory latency bandwidth XMP EXPO stability review",
    ),
    HardwareResearchFocus.STORAGE: (
        WebSearchCategory.REVIEWS,
        "SSD sustained performance endurance NAND controller temperature review",
    ),
    HardwareResearchFocus.PSU_QUALITY: (
        WebSearchCategory.REVIEWS,
        "PSU electrical test ripple protection transient response review",
    ),
    HardwareResearchFocus.PSU_TIER: (
        WebSearchCategory.PSU_TIER,
        "PSU Cultists Network tier list exact model series revision rating",
    ),
    HardwareResearchFocus.CASE_AIRFLOW: (
        WebSearchCategory.REVIEWS,
        "case airflow temperature noise clearance review",
    ),
    HardwareResearchFocus.KNOWN_ISSUES: (
        WebSearchCategory.HARDWARE_ISSUES,
        "known issues defect stability recall manufacturer advisory",
    ),
    HardwareResearchFocus.OFFICIAL_SPECS: (
        WebSearchCategory.VENDOR_SPECS,
        "official manufacturer specifications dimensions clearance support",
    ),
}
LEGACY_REVIEW_FOCUS = {
    "GAMING_FPS": HardwareResearchFocus.PERFORMANCE,
    "THERMAL_NOISE": HardwareResearchFocus.THERMAL_NOISE,
    "PRODUCTIVITY": HardwareResearchFocus.PRODUCTIVITY,
    "GENERAL": HardwareResearchFocus.GENERAL,
}
SOFTWARE_SEARCH_TERMS = {
    SoftwareType.VGA_DRIVER: "official graphics driver download",
    SoftwareType.BIOS_UPDATE: "motherboard BIOS update support",
    SoftwareType.CHIPSET_DRIVER: "official chipset driver support download",
    SoftwareType.RGB_CONTROL: "official RGB software utility control support",
}


class WebSearchPresets:
    """Helper factories converting domain-specific hardware needs into targeted search queries."""

    @staticmethod
    def compatibility(
        part_a: str, part_b: str, specific_concern: str | None = None
    ) -> str:
        query = f"{part_a} {part_b} compatibility official support BIOS clearance connectors"
        if specific_concern:
            query += f" {specific_concern}"
        return query.strip()

    @staticmethod
    def product_reviews(
        product_name: str,
        compare_with: str | None = None,
        focus_aspect: str = "GENERAL",
    ) -> str:
        subject = f"{product_name} vs {compare_with}" if compare_with else product_name
        return WebSearchPresets.research_hardware(
            subject, LEGACY_REVIEW_FOCUS[focus_aspect]
        )

    @staticmethod
    def research_hardware(
        product_name: str, focus: HardwareResearchFocus, question: str | None = None
    ) -> str:
        return " ".join(
            value
            for value in (product_name, HARDWARE_SEARCH_FOCUS[focus][1], question)
            if value
        )

    @staticmethod
    def build_guides(
        target_use_case: str = "Gaming AAA", budget_vnd: int | None = None
    ) -> str:
        query = f"cấu hình PC {target_use_case}"
        if budget_vnd:
            triệu = budget_vnd // 1_000_000
            query += f" {triệu} triệu"
        return query.strip()

    @staticmethod
    def game_requirements(game_title: str, target_resolution: str | None = None) -> str:
        return f"{game_title} official PC system requirements minimum recommended {target_resolution or ''}".strip()

    @staticmethod
    def hardware_issues(product_name: str, concern: str | None = None) -> str:
        return WebSearchPresets.research_hardware(
            product_name, HardwareResearchFocus.KNOWN_ISSUES, concern
        )

    @staticmethod
    def psu_tier(psu_model_name: str) -> str:
        return WebSearchPresets.research_hardware(
            psu_model_name, HardwareResearchFocus.PSU_TIER
        )

    @staticmethod
    def driver_and_software(
        component_name: str,
        software_type: SoftwareType | str = SoftwareType.VGA_DRIVER,
        version: str | None = None,
    ) -> str:
        return f"{component_name} {SOFTWARE_SEARCH_TERMS[SoftwareType(software_type)]} {version or 'latest'}".strip()

    @staticmethod
    def tech_specs(product_model: str, spec_attribute: str | None = None) -> str:
        return WebSearchPresets.research_hardware(
            product_model, HardwareResearchFocus.OFFICIAL_SPECS, spec_attribute
        )


__all__ = ["WebSearchPresets"]
