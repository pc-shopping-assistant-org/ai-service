"""Domain query presets for hardware research, specifications, and reviews."""

from __future__ import annotations


class WebSearchPresets:
    """Helper factories converting domain-specific hardware needs into targeted search queries."""

    @staticmethod
    def compatibility(part_a: str, part_b: str, specific_concern: str | None = None) -> str:
        query = f"{part_a} {part_b} compatibility"
        if specific_concern:
            query += f" {specific_concern}"
        return query.strip()

    @staticmethod
    def product_reviews(
        product_name: str,
        compare_with: str | None = None,
        focus_aspect: str = "GENERAL",
    ) -> str:
        query = product_name
        if compare_with:
            query += f" vs {compare_with}"
        if focus_aspect == "GAMING_FPS":
            query += " gaming FPS benchmark review"
        elif focus_aspect == "THERMAL_NOISE":
            query += " thermal temperature noise dBA review"
        elif focus_aspect == "PRODUCTIVITY":
            query += " blender premiere rendering benchmark"
        else:
            query += " benchmark review"
        return query.strip()

    @staticmethod
    def build_guides(target_use_case: str = "Gaming AAA", budget_vnd: int | None = None) -> str:
        query = f"cấu hình PC {target_use_case}"
        if budget_vnd:
            triệu = budget_vnd // 1_000_000
            query += f" {triệu} triệu"
        return query.strip()

    @staticmethod
    def game_requirements(game_title: str, target_resolution: str = "1080P") -> str:
        return f"{game_title} PC system requirements minimum recommended {target_resolution}".strip()

    @staticmethod
    def hardware_issues(product_name: str, concern: str | None = None) -> str:
        query = f"{product_name} defect issue stability recall"
        if concern:
            query += f" {concern}"
        return query.strip()

    @staticmethod
    def psu_tier(psu_model_name: str) -> str:
        return f"{psu_model_name} PSU Cultists Network tier list rating".strip()

    @staticmethod
    def driver_and_software(component_name: str, software_type: str = "VGA_DRIVER") -> str:
        kw = {
            "VGA_DRIVER": "official graphics driver download latest",
            "BIOS_UPDATE": "motherboard BIOS update download official",
            "CHIPSET_DRIVER": "chipset driver support download",
            "RGB_CONTROL": "RGB software utility control download",
        }.get(software_type, "official driver download")
        return f"{component_name} {kw}".strip()

    @staticmethod
    def tech_specs(product_model: str, spec_attribute: str | None = None) -> str:
        attr = f" {spec_attribute}" if spec_attribute else " specifications dimensions clearance official"
        return f"{product_model}{attr}".strip()


__all__ = ["WebSearchPresets"]
