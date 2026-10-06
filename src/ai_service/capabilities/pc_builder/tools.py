"""Agent tools for the PC Builder capability."""

from __future__ import annotations

from ai_service.application.ports.hardware import (
    BottleneckReport,
    BuildPurpose,
    CompatibilityReport,
    HardwareRuleEngine,
    PeripheralRecommendation,
    RecommendedBuild,
    UpgradePathReport,
    WattageReport,
)
from ai_service.capabilities.pc_builder.schemas import (
    AlternativeComponentView,
    AnalyzeBottleneckArgs,
    AssessUpgradePathArgs,
    CalculateWattageArgs,
    CheckCompatibilityArgs,
    FindAlternativesArgs,
    FindAlternativesOutput,
    RecommendBuildArgs,
    RecommendPeripheralsArgs,
)
from ai_service.infrastructure.hardware.rule_engine import LocalHardwareRuleEngine


class PCBuilderTools:
    """Executable tools callable by PydanticAI agents or orchestration workflows."""

    def __init__(self, rule_engine: HardwareRuleEngine | None = None) -> None:
        self.rule_engine = rule_engine or LocalHardwareRuleEngine()

    def check_pc_compatibility(self, args: CheckCompatibilityArgs) -> CompatibilityReport:
        """Kiểm tra tính tương thích vật lý và điện năng giữa các linh kiện PC đã chọn.

        Kiểm tra:
        - Socket CPU vs Mainboard (LGA1700, AM5...).
        - Chuẩn RAM vs Mainboard (DDR4 vs DDR5).
        - Kích thước Mainboard vs Khả năng lắp vừa của Vỏ Case (ATX, mATX, ITX).
        - Chiều dài Card màn hình VGA vs Giới hạn chiều dài của Case.
        - Chiều cao tản nhiệt khí vs Không gian hông case.
        - Công suất nguồn có đủ tải an toàn không.
        """
        return self.rule_engine.check_compatibility(args.components)

    def calculate_psu_wattage(self, args: CalculateWattageArgs) -> WattageReport:
        """Tính toán tổng công suất tiêu thụ điện (TDP) ước tính và gợi ý công suất nguồn (PSU) khuyên dùng.

        Giúp khách hàng biết bộ PC cần nguồn bao nhiêu Watt (550W, 650W, 750W...) để chạy ổn định và an toàn.
        """
        return self.rule_engine.calculate_wattage(
            components=args.components,
            selected_psu_watts=args.selected_psu_watts,
        )

    def recommend_pc_build(self, args: RecommendBuildArgs) -> RecommendedBuild:
        """Đề xuất trọn bộ cấu hình PC cân bằng tối ưu theo mức ngân sách và mục đích sử dụng.

        Tránh nghẽn cổ chai (bottleneck) giữa CPU và GPU, tối ưu hiệu năng trên từng đồng chi phí.
        """
        purpose_val = BuildPurpose.GAMING_AAA
        if args.purpose:
            try:
                purpose_val = BuildPurpose(args.purpose)
            except ValueError:
                purpose_val = BuildPurpose.GAMING_AAA
        return self.rule_engine.recommend_build(
            budget=args.budget_vnd,
            purpose=purpose_val,
        )

    def find_compatible_alternatives(self, args: FindAlternativesArgs) -> FindAlternativesOutput:
        """Tìm linh kiện thay thế tương thích khi một linh kiện bị hết hàng hoặc vượt ngân sách.

        Lọc theo cùng chuẩn socket, chuẩn RAM hoặc phân khúc giá.
        """
        # Catalog of standard fallback components indexed by socket/specs
        alternatives: list[AlternativeComponentView] = []
        sock = (args.required_socket or "").upper().replace(" ", "").replace("-", "")
        ram = (args.required_ram_type or "").upper()

        if args.slot == "MAINBOARD":
            if "AM5" in sock:
                alternatives = [
                    AlternativeComponentView(name="MSI PRO B650M-A WIFI", price=3600000, specs_summary="Socket AM5, DDR5, mATX"),
                    AlternativeComponentView(name="Gigabyte B650M GAMING PLUS", price=3250000, specs_summary="Socket AM5, DDR5, mATX"),
                    AlternativeComponentView(name="ASRock B650M Pro RS", price=3400000, specs_summary="Socket AM5, 4 khe DDR5"),
                ]
            elif "LGA1700" in sock:
                if ram == "DDR4":
                    alternatives = [
                        AlternativeComponentView(name="MSI PRO B760M-P DDR4", price=2450000, specs_summary="LGA1700, 4 khe DDR4"),
                        AlternativeComponentView(name="ASUS PRIME B760M-K D4", price=2350000, specs_summary="LGA1700, DDR4 giá tốt"),
                        AlternativeComponentView(name="ASRock H610M-HDV/M.2", price=1600000, specs_summary="LGA1700, DDR4 tiết kiệm"),
                    ]
                else:
                    alternatives = [
                        AlternativeComponentView(name="MSI B760M GAMING PLUS WIFI", price=3500000, specs_summary="LGA1700, DDR5, Wi-Fi"),
                        AlternativeComponentView(name="ASUS TUF GAMING B760M-PLUS", price=4100000, specs_summary="LGA1700, DDR5 cao cấp"),
                    ]
            else:
                alternatives = [
                    AlternativeComponentView(name="MSI B760M-E DDR4", price=2300000, specs_summary="Bo mạch chủ chuẩn phổ thông"),
                    AlternativeComponentView(name="Gigabyte B650M D3HP", price=2900000, specs_summary="Bo mạch chủ AM5 phổ thông"),
                ]
        elif args.slot == "RAM":
            if ram == "DDR4":
                alternatives = [
                    AlternativeComponentView(name="Kingston Fury Beast 16GB (2x8GB) 3200MHz", price=950000, specs_summary="DDR4 3200 CL16"),
                    AlternativeComponentView(name="Corsair Vengeance LPX 16GB (2x8GB) 3200MHz", price=1050000, specs_summary="DDR4 3200 tản nhôm"),
                ]
            else:
                alternatives = [
                    AlternativeComponentView(name="Kingston Fury Beast 32GB (2x16GB) 5600MHz", price=2600000, specs_summary="DDR5 5600 CL36"),
                    AlternativeComponentView(name="Corsair Vengeance 32GB (2x16GB) 6000MHz", price=2900000, specs_summary="DDR5 6000 CL30"),
                ]
        elif args.slot == "GPU":
            alternatives = [
                AlternativeComponentView(name="GeForce RTX 4060 8GB GDDR6", price=7800000, specs_summary="8GB GDDR6, DLSS 3, 115W"),
                AlternativeComponentView(name="AMD Radeon RX 6600 8GB", price=5300000, specs_summary="8GB GDDR6, 1080p gaming"),
                AlternativeComponentView(name="GeForce RTX 4070 SUPER 12GB", price=16800000, specs_summary="12GB GDDR6X, 2K/4K max settings"),
            ]
        else:
            alternatives = [
                AlternativeComponentView(name="Linh kiện tương đương tiêu chuẩn", price=1500000, specs_summary="Chất lượng và thông số đảm bảo"),
            ]

        if args.max_price:
            filtered = [alt for alt in alternatives if alt.price <= args.max_price]
            if filtered:
                alternatives = filtered

        return FindAlternativesOutput(
            slot=args.slot,
            requested_socket=args.required_socket,
            requested_ram_type=args.required_ram_type,
            alternatives=alternatives[: args.limit],
            message=f"Đã tìm thấy {len(alternatives)} linh kiện {args.slot} tương thích phù hợp yêu cầu.",
        )

    def analyze_bottleneck_balance(self, args: AnalyzeBottleneckArgs) -> BottleneckReport:
        """Phân tích nghẽn cổ chai (bottleneck) và sự cân xứng hiệu năng giữa CPU và Card đồ họa (GPU).

        Đo lường tỉ lệ phần trăm nghẽn theo từng độ phân giải mục tiêu (1080p, 1440p, 4K) và đưa ra lời khuyên tối ưu.
        """
        return self.rule_engine.analyze_bottleneck(
            cpu_name=args.cpu_name,
            gpu_name=args.gpu_name,
            resolution=args.resolution,
        )

    def assess_upgrade_path(self, args: AssessUpgradePathArgs) -> UpgradePathReport:
        """Đánh giá tiềm năng nâng cấp trong tương lai của dàn máy (vòng đời socket bo mạch chủ, công suất nguồn dư, chuẩn RAM).

        Giúp khách hàng biết cấu hình này có thể nâng cấp CPU/GPU trong 2-4 năm tới mà không cần thay toàn bộ hệ thống hay không.
        """
        return self.rule_engine.assess_upgrade_path(
            socket=args.socket,
            psu_wattage=args.psu_wattage,
            ram_type=args.ram_type,
            current_gpu=args.current_gpu,
        )

    def recommend_monitor_and_peripherals(self, args: RecommendPeripheralsArgs) -> PeripheralRecommendation:
        """Gợi ý màn hình (độ phân giải, tần số quét Hz, tấm nền IPS/OLED) và phụ kiện (chuột, phím, tai nghe) tương xứng với sức mạnh của bộ PC.

        Tránh lãng phí sức mạnh card đồ họa cao cấp hoặc chọn màn hình quá sức chịu đựng của linh kiện.
        """
        return self.rule_engine.recommend_peripherals(
            gpu_name=args.gpu_name,
            target_use_case=args.target_use_case,
        )


__all__ = ["PCBuilderTools"]
