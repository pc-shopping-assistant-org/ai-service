"""Agent tools for the PC Builder capability."""

from __future__ import annotations

from typing import Any

from ai_service.application.errors import BackendUnavailableError
from ai_service.application.ports.hardware import (
    BottleneckReport,
    CompatibilityReport,
    ComponentCategory,
    ComponentSpec,
    HardwareRuleEngine,
    PeripheralRecommendation,
    UpgradePathReport,
    WattageReport,
)
from ai_service.capabilities.pc_builder.application import (
    BuildApplicationResult,
    PCBuildApplicationService,
    build_explanation_context,
)
from ai_service.capabilities.pc_builder.schemas import (
    AlternativeComponentView,
    AnalyzeBottleneckArgs,
    AssessUpgradePathArgs,
    BuildObjective,
    CalculateWattageArgs,
    CheckCompatibilityArgs,
    ConstraintSource,
    ConstraintValue,
    FindAlternativesArgs,
    FindAlternativesOutput,
    PCBuildConstraints,
    RecommendBuildArgs,
    RecommendPeripheralsArgs,
)

ALTERNATIVES_FOUND_MESSAGE = "CATALOG_ALTERNATIVES_FOUND"
ALTERNATIVES_EMPTY_MESSAGE = "CATALOG_ALTERNATIVES_EMPTY"


class PCBuilderTools:
    """Deterministic operations callable by application/graph workflows."""

    def __init__(
        self,
        rule_engine: HardwareRuleEngine,
        application_service: PCBuildApplicationService | None = None,
    ) -> None:
        self.rule_engine = rule_engine
        self.application_service = application_service or PCBuildApplicationService()

    def build_pc(
        self,
        constraints: PCBuildConstraints,
        catalog: list[ComponentSpec],
        requested_objective: BuildObjective | None = None,
    ) -> BuildApplicationResult:
        """Run deterministic optimization and recommendation policy."""
        return self.application_service.build_pc(
            constraints=constraints,
            catalog=catalog,
            requested_objective=requested_objective,
        )

    def build_pc_from_args(
        self,
        args: RecommendBuildArgs,
        catalog: list[ComponentSpec],
        requested_objective: BuildObjective | None = None,
    ) -> BuildApplicationResult:
        """Convert RecommendBuildArgs into PCBuildConstraints and run the optimization pipeline."""
        constraints = PCBuildConstraints(
            target_budget_vnd=ConstraintValue(
                value=args.budget_vnd,
                source=ConstraintSource.USER,
                confidence=1.0,
            ),
            use_case=ConstraintValue(
                value=args.use_case,
                source=ConstraintSource.USER,
                confidence=1.0,
            ),
        )
        return self.build_pc(
            constraints=constraints,
            catalog=catalog,
            requested_objective=requested_objective,
        )

    def explain_build(
        self,
        result: BuildApplicationResult,
    ) -> dict[str, Any]:
        """Produce factual explanation context for downstream presentation."""
        return build_explanation_context(result)

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

    def recommend_pc_build(
        self, args: RecommendBuildArgs, catalog: list[ComponentSpec] | None = None,
    ) -> BuildApplicationResult:
        """Recommend only from explicitly supplied canonical component data."""
        if catalog is None:
            raise BackendUnavailableError("PC recommendation requires a canonical component catalog")
        return self.build_pc_from_args(args, catalog)

    def find_compatible_alternatives(
        self, args: FindAlternativesArgs, catalog: list[ComponentSpec] | None = None,
    ) -> FindAlternativesOutput:
        """Filter canonical candidates by requested standards, not full-build compatibility.

        The caller supplies active canonical rows, as for recommend_pc_build.
        ComponentSpec carries no stock evidence; availability remains unknown.
        """
        if catalog is None:
            raise BackendUnavailableError("Alternatives require a canonical component catalog")

        def normalized(value: str | None) -> str:
            return (value or "").strip().upper().replace(" ", "").replace("-", "")

        candidates = []
        for part in catalog:
            if part.id is None or part.is_integrated or part.category != args.slot or part.price < 0:
                continue
            if args.max_price is not None and part.price > args.max_price:
                continue
            if args.required_socket is not None:
                sockets: list[str | None] = [part.socket]
                if part.category == ComponentCategory.COOLER and part.supported_sockets:
                    sockets = list(part.supported_sockets)
                socket = normalized(args.required_socket)
                if not socket or socket not in {normalized(value) for value in sockets}:
                    continue
            if args.required_ram_type is not None:
                ram = normalized(args.required_ram_type)
                if not ram or ram != normalized(part.ram_type):
                    continue
            candidates.append(part)

        candidates.sort(key=lambda part: (part.price, str(part.id)))
        alternatives = [
            AlternativeComponentView(
                id=part.id, name=part.name, price=part.price,
                specs_summary="; ".join(
                    value for value in (part.socket, part.ram_type, part.form_factor) if value
                ),
                in_stock=None,
            )
            for part in candidates[:args.limit]
        ]
        return FindAlternativesOutput(
            slot=args.slot,
            requested_socket=args.required_socket,
            requested_ram_type=args.required_ram_type,
            alternatives=alternatives,
            message=ALTERNATIVES_FOUND_MESSAGE if alternatives else ALTERNATIVES_EMPTY_MESSAGE,
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
