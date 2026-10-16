"""Live web search tools for hardware research, reviews, and community insights."""

from __future__ import annotations

from ai_service.application.ports.web_search import (
    WebSearchCategory,
    WebSearchClient,
    WebSearchQuery,
    WebSearchResult,
)
from ai_service.capabilities.web_search.presets import (
    LEGACY_REVIEW_FOCUS,
    WebSearchPresets,
)
from ai_service.capabilities.web_search.research_contracts import (
    HardwareResearchFocus,
    ResearchSourceStatus,
)
from ai_service.capabilities.web_search.schemas import (
    BuildGuideSearchArgs,
    DriverSoftwareSearchArgs,
    GameRequirementSearchArgs,
    HardwareCompatibilitySearchArgs,
    HardwareIssueSearchArgs,
    LiveSearchResponse,
    LiveSearchResultView,
    ProductReviewSearchArgs,
    PsuTierSearchArgs,
    VendorSpecsSearchArgs,
)
from ai_service.capabilities.web_search.source_policy import (
    ResearchSourcePolicy,
    within_domains,
)

DISCOVERY_RESULT_LIMIT = 4


class WebSearchTools:
    """Search operations callable by application/graph workflows for live facts."""

    def __init__(
        self,
        search_client: WebSearchClient,
        source_policy: ResearchSourcePolicy | None = None,
    ) -> None:
        self.search_client = search_client
        self.source_policy = source_policy or ResearchSourcePolicy()

    async def _search(
        self,
        query: str,
        category: WebSearchCategory,
        topic: str,
        domains: list[str] | None = None,
    ) -> LiveSearchResponse:
        if domains == []:
            result = WebSearchResult(
                query=query,
                category=category,
                note=ResearchSourceStatus.SOURCE_UNRESOLVED,
            )
        else:
            result = await self.search_client.search(
                WebSearchQuery(
                    query=query,
                    category=category,
                    target_domains=domains or [],
                    limit=DISCOVERY_RESULT_LIMIT,
                )
            )
            if domains:
                items = [
                    item for item in result.items if within_domains(item.url, domains)
                ]
                result = result.model_copy(
                    update={"items": items, "total_results": len(items)}
                )
        return self._format_response(topic, result)

    async def search_hardware_compatibility(
        self, args: HardwareCompatibilitySearchArgs
    ) -> LiveSearchResponse:
        """Tra cứu thực tế từ Internet về tính tương thích vật lý / BIOS giữa 2 linh kiện.

        Ví dụ: Cấn chiều cao tản nhiệt và thanh RAM, vỏ case có vừa card 3 fan không, bo mạch chủ cần update BIOS trước khi nhận CPU không.
        """
        return await self._search(
            WebSearchPresets.compatibility(
                args.part_a, args.part_b, args.specific_concern
            ),
            WebSearchCategory.COMPATIBILITY,
            "Tương thích linh kiện thực tế",
            self.source_policy.review_domains(WebSearchCategory.COMPATIBILITY),
        )

    async def search_product_reviews(
        self, args: ProductReviewSearchArgs
    ) -> LiveSearchResponse:
        """Tra cứu review chuyên sâu, điểm benchmark và FPS chơi game thực tế từ các trang công nghệ uy tín.

        Ví dụ: So sánh FPS RTX 4060 vs RX 6700 XT ở độ phân giải 2K, nhiệt độ và độ ồn khi full load.
        """
        return await self._search(
            WebSearchPresets.product_reviews(
                args.product_name, args.compare_with, args.focus_aspect
            ),
            WebSearchCategory.REVIEWS,
            "Đánh giá & Benchmark thực tế",
            self.source_policy.review_domains(LEGACY_REVIEW_FOCUS[args.focus_aspect]),
        )

    async def search_build_guides(
        self, args: BuildGuideSearchArgs
    ) -> LiveSearchResponse:
        """Tìm kiếm các cấu hình PC mẫu và hướng dẫn build tối ưu từ cộng đồng cho một tầm ngân sách hoặc tựa game cụ thể.

        Ví dụ: 'Cấu hình 20 triệu chơi mượt Black Myth Wukong', 'Dàn máy đồ họa 3D kiến trúc 35 triệu'.
        """
        return await self._search(
            WebSearchPresets.build_guides(args.target_use_case, args.budget_vnd),
            WebSearchCategory.BUILD_GUIDES,
            "Cấu hình tham khảo & Hướng dẫn build",
        )

    async def search_game_requirements(
        self, args: GameRequirementSearchArgs
    ) -> LiveSearchResponse:
        """Tra cứu cấu hình yêu cầu phần cứng chính thức của một Game hoặc Phần mềm đồ họa (Tối thiểu, Đề nghị, 2K/4K).

        Ví dụ: Cấu hình yêu cầu của Black Myth Wukong, Cyberpunk 2077, AutoCAD 2026, Premiere Pro.
        """
        query = WebSearchPresets.game_requirements(
            args.game_title, args.target_resolution
        )
        topic = "Cấu hình yêu cầu của Game/Phần mềm"
        publisher = self.source_policy.publisher_domains(args.game_title)
        if publisher:
            result = await self._search(
                query, WebSearchCategory.GAME_REQUIREMENTS, topic, publisher
            )
            if result.results:
                return result
        return await self._search(
            query,
            WebSearchCategory.GAME_REQUIREMENTS,
            topic,
            self.source_policy.requirement_stores,
        )

    async def search_hardware_issues(
        self, args: HardwareIssueSearchArgs
    ) -> LiveSearchResponse:
        """Kiểm tra các lỗi kỹ thuật nổi cộm, sự cố mất ổn định, khuyến cáo nhiệt độ hoặc thông tin thu hồi sản phẩm.

        Ví dụ: Lỗi mất ổn định Vmin Shift của Intel Gen 13/14, sự cố cháy cáp 12VHPWR RTX 4090, tụt sức khỏe SSD.
        """
        return await self._search(
            WebSearchPresets.hardware_issues(args.product_name, args.concern),
            WebSearchCategory.HARDWARE_ISSUES,
            "Cảnh báo sự cố & Lỗi phần cứng",
            self.source_policy.official_domains(args.product_name)
            or self.source_policy.review_domains(HardwareResearchFocus.KNOWN_ISSUES),
        )

    async def search_psu_tier(self, args: PsuTierSearchArgs) -> LiveSearchResponse:
        """Tra cứu bảng xếp hạng an toàn của bộ nguồn (PSU Cultists Tier List: Tier A, Tier B, Tier C...).

        Giúp tư vấn khách hàng chọn nguồn an toàn, tránh mua phải nguồn kém chất lượng (Tier E/F) gây nguy cơ cháy nổ.
        """
        return await self._search(
            WebSearchPresets.psu_tier(args.psu_model_name),
            WebSearchCategory.PSU_TIER,
            "Bảng xếp hạng bộ nguồn (PSU Tier List)",
            self.source_policy.review_domains(HardwareResearchFocus.PSU_TIER),
        )

    async def search_driver_and_software(
        self, args: DriverSoftwareSearchArgs
    ) -> LiveSearchResponse:
        """Tra cứu link tải Driver chính thức, phiên bản cập nhật BIOS bo mạch chủ hoặc phần mềm điều khiển LED/quạt của hãng.

        Ví dụ: 'NVIDIA GeForce Game Ready Driver mới nhất', 'MSI B760M BIOS update fix lỗi', 'ASUS Armoury Crate download'.
        """
        return await self._search(
            WebSearchPresets.driver_and_software(
                args.component_name, args.software_type
            ),
            WebSearchCategory.DRIVER_SOFTWARE,
            "Driver & Phần mềm chính hãng",
            self.source_policy.official_domains(args.component_name),
        )

    async def search_tech_specs_from_vendor(
        self, args: VendorSpecsSearchArgs
    ) -> LiveSearchResponse:
        """Tra cứu bảng thông số kỹ thuật chuẩn từ trang chủ nhà sản xuất (khi kho nội bộ thiếu thông tin).

        Ví dụ: 'ASUS TUF RTX 4070 Ti SUPER chiều dài mm', 'Thermalright Assassin X 120 chiều cao mm', 'khe tản nhiệt m.2'.
        """
        return await self._search(
            WebSearchPresets.tech_specs(args.product_model, args.spec_attribute),
            WebSearchCategory.VENDOR_SPECS,
            "Thông số kỹ thuật từ nhà sản xuất",
            self.source_policy.official_domains(args.product_model),
        )

    def _format_response(
        self, topic: str, search_result: WebSearchResult
    ) -> LiveSearchResponse:
        items_view = [
            LiveSearchResultView(
                title=item.title,
                url=item.url,
                snippet=item.snippet,
            )
            for item in search_result.items
        ]
        snippets_text = "\n".join(
            f"- {it.title} ({it.url}): {it.snippet}" for it in items_view
        )
        prompt_instruction = (
            f"Kết quả tìm kiếm để chọn trang về '{topic}' ({search_result.query}):\n"
            f"{snippets_text}\n"
            "Snippet chỉ dùng discovery. Phải đọc trang và kiểm tra model/điều kiện trước khi trả lời hoặc dẫn nguồn."
        )
        if not items_view:
            prompt_instruction = "Không có bằng chứng từ web cho yêu cầu này. Nêu rõ giới hạn; không suy diễn hoặc bịa nguồn."
        return LiveSearchResponse(
            search_topic=topic,
            category=search_result.category.value,
            results_found=len(items_view),
            results=items_view,
            analysis_prompt=prompt_instruction,
            note=search_result.note,
        )


__all__ = ["WebSearchTools"]
