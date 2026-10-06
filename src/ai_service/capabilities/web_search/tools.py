"""Live web search tools for hardware research, reviews, and community insights."""

from __future__ import annotations

from ai_service.application.ports.web_search import (
    WebSearchCategory,
    WebSearchClient,
    WebSearchQuery,
    WebSearchResult,
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
from ai_service.infrastructure.search.duckduckgo_adapter import (
    DuckDuckGoSearchAdapter,
)


class WebSearchTools:
    """Executable search tools callable by PydanticAI agents to query live Internet facts."""

    def __init__(self, search_client: WebSearchClient | None = None) -> None:
        self.search_client = search_client or DuckDuckGoSearchAdapter()

    async def search_hardware_compatibility(
        self, args: HardwareCompatibilitySearchArgs
    ) -> LiveSearchResponse:
        """Tra cứu thực tế từ Internet về tính tương thích vật lý / BIOS giữa 2 linh kiện.

        Ví dụ: Cấn chiều cao tản nhiệt và thanh RAM, vỏ case có vừa card 3 fan không, bo mạch chủ cần update BIOS trước khi nhận CPU không.
        """
        query_str = f"{args.part_a} {args.part_b} compatibility"
        if args.specific_concern:
            query_str += f" {args.specific_concern}"

        search_query = WebSearchQuery(
            query=query_str,
            category=WebSearchCategory.COMPATIBILITY,
            target_domains=["reddit.com", "tomshardware.com", "techpowerup.com"],
            limit=4,
        )
        res = await self.search_client.search(search_query)
        return self._format_response("Tương thích linh kiện thực tế", res)

    async def search_product_reviews(
        self, args: ProductReviewSearchArgs
    ) -> LiveSearchResponse:
        """Tra cứu review chuyên sâu, điểm benchmark và FPS chơi game thực tế từ các trang công nghệ uy tín.

        Ví dụ: So sánh FPS RTX 4060 vs RX 6700 XT ở độ phân giải 2K, nhiệt độ và độ ồn khi full load.
        """
        query_str = args.product_name
        if args.compare_with:
            query_str += f" vs {args.compare_with}"
        if args.focus_aspect == "GAMING_FPS":
            query_str += " gaming FPS benchmark review"
        elif args.focus_aspect == "THERMAL_NOISE":
            query_str += " thermal temperature noise dbA review"
        elif args.focus_aspect == "PRODUCTIVITY":
            query_str += " blender premiere rendering benchmark"
        else:
            query_str += " techpowerup tomshardware review"

        search_query = WebSearchQuery(
            query=query_str,
            category=WebSearchCategory.REVIEWS,
            target_domains=["techpowerup.com", "tomshardware.com", "gamersnexus.net"],
            limit=4,
        )
        res = await self.search_client.search(search_query)
        return self._format_response("Đánh giá & Benchmark thực tế", res)

    async def search_build_guides(
        self, args: BuildGuideSearchArgs
    ) -> LiveSearchResponse:
        """Tìm kiếm các cấu hình PC mẫu và hướng dẫn build tối ưu từ cộng đồng cho một tầm ngân sách hoặc tựa game cụ thể.

        Ví dụ: 'Cấu hình 20 triệu chơi mượt Black Myth Wukong', 'Dàn máy đồ họa 3D kiến trúc 35 triệu'.
        """
        query_str = f"cấu hình PC {args.target_use_case}"
        if args.budget_vnd:
            triệu = args.budget_vnd // 1_000_000
            query_str += f" {triệu} triệu"

        search_query = WebSearchQuery(
            query=query_str,
            category=WebSearchCategory.BUILD_GUIDES,
            limit=4,
        )
        res = await self.search_client.search(search_query)
        return self._format_response("Cấu hình gợi ý & Hướng dẫn build", res)

    async def search_game_requirements(
        self, args: GameRequirementSearchArgs
    ) -> LiveSearchResponse:
        """Tra cứu cấu hình yêu cầu phần cứng chính thức của một Game hoặc Phần mềm đồ họa (Tối thiểu, Đề nghị, 2K/4K).

        Ví dụ: Cấu hình yêu cầu của Black Myth Wukong, Cyberpunk 2077, AutoCAD 2026, Premiere Pro.
        """
        query_str = f"{args.game_title} PC system requirements minimum recommended {args.target_resolution}"
        search_query = WebSearchQuery(
            query=query_str,
            category=WebSearchCategory.GAME_REQUIREMENTS,
            target_domains=["steampowered.com", "systemrequirementslab.com"],
            limit=4,
        )
        res = await self.search_client.search(search_query)
        return self._format_response("Cấu hình yêu cầu của Game/Phần mềm", res)

    async def search_hardware_issues(
        self, args: HardwareIssueSearchArgs
    ) -> LiveSearchResponse:
        """Kiểm tra các lỗi kỹ thuật nổi cộm, sự cố mất ổn định, khuyến cáo nhiệt độ hoặc thông tin thu hồi sản phẩm.

        Ví dụ: Lỗi mất ổn định Vmin Shift của Intel Gen 13/14, sự cố cháy cáp 12VHPWR RTX 4090, tụt sức khỏe SSD.
        """
        query_str = f"{args.product_name} defect issue stability recall"
        if args.concern:
            query_str += f" {args.concern}"

        search_query = WebSearchQuery(
            query=query_str,
            category=WebSearchCategory.HARDWARE_ISSUES,
            target_domains=["tomshardware.com", "techpowerup.com", "reddit.com"],
            limit=4,
        )
        res = await self.search_client.search(search_query)
        return self._format_response("Cảnh báo sự cố & Lỗi phần cứng nổi cộm", res)

    async def search_psu_tier(self, args: PsuTierSearchArgs) -> LiveSearchResponse:
        """Tra cứu bảng xếp hạng an toàn của bộ nguồn (PSU Cultists Tier List: Tier A, Tier B, Tier C...).

        Giúp tư vấn khách hàng chọn nguồn an toàn, tránh mua phải nguồn kém chất lượng (Tier E/F) gây nguy cơ cháy nổ.
        """
        query_str = f"{args.psu_model_name} PSU Cultists Network tier list rating"
        search_query = WebSearchQuery(
            query=query_str,
            category=WebSearchCategory.PSU_TIER,
            target_domains=["cultists.network", "linustechtips.com", "reddit.com"],
            limit=4,
        )
        res = await self.search_client.search(search_query)
        return self._format_response("Bảng xếp hạng an toàn bộ nguồn (PSU Tier List)", res)

    async def search_driver_and_software(
        self, args: DriverSoftwareSearchArgs
    ) -> LiveSearchResponse:
        """Tra cứu link tải Driver chính thức, phiên bản cập nhật BIOS bo mạch chủ hoặc phần mềm điều khiển LED/quạt của hãng.

        Ví dụ: 'NVIDIA GeForce Game Ready Driver mới nhất', 'MSI B760M BIOS update fix lỗi', 'ASUS Armoury Crate download'.
        """
        soft_kw = {
            "VGA_DRIVER": "official graphics driver download latest",
            "BIOS_UPDATE": "motherboard BIOS update download official",
            "CHIPSET_DRIVER": "chipset driver support download",
            "RGB_CONTROL": "RGB software utility control download",
        }.get(args.software_type, "official driver download")

        query_str = f"{args.component_name} {soft_kw}"
        search_query = WebSearchQuery(
            query=query_str,
            category=WebSearchCategory.DRIVER_SOFTWARE,
            limit=4,
        )
        res = await self.search_client.search(search_query)
        return self._format_response("Driver & Phần mềm chính hãng", res)

    async def search_tech_specs_from_vendor(
        self, args: VendorSpecsSearchArgs
    ) -> LiveSearchResponse:
        """Tra cứu bảng thông số kỹ thuật chuẩn từ trang chủ nhà sản xuất (khi kho nội bộ thiếu thông tin).

        Ví dụ: 'ASUS TUF RTX 4070 Ti SUPER chiều dài mm', 'Thermalright Assassin X 120 chiều cao mm', 'khe tản nhiệt m.2'.
        """
        attr_part = f" {args.spec_attribute}" if args.spec_attribute else " specifications dimensions clearance official"
        query_str = f"{args.product_model}{attr_part}"
        search_query = WebSearchQuery(
            query=query_str,
            category=WebSearchCategory.VENDOR_SPECS,
            limit=4,
        )
        res = await self.search_client.search(search_query)
        return self._format_response("Thông số kỹ thuật chính thức từ nhà sản xuất", res)

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
        snippets_text = "\n".join(f"- {it.title}: {it.snippet}" for it in items_view)
        prompt_instruction = (
            f"Dưới đây là thông tin tra cứu mới nhất từ Internet về '{topic}' ({search_result.query}):\n"
            f"{snippets_text}\n"
            f"Hãy dùng thông tin thực tế trên để trả lời khách hàng một cách khách quan, chính xác và có dẫn chứng."
        )
        return LiveSearchResponse(
            search_topic=topic,
            category=search_result.category.value,
            results_found=len(items_view),
            results=items_view,
            analysis_prompt=prompt_instruction,
        )


__all__ = ["WebSearchTools"]
