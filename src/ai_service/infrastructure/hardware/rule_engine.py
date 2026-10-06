"""Concrete implementation of HardwareRuleEngine."""

from __future__ import annotations

import re

from ai_service.application.ports.hardware import (
    BottleneckRating,
    BottleneckReport,
    BuildPurpose,
    CompatibilityIssue,
    CompatibilityReport,
    ComponentCategory,
    ComponentSpec,
    HardwareRuleEngine,
    PeripheralRecommendation,
    RecommendedBuild,
    RecommendedPart,
    ResolutionTier,
    UpgradePathReport,
    WattageReport,
)

STANDARD_PSU_SIZES = [450, 550, 650, 750, 850, 1000, 1200, 1600]

FORM_FACTOR_HIERARCHY = {
    "mini-itx": 1,
    "itx": 1,
    "micro-atx": 2,
    "matx": 2,
    "atx": 3,
    "e-atx": 4,
    "eatx": 4,
}


def _norm_str(val: str | None) -> str:
    if not val:
        return ""
    return re.sub(r"[\s\-_]+", "", val).upper()


def _norm_form_factor(ff: str | None) -> str:
    if not ff:
        return ""
    return ff.strip().lower()


class LocalHardwareRuleEngine(HardwareRuleEngine):
    """Deterministic hardware compatibility checker and sizing engine."""

    def check_compatibility(self, components: list[ComponentSpec]) -> CompatibilityReport:
        issues: list[CompatibilityIssue] = []

        by_cat: dict[ComponentCategory, list[ComponentSpec]] = {}
        for c in components:
            by_cat.setdefault(c.category, []).append(c)

        cpus = by_cat.get(ComponentCategory.CPU, [])
        mainboards = by_cat.get(ComponentCategory.MAINBOARD, [])
        rams = by_cat.get(ComponentCategory.RAM, [])
        gpus = by_cat.get(ComponentCategory.GPU, [])
        cases = by_cat.get(ComponentCategory.CASE, [])
        coolers = by_cat.get(ComponentCategory.COOLER, [])
        psus = by_cat.get(ComponentCategory.PSU, [])

        # 1. CPU vs Mainboard Socket Check
        if cpus and mainboards:
            cpu = cpus[0]
            mb = mainboards[0]
            cpu_socket = _norm_str(cpu.socket or cpu.extra_specs.get("socket"))
            mb_socket = _norm_str(mb.socket or mb.extra_specs.get("socket"))
            if cpu_socket and mb_socket and cpu_socket != mb_socket:
                issues.append(
                    CompatibilityIssue(
                        level="ERROR",
                        category="SOCKET_MISMATCH",
                        components_involved=[cpu.name, mb.name],
                        message=f"CPU socket ({cpu_socket}) không tương thích với bo mạch chủ socket ({mb_socket}).",
                        suggestion=f"Hãy chọn bo mạch chủ hỗ trợ socket {cpu_socket} hoặc đổi CPU sang socket {mb_socket}.",
                    )
                )

        # 2. Mainboard vs RAM Type Check (DDR4 vs DDR5)
        if mainboards and rams:
            mb = mainboards[0]
            mb_ram = _norm_str(mb.ram_type or mb.extra_specs.get("ram_type"))
            for ram in rams:
                ram_type = _norm_str(ram.ram_type or ram.extra_specs.get("ram_type"))
                if mb_ram and ram_type and mb_ram != ram_type:
                    issues.append(
                        CompatibilityIssue(
                            level="ERROR",
                            category="RAM_TYPE_MISMATCH",
                            components_involved=[mb.name, ram.name],
                            message=f"Bo mạch chủ yêu cầu chuẩn {mb_ram} nhưng thanh RAM là chuẩn {ram_type}.",
                            suggestion=f"Đổi RAM sang chuẩn {mb_ram} để cắm vừa khe cắm trên bo mạch chủ.",
                        )
                    )

        # 3. Mainboard Form Factor vs Case Support
        if mainboards and cases:
            mb = mainboards[0]
            case = cases[0]
            mb_ff = _norm_form_factor(mb.form_factor or mb.extra_specs.get("form_factor"))
            case_supported = [
                _norm_form_factor(s)
                for s in (case.supported_form_factors or case.extra_specs.get("supported_form_factors", []))
            ]
            case_ff = _norm_form_factor(case.form_factor or case.extra_specs.get("form_factor"))
            if case_ff and case_ff not in case_supported:
                case_supported.append(case_ff)

            if mb_ff:
                mb_rank = FORM_FACTOR_HIERARCHY.get(mb_ff)
                case_ranks = [FORM_FACTOR_HIERARCHY[s] for s in case_supported if s in FORM_FACTOR_HIERARCHY]
                max_case_rank = max(case_ranks) if case_ranks else None

                if max_case_rank and mb_rank and mb_rank > max_case_rank:
                    issues.append(
                        CompatibilityIssue(
                            level="ERROR",
                            category="FORM_FACTOR_MISMATCH",
                            components_involved=[mb.name, case.name],
                            message=f"Kích thước bo mạch chủ ({mb_ff.upper()}) quá lớn so với vỏ case ({case_ff.upper() if case_ff else 'vỏ case'}).",
                            suggestion=f"Chọn vỏ case hỗ trợ kích thước {mb_ff.upper()} trở lên.",
                        )
                    )

        # 4. GPU Length vs Case Max GPU Length
        if gpus and cases:
            gpu = gpus[0]
            case = cases[0]
            gpu_len = gpu.gpu_length_mm or gpu.extra_specs.get("length_mm")
            case_max_gpu = case.max_gpu_length_mm or case.extra_specs.get("max_gpu_length_mm")
            if gpu_len and case_max_gpu and int(gpu_len) > int(case_max_gpu):
                issues.append(
                    CompatibilityIssue(
                        level="ERROR",
                        category="GPU_CLEARANCE",
                        components_involved=[gpu.name, case.name],
                        message=f"Chiều dài card đồ họa ({gpu_len}mm) vượt quá chiều dài tối đa của vỏ case ({case_max_gpu}mm).",
                        suggestion="Chọn vỏ case rộng hơn hoặc chọn model VGA 2 quạt/kích thước ngắn hơn.",
                    )
                )

        # 5. CPU Cooler Height vs Case Max Cooler Height
        if coolers and cases:
            cooler = coolers[0]
            case = cases[0]
            c_height = cooler.cooler_height_mm or cooler.extra_specs.get("height_mm")
            case_max_c = case.max_cooler_height_mm or case.extra_specs.get("max_cooler_height_mm")
            if c_height and case_max_c and int(c_height) > int(case_max_c):
                issues.append(
                    CompatibilityIssue(
                        level="ERROR",
                        category="COOLER_CLEARANCE",
                        components_involved=[cooler.name, case.name],
                        message=f"Chiều cao tản nhiệt ({c_height}mm) cấn nắp hông của vỏ case ({case_max_c}mm).",
                        suggestion="Chọn tản nhiệt khí thấp hơn hoặc dùng tản nhiệt nước AIO.",
                    )
                )

        # 6. PSU Wattage Check vs Recommended
        if psus:
            psu = psus[0]
            psu_watts = psu.wattage or psu.extra_specs.get("wattage")
            if psu_watts:
                watt_report = self.calculate_wattage(components, selected_psu_watts=int(psu_watts))
                if watt_report.is_sufficient is False:
                    issues.append(
                        CompatibilityIssue(
                            level="WARNING",
                            category="INSUFFICIENT_PSU",
                            components_involved=[psu.name],
                            message=f"Công suất nguồn ({psu_watts}W) có thể không đủ an toàn cho cấu hình (khuyên dùng tối thiểu {watt_report.recommended_psu_watts}W).",
                            suggestion=f"Nâng cấp nguồn lên mức tối thiểu {watt_report.recommended_psu_watts}W để tránh sập nguồn khi tải nặng.",
                        )
                    )

        errors_count = sum(1 for i in issues if i.level == "ERROR")
        warnings_count = sum(1 for i in issues if i.level == "WARNING")
        return CompatibilityReport(
            is_compatible=(errors_count == 0),
            errors_count=errors_count,
            warnings_count=warnings_count,
            issues=issues,
        )

    def calculate_wattage(
        self,
        components: list[ComponentSpec],
        selected_psu_watts: int | None = None,
    ) -> WattageReport:
        cpu_tdp = 65
        gpu_tdp = 0
        has_gpu = False

        for c in components:
            if c.category == ComponentCategory.CPU:
                cpu_tdp = c.tdp_watts or c.extra_specs.get("tdp_watts", 95)
            elif c.category == ComponentCategory.GPU:
                has_gpu = True
                gpu_tdp = c.tdp_watts or c.extra_specs.get("tdp_watts", 180)

        # Base system load: Mainboard (30W) + RAM (10W) + NVMe SSD (10W) + Fans/RGB (20W) = 70W
        base_watts = 70
        estimated_tdp = cpu_tdp + (gpu_tdp if has_gpu else 15) + base_watts

        # Recommended PSU: 1.35x headroom factor for transient spikes
        raw_rec = int(estimated_tdp * 1.35)
        recommended_psu = next(
            (size for size in STANDARD_PSU_SIZES if size >= raw_rec),
            STANDARD_PSU_SIZES[-1],
        )

        is_sufficient: bool | None = None
        headroom: int | None = None
        if selected_psu_watts is not None:
            is_sufficient = selected_psu_watts >= recommended_psu
            headroom = selected_psu_watts - estimated_tdp

        note = (
            f"Tổng công suất tiêu thụ tối đa ước tính ~{estimated_tdp}W (CPU: {cpu_tdp}W, GPU: {gpu_tdp}W). "
            f"Mức nguồn khuyến nghị là {recommended_psu}W chuẩn 80 Plus để đảm bảo độ bền và nâng cấp."
        )

        return WattageReport(
            estimated_tdp_watts=estimated_tdp,
            recommended_psu_watts=recommended_psu,
            selected_psu_watts=selected_psu_watts,
            is_sufficient=is_sufficient,
            headroom_watts=headroom,
            note=note,
        )

    def recommend_build(
        self,
        budget: int,
        purpose: BuildPurpose = BuildPurpose.GAMING_AAA,
    ) -> RecommendedBuild:
        """Rule-based preset recommendation based on budget tier and use case."""
        if budget <= 16_000_000:
            parts = [
                RecommendedPart(slot=ComponentCategory.CPU, name="Intel Core i3-12100F", estimated_price=1900000, key_specs="4C/8T, LGA1700"),
                RecommendedPart(slot=ComponentCategory.MAINBOARD, name="ASRock H610M-HDV/M.2", estimated_price=1600000, key_specs="LGA1700, DDR4"),
                RecommendedPart(slot=ComponentCategory.RAM, name="Kingston Fury Beast 16GB (2x8GB) DDR4 3200MHz", estimated_price=950000, key_specs="16GB DDR4"),
                RecommendedPart(slot=ComponentCategory.STORAGE, name="Kingston NV2 500GB PCIe 4.0 NVMe", estimated_price=950000, key_specs="500GB NVMe"),
                RecommendedPart(slot=ComponentCategory.GPU, name="AMD Radeon RX 6500 XT 4GB", estimated_price=3800000, key_specs="4GB GDDR6"),
                RecommendedPart(slot=ComponentCategory.PSU, name="MSI MAG A550BN 550W 80 Plus Bronze", estimated_price=1100000, key_specs="550W Bronze"),
                RecommendedPart(slot=ComponentCategory.CASE, name="Xigmatek NYX Air Arctic (mATX)", estimated_price=650000, key_specs="Micro-ATX"),
            ]
            summary = "Cấu hình PC Gaming Esport tiết kiệm: chiến mượt Liên Minh, Valorant, CS2, FO4 ở độ phân giải 1080p."

        elif budget < 22_000_000:
            parts = [
                RecommendedPart(slot=ComponentCategory.CPU, name="Intel Core i5-12400F", estimated_price=2800000, key_specs="6C/12T, LGA1700"),
                RecommendedPart(slot=ComponentCategory.MAINBOARD, name="MSI PRO B760M-E DDR4", estimated_price=2400000, key_specs="LGA1700, DDR4"),
                RecommendedPart(slot=ComponentCategory.RAM, name="Corsair Vengeance LPX 16GB (2x8GB) DDR4 3200", estimated_price=1050000, key_specs="16GB DDR4"),
                RecommendedPart(slot=ComponentCategory.STORAGE, name="Samsung 980 1TB NVMe PCIe 3.0", estimated_price=1750000, key_specs="1TB NVMe"),
                RecommendedPart(slot=ComponentCategory.GPU, name="GeForce RTX 4060 8GB GDDR6", estimated_price=7800000, key_specs="8GB GDDR6, DLSS 3"),
                RecommendedPart(slot=ComponentCategory.PSU, name="Corsair CV650 650W 80 Plus Bronze", estimated_price=1450000, key_specs="650W Bronze"),
                RecommendedPart(slot=ComponentCategory.COOLER, name="Thermalright Assassin X 120 Refined SE", estimated_price=450000, key_specs="4 heatpipes 120mm"),
                RecommendedPart(slot=ComponentCategory.CASE, name="Montech Air 100 ARGB (mATX)", estimated_price=1100000, key_specs="Kèm 4 fan ARGB"),
            ]
            summary = "Cấu hình Quốc Dân tầm trung: Chiến tốt game AAA với DLSS 3, dựng video Full HD / 2K mượt mà."

        elif budget < 38_000_000:
            parts = [
                RecommendedPart(slot=ComponentCategory.CPU, name="AMD Ryzen 5 7600X", estimated_price=5600000, key_specs="6C/12T, AM5, 5.3GHz"),
                RecommendedPart(slot=ComponentCategory.MAINBOARD, name="MSI B650M GAMING PLUS WIFI", estimated_price=4200000, key_specs="Socket AM5, DDR5, Wi-Fi 6E"),
                RecommendedPart(slot=ComponentCategory.RAM, name="Corsair Vengeance RGB 32GB (2x16GB) DDR5 6000MHz", estimated_price=2900000, key_specs="32GB DDR5"),
                RecommendedPart(slot=ComponentCategory.STORAGE, name="Kingston KC3000 1TB PCIe 4.0 NVMe", estimated_price=2300000, key_specs="Tốc độ 7000MB/s"),
                RecommendedPart(slot=ComponentCategory.GPU, name="GeForce RTX 4070 SUPER 12GB GDDR6X", estimated_price=16800000, key_specs="12GB, 2K/4K Gaming"),
                RecommendedPart(slot=ComponentCategory.PSU, name="Super Flower Leadex III Gold 750W", estimated_price=2500000, key_specs="750W 80 Plus Gold, Full Modular"),
                RecommendedPart(slot=ComponentCategory.COOLER, name="Deepcool AK620 Digital", estimated_price=1650000, key_specs="Tháp đôi hiển thị nhiệt độ"),
                RecommendedPart(slot=ComponentCategory.CASE, name="NZXT H5 Flow", estimated_price=2100000, key_specs="ATX, airflow tối ưu"),
            ]
            summary = "Dàn máy hiệu năng cao chuẩn AM5: Cân mọi game AAA 2K max setting, Render 3D, Premiere, After Effects chuyên nghiệp."

        else:
            parts = [
                RecommendedPart(slot=ComponentCategory.CPU, name="AMD Ryzen 7 7800X3D", estimated_price=10500000, key_specs="8C/16T, 3D V-Cache đỉnh gaming"),
                RecommendedPart(slot=ComponentCategory.MAINBOARD, name="ASUS ROG STRIX B650-A GAMING WIFI", estimated_price=6400000, key_specs="AM5, PCIe 5.0, VRM khủng"),
                RecommendedPart(slot=ComponentCategory.RAM, name="G.Skill Trident Z5 Neo RGB 64GB (2x32GB) DDR5 6000", estimated_price=5800000, key_specs="64GB DDR5 Expo"),
                RecommendedPart(slot=ComponentCategory.STORAGE, name="Samsung 990 PRO 2TB PCIe 4.0 NVMe", estimated_price=4600000, key_specs="2TB, 7450MB/s"),
                RecommendedPart(slot=ComponentCategory.GPU, name="GeForce RTX 4080 SUPER 16GB GDDR6X", estimated_price=28500000, key_specs="16GB, Đỉnh cao 4K / AI Training"),
                RecommendedPart(slot=ComponentCategory.PSU, name="Corsair RM850e 850W ATX 3.0 PCIe 5.0", estimated_price=3400000, key_specs="850W Gold, cáp 12VHPWR"),
                RecommendedPart(slot=ComponentCategory.COOLER, name="Thermalright Frozen Warframe 360 ARGB", estimated_price=2600000, key_specs="AIO 360mm có màn hình LCD"),
                RecommendedPart(slot=ComponentCategory.CASE, name="Lian Li O11 Dynamic EVO", estimated_price=3900000, key_specs="Bể cá cao cấp, thoáng khí"),
            ]
            summary = "Cấu hình Flagship cao cấp nhất: Vua chơi game 4K, xử lý đồ họa kiến trúc nặng và huấn luyện mô hình AI Local."

        total_price = sum(p.estimated_price for p in parts)
        return RecommendedBuild(
            purpose=purpose,
            target_budget=budget,
            total_estimated_price=total_price,
            parts=parts,
            summary=summary,
            compatibility_guaranteed=True,
        )

    def analyze_bottleneck(
        self,
        cpu_name: str,
        gpu_name: str,
        resolution: ResolutionTier = ResolutionTier.RES_1440P,
    ) -> BottleneckReport:
        cpu_clean = cpu_name.strip()
        gpu_clean = gpu_name.strip()
        cpu_tier = self._eval_cpu_tier(cpu_clean)
        gpu_tier = self._eval_gpu_tier(gpu_clean)

        # Resolution adjustment: at 1080p CPU matters more; at 4K GPU dominates
        res_mult = 1.2 if resolution == ResolutionTier.RES_1080P else (0.9 if resolution == ResolutionTier.RES_1440P else 0.5)

        tier_diff = gpu_tier - cpu_tier
        if abs(tier_diff) <= 1:
            bottleneck_pct = round(max(2.5, min(7.5, 3.0 + abs(tier_diff) * 2.5)), 1)
            status = BottleneckRating.BALANCED
            explanation = (
                f"Sự kết hợp giữa {cpu_clean} và {gpu_clean} ở độ phân giải {resolution} rất cân bằng. "
                "Cả hai linh kiện đều phát huy tối đa công suất mà không bị kìm hãm lẫn nhau."
            )
            recommendation = "Cấu hình đạt độ tương thích và cân bằng hiệu năng tối ưu. Có thể yên tâm sử dụng!"
        elif tier_diff > 1:
            bottleneck_pct = round(min(45.0, tier_diff * 11.5 * res_mult), 1)
            status = BottleneckRating.CPU_BOTTLENECK
            explanation = (
                f"Phát hiện nghẽn cổ chai CPU khoảng {bottleneck_pct}% ở {resolution}. "
                f"Bộ xử lý {cpu_clean} không kịp chuẩn bị khung hình cho card đồ họa {gpu_clean}, "
                "khiến GPU không thể chạy hết 100% hiệu năng."
            )
            recommendation = (
                "Khuyên bạn nên nâng cấp CPU lên dòng cao hơn (ví dụ Core i5/i7 thế hệ mới hoặc Ryzen 5/7 AM5) "
                "hoặc chuyển sang màn hình 2K/4K để đẩy tải xử lý về phía GPU."
            )
        else:
            bottleneck_pct = round(min(40.0, abs(tier_diff) * 10.0), 1)
            status = BottleneckRating.GPU_BOTTLENECK
            explanation = (
                f"Phát hiện giới hạn hiệu năng do GPU (khoảng {bottleneck_pct}%). "
                f"CPU {cpu_clean} có sức mạnh vượt trội so với {gpu_clean}. Trong các tác vụ đồ họa và game, "
                "GPU sẽ luôn hoạt động 100% trong khi CPU chỉ dùng một phần công suất."
            )
            recommendation = (
                "Nếu mục đích chính là chơi game đồ họa cao hoặc render 3D, hãy cân nhắc nâng cấp GPU mạnh hơn "
                "để tận dụng hết sức mạnh của CPU."
            )

        return BottleneckReport(
            cpu_name=cpu_clean,
            gpu_name=gpu_clean,
            resolution=resolution,
            bottleneck_percentage=bottleneck_pct,
            status=status,
            explanation=explanation,
            recommendation=recommendation,
        )

    def assess_upgrade_path(
        self,
        socket: str,
        psu_wattage: int,
        ram_type: str,
        current_gpu: str | None = None,
    ) -> UpgradePathReport:
        norm_socket = _norm_str(socket)
        norm_ram = _norm_str(ram_type)

        if "AM5" in norm_socket:
            lifecycle = "ACTIVE (Hỗ trợ dài hạn ít nhất đến 2027+)"
            cpu_options = [
                "Ryzen 7 7800X3D / Ryzen 7 9800X3D (Vua chơi game 3D V-Cache)",
                "Ryzen 9 7950X / 9950X (16 nhân 32 luồng cho đồ họa & AI)",
                "Các thế hệ Zen 6 tiếp theo trong tương lai",
            ]
            verdict = "Nền tảng AM5 có tiềm năng nâng cấp vượt trội nhất hiện nay, không cần thay main/RAM trong 3-5 năm tới."
        elif "LGA1700" in norm_socket:
            lifecycle = "MATURE_END_OF_LIFE (Intel đã kết thúc vòng đời LGA1700)"
            cpu_options = [
                "Intel Core i7-13700K / 14700K (16C/24T hoặc 20C/28T)",
                "Intel Core i9-13900K / 14900K (Đỉnh cao của socket LGA1700)",
            ]
            verdict = "LGA1700 đã hết vòng đời; nếu muốn lên đời CPU thế hệ 15 (Core Ultra 200) sẽ phải thay bo mạch chủ mới."
        elif "AM4" in norm_socket:
            lifecycle = "LEGACY_MATURE (Vòng đời hoàn thành, giá cực kỳ tiết kiệm)"
            cpu_options = [
                "Ryzen 7 5700X3D / 5800X3D (Cứu cánh nâng cấp game đỉnh cao trên DDR4)",
                "Ryzen 9 5900X / 5950X (Tối ưu cho làm việc đa nhiệm)",
            ]
            verdict = "AM4 rất bền bỉ và giá rẻ. 5700X3D là bước nâng cấp cuối cùng hoàn hảo trước khi đổi cả dàn sang AM5."
        else:
            lifecycle = f"CURRENT_SOCKET ({socket})"
            cpu_options = ["Các dòng CPU cùng socket với xung nhịp và số nhân cao hơn"]
            verdict = "Cần kiểm tra danh sách hỗ trợ CPU (CPU Support List) trên trang chủ của hãng sản xuất bo mạch chủ."

        # PSU Headroom evaluation
        if psu_wattage >= 850:
            gpu_headroom = "Dồi dào. Cân tốt mọi card đồ họa cao cấp nhất (RTX 4080 Super, RTX 4090, RX 7900 XTX)."
        elif psu_wattage >= 750:
            gpu_headroom = "Rất tốt. Nâng cấp thoải mái lên RTX 4070 Ti Super, RTX 4070 Super, RX 7800 XT mà không cần thay nguồn."
        elif psu_wattage >= 600:
            gpu_headroom = "Đủ dùng cho phân khúc tầm trung (RTX 4060 Ti, RTX 4070, RX 7600 XT). Nếu lên RTX 4070 Ti trở lên nên đổi nguồn."
        else:
            gpu_headroom = f"Công suất {psu_wattage}W ở mức cơ bản. Nâng cấp GPU công suất cao sẽ cần thay nguồn mới tối thiểu 650W - 750W."

        # RAM upgradeability
        if "DDR5" in norm_ram:
            ram_upgrade = "Chuẩn DDR5 hiện đại, dễ dàng nâng cấp dung lượng (32GB/64GB) hoặc bus cao hơn (6000-6400MHz) và tái sử dụng sau này."
        else:
            ram_upgrade = "Chuẩn DDR4 chi phí rẻ, dễ cắm thêm thanh RAM nâng cấp dung lượng, nhưng sẽ không tái sử dụng được nếu đổi sang nền tảng CPU mới."

        return UpgradePathReport(
            platform_socket=socket,
            platform_lifecycle=lifecycle,
            cpu_upgrade_options=cpu_options,
            gpu_upgrade_headroom=gpu_headroom,
            ram_upgradeability=ram_upgrade,
            verdict=verdict,
        )

    def recommend_peripherals(
        self,
        gpu_name: str,
        target_use_case: str = "Gaming",
    ) -> PeripheralRecommendation:
        gpu_tier = self._eval_gpu_tier(gpu_name)
        use_case = target_use_case.lower()

        if gpu_tier >= 4:
            res = "4K UHD (3840x2160) hoặc 2K QHD (2560x1440)"
            hz = "165Hz - 240Hz (hoặc 360Hz QD-OLED)"
            panel = "QD-OLED hoặc Fast IPS Mini-LED"
            examples = [
                "LG UltraGear 27GR95QE 27'' OLED 240Hz 0.03ms",
                "ASUS ROG Swift PG32UCDM 32'' 4K QD-OLED 240Hz",
                "Alienware AW3423DWF 34'' QD-OLED Curved 165Hz",
            ]
            accessories = [
                "Chuột Gaming siêu nhẹ cao cấp: Logitech G Pro X Superlight 2 / Razer Viper V3 Pro",
                "Bàn phím cơ Custom không dây: MonsGeek M1W V3 / Keychron Q1 Pro",
                "Tai nghe Gaming không dây: SteelSeries Arctis Nova Pro Wireless / HyperX Cloud III Wireless",
            ]
            advice = f"GPU {gpu_name} thuộc phân khúc cao cấp, bắt buộc đi kèm màn hình 2K 240Hz hoặc 4K để không lãng phí sức mạnh đồ họa đỉnh cao."
        elif gpu_tier == 3:
            res = "2K QHD (2560x1440)"
            hz = "165Hz - 180Hz"
            panel = "Fast IPS 10-bit màu (sRGB 100%, DCI-P3 95%)"
            examples = [
                "LG 27GP850-B 27'' Nano IPS 2K 165Hz",
                "ASUS TUF Gaming VG27AQL3A 27'' 2K Fast IPS 180Hz",
                "ViewSonic VX2780-2K-PRO 27'' 2K 170Hz",
            ]
            accessories = [
                "Chuột Gaming: Logitech G502 X / Pulsar X2V2",
                "Bàn phím cơ: Akko 5087B Plus Multi-modes / RK Royal Kludge R87",
                "Tai nghe: HyperX Cloud II / Corsair HS65 Surround",
            ]
            advice = f"GPU {gpu_name} là 'vua' độ phân giải 2K. Màn hình 27 inch 2K 165Hz Fast IPS là sự kết hợp chuẩn mực nhất về độ sắc nét và mượt mà."
        else:
            if "đồ họa" in use_case or "graphic" in use_case:
                res = "2K QHD (2560x1440) hoặc Full HD (1920x1080)"
                hz = "75Hz - 100Hz"
                panel = "IPS chuẩn màu chuyên nghiệp (Delta E < 2, 100% sRGB)"
                examples = [
                    "ASUS ProArt PA278CV 27'' 2K IPS Chuyên Đồ Họa",
                    "Dell UltraSharp U2424H 24'' IPS 120Hz chuẩn màu",
                ]
            else:
                res = "Full HD (1920x1080)"
                hz = "144Hz - 180Hz"
                panel = "Fast IPS"
                examples = [
                    "AOC 24G2SP 23.8'' Fast IPS 165Hz",
                    "ViewSonic Omni VX2479-HD-PRO 24'' 165Hz",
                    "MSI G244F E2 24'' Rapid IPS 180Hz",
                ]
            accessories = [
                "Chuột Gaming/Văn phòng: Logitech G102 Gen2 Lightsync / Razer Cobra",
                "Bàn phím: DareU EK87 V2 / Fuhlen D87S RGB",
                "Tai nghe: E-Dra EH492 / DareU EH416",
            ]
            advice = f"Cấu hình với {gpu_name} thích hợp nhất với màn hình Full HD Fast IPS 144Hz-180Hz để đạt mức FPS cao nhất khi chơi game Esport & AAA."

        return PeripheralRecommendation(
            gpu_model=gpu_name,
            target_use_case=target_use_case,
            recommended_display_resolution=res,
            recommended_refresh_rate=hz,
            recommended_panel=panel,
            display_examples=examples,
            recommended_peripherals=accessories,
            advice=advice,
        )

    def _eval_cpu_tier(self, name: str) -> int:
        n = name.upper()
        if any(k in n for k in ["7800X3D", "7950X3D", "9800X3D", "13900", "14900", "9950X", "THREAD"]):
            return 4
        if any(k in n for k in ["I7", "RYZEN 7", "13700", "14700", "7700", "9700", "5700X3D", "5800X3D"]):
            return 3
        if any(k in n for k in ["I5", "RYZEN 5", "12400", "12600", "13400", "13600", "14400", "14600", "5600", "7500F", "7600"]):
            return 2
        return 1

    def _eval_gpu_tier(self, name: str) -> int:
        n = name.upper()
        if any(k in n for k in ["4090", "5090", "7900 XTX", "7900XTX"]):
            return 5
        if any(k in n for k in ["4080", "4070 TI", "7900 XT", "7900XT", "3080", "3090"]):
            return 4
        if any(k in n for k in ["4070", "7800 XT", "7800XT", "7700 XT", "3070", "6800"]):
            return 3
        if any(k in n for k in ["4060", "3060", "7600", "6600", "3050", "2060"]):
            return 2
        return 1


__all__ = ["LocalHardwareRuleEngine"]
