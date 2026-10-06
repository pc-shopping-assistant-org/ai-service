# Architecture Decision Records (ADR) — AI Service

Tài liệu này ghi lại toàn bộ các **quyết định kiến trúc cốt lõi (Key Decisions)** đã được thống nhất, kiểm chứng qua các vòng phản biện kỹ thuật và đóng băng cho Python `ai-service` (đặc biệt là Core Optimizer & AI Capabilities).

---

## Danh mục quyết định

- [ADR-001: Phân tách ranh giới tuyệt đối giữa Deterministic Core và AI/LLM](#adr-001-phân-tách-ranh-giới-tuyệt-đối-giữa-deterministic-core-và-aillm)
- [ADR-002: Nguyên tắc Zero Spec Guessing & Zero Fabricated Components](#adr-002-nguyên-tắc-zero-spec-guessing--zero-fabricated-components)
- [ADR-003: Cơ chế tương thích Fail-Safe và phân loại UNKNOWN](#adr-003-cơ-chế-tương-thích-fail-safe-và-phân-loại-unknown)
- [ADR-004: Tối ưu Pareto bảo toàn Feasibility và Global Optimum](#adr-004-tối-ưu-pareto-bảo-toàn-feasibility-và-global-optimum)
- [ADR-005: Tiêu chuẩn an toàn điện và xử lý PSU Overflow](#adr-005-tiêu-chuẩn-an-toàn-điện-và-xử-lý-psu-overflow)
- [ADR-006: Phân tách Boundary Schema (TOOL_CONFIG) vs Core Domain Schema (CORE_CONFIG)](#adr-006-phân-tách-boundary-schema-tool_config-vs-core-domain-schema-core_config)
- [ADR-007: Tách rời Recommendation Policy khỏi Mathematical Core](#adr-007-tách-rời-recommendation-policy-khỏi-mathematical-core)
- [ADR-008: Bằng chứng số liệu (MetricEvidence) thay thế hoàn toàn văn bản cảm tính](#adr-008-bằng-chứng-số-liệu-metricevidence-thay-thế-hoàn-toàn-văn-bản-cảm-tính)
- [ADR-009: Tính xác định toàn cục (Deterministic Identity & Tie-Breaking)](#adr-009-tính-xác-định-toàn-cục-deterministic-identity--tie-breaking)
- [ADR-010: Quản lý ràng buộc có nguồn gốc và cấm khóa giả định (Constraint Provenance)](#adr-010-quản-lý-ràng-buộc-có-nguồn-gốc-và-cấm-khóa-giả-định-constraint-provenance)
- [ADR-011: Đóng băng Deterministic Core v1 — Khóa linh kiện sở hữu và Minh bạch Điểm số](#adr-011-đóng-băng-deterministic-core-v1--khóa-linh-kiện-sở-hữu-và-minh-bạch-điểm-số)

---

## ADR-001: Phân tách ranh giới tuyệt đối giữa Deterministic Core và AI/LLM

### Bối cảnh & Vấn đề

Các mô hình ngôn ngữ lớn (LLM) có điểm yếu cố hữu: tính toán số học kém chính xác, dễ ảo giác thông số phần cứng, không thể cam kết 100% về độ tương thích vật lý/điện năng và kết quả thiếu tính tái lặp (non-deterministic). Nếu để LLM trực tiếp ráp cấu hình hoặc tự chọn linh kiện, hệ thống thương mại điện tử sẽ gặp rủi ro tư vấn sai, dẫn đến đơn hàng bị trả lại hoặc chập cháy linh kiện.

### Quyết định

Thiết lập ranh giới tách bạch tuyệt đối:

1. **LLM chỉ làm 2 việc:**
   - **Requirement Extraction**: Đọc hiểu hội thoại tự nhiên của khách hàng, trích xuất nhu cầu thành cấu trúc ràng buộc chuẩn (`PCBuildConstraints`).
   - **Explanation Layer**: Diễn giải kết quả cấu hình bằng văn phong thân thiện dựa trên danh sách bằng chứng kỹ thuật (`MetricEvidence`) do Core cung cấp.
2. **Deterministic Optimizer đảm nhận 100%:**
   - Tổ hợp và duyệt không gian linh kiện (Combinatorial search).
   - Kiểm tra tương thích cơ khí, socket, RAM, kích thước vỏ case.
   - Tính toán công suất nguồn và transient spike margin.
   - Đánh giá Pareto dominance và chấm điểm đa tiêu chí (MAUT).

### Trạng thái

**APPROVED & FROZEN (V1)**.

---

## ADR-002: Nguyên tắc Zero Spec Guessing & Zero Fabricated Components

### Bối cảnh & Vấn đề

Trong các phiên bản trước:

- Khi CPU không có GPU, hệ thống tự động sinh ra một "iGPU ảo 15W" hoặc "Stock Cooler ảo 55mm". Điều này sai vật lý vì CPU dòng `F` của Intel hoàn toàn không có nhân đồ họa, và các CPU cao cấp (như Ryzen 7 7800X3D) không hề đi kèm tản nhiệt hộp.
- Trong thuật toán tính điểm, khi linh kiện thiếu benchmark, hệ thống gán ngầm `50.0/100`, biến kết quả tối ưu thành "chọn build dựa trên dữ liệu + giả định 50 điểm".

### Quyết định

1. Bổ sung trường dữ liệu thực tế vào `ComponentSpec`:
   - `has_integrated_graphics: bool`
   - `includes_stock_cooler: bool`
   - `stock_cooler_tdp_watts: int | None`
2. **Tuyệt đối không chế tạo linh kiện ảo (No component fabrication):**
   - Chỉ cho phép build không card rời nếu `cpu.has_integrated_graphics == True`. Nếu CPU không có iGPU và catalog hết card rời $\to$ Báo không tìm thấy cấu hình khả thi.
   - Chỉ cho phép build không tản rời nếu `cpu.includes_stock_cooler == True`.
3. **Policy-driven metric validation trong chấm điểm:**
   - Nếu một chiều chất lượng tham gia objective với trọng số $W > 0$ mà linh kiện thiếu dữ liệu (`None`) $\to$ Trả về `-1.0` (Invalidated), loại bỏ cấu hình khỏi danh sách đề cử. Tuyệt đối không gán mặc định ngầm.

### Trạng thái

**APPROVED & FROZEN (V1)**.

---

## ADR-003: Cơ chế tương thích Fail-Safe và phân loại UNKNOWN

### Bối cảnh & Vấn đề

Khi một linh kiện trong catalog bị thiếu dữ liệu kỹ thuật (ví dụ VGA không ghi chiều dài, Case không ghi chiều dài VGA tối đa, hoặc PSU không ghi công suất), nếu bỏ qua kiểm tra thì hệ thống có thể tạo ra build không thể lắp ráp ngoài đời thực.

### Quyết định

1. Phân loại tương thích thành 3 trạng thái rõ ràng:
   - `COMPATIBLE`: Mọi thông số đối chiếu đầy đủ và đạt chuẩn.
   - `INCOMPATIBLE`: Thông số đối chiếu vi phạm giới hạn vật lý/điện năng.
   - `UNKNOWN`: Thiếu ít nhất một thông số trọng yếu để kết luận.
2. **Quy tắc Fail-Safe:**
   - Thiếu socket, thiếu chuẩn RAM, thiếu form factor $\to$ `UNKNOWN`.
   - Card rời tồn tại nhưng thiếu `gpu_length_mm` hoặc Case thiếu `max_gpu_length_mm` $\to$ `UNKNOWN`.
   - Tản nhiệt khí thiếu `cooler_height_mm` hoặc Case thiếu `max_cooler_height_mm` $\to$ `UNKNOWN`.
   - Nguồn PSU thiếu `wattage` $\to$ `UNKNOWN`.
3. **Chỉ giữ lại cấu hình `COMPATIBLE`:** Toàn bộ cấu hình `UNKNOWN` hoặc `INCOMPATIBLE` đều bị loại bỏ ngay trước vòng chấm điểm. `RankedBuild.compatibility_status` bắt buộc phải có giá trị cụ thể, không được đặt mặc định là `COMPATIBLE`.

### Trạng thái

**APPROVED & FROZEN (V1)**.

---

## ADR-004: Tối ưu Pareto bảo toàn Feasibility và Global Optimum

### Bối cảnh & Vấn đề

- Nếu áp dụng Pareto dominance ngây thơ `Price(B) <= Price(A) and Perf(B) >= Perf(A)`, một GPU rẻ hơn có điểm benchmark nhỉnh hơn có thể loại bỏ một GPU có 16GB VRAM. Điều này làm mất nghiệm tối ưu cho `AI_DATA_SCIENCE`.
- Nếu lọc ứng viên quá chặt theo Budget Envelope ban đầu, các cấu hình tối ưu toàn cục (ví dụ GPU vượt mức envelope nhưng kết hợp linh kiện khác rẻ tạo nên tổng chi phí hợp lệ) sẽ bị loại bỏ sớm.

### Quyết định

1. **Dominance Vector phụ thuộc `(UseCaseProfile, BuildObjective)`:**
   - `AI_DATA_SCIENCE`: Bắt buộc đưa `vram_gb` vào vector so sánh.
   - `UPGRADE_FRIENDLY`: Đưa `psu_quality`, `ram_slots` vào so sánh.
2. **Bảo tồn Feasibility:**
   - Không bao giờ cho phép 2 linh kiện khác Socket, khác chuẩn RAM, hoặc khác Form Factor triệt tiêu lẫn nhau qua Pareto dominance.
   - Với Bo mạch chủ: Không prune purely on price khi chưa có thông số đánh giá chất lượng VRM/tản nhiệt.
3. **Envelope chỉ sắp xếp thứ tự ưu tiên:**
   - `CandidatePool` chứa cả `preferred` (để duyệt trước) và `all_affordable` (để vét cạn không sót nghiệm).
   - Invariant: **Bật hay tắt Pareto Pruning đều phải cho ra cùng một cấu hình tối ưu toàn cục (Global Optimum)**.

### Trạng thái

**APPROVED & FROZEN (V1)**.

---

## ADR-005: Tiêu chuẩn an toàn điện và xử lý PSU Overflow

### Bối cảnh & Vấn đề

Thuật toán làm tròn nguồn trước đây dùng `next((s for s in SIZES if s >= target), SIZES[-1])`. Khi hệ thống yêu cầu 1270W, hàm trả về 1200W (kích thước cuối cùng), dẫn đến nguồn bị non tải so với yêu cầu, có nguy cơ ngắt điện hoặc chập cháy.

### Quyết định

1. Công thức tính công suất thực tế:
   $$\text{Sustained Peak} = \text{CPU Peak} + \text{GPU Peak} + \text{Platform Base (65W)}$$
   $$\text{Minimum PSU} = \text{Sustained Peak} + \text{Transient Allowance (12\% - 30\%)}$$
   $$\text{Recommended PSU} = \text{Minimum PSU} \times 1.30 \quad (\text{Headroom 30\%})$$
2. Thang công suất thương mại chuẩn: `(450, 500, 550, 650, 750, 850, 1000, 1200)`.
3. **Cơ chế Overflow Guard:** Hàm `round_up_psu(target)` ném ngay lập tức `ValueError` nếu công suất yêu cầu vượt quá 1200W, kiên quyết từ chối cấu hình thay vì hạ thấp công suất.

### Trạng thái

**APPROVED & FROZEN (V1)**.

---

## ADR-006: Phân tách Boundary Schema (TOOL_CONFIG) vs Core Domain Schema (CORE_CONFIG)

### Bối cảnh & Vấn đề

Nếu toàn bộ schemas dùng `ConfigDict(extra="ignore")`, lỗi đánh máy như `target_buget_vnd` sẽ bị bỏ qua trong im lặng, dẫn đến việc engine chạy bằng giá trị fallback mặc định. Ngược lại, nếu schema tiếp nhận tool call của LLM quá khắt khe, tool call sẽ dễ gãy do LLM sinh thêm trường ngoài ý muốn.

### Quyết định

Tách làm 2 tầng cấu hình Pydantic:

1. **`TOOL_CONFIG = ConfigDict(extra="ignore")`**:
   - Dùng cho các schema tầng biên (Boundary schemas): `CheckCompatibilityArgs`, `CalculateWattageArgs`, `RecommendBuildArgs`, `FindAlternativesArgs`...
   - Tiếp nhận input linh hoạt từ LLM hoặc REST request.
2. **`CORE_CONFIG = ConfigDict(extra="forbid")`**:
   - Dùng cho toàn bộ Core Domain Models: `ConstraintValue`, `PCBuildConstraints`, `RankedBuild`, `OptimizationResult`, `PowerEstimate`, `MetricEvidence`...
   - Bất kỳ trường lạ hoặc lỗi chính tả nào đều ném `ValidationError` ngay lập tức.
3. **Cách ly hoàn toàn legacy types:** `BuildPurpose` chỉ được chấp nhận như alias tạm thời ở tầng boundary của `RecommendBuildArgs`, tự động chuẩn hóa sang `UseCaseProfile`. Core domain bên dưới hoàn toàn không import `BuildPurpose`.

### Trạng thái

**APPROVED & FROZEN (V1)**.

---

## ADR-007: Tách rời Recommendation Policy khỏi Mathematical Core

### Bối cảnh & Vấn đề

Trước đây `OptimizationResult` chứa sẵn trường `recommended_objective = BuildObjective.BALANCED` và `recommended_build = builds[BALANCED]`. Điều này áp đặt chủ quan của chính sách bán hàng vào tầng tính toán thuần túy. Core toán học không có cơ sở để khẳng định `BALANCED` tốt hơn `PERFORMANCE` hay `UPGRADE_FRIENDLY`.

### Quyết định

1. `OptimizationResult` chỉ phản ánh sự thật toán học:
   ```python
   class OptimizationResult(BaseModel):
       target_budget_vnd: int
       use_case: UseCaseProfile
       builds: dict[BuildObjective, RankedBuild]
       candidates_evaluated: int
       pruned_count: int
   ```
2. Quyết định chọn cấu hình nào làm mặc định giới thiệu cho người dùng (`BALANCED` hoặc objective người dùng chọn) được chuyển ra ngoài: thuộc trách nhiệm của **Application / Recommendation Policy Layer**.

### Trạng thái

**APPROVED & FROZEN (V1)**.

---

## ADR-008: Bằng chứng số liệu (MetricEvidence) thay thế hoàn toàn văn bản cảm tính

### Bối cảnh & Vấn đề

Trong core model trước đây có các trường văn bản tự do: `key_strengths`, `trade_offs`, `bottleneck_analysis` chứa các câu khẳng định chung chung như "Tản nhiệt êm ái", "Bảo đảm nâng cấp sau này". Những nhận định này không có số liệu chứng minh từ thuật toán.

### Quyết định

1. Xóa bỏ toàn bộ các trường chuỗi văn bản tự do khỏi `RankedBuild`.
2. Thay thế bằng danh sách bằng chứng kỹ thuật có cấu trúc:
   ```python
   class MetricEvidence(BaseModel):
       metric: str
       raw_value: float | int | str
       normalized_score: float | None = None
       unit: str | None = None
       source: str | None = None
       label: str = ""
   ```
3. Tầng Agent / Explanation Layer phía ngoài chỉ được phép sinh lời tư vấn dựa trên các `MetricEvidence` thực tế này (Anti-hallucination ground truth).

### Trạng thái

**APPROVED & FROZEN (V1)**.

---

## ADR-009: Tính xác định toàn cục (Deterministic Identity & Tie-Breaking)

### Bối cảnh & Vấn đề

Nếu hai cấu hình có cùng điểm số và cùng mức giá, thứ tự trả về có thể phụ thuộc vào thứ tự danh sách linh kiện trong catalog đầu vào. Khi xáo trộn catalog (`shuffle`), kết quả trả về có thể khác nhau. Hơn nữa, trong môi trường test fixture, linh kiện có thể không có `id` (None).

### Quyết định

1. Định danh linh kiện ổn định (Deterministic Component Fingerprint):
   - Nếu `id` có giá trị: Sử dụng `str(comp.id)`.
   - Nếu `id is None`: Tạo fingerprint kết hợp đầy đủ:
     `category|brand|name|socket|ram_type|form_factor|price|tdp_watts|performance_score`.
2. Khóa phân định hòa (Tie-Breaking Key):
   $$\text{TieKey} = (-\text{objective\_score}, \text{total\_price}, \text{sorted\_parts\_fingerprint})$$
   Đảm bảo 100% test xáo trộn catalog với bất kỳ seed ngẫu nhiên nào đều cho ra cùng một cấu hình tối ưu duy nhất.

### Trạng thái

**APPROVED & FROZEN (V1)**.

---

## ADR-010: Quản lý ràng buộc có nguồn gốc và cấm khóa giả định (Constraint Provenance)

### Bối cảnh & Vấn đề

Hệ thống AI dễ gặp lỗi: mô hình tự suy luận một sở thích (ví dụ khách nói chơi game $\to$ AI đoán thích card NVIDIA), sau đó tự động đánh dấu ràng buộc này là bất biến (`locked=True`), tước đoạt quyền lựa chọn của khách hàng hoặc làm gãy thuật toán tối ưu.

### Quyết định

1. Định nghĩa rõ nguồn gốc ràng buộc qua `ConstraintSource`:
   - `USER`: Khách hàng trực tiếp yêu cầu (ví dụ: "Tôi chỉ mua NVIDIA", "Ngân sách 25 triệu").
   - `SYSTEM`: Chính sách an toàn hoặc nghiệp vụ bắt buộc (ví dụ: công suất PSU tối thiểu).
   - `INFERRED`: Do LLM hoặc rule engine suy luận từ ngữ cảnh hội thoại.
   - `DEFAULT`: Giá trị dự phòng an toàn ban đầu.
2. **Quy tắc bất biến:**
   Chỉ các ràng buộc có nguồn `USER` hoặc `SYSTEM` mới được phép thiết lập `locked=True`.
   Nếu ràng buộc có nguồn `INFERRED` hoặc `DEFAULT` mà cố tình set `locked=True`, hệ thống ném `ValidationError` ngay lập tức.

---

## ADR-011: Đóng băng Deterministic Core v1 — Khóa linh kiện sở hữu và Minh bạch Điểm số

### Bối cảnh & Vấn đề

1. Người dùng có thể đã sở hữu sẵn linh kiện (GPU, nguồn, case...) và chỉ muốn chi ngân sách mua các linh kiện còn thiếu. Trước đây `owned_parts` chưa được optimizer sử dụng, khiến slot bị duyệt tự do và ngân sách tính sai.
2. Khi append synthetic iGPU hoặc stock cooler giá 0 VNĐ vào cuối danh sách ứng viên đã sort, logic cắt tỉa nhánh `if sub > budget: break` gặp linh kiện giá cao sẽ dừng sớm và bỏ qua phương án 0 VNĐ.
3. Việc chế tạo linh kiện giả định (fake TDP, fake benchmark, fake UUID) vi phạm nguyên tắc Zero-Guessing.
4. Điểm số tổng quát của objective là một con số tổng `float`, thiếu minh bạch về tỷ trọng đóng góp của từng chiều kỹ thuật.

### Quyết định

1. **Xử lý Linh kiện có sẵn (`OwnedComponent`) & Hạch toán Ngân sách Chính xác:**
   - Khi slot thuộc `owned_parts`, khóa hoàn toàn candidate pool của slot đó vào `[owned.component]`.
   - Nếu `exclude_from_budget == True`, linh kiện có sẵn đóng góp 0 VNĐ vào chi phí thực chi (`spending`). `RankedBuild.total_price` phản ánh chính xác số tiền thực tế khách hàng cần thanh toán.
2. **Sắp xếp Toàn cục Bảo toàn Nghiệm 0 VNĐ:**
   - Sau khi bổ sung synthetic candidate (iGPU hoặc stock cooler), danh sách ứng viên bắt buộc phải được sắp xếp lại theo `(spending_price asc, stable_id asc)`.
   - Đảm bảo candidate 0 VNĐ luôn nằm ở vị trí đầu tiên, không bị bỏ qua bởi `break`.
3. **Triệt tiêu Hoàn toàn Linh kiện Ảo (Zero Spec Fabrication):**
   - Chỉ tạo synthetic iGPU khi CPU có đầy đủ `has_integrated_graphics == True`, `integrated_graphics_score != None`, và `integrated_graphics_power_watts != None`.
   - Chỉ tạo synthetic stock cooler khi CPU có đầy đủ `includes_stock_cooler == True`, `stock_cooler_height_mm != None`, và `stock_cooler_score != None`.
   - Định danh synthetic candidate sử dụng UUIDv5 determinism: `uuid5(SYNTHETIC_NAMESPACE, f"{stable_component_id(cpu)}:igpu")`.
4. **Kiểm tra Tương thích Cơ - Điện Tuyệt đối:**
   - Chiều dài GPU chỉ kiểm tra với card rời (`not gpu.is_integrated`), không kiểm tra với iGPU.
   - Kích thước Case kiểm tra strict membership trong `case.supported_form_factors` trước khi fallback theo thứ bậc ranking.
   - Tản nhiệt khí kiểm tra nghiêm ngặt Socket hỗ trợ (`cooler.supported_sockets`).
5. **Minh bạch Toán học Đa Mục tiêu (`ScoreBreakdown` & `ScoreDimension`):**
   - Hàm `score_build` trả về đối tượng `ScoreBreakdown` chứa danh sách chi tiết `ScoreDimension` (bao gồm `raw_value`, `normalized_score`, `weight`, và `contribution = normalized_score * weight`).
   - `ScoreBreakdown` hỗ trợ toán tử số học/so sánh để tương thích ngược 100% với các phép so sánh số thực.
   - Thêm assertion kiểm tra toàn bộ 18 chính sách `(UseCaseProfile, BuildObjective)` có tổng trọng số bằng đúng 1.0 lúc khởi động module; fail-fast ngay lập tức nếu thiếu chính sách.

### Trạng thái

**APPROVED & FROZEN (V1)**.

---

## ADR-012: Quản trị Tri thức Phần cứng theo 3 Tầng (3-Tier Knowledge Governance) & Đồng bộ Tri thức - Mã nguồn

### Bối cảnh & Vấn đề

1. **Pha trộn ranh giới tri thức:** Phiên bản Knowledge Base ban đầu gom chung các bất biến vật lý (socket, RAM generation, physical clearances), chính sách kỹ thuật điện toán (PSU margins, transient spikes), và cấu hình thuật toán (MAUT weights, normalization constants) dưới nhãn "Ground-Truth Knowledge Base", làm suy yếu tính minh định khi phản biện kỹ thuật.
2. **Sai lệch (Drift) giữa Tài liệu và Mã nguồn Core:**
   - Công thức tính nguồn và phụ cấp dòng đột biến trong tài liệu lệch với triển khai thực tế trong `optimizer.py`.
   - Danh sách công suất nguồn thương mại trong tài liệu thiếu mốc `600W`.
   - Form-factor case chỉ dựa vào phân cấp thứ bậc (rank) mà bỏ qua kiểm tra thành viên rõ ràng (`explicit membership`).
   - `Cost Efficiency` bị dùng chung để gọi hai công thức có bản chất khác nhau (`budget_saving` vs `performance_per_cost`).
   - Đánh giá khả năng nâng cấp bị phân đôi giữa quy tắc cộng điểm tĩnh ($+35/+20/+25$) và mô hình 3 chiều trong MAUT.
3. **Lỗ hổng Cắt tỉa Pareto CPU (Capability Preservation):** CPU có iGPU hoặc stock cooler cho phép tạo cấu hình khả thi với GPU/Cooler 0 VNĐ. Nếu cắt tỉa CPU thuần túy theo `(performance_score, tdp_watts)`, một CPU rẻ hơn hoặc xung cao hơn nhưng không có iGPU/cooler sẽ loại bỏ CPU có iGPU/cooler, làm mất không gian nghiệm 0 VNĐ.
4. **Nguy cơ Đụng độ Định danh (Identity Collision):** Khi linh kiện không có `id` (`None`), fingerprint cũ chỉ gồm 9 trường, có thể gây đụng độ giữa 2 SKU cùng tên/giá nhưng khác VRAM hoặc công suất.

### Quyết định

1. **Phân chia Tri thức theo Mô hình 3 Tầng Nghiêm ngặt (3-Tier Governance):**
   - **Tầng 1 - Physical Ground Truth (Bất biến Vật lý & Tiêu chuẩn Phần cứng):** Socket, chuẩn RAM cơ học, không gian vật lý (clearance; thiếu dữ liệu $\to$ `UNKNOWN`), kiểm tra thành viên rõ ràng cho kích thước vỏ case, năng lực phần cứng thực chứng của CPU (iGPU/Stock Cooler).
   - **Tầng 2 - Engineering Policy (`eng-power-sizing-v1`):** Công thức tải đỉnh duy trì kết hợp hệ số boost TDP, phụ cấp dòng đột biến theo ngưỡng TDP card đồ họa, bậc thang thương mại 9 kích thước chuẩn (gồm `600W`), công thức nguồn tối thiểu (+10%) và khuyến nghị (+transient +50W buffer), cơ chế chống tràn tải an toàn (>1200W ném `ValueError`), và ánh xạ chất lượng nguồn Fail-Safe (không gán ngầm điểm 50.0).
   - **Tầng 3 - Optimization Policy (`opt-budget-envelope-v1`, `opt-pareto-dominance-v1`, `opt-maut-scoring-v1`):** Khung phân bổ ngân sách 6 hồ sơ, chiều cắt tỉa Pareto theo hồ sơ/mục tiêu, 18 chính sách MAUT chuẩn hóa $\sum W_i = 1.0$, phân biệt rõ `performance_per_cost` và `budget_saving`, và mô hình nâng cấp 3 chiều độc lập (`platform_longevity`, `ram_slots`, `psu_headroom`).
2. **Bảo toàn Năng lực Tùy chọn của CPU trong Pareto Dominance (`preserves_cpu_optional_capabilities`):**
   - CPU $B$ chỉ được dominate CPU $A$ nếu $B$ không làm mất bất kỳ năng lực tùy chọn nào mà $A$ cung cấp.
   - Nếu $A$ có iGPU $\to$ $B$ bắt buộc phải có iGPU với hiệu năng $\ge$ và công suất $\le$.
   - Nếu $A$ có stock cooler $\to$ $B$ bắt buộc phải có stock cooler với chiều cao $\le$ và công suất tản $\ge$.
3. **Đồng bộ Toàn diện Fingerprint Linh kiện (Canonical 29-Attribute Tuple):**
   - Khi `id is None`, cả `pruning.py` (`stable_component_key`) và `optimizer.py` (`stable_component_id`) serialize đầy đủ toàn bộ 29 thuộc tính chức năng. Loại bỏ 100% nguy cơ đụng độ giữa các SKU cùng dòng nhưng khác biến thể thông số kỹ thuật.
4. **Kiểm tra Tương thích Form-Factor Hai Lớp (Two-Tier Form Factor Invariant):**
   - Vừa bắt buộc thỏa mãn thứ bậc Rank, vừa bắt buộc `_norm(mb.form_factor)` phải nằm trong `case.supported_form_factors`. Trường hợp thiếu dữ liệu trả về `UNKNOWN`.

### Trạng thái

**APPROVED & FROZEN (V1)**.

---

## ADR-013: Mô hình hóa Bài toán Tối ưu Cấu hình theo Biến thể Multiple-Choice Knapsack Problem with Pairwise Compatibility Constraints (MCKP-PCC)

### Bối cảnh & Vấn đề

Bài toán tối ưu hóa cấu hình PC thường bị nhầm lẫn là bài toán lọc tìm kiếm thông thường (Search Filtering) hoặc bài toán sinh văn bản của LLM. Tuy nhiên:
1. Bản chất bài toán là chọn đúng 1 linh kiện từ mỗi danh mục trong 8 danh mục linh kiện rời rạc sao cho tổng chi phí $\le$ Ngân sách tối đa, đồng thời thỏa mãn toàn bộ các ràng buộc tương thích cơ học, vật lý và điện năng.
2. Tổng số trạng thái tổ hợp đạt mức $10^{11} - 10^{13}$, thuộc nhóm bài toán tối ưu hóa tổ hợp kinh điển có độ phức tạp NP-hard.
3. Cần một cơ sở lý thuyết toán học và khoa học máy tính chuẩn mực để:
   - Chứng minh sự cần thiết của việc tách biệt Deterministic Core khỏi Probabilistic LLM.
   - Định hướng cấu trúc dữ liệu và giải thuật cắt tỉa (Pareto Pruning + Branch-and-Bound).
   - Làm cơ sở bảo vệ học thuật và báo cáo kỹ thuật cho đồ án tốt nghiệp.

### Quyết định

1. **Định danh & Mô hình hóa Toán học Chính thức:**
   Bài toán xây dựng cấu hình PC được định danh chính thức là:
   > **"A Multiple-Choice Knapsack Problem with Pairwise Compatibility Constraints, solved using deterministic constrained search with Pareto pruning."**

   Hệ thống mô hình hóa bài toán thành biến thể **Multiple-Choice Knapsack Problem with Pairwise Compatibility Constraints (MCKP-PCC / MCKPC)** kết hợp với **Multi-Attribute Utility Theory (MAUT)**:
   - **Tập lớp rời nhau:** $K = 8$ danh mục $\{\text{CPU, MB, RAM, GPU, Storage, PSU, Case, Cooler}\}$.
   - **Ràng buộc Multiple-Choice:** $\sum_{j \in N_k} x_{kj} = 1 \quad \forall k \in \{1, \dots, 8\}$.
   - **Ràng buộc Sức chứa (Knapsack Capacity):** $\sum_{k=1}^8 \sum_{j \in N_k} c_{kj} x_{kj} \le B_{\text{target}}$.
   - **Đồ thị Xung đột $G = (V, E)$:** $x_u + x_v \le 1 \quad \forall (u, v) \in E$ (đại diện cho các cặp linh kiện không tương thích socket, chuẩn DDR, form factor case, clearance VGA/Cooler).
   - **Ràng buộc Ghép cặp Phi tuyến tính (Dynamic Coupled Constraint):** Công suất PSU phụ thuộc vào tổng tải đỉnh duy trì tức thời của CPU và GPU được chọn trong nhánh tìm kiếm:
     $\text{wattage}_{\text{PSU}} \ge \text{round\_up\_psu}\big((P_{\text{cpu\_peak}} + P_{\text{gpu\_peak}} + 65.0) \times 1.10\big)$.
   - **Hàm Mục tiêu:** $\max_{\mathbf{x}} U(\mathbf{x}) = \sum W_m S_m(\mathbf{x})$ chuẩn hóa theo MAUT.

2. **Chiến lược Thuật toán Giải quyết (Complexity Reduction Pipeline):**
   - **Tầng 1 (Pareto Reduction):** Giảm kích thước mỗi lớp $N_k$ bằng cắt tỉa Pareto đa chiều có bảo toàn tính khả thi (`preserves_cpu_optional_capabilities`).
   - **Tầng 2 (Adaptive Budget Envelopes):** Giới hạn không gian tìm kiếm sơ cấp trong dải tỷ lệ ngân sách chuẩn `preferred`, tự động mở rộng sang `all_affordable` nếu thiếu ứng viên.
   - **Tầng 3 (Constrained Branch-and-Bound with Conflict Pruning):** Duyệt theo thứ tự chiến lược $\text{CPU} \to \text{MB} \to \text{RAM} \to \text{GPU} \to \text{Case} \to \text{Cooler} \to \text{Storage} \to \text{PSU}$. Phát hiện xung đột sớm (Early Conflict Detection) và cắt nhánh ngân sách (Bounding).
   - **Tầng 4 (Deterministic Tie-Breaking):** Khóa phân định hòa bằng fingerprint 29 thuộc tính đảm bảo tính duy nhất và tái lập 100%.

### Trạng thái

**APPROVED & FROZEN (V1)**.


