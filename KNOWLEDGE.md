# Cơ Sở Tri Thức Kỹ Thuật Phần Cứng (Hardware Knowledge Base) — 3-Tier Governance

Cơ sở Tri thức Phần cứng của `ai-service` được tổ chức và quản trị nghiêm ngặt thành **3 Tầng Độc lập (3-Tier Governance)** nhằm tách biệt tuyệt đối giữa:
1. Các quy luật vật lý bất biến (không thể vi phạm);
2. Các chính sách kỹ thuật điện toán (được chuẩn hóa và version hóa);
3. Các tham số cấu hình thuật toán tối ưu (có thể tinh chỉnh theo nghiệp vụ).

```text
ai-service/knowledge/
├── 01-physical-ground-truth.md     # Tầng 1: Physical Ground Truth (Bất biến Vật lý)
├── 02-engineering-policies.md      # Tầng 2: Engineering Policy (Chính sách Kỹ thuật Điện & Nguồn)
└── 03-optimization-policies.md     # Tầng 3: Optimization Policy (Chính sách Thuật toán & MAUT)
```

---

## Danh mục Tài liệu Thành phần

### [1. Tầng 1: Physical Ground Truth (Bất biến Vật lý & Tiêu chuẩn Phần cứng)](knowledge/01-physical-ground-truth.md)
*Tập trung các quy tắc cơ học và điện tử không thể thương lượng:*
- **Socket CPU & Bo mạch chủ:** Ma trận tương thích AM5, AM4, LGA1700, LGA1851, LGA1200.
- **Chuẩn RAM:** Cơ chế cắm và điện áp DDR4 vs DDR5; bắt buộc Desktop DIMM.
- **Form Factor 2 Lớp:** Bắt buộc vừa thỏa mãn thứ bậc Rank, vừa nằm trong danh sách `case.supported_form_factors`.
- **Giới hạn Không gian Vật lý (Clearance):** Chiều dài GPU, chiều cao tản nhiệt, socket tản nhiệt. Triết lý Fail-Safe: *Thiếu dữ liệu $\to$ `UNKNOWN`*.
- **Năng lực Tích hợp Thực chứng (Zero Spec Fabrication):** Cấm tạo linh kiện ảo; chỉ cho phép iGPU / Stock cooler 0 VNĐ khi CPU mang đầy đủ thông số thực nghiệm grounded.

👉 Chi tiết xem tại: [`knowledge/01-physical-ground-truth.md`](knowledge/01-physical-ground-truth.md)

---

### [2. Tầng 2: Engineering Policy (Chính sách Kỹ thuật Điện toán & Bộ nguồn)](knowledge/02-engineering-policies.md)
*Quy chuẩn tính toán công suất và biên độ an toàn (`eng-power-sizing-v1`):*
- **Tải Đỉnh Duy trì (Sustained Peak Load):** $P_{\text{base}} = 65\text{W}$, TDP CPU kết hợp hệ số boost ($1.20$ nếu $\ge 105\text{W}$, $1.10$ nếu $< 105\text{W}$), TDP GPU.
- **Phụ cấp Dòng Đột biến (Transient Spike):** Card rời bù thêm $+30\%$ ($\ge 280\text{W}$), $+20\%$ ($\ge 180\text{W}$), $+12\%$ ($< 180\text{W}$); iGPU bù $0\%$.
- **Quy chuẩn Công suất Nguồn:**
  - Nguồn tối thiểu: $P_{\text{sustained}} \times 1.10$.
  - Nguồn khuyến nghị: $P_{\text{sustained}} + P_{\text{transient}} + 50.0\text{W}$.
- **Bậc thang Thương mại:** 9 kích thước chuẩn `(450, 500, 550, 600, 650, 750, 850, 1000, 1200) W`.
- **Cơ chế Chống Tràn tải (Overflow Guard):** Yêu cầu $> 1200\text{W}$ ném `ValueError` ngay lập tức.
- **Ánh xạ Chất lượng Nguồn:** Tier A=100, B=85, C=70, D=50; Bậc lạ $\to$ `None` (Fail-Safe, không gán ngầm điểm 50.0).

👉 Chi tiết xem tại: [`knowledge/02-engineering-policies.md`](knowledge/02-engineering-policies.md)

---

### [3. Tầng 3: Optimization Policy (Chính sách Tối ưu hóa & Thuật toán MAUT)](knowledge/03-optimization-policies.md)
*Chính sách phân bổ ngân sách, cắt tỉa Pareto và hàm chấm điểm:*
- **Khung Phân bổ Ngân sách (`opt-budget-envelope-v1`):** Tỷ lệ linh kiện theo 6 hồ sơ người dùng (`GAMING_1080P`, `GAMING_1440P`, `GAMING_4K`, `CONTENT_CREATION_3D`, `AI_DATA_SCIENCE`, `OFFICE_BUDGET`).
- **Cắt tỉa Pareto Bảo toàn Khả thi (`opt-pareto-dominance-v1`):** CPU không thể dominate CPU khác nếu làm mất phương án iGPU hoặc Stock Cooler 0 VNĐ (`preserves_cpu_optional_capabilities`). Thiếu metric so sánh $\to$ `False` (không prune).
- **Mô hình Chấm điểm Đa Mục tiêu MAUT (`opt-maut-scoring-v1`):** 18 bộ trọng số độc lập chuẩn hóa $\sum W_i = 1.0$.
  - Phân tách rõ ràng giữa `performance_per_cost` và `budget_saving`.
  - Mô hình nâng cấp 3 chiều: `platform_longevity`, `ram_slots`, `psu_headroom`.
  - Nguyên tắc Zero-Guessing: Thiếu dữ liệu của chiều có trọng số $> 0 \to$ Trả về `-1.0` (Invalidated).
- **Định danh Bất biến & Phân định Hòa:** Fingerprint 29 thuộc tính khi `id is None` và Tie-breaking key loại bỏ 100% rủi ro phụ thuộc thứ tự catalog.

👉 Chi tiết xem tại: [`knowledge/03-optimization-policies.md`](knowledge/03-optimization-policies.md)
