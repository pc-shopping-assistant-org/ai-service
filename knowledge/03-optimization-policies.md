# Tầng 3: Optimization Policy (Chính sách Tối ưu hóa & Thuật toán MAUT)

> **Phân loại Quản trị:** Tầng 3 — Chính sách Tối ưu hóa & Tham số Thuật toán (Optimization Policy)  
> **Mã Chính sách:** `opt-budget-envelope-v1`, `opt-pareto-dominance-v1`, `opt-maut-scoring-v1`  
> **Phiên bản:** 1.0.0  
> **Ngày hiệu lực:** 2026-10-06  
> **Căn cứ & Động lực:** Tối ưu hóa đa mục tiêu MAUT (Multi-Attribute Utility Theory), phân bổ ngân sách thích ứng theo hồ sơ người dùng, và cắt tỉa Pareto bảo toàn không gian nghiệm khả thi.

---

## 3.1. Khung Phân bổ Ngân sách theo Hồ sơ (`opt-budget-envelope-v1`)

Mỗi hồ sơ nhu cầu (`UseCaseProfile`) sở hữu một khung phân bổ tỷ lệ phần trăm ngân sách cho từng loại linh kiện. Khung này gồm hai dải:
- `Preferred Range`: Dải ưu tiên tối ưu tỷ lệ lý tưởng.
- `Absolute Range`: Biên giới mở rộng khả thi tối đa.

| Nhóm Linh Kiện    | GAMING_1080P  | GAMING_1440P  |   GAMING_4K   | CONTENT_CREATION_3D | AI_DATA_SCIENCE | OFFICE_BUDGET |
| :---------------- | :-----------: | :-----------: | :-----------: | :-----------------: | :-------------: | :-----------: |
| **GPU (VGA)**     | $30\% - 48\%$ | $35\% - 55\%$ | $42\% - 62\%$ |    $28\% - 45\%$    |  $40\% - 65\%$  |  $0\% - 15\%$  |
| **CPU**           | $18\% - 28\%$ | $15\% - 25\%$ | $14\% - 22\%$ |    $22\% - 35\%$    |  $12\% - 22\%$  | $22\% - 38\%$  |
| **Mainboard**     |  $8\% - 16\%$ |  $8\% - 16\%$ |  $8\% - 15\%$ |     $8\% - 16\%$    |   $7\% - 15\%$  | $12\% - 24\%$  |
| **RAM**           |  $5\% - 12\%$ |  $5\% - 12\%$ |  $5\% - 12\%$ |     $8\% - 18\%$    |   $8\% - 18\%$  |  $8\% - 18\%$  |
| **Storage (SSD)** |  $4\% - 10\%$ |  $4\% - 10\%$ |  $4\% - 10\%$ |     $6\% - 14\%$    |   $5\% - 12\%$  |  $8\% - 18\%$  |
| **PSU (Nguồn)**   |  $5\% - 10\%$ |  $5\% - 10\%$ |  $6\% - 12\%$ |     $5\% - 11\%$    |   $6\% - 12\%$  |  $8\% - 16\%$  |
| **Case (Vỏ)**     |  $3\% - 8\%$  |  $3\% - 8\%$  |  $4\% - 9\%$  |     $3\% - 8\%$     |   $3\% - 8\%$   |  $5\% - 14\%$  |
| **Cooler (Tản)**  |  $2\% - 6\%$  |  $2\% - 7\%$  |  $3\% - 8\%$  |     $3\% - 9\%$     |   $2\% - 7\%$   |  $0\% - 8\%$   |

---

## 3.2. Quy tắc Cắt tỉa Pareto & Bảo toàn Khả thi (`opt-pareto-dominance-v1`)

Linh kiện $B$ được gọi là **Pareto Dominates** linh kiện $A$ khi và chỉ khi:
1. $B$ không đắt hơn $A$: $\text{Price}(B) \le \text{Price}(A)$.
2. Có cùng giao tiếp vật lý bắt buộc (socket, chuẩn RAM, form factor bo mạch).
3. **Bảo toàn Năng lực Tùy chọn của CPU (CPU Feasibility Capability Preservation):**
   - Nếu $A$ có iGPU $\to$ $B$ bắt buộc phải có iGPU với hiệu năng không kém hơn và điện năng không lớn hơn.
   - Nếu $A$ có stock cooler $\to$ $B$ bắt buộc phải có stock cooler với độ cao không lớn hơn và công suất giải nhiệt không kém hơn.
   *(CPU $B$ không thể loại bỏ CPU $A$ nếu làm mất đi phương án tiết kiệm 0 VNĐ cho GPU hoặc Cooler).*
4. Trên tất cả các chiều đặc trưng của `(UseCaseProfile, BuildObjective)`: $B$ không kém hơn $A$.
5. Trên ít nhất một chiều (giá hoặc chỉ số kỹ thuật): $B$ vượt trội hơn hẳn $A$.
6. **Fail-Safe:** Nếu bất kỳ chỉ số so sánh nào bị thiếu (`None`) $\to$ Dominance trả về `False` (không bao giờ cắt tỉa trên dữ liệu thiếu).
7. **Mainboard Policy:** Chiều so sánh của Mainboard là `[]` (danh sách rỗng) $\to$ Không bao giờ cắt tỉa bo mạch chủ chỉ vì giá rẻ hơn khi chưa có đủ dữ liệu VRM/kết nối.

---

## 3.3. Mô hình Chấm điểm Đa Mục tiêu MAUT (`opt-maut-scoring-v1`)

Hệ thống vận hành 18 bộ trọng số tối ưu hóa độc lập, được xác thực tự động $\sum W_i = 1.0$ khi khởi động.

Điểm số cấu hình là hàm tuyến tính chuẩn hóa:
$$\text{Total Score} = \sum_{i} W_i \times S_i \quad \in [0.0, 100.0]$$

### 3.3.1. Phân biệt Rõ ràng giữa Tiết kiệm Ngân sách & Hiệu năng trên Giá thành
Hệ thống phân tách hai khái niệm hoàn toàn khác biệt:

1. **Hiệu năng trên Giá thành (`performance_per_cost`):**
   Đánh giá mật độ sức mạnh tính toán mang lại trên mỗi đồng chi phí bỏ ra:
   $$S_{\text{cost\_eff}} = \min\left(100.0, \frac{\frac{\text{score}_{\text{gpu}} + \text{score}_{\text{cpu}}}{\text{Total Price (triệu VNĐ)}}}{12.0} \times 100.0\right)$$
2. **Tiết kiệm Ngân sách (`budget_saving`):**
   Đánh giá tỷ lệ ngân sách dôi dư chưa sử dụng (phù hợp nhu cầu văn phòng):
   $$S_{\text{budget\_saving}} = \max\left(0.0, \left(1.0 - \frac{\text{Total Price}}{\text{Budget}}\right) \times 100.0\right)$$

### 3.3.2. Mô hình Đánh giá Nâng cấp 3 Chiều Không Chồng Chéo
Thay vì dùng một công thức cộng điểm tĩnh ($+35/+20/+25$), kiến trúc MAUT model hóa khả năng nâng cấp thành **3 chiều đo lường chuẩn hóa độc lập**:

1. **Độ bền Nền tảng Socket (`platform_longevity`):**
   - AM5: `100.0`
   - LGA1851: `90.0`
   - LGA1700: `40.0`
   - AM4: `30.0`
   - LGA1200 / Khác: `20.0`
2. **Dư địa Khe RAM (`ram_slots`):**
   - 4 khe RAM: `100.0`
   - 2 khe RAM: `40.0`
3. **Dư địa Công suất Nguồn (`psu_headroom`):**
   $$S_{\text{psu\_headroom}} = \min\left(100.0, \max\left(0.0, \frac{\text{Wattage} - \text{Recommended PSU}}{300.0} \times 100.0\right)\right)$$

### 3.3.3. Nguyên tắc Không Giả định Dữ liệu (Zero-Guessing Invariant)
Nếu bất kỳ chiều chất lượng nào có trọng số $W_i > 0$ nhưng linh kiện tương ứng thiếu dữ liệu thông số (`None`) $\to$ **Thuật toán lập tức trả về điểm âm `-1.0` (Invalidated)**. Tuyệt đối không gán điểm mặc định 50.0.

---

## 3.4. Định danh Ổn định & Phân định Hòa Tuyệt đối (Deterministic Tie-Break)

Để đảm bảo tính lặp lại 100% trong môi trường phân tán:
1. **Fingerprint Linh kiện:** Kết hợp toàn bộ 29 thuộc tính kỹ thuật chuẩn khi linh kiện không có `id`.
2. **Khóa Phân định Hòa (Tie-Breaking Key):**
   $$\text{TieKey} = (-\text{round}(\text{objective\_score}, 4), \text{total\_price}, \text{sorted\_parts\_stable\_ids})$$
   Đảm bảo dù catalog bị xáo trộn (`shuffle`) ngẫu nhiên, thuật toán luôn tìm ra cùng một cấu hình tối ưu duy nhất.
