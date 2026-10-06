# Tầng 1: Physical Ground Truth (Bất biến Vật lý & Tiêu chuẩn Phần cứng)

> **Phân loại Quản trị:** Tầng 1 — Bất biến Miền Vật lý (Physical Domain Invariants)  
> **Tính chất:** Các định luật cơ học, điện tử và tiêu chuẩn chân cắm của nhà sản xuất phần cứng. Tuyệt đối không thay đổi theo thời gian, sở thích người dùng hay chính sách thuật toán.

---

## 1.1. Ma trận Tương thích CPU Socket & Bo mạch chủ (Mainboard)

CPU và Bo mạch chủ chỉ tương thích khi có giá trị `socket` chuẩn hóa trùng khớp hoàn toàn:

$$\text{NormStr}(\text{cpu.socket}) == \text{NormStr}(\text{mb.socket})$$

*(Chuẩn hóa: Chuyển toàn bộ thành chữ in hoa, loại bỏ khoảng trắng và dấu gạch nối. Ví dụ: `"am-5"` $\to$ `"AM5"`).*

| Nền tảng Socket | Thế hệ Vi xử lý Hỗ trợ                                 | Chuẩn RAM Hỗ trợ                         | Trạng thái Vòng đời (Lifecycle)              |
| :-------------- | :----------------------------------------------------- | :--------------------------------------- | :------------------------------------------- |
| **AM5**         | AMD Ryzen 7000, 8000, 9000 series                      | **DDR5 duy nhất**                        | `ACTIVE` (Hỗ trợ dài hạn 2027+)              |
| **AM4**         | AMD Ryzen 1000, 2000, 3000, 4000, 5000 series          | **DDR4 duy nhất**                        | `LEGACY_MATURE` (Hoàn tất vòng đời, kinh tế) |
| **LGA1700**     | Intel Gen 12 (Alder Lake), Gen 13 (Raptor), Gen 14     | **DDR4 hoặc DDR5** (tùy model mainboard) | `MATURE_END_OF_LIFE` (Đã đóng vòng đời)      |
| **LGA1851**     | Intel Core Ultra 200 series (Arrow Lake)               | **DDR5 duy nhất**                        | `ACTIVE` (Nền tảng thế hệ mới nhất của Intel)|
| **LGA1200**     | Intel Gen 10 (Comet Lake), Gen 11 (Rocket Lake)        | **DDR4 duy nhất**                        | `LEGACY` (Ngừng sản xuất mới)                |

> **Bất biến Cơ học:** Tuyệt đối không bao giờ cho phép kết hợp CPU Socket AM5 với Mainboard LGA1700 hay AM4, bất kể cùng phân khúc giá hay mục đích sử dụng.

---

## 1.2. Chuẩn Bộ nhớ trong (RAM Invariants)

1. **Thế hệ Bộ nhớ (RAM Generation):** DDR4 và DDR5 không thể cắm lẫn do vị trí rãnh khuyết (notch) cơ học lệch nhau và điện áp cung cấp khác biệt hoàn toàn (DDR4 1.2V vs DDR5 1.1V với IC quản lý nguồn PMIC tích hợp trên thanh RAM). Bo mạch chủ chuẩn nào chỉ cắm được thanh RAM đúng chuẩn đó:
   $$\text{NormStr}(\text{mb.ram\_type}) == \text{NormStr}(\text{ram.ram\_type})$$
2. **Hình dạng Chân cắm:** Bắt buộc Desktop DIMM 288-pin (không hỗ trợ chuẩn laptop SO-DIMM).
3. **Số lượng Khe cắm (Slots):** Mainboard có 4 khe RAM cho phép nâng cấp dung lượng linh hoạt hơn mainboard 2 khe RAM.

---

## 1.3. Phân cấp Form Factor & Kiểm tra Thành viên Rõ ràng (Explicit Membership)

Phân cấp kích thước chuẩn hóa từ lớn đến nhỏ theo thứ bậc (Rank):

$$\text{E-ATX (Rank 4)} > \text{ATX (Rank 3)} > \text{Micro-ATX (Rank 2)} > \text{Mini-ITX (Rank 1)}$$

**Bất biến tương thích Kích thước Case - Mainboard:**
Để một Bo mạch chủ lắp vừa một Vỏ Case, cấu hình bắt buộc phải thỏa mãn **cả hai điều kiện**:

1. **Điều kiện Phân cấp (Rank Hierarchy):**
   $$\text{Rank}(\text{mb.form\_factor}) \le \max_{s \in \text{case.supported\_form\_factors}}(\text{Rank}(s))$$
2. **Điều kiện Thành viên Rõ ràng (Explicit Membership):**
   $$\text{NormStr}(\text{mb.form\_factor}) \in \{\text{NormStr}(s) \mid s \in \text{case.supported\_form\_factors}\}$$

*(Nếu Vỏ Case chỉ khai báo hỗ trợ `["ATX"]` mà không liệt kê `Micro-ATX`, hệ thống đánh dấu `INCOMPATIBLE`, tuyệt đối không tự ý suy diễn Rank 2 < Rank 3 để cho phép).*

3. **Nguyên tắc Thiếu dữ liệu:** Nếu một trong hai linh kiện thiếu thông tin `form_factor` $\to$ Trả về `CompatibilityStatus.UNKNOWN`.

---

## 1.4. Giới hạn Không gian Vật lý (Clearance Constraints) & Fail-Safe

Hệ thống tuân thủ nghiêm ngặt triết lý:
$$\text{Không chứng minh được tương thích} \ne \text{Tương thích}$$

- **Chiều dài Card Đồ họa (GPU Length Clearance):**
  - Đối với GPU rời:
    $$\text{gpu.gpu\_length\_mm} \le \text{case.max\_gpu\_length\_mm}$$
  - Nếu `gpu.gpu_length_mm is None` hoặc `case.max_gpu_length_mm is None` $\to$ Trả về `CompatibilityStatus.UNKNOWN`.
  - Đối với GPU tích hợp (`gpu.is_integrated == True`): Bỏ qua kiểm tra chiều dài card.
- **Chiều cao Tản nhiệt CPU (Cooler Height Clearance):**
  $$\text{cooler.cooler\_height\_mm} \le \text{case.max\_cooler\_height\_mm}$$
  - Nếu `cooler.cooler_height_mm is None` hoặc `case.max_cooler_height_mm is None` $\to$ Trả về `CompatibilityStatus.UNKNOWN`.
- **Socket Tản nhiệt (Cooler Socket Support):**
  - Tản nhiệt rời phải hỗ trợ rõ ràng socket của CPU:
    $$\text{NormStr}(\text{cpu.socket}) \in \{\text{NormStr}(s) \mid s \in \text{cooler.supported\_sockets}\}$$
  - Nếu tản nhiệt không cung cấp danh sách socket hỗ trợ $\to$ Trả về `CompatibilityStatus.UNKNOWN`.

---

## 1.5. Năng lực Vật lý Tích hợp của CPU (iGPU & Stock Cooler Grounding)

Hệ thống cấm tuyệt đối việc tạo linh kiện giả định (Zero Spec Fabrication) khi danh mục thiếu dữ liệu. Cấu hình không card rời hoặc không tản rời chỉ được chấp nhận khi CPU mang đầy đủ dữ liệu thực chứng:

1. **Đồ họa Tích hợp (iGPU Option - 0 VNĐ):**
   Chỉ được phép tạo phương án GPU tích hợp khi CPU thỏa mãn đầy đủ 3 thuộc tính có căn cứ:
   - `cpu.has_integrated_graphics == True`
   - `cpu.integrated_graphics_score is not None`
   - `cpu.integrated_graphics_power_watts is not None`
2. **Tản nhiệt Kèm theo (Stock Cooler Option - 0 VNĐ):**
   Chỉ được phép tạo phương án tản nhiệt hộp khi CPU thỏa mãn đầy đủ 4 thuộc tính có căn cứ:
   - `cpu.includes_stock_cooler == True`
   - `cpu.stock_cooler_height_mm is not None`
   - `cpu.stock_cooler_tdp_watts is not None`
   - `cpu.stock_cooler_score is not None`
3. **Định danh Bất biến:** Cả hai phương án trên được định danh bằng UUIDv5 determinism gắn liền với CPU:
   `uuid5(SYNTHETIC_NAMESPACE, f"{stable_component_id(cpu)}:igpu")`
   `uuid5(SYNTHETIC_NAMESPACE, f"{stable_component_id(cpu)}:stock_cooler")`
