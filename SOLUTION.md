# Đặc tả Giải pháp Kỹ thuật Toàn diện (System Solution Specification) — AI Service

Tài liệu này trình bày tổng quan và chi tiết về **Giải pháp Kỹ thuật (Technical Solution)** được thiết kế và triển khai trong Python AI Service (`ai-service`) nhằm giải quyết bài toán tư vấn mua sắm và tối ưu hóa cấu hình PC tự động cho dự án `pc-shopping-assistant-org`.

Tương tự như [DECISIONS.md](DECISIONS.md) (nơi lưu trữ các quyết định kiến trúc cụ thể ADR-001 đến ADR-012), tài liệu này tập trung vào **bức tranh giải pháp tổng thể**: từ việc định nghĩa bài toán cốt lõi, thách thức toán học/kỹ thuật, kiến trúc phân tầng, đến các thuật toán và bất biến bảo đảm hệ thống vận hành chính xác 100%, có thể kiểm chứng và tái lập.

---

## 1. Định nghĩa Bài toán & Thách thức Kỹ thuật (Problem Statement)

Bài toán xây dựng cấu hình PC tối ưu tự động từ danh mục bán hàng thực tế đối mặt với 5 thách thức kỹ thuật lớn:

```text
                                BÀI TOÁN XÂY DỰNG CẤU HÌNH PC
                                              │
      ┌──────────────────┬────────────────────┼───────────────────┬──────────────────┐
      ▼                  ▼                    ▼                   ▼                  ▼
1. Bùng nổ tổ hợp   2. Ràng buộc       3. Rủi ro ảo giác   4. Đa mục tiêu     5. Nghiệm biên &
   (Combinatorial      vật lý & điện       & bất định LLM     xung đột           Linh kiện sở hữu
   Explosion: 10¹²)    (Hard Fits)         (Hallucination)    (MAUT Trade-offs)  (0đ & owned_parts)
```

### 1.1. Thách thức 1: Tính bùng nổ tổ hợp (Combinatorial Explosion)
Một hệ thống PC hoàn chỉnh bao gồm 8 danh mục linh kiện bắt buộc:
$$\text{Build} = \{\text{CPU, Mainboard, RAM, GPU, Storage, PSU, Case, Cooler}\}$$
Trong một danh mục thương mại điện tử thực tế có quy mô trung bình:
$$N_{\text{CPU}} \approx 80, \quad N_{\text{MB}} \approx 120, \quad N_{\text{RAM}} \approx 60, \quad N_{\text{GPU}} \approx 90, \dots$$
Tổng không gian trạng thái tìm kiếm (Search Space) đạt mức:
$$\prod_{i=1}^{8} N_i \approx 10^{11} - 10^{13} \text{ cấu hình tiềm năng}$$
Nếu sử dụng phương pháp duyệt vét cạn (Brute-Force Enumeration), thời gian tính toán sẽ mất hàng giờ đến hàng ngày. Hệ thống yêu cầu phải phản hồi trong thời gian thực (**dưới 300ms - 500ms**) để phục vụ người dùng qua luồng Server-Sent Events (SSE).

### 1.2. Thách thức 2: Ràng buộc cơ học, vật lý & điện năng phức tạp (Hard Constraints)
Không phải linh kiện nào ghép với nhau cũng tạo thành một chiếc máy tính hoạt động được:
- **Tương thích cơ học & chân cắm:** CPU Socket AM5 không thể lắp trên Mainboard LGA1700; RAM DDR4 không thể cắm vào khe DDR5; Mainboard ATX không thể vừa vỏ case Mini-ITX.
- **Giới hạn không gian lắp đặt (Clearance Constraints):** Chiều dài GPU không được vượt quá khoảng trống bên trong case; chiều cao tản nhiệt khí tháp không được cấn nắp kính case.
- **Quy chuẩn điện toán & an toàn nguồn (Power Dynamics):** Tải đỉnh duy trì (Sustained Peak Load), hiện tượng quá dòng đột biến mili-giây (Transient Spikes) của GPU hiện đại, và giới hạn công suất nguồn thương mại (450W - 1200W).

### 1.3. Thách thức 3: Ảo giác và tính bất định của Mô hình Ngôn ngữ Lớn (LLM Pitfalls)
Nếu giao toàn bộ việc tư vấn cấu hình cho LLM:
- **Ảo giác thông số (Hallucination):** LLM tự bịa giá, tự bịa linh kiện không có trong kho, hoặc tự ghép CPU Intel đuôi F (không có iGPU) với cấu hình không card đồ họa rời.
- **Tính bất định (Non-Determinism):** Cùng một yêu cầu "Tư vấn máy 25 triệu chơi game", mỗi lần hỏi LLM lại trả về một cấu hình khác nhau, không có cơ chế chứng minh cấu hình nào là tối ưu nhất.
- **Tính toán sai số học:** LLM tính nhầm tổng giá tiền, nhầm công suất nguồn khuyến nghị.

### 1.4. Thách thức 4: Đa mục tiêu xung đột (Multi-Objective Optimization)
Các yêu cầu của khách hàng luôn mâu thuẫn nhau:
- **Hiệu năng (`PERFORMANCE`):** Đòi hỏi dồn tối đa ngân sách vào GPU/CPU, chấp nhận nguồn và vỏ case vừa đủ.
- **Cân đối (`BALANCED`):** Phân bổ hài hòa giữa hiệu năng, chất lượng linh kiện, khả năng tản nhiệt và độ bền.
- **Nâng cấp lâu dài (`UPGRADE_FRIENDLY`):** Ưu tiên socket mới (AM5/LGA1851), mainboard 4 khe RAM, nguồn dư công suất lớn để sau này cắm thêm linh kiện mà không phải thay toàn bộ máy.

### 1.5. Thách thức 5: Nghiệm biên 0 VNĐ & Linh kiện người dùng đã sở hữu sẵn
- Khách hàng văn phòng hoặc ngân sách hạn hẹp muốn tận dụng **iGPU** hoặc **Stock Cooler** kèm theo CPU (chi phí 0 VNĐ).
- Khách hàng nâng cấp máy có thể đã có sẵn card đồ họa hoặc nguồn (`owned_parts`).
- Nếu thuật toán sắp xếp ứng viên theo giá rồi ngắt nhánh tìm kiếm sớm (`break` khi vượt ngân sách), các phương án 0 VNĐ nếu bị xếp ở cuối sẽ bị bỏ qua hoàn toàn, dẫn đến kết luận sai rằng "không tìm thấy cấu hình phù hợp".

### 1.6. Mô hình hóa Toán học: Multiple-Choice Knapsack Problem with Pairwise Compatibility Constraints (MCKP-PCC)

> **Mô tả Học thuật Chuẩn mực (Formal Theoretical Descriptor):**  
> **"A Multiple-Choice Knapsack Problem with Pairwise Compatibility Constraints, solved using deterministic constrained search with Pareto pruning."**

Về mặt lý thuyết Khoa học Máy tính và Tối ưu hóa Tổ hợp (Operations Research), bài toán xây dựng cấu hình PC chính là một biến thể mở rộng của bài toán **Multiple-Choice Knapsack Problem with Pairwise Compatibility Constraints (MCKP-PCC / MCKPC)** kết hợp với **Multi-Attribute Utility Theory (MAUT)**:

1. **Các tập lớp rời nhau (Disjoint Classes):**
   Danh mục sản phẩm được phân hoạch thành $K = 8$ tập lớp rời nhau tương ứng 8 danh mục linh kiện bắt buộc:
   $$N_1 = \text{CPU}, \quad N_2 = \text{MB}, \quad N_3 = \text{RAM}, \quad N_4 = \text{GPU}, \quad N_5 = \text{SSD}, \quad N_6 = \text{PSU}, \quad N_7 = \text{Case}, \quad N_8 = \text{Cooler}$$
2. **Biến quyết định nhị phân (Binary Decision Variables):**
   $$x_{kj} \in \{0, 1\} \quad \forall k \in \{1, \dots, K\}, \, j \in N_k$$
   Trong đó $x_{kj} = 1$ nếu linh kiện thứ $j$ thuộc nhóm $k$ được lựa chọn, ngược lại bằng $0$.
3. **Ràng buộc Multiple-Choice (Chọn duy nhất 1 linh kiện mỗi slot):**
   $$\sum_{j \in N_k} x_{kj} = 1 \quad \forall k \in \{1, \dots, K\}$$
4. **Ràng buộc Sức chứa Ngân sách (Knapsack Budget Capacity):**
   $$\sum_{k=1}^{K} \sum_{j \in N_k} c_{kj} x_{kj} \le B_{\text{target}}$$
   *(Với $c_{kj}$ là giá tiền thực tế phải chi trả của linh kiện $j$ thuộc nhóm $k$, $c_{kj} = 0$ nếu là iGPU/stock cooler hoặc linh kiện sở hữu sẵn).*
5. **Ràng buộc Xung đột Đa chiều (Conflict Graph Constraints):**
   Định nghĩa đồ thị xung đột $G = (V, E)$ với tập đỉnh $V = \bigcup N_k$ và tập cạnh xung đột $E$. Nếu hai linh kiện $(u, v)$ không tương thích vật lý (khác Socket, khác chuẩn DDR, cấn chiều dài GPU, cấn chiều cao tản, không hỗ trợ form factor):
   $$x_u + x_v \le 1 \quad \forall (u, v) \in E$$
6. **Ràng buộc Ghép cặp Phi tuyến tính (Dynamic Coupled Constraint - Nguồn PSU):**
   Công suất nguồn không độc lập mà phụ thuộc vào trạng thái ghép cặp của CPU và GPU:
   $$\sum_{j \in N_{\text{PSU}}} \text{wattage}_j \cdot x_{\text{PSU}, j} \ge \text{round\_up\_psu}\Big(\big(P_{\text{cpu\_peak}}(\mathbf{x}) + P_{\text{gpu\_peak}}(\mathbf{x}) + 65.0\big) \times 1.10\Big)$$
7. **Hàm Mục tiêu Đa thuộc tính (Multi-Attribute Objective Function):**
   $$\max_{\mathbf{x}} \quad U(\mathbf{x}) = \sum_{m=1}^{M} W_m \cdot S_m(\mathbf{x})$$
   Thỏa mãn $\sum_{m=1}^M W_m = 1.0$ và $S_m(\mathbf{x}) \in [0.0, 100.0]$.

> **Độ phức tạp tính toán:** MCKPC thuộc lớp bài toán **NP-hard**. Ngay cả bài toán tìm một nghiệm khả thi thỏa mãn đồ thị xung đột $G$ đã tương đương bài toán *Independent Set*. Do đó, việc giải quyết triệt để trong $< 300\text{ms}$ đòi hỏi phương pháp tiếp cận kết hợp: **Pareto Reduction + Adaptive Envelopes + Constrained Branch-and-Bound**.

---

## 2. Kiến trúc Giải pháp Tổng thể (Architectural Blueprint)

Để giải quyết triệt để các thách thức trên, hệ thống được xây dựng trên một nguyên tắc cốt lõi:

> **"Tách biệt Tuyệt đối giữa Tầng Giao tiếp Xác suất (Probabilistic Interface) và Lõi Tối ưu Xác định (Deterministic Core Engine)"**

```mermaid
flowchart TD
    subgraph UserInterface [Tầng Giao tiếp & Thu nhận Yêu cầu]
        UserPrompt["Yêu cầu tự nhiên của khách hàng\n(VD: 'Build máy 25tr chơi game 1440p, tôi có sẵn nguồn 750W')"]
        LLMAgent["LLM Nodes (LangGraph + Pydantic)\n- Bóc tách yêu cầu khách quan\n- Nhận diện hồ sơ nhu cầu (UseCaseProfile)\n- Xác lập ConstraintSource (USER, SYSTEM, INFERRED)"]
        Constraints["PCBuildConstraints (Structured)\n- target_budget: 25.000.000 (USER)\n- use_case: GAMING_1440P\n- owned_parts: [PSU 750W (exclude_from_budget=True)]"]
    end

    subgraph DeterministicEngine [Lõi Tối ưu Xác định - Deterministic Core Engine]
        Catalog[(Active Store Catalog)]
        
        Step1["1. Candidate Retrieval & Adaptive Envelopes\n(Lọc linh kiện khả thi theo dải ngân sách hồ sơ)"]
        Step2["2. Profile-Aware Pareto Dominance Pruning\n(Cắt tỉa ứng viên bị thống trị toàn diện & Bảo toàn Feasibility)"]
        Step3["3. Branch-and-Bound Combinatorial Search\n(Duyệt có ràng buộc ngân sách & Bảo toàn nghiệm 0đ)"]
        Step4["4. Two-Tier Physical & Electrical Verification\n(Kiểm tra Socket, RAM, Form Factor, Clearance, PSU Sizing)"]
        Step5["5. Multi-Attribute Utility (MAUT) Scoring\n(Chấm điểm 18 chính sách chuẩn hóa & Deterministic Tie-Break)"]
    end

    subgraph GroundedExplanation [Tầng Thuyết minh Dựa trên Thực chứng]
        RankedResult["OptimizationResult (3 Cấu hình Chuẩn)\n- PERFORMANCE Build\n- BALANCED Build\n- UPGRADE_FRIENDLY Build\n(Kèm MetricEvidence & ScoreBreakdown chi tiết)"]
        ExplainerAgent["Grounding Explainer (LLM)\n- Thuyết minh dựa trên số liệu thực tế\n- Không bịa thông số, trích dẫn MetricEvidence"]
        SSEStream["Server-Sent Events (SSE) Stream\n- Token text giải thích\n- JSON structured build data"]
    end

    UserPrompt --> LLMAgent
    LLMAgent --> Constraints
    Constraints --> Step1
    Catalog --> Step1
    Step1 --> Step2
    Step2 --> Step3
    Step3 --> Step4
    Step4 --> Step5
    Step5 --> RankedResult
    RankedResult --> ExplainerAgent
    ExplainerAgent --> SSEStream
```

---

## 3. Chi tiết các Khối Thuật toán & Giải pháp Kỹ thuật

### 3.1. Khối 1: Khung Ngân sách Thích ứng 2 Tầng (Adaptive Two-Tier Envelopes)

- **Vấn đề:** Nếu lọc cứng catalog theo ngân sách tỷ lệ, có thể loại bỏ nhầm linh kiện giá rẻ tạo nên cấu hình tối ưu. Nếu không lọc gì cả, không gian tìm kiếm sẽ quá lớn.
- **Giải pháp:** Thiết kế cấu trúc `CandidatePool` gồm 2 danh sách riêng biệt:
  - `preferred`: Các ứng viên nằm trong dải tỷ lệ ngân sách chuẩn của hồ sơ (`preferred_min` đến `preferred_max`).
  - `all_affordable`: Toàn bộ các ứng viên có giá tiền mua được ($\le$ Ngân sách tổng).
- **Cơ chế thích ứng (Adaptive Fallback):**
  - Thuật toán ưu tiên thử nghiệm trên danh sách `preferred`.
  - Nếu số lượng ứng viên trong `preferred` không đủ tạo ra cấu hình khả thi, thuật toán tự động mở rộng sang dải mở rộng (`absolute_min` đến `absolute_max`), và cuối cùng là `all_affordable`.
  - **Bảo đảm:** Không bao giờ để mất cấu hình tối ưu toàn cục chỉ vì linh kiện nằm ngoài dải dự kiến ban đầu.

---

### 3.2. Khối 2: Cắt tỉa Pareto Đa chiều & Bảo toàn Năng lực Khả thi (Pareto Pruning)

- **Vấn đề:** Trong cùng một danh mục (ví dụ GPU), có nhiều model đắt hơn nhưng yếu hơn, hoặc cùng hiệu năng nhưng ăn điện hơn và to hơn. Cần loại bỏ chúng trước khi ghép tổ hợp để giảm không gian tìm kiếm từ $10^{12}$ xuống dưới $10^4$.
- **Quy tắc Pareto Cổ điển:** Linh kiện $B$ thống trị (dominates) $A$ khi $B$ có giá $\le A$ và trên mọi chiều chất lượng $B$ đều $\ge A$.
- **Cải tiến Kỹ thuật Đột phá của Giải pháp:**
  1. **Bảo toàn Năng lực Khả thi (Feasibility Capability Preservation):**
     - CPU không chỉ có điểm benchmark và điện năng (TDP). CPU còn quyết định sự tồn tại của **iGPU** và **Stock Cooler**.
     - Nếu một CPU $B$ rẻ hơn và benchmark cao hơn một CPU $A$, nhưng $A$ có iGPU còn $B$ thì không $\to$ **$B$ TUYỆT ĐỐI KHÔNG ĐƯỢC DOMINATE $A$**.
     - Lý do: Việc chọn CPU $A$ cho phép tạo ra cấu hình chạy được với GPU giá 0 VNĐ. Nếu loại bỏ $A$, hệ thống sẽ bị mất toàn bộ không gian nghiệm ngân sách thấp.
     - Hàm [preserves_cpu_optional_capabilities](file:///home/lamdx4/Projects/pc-shopping-assistant-org/ai-service/src/ai_service/capabilities/pc_builder/pruning.py#L149-L191) đảm bảo CPU $B$ chỉ có thể dominate $A$ khi $B$ cũng có iGPU/cooler với thông số không thua kém $A$.
  2. **Chính sách Bo mạch chủ Bảo thủ (Motherboard Conservative Policy):**
     - Không prune bo mạch chủ dựa trên giá tiền (`dimensions = []`) khi chưa có đầy đủ thông số phase nguồn VRM, tản nhiệt VRM và số cổng kết nối.
  3. **Nguyên tắc Fail-Safe trên Dữ liệu Thiếu:**
     - Nếu bất kỳ chiều đánh giá nào bị thiếu (`None`) trên một trong hai linh kiện $\to$ `dominates()` lập tức trả về `False`. Không bao giờ cắt tỉa trên cơ sở dữ liệu phỏng đoán.
  4. **Liên hiệp Nghiệm trên 3 Mục tiêu (Union across Objectives):**
     - Một linh kiện chỉ bị loại khỏi catalog khi nó bị thống trị trên **cả 3 mục tiêu** (`PERFORMANCE`, `BALANCED`, `UPGRADE_FRIENDLY`). Chỉ cần có giá trị trên ít nhất 1 mục tiêu, linh kiện sẽ được giữ lại (`pareto_prune_all_objectives`).

---

### 3.3. Khối 3: Tìm kiếm Nhánh - Cận với Bảo toàn Nghiệm 0 VNĐ & Linh kiện Có sẵn

- **Vấn đề:** Khi ghép 8 danh mục linh kiện, nếu chi phí hiện tại vượt quá ngân sách, thuật toán sẽ cắt tỉa nhánh (`if current_cost > budget: break`). Tuy nhiên, phương án iGPU hoặc Stock Cooler có giá 0 VNĐ. Nếu danh sách ứng viên chỉ sort theo giá trước khi thêm phương án 0 VNĐ, phương án 0 VNĐ nằm ở cuối sẽ bị `break` bỏ qua.
- **Giải pháp Toàn cục:**
  1. **Xử lý Linh kiện Sở hữu sẵn (`OwnedComponent`):**
     - Khi người dùng khai báo linh kiện có sẵn ở slot nào (ví dụ GPU hoặc nguồn), pool của slot đó được khóa cứng vào `[owned.component]`.
     - Nếu `exclude_from_budget == True`, linh kiện có sẵn đóng góp 0 VNĐ vào chi phí thực chi (`spending = 0`). `RankedBuild.total_price` phản ánh chính xác số tiền khách hàng cần thanh toán.
  2. **Bảo toàn Thứ tự Toàn cục (Global Price Sort Invariant):**
     - Sau khi bổ sung phương án synthetic 0 VNĐ (chỉ tạo khi CPU có dữ liệu grounded), danh sách ứng viên **bắt buộc phải được tái sắp xếp toàn cục**:
       $$\text{sort}(\text{key} = (\text{get\_spending\_price}(\text{item}), \text{stable\_component\_id}(\text{item})))$$
     - Đảm bảo phương án 0 VNĐ luôn nằm ở vị trí đầu tiên ($index = 0$), triệt tiêu 100% rủi ro bị `break` bỏ qua khi các card đồ họa rời giá cao vượt ngân sách.

---

### 3.4. Khối 4: Thẩm định Tương thích Vật lý & Điện toán 2 Lớp (Two-Tier Verification)

Tất cả các tổ hợp tìm được đều phải vượt qua bộ lọc tương thích nghiêm ngặt được quy định trong [01-physical-ground-truth.md](knowledge/01-physical-ground-truth.md) và [02-engineering-policies.md](knowledge/02-engineering-policies.md):

| Khía cạnh Kiểm tra | Quy tắc Kỹ thuật Thẩm định | Hành vi khi Thiếu dữ liệu |
| :----------------- | :------------------------- | :------------------------ |
| **CPU vs Mainboard Socket** | `NormStr(cpu.socket) == NormStr(mb.socket)` | `UNKNOWN` |
| **RAM vs Mainboard Type** | `NormStr(mb.ram_type) == NormStr(ram.ram_type)` | `UNKNOWN` |
| **Case vs Mainboard Size** | $\text{Rank}(mb) \le \max(\text{Rank}(\text{case})) \quad\mathbf{VÀ}\quad \text{NormStr}(mb.ff) \in \text{case.supported\_ff}$ | `UNKNOWN` (Strict Membership) |
| **GPU Length Clearance** | `gpu.gpu_length_mm <= case.max_gpu_length_mm` (chỉ xét card rời) | `UNKNOWN` |
| **Cooler Height Clearance** | `cooler.cooler_height_mm <= case.max_cooler_height_mm` | `UNKNOWN` |
| **Cooler Socket Support** | `NormStr(cpu.socket) in cooler.supported_sockets` | `UNKNOWN` |
| **PSU Minimum Capacity** | $\text{psu.wattage} \ge \text{round\_up\_psu}(P_{\text{sustained}} \times 1.10)$ | `INCOMPATIBLE` nếu thiếu tải |
| **PSU Transient Safety** | Phụ cấp đột biến $+30\%$ ($>280\text{W}$), $+20\%$ ($>180\text{W}$), $+12\%$ ($<180\text{W}$) | Tính vào Recommended PSU |
| **Overflow Guard** | $P_{\text{required}} > 1200\text{W} \to \text{Ném ngoại lệ } \mathbf{ValueError}$ | Ngăn chặn ép non tải nguy hiểm |

> **Nguyên tắc Fail-Safe Bất biến:** Nếu một cấu hình không chứng minh được tương thích (trả về `UNKNOWN`), hệ thống **tuyệt đối không đánh dấu là `COMPATIBLE`**. Chỉ các cấu hình đạt trạng thái `COMPATIBLE` mới được xem xét xếp hạng.

---

### 3.5. Khối 5: Chấm điểm Đa Mục tiêu MAUT & Phân định Hòa Tuyệt đối

- **Hàm Mục tiêu Tuyến tính Chuẩn hóa:**
  $$\text{Total Score} = \sum_{i=1}^{k} W_i \times S_i \quad \in [0.0, 100.0]$$
  Hệ thống quản lý 18 bộ trọng số độc lập tương ứng với $6 \text{ hồ sơ} \times 3 \text{ mục tiêu}$, được kiểm tra tự động $\sum W_i = 1.0$ khi khởi động ứng dụng.
- **Phân tách Khái niệm Tiết kiệm và Hiệu năng trên Giá thành:**
  - `performance_per_cost`: Đo lường mật độ hiệu năng trên mỗi triệu VNĐ bỏ ra.
  - `budget_saving`: Đo lường tỷ lệ ngân sách dôi dư chưa sử dụng.
- **Mô hình Nâng cấp 3 Chiều Không Chồng Chéo:**
  - Chiều 1: `platform_longevity` (AM5=100, LGA1851=90, LGA1700=40, AM4=30, LGA1200=20).
  - Chiều 2: `ram_slots` (4 khe = 100, 2 khe = 40).
  - Chiều 3: `psu_headroom` (Đo khoảng cách công suất nguồn thực tế so với công suất khuyến nghị).
- **Phân định Hòa Tuyệt đối (Deterministic Tie-Break):**
  Khi hai cấu hình có cùng điểm số và cùng mức giá:
  $$\text{TieKey} = (-\text{round}(\text{objective\_score}, 4), \text{total\_price}, \text{sorted\_parts\_stable\_ids})$$
  - Khi linh kiện thiếu ID (`None`), hệ thống sinh fingerprint ổn định từ **29 thuộc tính chức năng chuẩn** (category, brand, name, price, socket, ram_type, form_factor, wattage, vram, tdp, clearances...).
  - **Bảo đảm:** Bất kể catalog đầu vào bị xáo trộn (`shuffle`) theo thứ tự nào, thuật toán luôn tìm ra cùng một cấu hình tối ưu duy nhất (tái lập 100%).

---

## 4. Bảng So sánh Trước và Sau khi Áp dụng Giải pháp

| Tiêu chí Đánh giá | Cách làm Thông thường (Pure LLM / Heuristics) | Giải pháp AI Service (Deterministic Engine + Grounding) |
| :---------------- | :-------------------------------------------- | :------------------------------------------------------- |
| **Tính Đúng đắn Cơ - Điện** | Dễ sai sót socket, cấn tản, non tải nguồn | **Chính xác 100%**, có kiểm chứng ràng buộc vật lý |
| **Tốc độ Xử lý** | Chờ LLM sinh token (5s - 15s) | **Toán học tối ưu trong 50ms - 200ms** |
| **Khả năng Tái lập (Reproducibility)** | Thấp; hỏi lại lần 2 cho kết quả khác | **100% tái lập**, catalog xáo trộn vẫn ra 1 kết quả |
| **Bảo vệ Nghiệm Biên 0 VNĐ** | Thường bỏ quên iGPU / Stock cooler | **Bảo toàn toàn cục**, sort giá sau append synthetic |
| **Minh bạch Điểm số** | LLM giải thích chung chung, cảm tính | **Minh bạch từng chiều qua `ScoreBreakdown` & `MetricEvidence`** |
| **Quản trị Kiến thức** | Trộn lẫn quy tắc vật lý và kinh nghiệm | **Tách biệt 3 Tầng rõ ràng (Physical, Engineering, Optimization)** |

---

## 5. Bằng chứng Kiểm chứng Kỹ thuật (Empirical Verification)

Toàn bộ giải pháp đã được chứng minh và kiểm thử tự động với mức độ bao phủ toàn diện:

1. **45 Property-Based & Invariant Tests trong [test_pc_optimizer.py](tests/test_pc_optimizer.py):**
   - Kiểm thử tính bất biến của tương thích Socket, RAM DDR4/DDR5, kích thước Case.
   - Kiểm thử tính xác định khi xáo trộn catalog (`shuffle invariance`).
   - Kiểm thử công thức tải điện, quá tải $>1200\text{W}$ ném `ValueError`.
   - Kiểm thử bảo toàn năng lực CPU (iGPU, stock cooler) trong Pareto dominance.
   - Kiểm thử chống đụng độ định danh (identity collision) khi linh kiện không có ID.
   - Kiểm thử tính tương đương của kết quả khi bật và tắt pruning.
2. **Hệ thống Kiểm thử Toàn bộ `ai-service`:**
   - **116 passed tests** (100% pass, thời gian thực thi ~7 giây).
   - **Type Checking (Mypy):** Không có lỗi nào trên 80 file mã nguồn (`Strict Typing`).
   - **Linter (Ruff):** Đạt chuẩn 100% PEP 8 và Clean Code.
   - **Dependency Lock:** `uv lock --check` hoàn toàn nhất quán.
