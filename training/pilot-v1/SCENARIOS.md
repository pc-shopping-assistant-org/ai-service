# Pilot scenarios — gearpc-pilot-v1

51 mẫu / 50 scenario; toàn bộ DRAFT, chưa được owner/BA duyệt.
Hai mẫu react-21/react-23 là hai bước cùng scenario build-demo, luôn ở cùng train split.
Context, giá, thông số và web results là dữ liệu mô phỏng; không phải benchmark/catalog thật.

Đọc hướng dẫn tại [README](README.md). Nhãn máy đọc: [samples.jsonl](samples.jsonl).
Một target là lựa chọn minh họa; không khẳng định mọi action khác đều sai.

## TASK_EXTRACTION

| ID | Split | User message | Hành vi cần duyệt |
| --- | --- | --- | --- |
| task-01 | train | Build PC 25 triệu. | Giữ task BUILD dù chưa rõ use case; không đổi thành ANSWER. |
| task-02 | train | Cho tôi PC làm Blender 30 triệu. | Đăng ký BUILD; chưa tự chọn linh kiện. |
| task-03 | train | So sánh Ryzen 5 7600 với Core i5-14400. | Hai mentions riêng, không tự gán catalog ID. |
| task-04 | train | Corsair RM850x có những ưu nhược điểm gì? | Câu hỏi đánh giá nguồn, không phải yêu cầu build. |
| task-05 | train | Tìm trang BIOS của MSI MAG B650 TOMAHAWK WIFI. | Đăng ký thông tin support, không thực hiện cài BIOS. |
| task-06 | train | Build PC 30 triệu và tìm benchmark GPU được chọn. | Hai task phải được giữ; build xong không đồng nghĩa benchmark xong. |
| task-07 | train | Giải thích cấu hình tôi đã chốt trước đó. | Đăng ký giải thích snapshot cũ, không tạo build mới. |
| task-08 | train | Blender cần cấu hình tối thiểu như thế nào? | Tìm requirements ứng dụng, không tự coi là task BUILD. |
| task-09 | validation | So sánh RTX 4060, RX 7600 và Arc A750. | Giữ đủ ba subject; resolver quyết định canonical IDs. |
| task-10 | test | Tìm thông số kích thước ASUS Dual RTX 4060 và cho tôi ưu nhược điểm. | Giữ hai nhu cầu thông tin, không tự chứng nhận card vừa case. |

## REQUIREMENT_EXTRACTION

| ID | Split | User message | Hành vi cần duyệt |
| --- | --- | --- | --- |
| req-01 | train | Ngân sách tối đa 25 triệu. | 25 triệu là budget cap, không phải giá target buộc tiêu hết. |
| req-02 | train | RAM ít nhất 32GB. | Hard capacity constraint, không pin SKU. |
| req-03 | train | SSD ít nhất 1TB. | Giữ TB theo source; compiler/resolver xử lý unit. |
| req-04 | train | Bắt buộc dùng CPU AMD. | HARD brand constraint vì user nói bắt buộc. |
| req-05 | train | Ưu tiên card ASUS. | SOFT preference, không khóa một variant. |
| req-06 | train | Tôi có sẵn ASUS Dual RTX 4060. | Owned resource mention; chưa có UUID và chưa tự trừ tiền. |
| req-07 | train | Giữ đúng ASUS TUF RTX 5070 Ti OC. | Exact identity requirement; catalog resolver mới được bind SKU. |
| req-08 | train | Đổi ngân sách lên 30 triệu. | Chỉ đề xuất sửa budget; không xóa RAM/preference cũ. |
| req-09 | train | Tăng RAM tối thiểu lên 64GB. | Sửa requirement RAM đã có, giữ budget. |
| req-10 | train | Bỏ ưu tiên ASUS. | Explicit CLEAR khác với không nhắc lại. |
| req-11 | train | Không cần giữ yêu cầu RAM 32GB nữa. | Xóa đúng requirement RAM, không xóa cả state. |
| req-12 | train | SSD đúng 512GB. | Dung lượng đúng như yêu cầu, không tự chọn sản phẩm. |
| req-13 | validation | RAM không quá 64GB. | Giữ đúng chiều LTE, không biến thành ít nhất. |
| req-14 | test | Cảm ơn bạn. | Không có yêu cầu sửa state; draft rỗng không xóa yêu cầu trước. |
| req-15 | train | Tối đa 28 triệu, RAM ít nhất 32GB. | Hai yêu cầu độc lập, source spans không kéo từ history. |
| req-16 | train | Ổ SSD tối đa 2TB. | Không đảo min/max capacity. |

## REACT_DECISION

| ID | Split | User message | Hành vi cần duyệt |
| --- | --- | --- | --- |
| react-01 | train | Tìm laptop làm việc văn phòng. | Khám phá catalog trước khi đòi đủ budget build. |
| react-02 | train | Có CPU AM5 nào đang bán? | Tra catalog cho dữ liệu đang bán; không dùng web để thay tồn kho. |
| react-03 | train | Xem chi tiết GPU mô phỏng A vừa tìm được. | Dùng đúng product ID từ result trước. |
| react-04 | train | So sánh hai GPU mô phỏng vừa tìm. | Hai product IDs đều từ snapshot; không dùng variant ID thay product ID. |
| react-05 | train | Tìm thông số chính hãng ASUS Dual RTX 4060. | Tra hãng trước; không coi READABLE là xác thực mọi thông số. |
| react-06 | train | Corsair RM850x có ripple và protection thế nào? | Chọn bài kiểm thử điện/nguồn, không chỉ benchmark GPU. |
| react-07 | train | Ryzen 5 7600 và MSI B650M GAMING PLUS có cần BIOS nào? | Nghiên cứu cặp model; chưa tự certify compatibility toàn build. |
| react-08 | train | RTX 4070 Super chơi Black Myth Wukong 1440p có benchmark không? | Không tự điền preset/upscaling còn thiếu. |
| react-09 | validation | Tìm benchmark RTX 4070 trong Blender. | Dùng productivity sources, không ép resolution/DLSS. |
| react-10 | train | Black Myth Wukong yêu cầu cấu hình tối thiểu gì? | Publisher/Steam requirements, không review sites. |
| react-11 | train | Tìm BIOS của MSI MAG B650 TOMAHAWK WIFI. | Chỉ đọc trang hãng, không download/install hay tự chọn OS. |
| react-12 | test | Driver cho AMD Radeon RX 7900 XT trên Windows 11 ở đâu? | OS do user cung cấp; không tự khẳng định phiên bản mới nhất. |
| react-13 | validation | Card của tôi có vừa case không? | Thiếu hai model; không đoán kích thước hoặc fit. |
| react-14 | test | Build PC gaming cho tôi. | Hỏi thông tin cần thiết, không tự chia budget hoặc chọn GPU. |
| react-15 | train | Tôi có sẵn GPU rồi, build phần còn lại. | Owned chưa rõ model không thành SKU pin mặc định. |
| react-16 | train | Build tối đa 25 triệu, RAM ít nhất 32GB. | Đề xuất user-sourced draft, không tự merge/publish. |
| react-17 | validation | Giảm chi phí có cách nào không? | Đây là agent proposal khác RAM user yêu cầu; application phải xin confirmation. |
| react-18 | train | Thông tin giá của hai GPU mô phỏng vừa tìm là gì? | Chọn evidence; server render giá, không tự viết lại số tiền. |
| react-19 | validation | Bài review mô phỏng nói gì về tiếng quạt? | Quote phải đúng đoạn đã đọc; không kết luận mọi sản phẩm cùng model đều như vậy. |
| react-20 | test | Đọc thông tin từ trang này giúp tôi. | UNREADABLE không chứng minh thông tin/sản phẩm không tồn tại. |
| react-21 | train | Build PC 30 triệu cho gaming 1080p. | Requirements đã compile-ready; gọi deterministic optimizer. |
| react-22 | train | Build PC 30 triệu gaming 1080p và nhận xét nguồn, độ ồn của cấu hình. | Dùng build_ref từ optimizer result hiện tại; research không đổi build. |
| react-23 | train | Build PC 30 triệu cho gaming 1080p. | Chỉ BUILT khi có receipt/verifier và đúng reference. |
| react-24 | train | Đọc trang https://example.com/gearpc/demo-review giúp tôi. | User đã cấp URL; đọc nội dung trước khi trích dẫn. |
| react-25 | test | Đổi ngân sách lên 28 triệu, giữ nguyên RAM. | Chỉ sửa budget; absence của RAM không là CLEAR. |
