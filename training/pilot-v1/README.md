# gearPC training pilot v1

Đọc [plan](../../../docs/04-ai/training-plan.md), sau đó [bảng scenario](SCENARIOS.md).
51 model-call samples / 50 scenario. Tất cả là **AUTHORED_SYNTHETIC, DRAFT**;
không phải dữ liệu khách hàng hay kiến thức hardware đã kiểm chứng.

## Các file

| File | Vai trò |
| --- | --- |
| manifest.json | Dataset/contract/source version và SHA-256 của hai data files |
| samples.jsonl | Một model-call sample mỗi dòng; target typed theo task |
| snapshots.json | Context và committed tool results trước model call; không fetch live |
| SCENARIOS.md | Bảng mô tả để reviewer đọc nhanh; JSONL giữ nhãn chính xác |

Mỗi sample giữ sample_id, scenario_id, split, snapshot_id, message,
expected_behavior, review_status và target. APPROVED yêu cầu reviewed_by.
scenario_id đại diện cùng nguồn/hành trình/paraphrases, không cấp ID mới chỉ để
đẩy một biến thể sang test. Không sửa message sau khi đã gắn source offsets.

## Hai kiểu nhãn điển hình

**req-02:** “RAM ít nhất 32GB.” → REQUIREMENT_EXTRACTION: subject RAM,
predicate capacity, GTE, HARD, QUANTITY32GB. Không chọn variant.

**react-11:** “Tìm BIOS của MSI MAG B650 TOMAHAWK WIFI.” → REACT_DECISION:
CALL_TOOL/research_driver với software_type BIOS_UPDATE. Không tự thêm OS/version
và không tải/cài BIOS.

Đây là canonical labels để validate, chưa là provider/tokenizer training format.
Exporter phải render đúng prompts/context như runtime và native tool protocol.
Không train JSON CALL_TOOL trực tiếp; không đưa owner/run IDs, future results,
review notes hoặc split vào model prompt chỉ vì chúng nằm trong dataset metadata.

Exporter và captured-response evaluator đã có; xem [workflow và lệnh chạy](../README.md).
Chúng bắt buộc review gate, không gọi model và không tự train. Canonical message
export vẫn cần model-specific chat template/schema mapping trước khi SFT.

## Kiểm tra từ ai-service/

```bash
uv run python -m ai_service.training training/pilot-v1
uv run pytest -q tests/test_training_dataset.py
```

Không cần API key, database, GPU hay training dependencies. Để xem labels dễ đọc:

```bash
uv run python -m json.tool --json-lines training/pilot-v1/samples.jsonl
```

Gate duyệt dữ liệu (hiện phải exit1 vì toàn bộ còn DRAFT):

```bash
uv run python -m ai_service.training training/pilot-v1 --require-approved
```

Sửa data phải cập nhật SHA-256 trong manifest và tạo version mới trước release.
Validator kiểm tra hashes/schema/policy/source quantities/current results và final
grounding. Nó không chứng minh prose/intent đúng, không tự review hay train.
train_ready chỉ tổng hợp review và split coverage; không chứng nhận chất lượng,
provider export hoặc khả năng triển khai model.

Pilot gồm train39/validation6/test6. Đây là phân chia để kiểm tra pipeline,
không phải bộ benchmark đủ lớn để tuyên bố chất lượng production.
