# Qwen training assets

Hướng dẫn chuẩn: [Qwen baseline → SFT/QLoRA → evaluate → serving](../../docs/04-ai/training.md).
[Plan và tiến độ](../../docs/04-ai/training-plan.md) · [Evaluation rubric](../../docs/06-evaluation/evaluation-plan.md).

| Asset | Vai trò |
| --- | --- |
| qwen3-8b.json | Pinned model revision, batch/context/LoRA/decoding config |
| serving-requirements.txt | vLLM pin cho môi trường serving riêng |
| pilot-v1/ | 51 authored DRAFT labels, snapshots và manifest; chưa được duyệt |
| artifacts/ | Local outputs được git-ignore; không có sẵn trained weights |

Python commands nằm ở src/ai_service/training/: dataset validation, export,
inference, train, compare và serve. Normal CI không chạy GPU jobs.

Canonical exporter không xuất test targets thành SFT. Qwen preprocessing giữ
native chat template, schema context và loss ở completion cuối; không silently
truncate. Chỉ inference/train chủ động mới gọi serving/tải weights. serve mặc định
chỉ in command, cần --execute để chạy. Không có automatic approval/promotion.

Code tests không thay thế native tokenizer, GPU/vLLM smoke hoặc semantic review.

ReAct export hiện dùng `text-native-controls-v1`: plain text hoặc native control/tool
call, không JSON FlexibleDecision. Manifest ghi protocol; artifacts/adapter cũ cần
regenerate/retrain, không tự overwrite. Pilot react-13/14/15 còn DRAFT cần review
context/nhãn clarification; exporter chặn action không có trong serving tools.
