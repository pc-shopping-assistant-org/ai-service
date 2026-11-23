"""Render build blockers using compiler diagnostics and canonical catalog data."""

from ai_service.application.ports.hardware import ComponentCategory
from ai_service.capabilities.assistant.provenance import CatalogRowV1
from ai_service.capabilities.assistant.requirements.catalog_policy import (
    build_clarification_items,
)
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.assistant.requirements.diagnostics import CompilationReason

from .actions import AskClarificationAction

INPUT_QUESTIONS: dict[CompilationReason, str] = {
    CompilationReason.BUDGET_REQUIRED: "Ngân sách tối đa của bạn cho bộ PC là bao nhiêu?",
    CompilationReason.USE_CASE_REQUIRED: "Bạn dùng PC chủ yếu để làm gì?",
    CompilationReason.ACCESSORY_BUDGET_REQUIRED: "Bạn muốn dành tối đa bao nhiêu cho phụ kiện?",
    CompilationReason.ACCESSORY_REQUIREMENTS_REQUIRED: "Bạn cần thêm những phụ kiện nào cho bộ setup?",
}
SPEC_LABELS: dict[tuple[str, str], str] = {
    (ComponentCategory.RAM, "capacity"): "dung lượng RAM",
    (ComponentCategory.STORAGE, "capacity"): "dung lượng ổ lưu trữ",
    (ComponentCategory.STORAGE, "media"): "loại ổ lưu trữ",
}
CHOICE_QUESTIONS: dict[CompilationReason, str] = {
    CompilationReason.ACCESSORY_SCOPE_CONFIRMATION: "Bạn muốn tính chung phụ kiện vào ngân sách bộ PC không?",
    CompilationReason.ACCESSORY_BUDGET_EXCEEDED: (
        "Các phụ kiện phù hợp vượt mức chi đã đặt. Bạn muốn điều chỉnh ngân sách "
        "hay xem lại danh sách phụ kiện?"
    ),
    CompilationReason.ACCESSORY_UNAVAILABLE_OR_METADATA_MISSING: (
        "Mình chưa xác nhận được phụ kiện nào đáp ứng yêu cầu. "
        "Bạn có muốn xem các lựa chọn khác không?"
    ),
    CompilationReason.VARIANT_NOT_RESOLVED: (
        "Mình chưa xác nhận được linh kiện bạn đã chọn trong catalog hiện tại. "
        "Bạn có muốn chọn lại mẫu không?"
    ),
    CompilationReason.UNRESOLVED_REQUIREMENTS: "Bạn có thể nói rõ phần yêu cầu còn bỏ ngỏ không?",
}


def build_clarification(
    state: CanonicalRequirementState,
    catalog: list[CatalogRowV1],
    *,
    catalog_ready: bool,
    selected_reasons: set[str] | None = None,
) -> AskClarificationAction | None:
    items = build_clarification_items(state, catalog, catalog_ready=catalog_ready)
    if selected_reasons is not None:
        items = [item for item in items if item.reason in selected_reasons]
    if not items:
        return None

    reasons = sorted({item.reason for item in items})
    questions = [text for reason, text in INPUT_QUESTIONS.items() if reason in reasons]
    if not questions and CompilationReason.PIN_UNAVAILABLE in reasons:
        unavailable = {
            variant_id
            for item in items
            if item.reason == CompilationReason.PIN_UNAVAILABLE
            for variant_id in item.target_bindings
        }
        part = next(
            (row.component.name for row in catalog if row.variant_id in unavailable),
            None,
        )
        label = part or "Linh kiện bạn muốn giữ"
        questions.append(f"{label} hiện hết hàng. Bạn có muốn xem mẫu thay thế không?")
    if not questions and CompilationReason.METADATA_INSUFFICIENT in reasons:
        label = next(
            (
                SPEC_LABELS[(item.subject, item.predicate)]
                for item in items
                if item.reason == CompilationReason.METADATA_INSUFFICIENT
                and item.subject is not None
                and item.predicate is not None
                and (item.subject, item.predicate) in SPEC_LABELS
            ),
            "thông số linh kiện bạn yêu cầu",
        )
        questions.append(
            f"Catalog chưa đủ dữ liệu để xác minh {label}. "
            "Bạn có muốn xem các lựa chọn có thông số đã xác minh không?"
        )
    if not questions:
        questions.append(
            next(
                (
                    text
                    for reason, text in CHOICE_QUESTIONS.items()
                    if reason in reasons
                ),
                "Mình cần làm rõ thêm yêu cầu để tiếp tục chọn cấu hình. Bạn có thể nói rõ hơn không?",
            )
        )
    return AskClarificationAction(
        question=" ".join(questions),
        missing_fields=[reason.value for reason in reasons],
    )
