from datetime import UTC, datetime
from uuid import uuid4

import pytest
from test_requirement_pipeline import rows, state

from ai_service.capabilities.assistant.react.contracts import (
    TOOL_SCHEMAS,
    AnswerCandidate,
    CompletionRequirement,
    ProductEvidence,
    ToolCall,
    ToolContext,
    ToolName,
)
from ai_service.capabilities.assistant.react.tools import (
    ToolDispatcher,
    allowed_tools,
    validate_answer,
)
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)


def context(requirements=None, catalog=None):
    return ToolContext(
        account_id=uuid4(),
        conversation_id=uuid4(),
        run_id=uuid4(),
        requirements=requirements or CanonicalRequirementState(),
        catalog=catalog or [],
        read_catalog_allowed=True,
        execution_allowed=True,
    )


class Catalog:
    def __init__(self):
        self.calls = 0
        self.product = ProductEvidence(
            product_id=uuid4(),
            name="Canonical GPU",
            price_vnd=8_000_000,
            evidence_ref="catalog:gpu:1",
            retrieved_at=datetime.now(UTC),
        )

    async def search(self, query, limit):
        self.calls += 1
        return [self.product]

    async def get(self, product_id):
        self.calls += 1
        return self.product if product_id == self.product.product_id else None


@pytest.mark.asyncio
async def test_consultation_explanation_keeps_canonical_facts_and_bound_sources():
    catalog, trusted = Catalog(), context()
    result = await ToolDispatcher(catalog, PCBuildApplicationService()).execute(
        ToolCall(
            call_id="details",
            name="get_product_details",
            arguments={"product_id": str(catalog.product.product_id)},
        ),
        trusted,
    )
    explanation = {
        "text": "Lựa chọn này cần cân nhắc cùng nhu cầu và các linh kiện bạn đã có.",
        "evidence_refs": result.evidence_refs,
    }
    candidate = AnswerCandidate(
        text="unused narration",
        evidence_refs=result.evidence_refs,
        explanation=explanation,
    )
    answer = validate_answer(
        candidate, [result], trusted, [CompletionRequirement(kind="ANSWER")]
    )
    assert "8,000,000 VND" in answer.text and explanation["text"] in answer.text
    assert "unused narration" not in answer.text
    forged = candidate.model_copy(deep=True)
    forged.explanation.evidence_refs = ["catalog:unread"]
    with pytest.raises(ValueError, match="Explanation"):
        validate_answer(
            forged, [result], trusted, [CompletionRequirement(kind="ANSWER")]
        )


@pytest.mark.asyncio
async def test_catalog_specs_render_as_readable_markdown_instead_of_python_repr():
    catalog, trusted = Catalog(), context()
    catalog.product.specifications = {
        "memory_gb": 32,
        "connections": ["USB-C", "3.5 mm"],
        "dimensions": {"width_mm": 100},
    }
    result = await ToolDispatcher(catalog, PCBuildApplicationService()).execute(
        ToolCall(
            call_id="details",
            name="get_product_details",
            arguments={"product_id": str(catalog.product.product_id)},
        ),
        trusted,
    )
    answer = validate_answer(
        AnswerCandidate(text="unused", evidence_refs=result.evidence_refs),
        [result],
        trusted,
        [CompletionRequirement(kind="ANSWER")],
    )
    assert "memory gb: 32" in answer.text
    assert "USB-C · 3.5 mm" in answer.text
    assert "width mm: 100" in answer.text
    assert "{'" not in answer.text and "['" not in answer.text


def test_state_aware_exposure_keeps_reads_and_requires_compiled_build(
    realistic_catalog,
):
    assert allowed_tools(context()) == (
        ToolName.SEARCH_CATALOG,
        ToolName.GET_PRODUCT_DETAILS,
        ToolName.COMPARE_PRODUCTS,
        ToolName.RESEARCH_HARDWARE,
        ToolName.RESEARCH_COMPATIBILITY,
        ToolName.RESEARCH_WORKLOAD,
        ToolName.RESEARCH_DRIVER,
    )
    ready = context(state(), rows(realistic_catalog))
    assert ToolName.BUILD_PC in allowed_tools(ready)
    for name in allowed_tools(ready):
        doc = TOOL_SCHEMAS[name].__doc__
        assert doc and doc.strip(), f"Missing docstring for {name}"
    assert allowed_tools(ready.model_copy(update={"execution_allowed": False})) == ()
    assert allowed_tools(ready.model_copy(update={"read_catalog_allowed": False})) == ()


@pytest.mark.asyncio
async def test_rejected_arguments_and_preconditions_never_invoke_handlers():
    catalog = Catalog()
    dispatcher = ToolDispatcher(catalog, PCBuildApplicationService())
    trusted = context()
    for name, arguments in [
        ("search_catalog", {"query": "GPU", "account_id": str(uuid4())}),
        ("compare_products", {"product_ids": [str(catalog.product.product_id)] * 2}),
        ("compare_products", {"queries": ["RTX 4070 Super", "  rtx 4070 super "]}),
        ("compare_products", {"queries": ["RTX 4070 Super"]}),
        ("build_pc", {}),
        ("shell", {}),
    ]:
        result = await dispatcher.execute(
            ToolCall(call_id=str(uuid4()), name=name, arguments=arguments), trusted
        )
        assert result.status == "REJECTED"
    assert catalog.calls == 0


@pytest.mark.asyncio
async def test_build_result_is_verified_and_generic_advice_cannot_complete_build(
    realistic_catalog,
):
    trusted = context(state(), rows(realistic_catalog))
    dispatcher = ToolDispatcher(Catalog(), PCBuildApplicationService())
    result = await dispatcher.execute(
        ToolCall(call_id="build-1", name="build_pc", arguments={}), trusted
    )
    assert result.status == "COMPLETED"
    assert result.result.receipt.outcome == "BUILT"
    assert result.result.satisfaction.hard_satisfied
    required = [
        CompletionRequirement(
            kind="BUILD", constraints_revision=trusted.requirements.revision
        )
    ]
    with pytest.raises(ValueError, match="completion"):
        validate_answer(AnswerCandidate(text="Ưu tiên GPU mạnh"), [], trusted, required)
    answer = validate_answer(
        AnswerCandidate(
            text="mọi game 2K đều mượt",
            evidence_refs=result.evidence_refs,
            build_ref=result.result.build_ref,
        ),
        [result],
        trusted,
        required,
    )
    assert "mọi game" not in answer.text
    assert "Tổng chi phí" in answer.text
    assert answer.status == "COMPLETED"


@pytest.mark.asyncio
@pytest.mark.parametrize("product_count", [2, 5])
async def test_comparison_cannot_complete_with_unselected_product_evidence(
    product_count,
):
    catalog = Catalog()
    products = [
        catalog.product.model_copy(
            update={
                "product_id": uuid4(),
                "name": f"GPU {index}",
                "evidence_ref": f"catalog:gpu:{index}",
            }
        )
        for index in range(product_count)
    ]

    class ComparisonCatalog:
        async def get(self, product_id):
            return next((p for p in products if p.product_id == product_id), None)

        async def search(self, query, limit):
            return products[:limit]

    trusted = context()
    ids = [p.product_id for p in products]
    result = await ToolDispatcher(
        ComparisonCatalog(), PCBuildApplicationService()
    ).execute(
        ToolCall(
            call_id="compare",
            name="compare_products",
            arguments={"product_ids": [str(identifier) for identifier in ids]},
        ),
        trusted,
    )
    required = [CompletionRequirement(kind="COMPARE", product_ids=ids)]
    with pytest.raises(ValueError, match="Comparison completion"):
        validate_answer(
            AnswerCandidate(
                text="Comparison done", evidence_refs=result.evidence_refs[:-1]
            ),
            [result],
            trusted,
            required,
        )
    answer = validate_answer(
        AnswerCandidate(text="Comparison done", evidence_refs=result.evidence_refs),
        [result],
        trusted,
        required,
    )
    assert answer.status == "COMPLETED"
    assert all(product.name in answer.text for product in products)


@pytest.mark.asyncio
async def test_comparison_requires_all_requested_products_and_only_server_renders_facts():
    catalog = Catalog()
    trusted = context()
    dispatcher = ToolDispatcher(catalog, PCBuildApplicationService())
    result = await dispatcher.execute(
        ToolCall(
            call_id="get-1",
            name="get_product_details",
            arguments={"product_id": str(catalog.product.product_id)},
        ),
        trusted,
    )
    assert result.status == "COMPLETED"
    missing = uuid4()
    with pytest.raises(ValueError, match="completion"):
        validate_answer(
            AnswerCandidate(text="Comparison done", evidence_refs=result.evidence_refs),
            [result],
            trusted,
            [
                CompletionRequirement(
                    kind="COMPARE", product_ids=[catalog.product.product_id, missing]
                )
            ],
        )
    with pytest.raises(ValueError, match="evidence"):
        validate_answer(
            AnswerCandidate(text="8 triệu", evidence_refs=["invented"]),
            [result],
            trusted,
            [CompletionRequirement(kind="ANSWER")],
        )
    answer = validate_answer(
        AnswerCandidate(
            text="invented marketing claim", evidence_refs=result.evidence_refs
        ),
        [result],
        trusted,
        [CompletionRequirement(kind="ANSWER")],
    )
    assert "8,000,000" in answer.text
    assert "invented" not in answer.text


@pytest.mark.asyncio
async def test_compare_products_resolves_keywords_in_single_turn_without_prior_search():
    catalog = Catalog()
    gpu_a = catalog.product.model_copy(
        update={
            "product_id": uuid4(),
            "name": "ASUS Dual GeForce RTX 4070 SUPER",
            "price_vnd": 16_990_000,
            "evidence_ref": "catalog:gpu:a",
        }
    )
    gpu_b = catalog.product.model_copy(
        update={
            "product_id": uuid4(),
            "name": "Sapphire PULSE Radeon RX 7800 XT",
            "price_vnd": 14_490_000,
            "evidence_ref": "catalog:gpu:b",
        }
    )

    class KeywordComparisonCatalog:
        async def get(self, product_id):
            return next(
                (p for p in (gpu_a, gpu_b) if p.product_id == product_id), None
            )

        async def search(self, query, limit):
            lowered = query.casefold()
            if "4070" in lowered:
                return [gpu_a][:limit]
            if "7800" in lowered:
                return [gpu_b][:limit]
            if "both" in lowered:
                return [gpu_a, gpu_b][:limit]
            return []

    trusted = context()
    dispatcher = ToolDispatcher(KeywordComparisonCatalog(), PCBuildApplicationService())

    # 1. Pure keyword comparison in a single tool call
    result = await dispatcher.execute(
        ToolCall(
            call_id="compare-keywords",
            name="compare_products",
            arguments={"queries": ["RTX 4070 Super", "RX 7800 XT"]},
        ),
        trusted,
    )
    assert result.status == "COMPLETED"
    assert [p.product_id for p in result.result.products] == [
        gpu_a.product_id,
        gpu_b.product_id,
    ]
    assert result.evidence_refs == ["catalog:gpu:a", "catalog:gpu:b"]

    answer = validate_answer(
        AnswerCandidate(
            text="So sánh 2 card đồ họa", evidence_refs=result.evidence_refs
        ),
        [result],
        trusted,
        [
            CompletionRequirement(
                kind="COMPARE", product_ids=[gpu_a.product_id, gpu_b.product_id]
            )
        ],
    )
    assert answer.status == "COMPLETED"
    assert gpu_a.name in answer.text and gpu_b.name in answer.text

    # 2. Mixed attached product_id + keyword query deduplicates already-selected product
    mixed = await dispatcher.execute(
        ToolCall(
            call_id="compare-mixed",
            name="compare_products",
            arguments={
                "product_ids": [str(gpu_a.product_id)],
                "queries": ["both"],
            },
        ),
        trusted,
    )
    assert mixed.status == "COMPLETED"
    assert [p.product_id for p in mixed.result.products] == [
        gpu_a.product_id,
        gpu_b.product_id,
    ]

    # 3. Missing keyword match returns PRODUCT_NOT_FOUND
    not_found = await dispatcher.execute(
        ToolCall(
            call_id="compare-missing",
            name="compare_products",
            arguments={"queries": ["RTX 4070 Super", "RTX 5090 Ti Missing"]},
        ),
        trusted,
    )
    assert not_found.status == "REJECTED"
    assert not_found.error == "PRODUCT_NOT_FOUND"

