from uuid import uuid4

import pytest
from test_requirement_pipeline import requirement, rows, state

from ai_service.capabilities.assistant.requirements.build import (
    optimize_requirements,
    verify_receipt,
)
from ai_service.capabilities.assistant.requirements.contracts import (
    QuantityValue,
    ReferenceValue,
    TextValue,
)
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)


def accessory(catalog, *, kind="MOUSE", price=500_000):
    from uuid import uuid4

    from ai_service.application.ports.accessories import AccessorySpec

    identifier = uuid4()
    return catalog[0].model_copy(
        update={
            "variant_id": identifier,
            "component": AccessorySpec(
                id=identifier, category=kind, name="Canonical gear", price=price
            ),
        }
    )


def test_full_setup_owned_and_pinned_share_total_budget_and_publish_history(
    realistic_catalog,
):
    from ai_service.capabilities.assistant.requirements.contracts import (
        ReferenceValue,
        Resource,
    )
    from ai_service.capabilities.assistant.requirements.publication import (
        public_projection,
    )

    catalog = rows(realistic_catalog)
    mouse = accessory(catalog)
    monitor = accessory(catalog, kind="MONITOR", price=3_000_000)
    owned = requirement(
        "MOUSE", "use_resource", ReferenceValue(variant_id=mouse.variant_id)
    )
    canonical = state(
        requirement("SETUP", "scope", TextValue(text="FULL_SETUP")),
        owned,
        requirement(
            "MONITOR", "identity", ReferenceValue(variant_id=monitor.variant_id)
        ),
    )
    resource = Resource(
        id=uuid4(),
        variant_id=mouse.variant_id,
        evidence=owned.evidence,
    )
    canonical.available_resources[resource.id] = resource
    catalog += [mouse, monitor]
    receipt = optimize_requirements(canonical, catalog, PCBuildApplicationService())
    assert receipt.outcome == "BUILT"
    assert (
        receipt.compilation.optimizer_input.constraints.target_budget_vnd.value
        == 27_000_000
    )
    assert verify_receipt(canonical, receipt, catalog).hard_satisfied
    _, public = public_projection(canonical, receipt, None)
    assert public.total_spending_vnd == receipt.build.total_price + 3_000_000
    assert next(p for p in public.parts if p.slot == "MOUSE").spending_vnd == 0
    _, historical = public_projection(canonical, None, public)
    assert historical.stale and historical.parts == public.parts


def test_recommended_accessories_need_explicit_cap_not_implicit_split(
    realistic_catalog,
):
    catalog = rows(realistic_catalog)
    catalog.append(accessory(catalog))
    canonical = state(
        requirement("SETUP", "scope", TextValue(text="FULL_SETUP")),
        requirement("MOUSE", "selection", TextValue(text="RECOMMEND")),
    )
    receipt = optimize_requirements(canonical, catalog, PCBuildApplicationService())
    assert receipt.outcome == "CLARIFICATION"
    assert any(
        m.reason == "ACCESSORY_BUDGET_REQUIRED" for m in receipt.compilation.mapping
    )
    from ai_service.capabilities.assistant.requirements.publication import (
        public_projection,
    )

    consultation, public = public_projection(canonical, receipt, None)
    assert consultation.constraints.budget_scope.value == "FULL_SETUP"
    assert public is None


@pytest.mark.parametrize("predicate", ["brand", "model"])
def test_soft_brand_is_ranked_not_filtered_and_survives_dominance(
    realistic_catalog, predicate
):
    catalog = rows(realistic_catalog)
    ram = next(row for row in catalog if row.component.category == "RAM")
    # Equal hardware, but slightly dearer preferred brand would formerly be pruned.
    import uuid

    preferred_id = uuid.uuid4()
    preferred = ram.model_copy(
        update={
            "variant_id": preferred_id,
            "component": ram.component.model_copy(
                update={
                    "id": preferred_id,
                    "brand": "Preferred",
                    "name": "Preferred RAM",
                    "price": ram.component.price + 1,
                }
            ),
        }
    )
    catalog = [row for row in catalog if row.component.category != "RAM"] + [
        ram,
        preferred,
    ]
    canonical = state(
        requirement("RAM", predicate, TextValue(text="Preferred"), strength="SOFT")
    )
    receipt = optimize_requirements(canonical, catalog, PCBuildApplicationService())
    assert receipt.outcome == "BUILT"
    assert receipt.build.parts["RAM"].id == preferred_id
    assert {p.id for p in receipt.compilation.optimizer_input.candidates} >= {
        ram.variant_id,
        preferred_id,
    }
    assert verify_receipt(canonical, receipt, catalog).hard_satisfied


def test_unavailable_soft_preference_does_not_block_build(realistic_catalog):
    canonical = state(
        requirement("RAM", "brand", TextValue(text="NotInCatalog"), strength="SOFT")
    )
    catalog = rows(realistic_catalog)
    receipt = optimize_requirements(canonical, catalog, PCBuildApplicationService())
    assert receipt.outcome == "BUILT"
    report = verify_receipt(canonical, receipt, catalog)
    assert report.hard_satisfied
    assert any(
        not result.satisfied and result.strength == "SOFT" for result in report.results
    )
    from ai_service.capabilities.assistant.requirements.build import grounded_summary

    assert "Ưu tiên chưa đáp ứng: RAM (brand)" in grounded_summary(receipt, report)


def test_four_accessories_recommend_with_cap_and_never_change_core_slots(
    realistic_catalog,
):
    from ai_service.application.ports.accessories import AccessoryType
    from ai_service.capabilities.assistant.requirements.catalog_policy import (
        build_needs_catalog,
    )
    from ai_service.capabilities.assistant.requirements.publication import (
        public_projection,
    )

    catalog = rows(realistic_catalog)
    requests = [
        requirement("SETUP", "scope", TextValue(text="FULL_SETUP")),
        requirement(
            "SETUP",
            "accessory_budget",
            QuantityValue(amount=3_000_000, unit="VND"),
            operator="LTE",
        ),
    ]
    for slot in AccessoryType:
        requests.append(requirement(slot, "selection", TextValue(text="RECOMMEND")))
        catalog.append(accessory(catalog, kind=slot, price=500_000))
    preferred = accessory(catalog, kind="MOUSE", price=600_000)
    preferred.component.brand = "Preferred"
    catalog.append(preferred)
    requests.append(
        requirement("MOUSE", "brand", TextValue(text="Preferred"), strength="SOFT")
    )
    canonical = state(*requests)
    assert build_needs_catalog(canonical)
    receipt = optimize_requirements(canonical, catalog, PCBuildApplicationService())
    assert receipt.outcome == "BUILT"
    assert len(receipt.build.parts) == 8
    assert receipt.compilation.setup.spending == 2_100_000
    assert verify_receipt(canonical, receipt, catalog).hard_satisfied
    _, public = public_projection(canonical, receipt, None)
    assert len(public.parts) == 12
    assert (
        public.total_spending_vnd == receipt.build.total_price + 2_100_000 <= 30_000_000
    )
    assert (
        next(p for p in public.parts if p.slot == "MOUSE").variant_id
        == preferred.variant_id
    )
    # Independent verifier rejects a forged financial snapshot even if shape-valid.
    forged = receipt.model_copy(deep=True)
    forged.compilation.setup.accessories[0].row.component.price += 1
    with pytest.raises(ValueError, match="satisfaction"):
        verify_receipt(canonical, forged, catalog)


def test_soft_brand_cannot_override_hard_pin(realistic_catalog):
    catalog = rows(realistic_catalog)
    ram = next(row for row in catalog if row.component.category == "RAM")
    canonical = state(
        requirement("RAM", "identity", ReferenceValue(variant_id=ram.variant_id)),
        requirement("RAM", "brand", TextValue(text="Absent"), strength="SOFT"),
    )
    receipt = optimize_requirements(canonical, catalog, PCBuildApplicationService())
    assert receipt.outcome == "BUILT"
    assert receipt.build.parts["RAM"].id == ram.variant_id
    report = verify_receipt(canonical, receipt, catalog)
    assert report.hard_satisfied
    assert not next(r for r in report.results if r.strength == "SOFT").satisfied
    dimensions = [e for e in receipt.build.evidence if e.source == "scoring_breakdown"]
    assert sum(e.contribution for e in dimensions) == pytest.approx(
        receipt.build.objective_score
    )


def test_accessory_name_resolves_without_model_inventing_uuid(realistic_catalog):
    from ai_service.capabilities.assistant.requirements.contracts import (
        CanonicalRequirementState,
    )
    from ai_service.capabilities.assistant.requirements.resolver import (
        Draft,
        DraftItem,
        resolve_draft,
    )

    catalog = rows(realistic_catalog)
    mouse = accessory(catalog)
    catalog.append(mouse)
    message = "Giữ đúng Canonical gear"
    result = resolve_draft(
        Draft(
            items=[
                DraftItem(
                    subject="MOUSE",
                    predicate="identity",
                    value=TextValue(text="Canonical gear"),
                    start=9,
                    end=len(message),
                )
            ]
        ),
        message_id=uuid4(),
        message=message,
        state=CanonicalRequirementState(),
        catalog=catalog,
    )
    assert not result.pending
    assert result.patch.set_requirements[0].value.variant_id == mouse.variant_id


@pytest.mark.asyncio
async def test_react_build_tool_renders_full_setup_from_verified_receipt(
    realistic_catalog,
):
    from test_react_tools import Catalog, context

    from ai_service.capabilities.assistant.react.contracts import (
        AnswerCandidate,
        CompletionRequirement,
        ToolCall,
    )
    from ai_service.capabilities.assistant.react.tools import (
        ToolDispatcher,
        validate_answer,
    )

    catalog = rows(realistic_catalog)
    mouse = accessory(catalog)
    catalog.append(mouse)
    canonical = state(
        requirement("SETUP", "scope", TextValue(text="FULL_SETUP")),
        requirement("MOUSE", "identity", ReferenceValue(variant_id=mouse.variant_id)),
    )
    ctx = context(canonical, catalog)
    result = await ToolDispatcher(Catalog(), PCBuildApplicationService()).execute(
        ToolCall(call_id="build", name="build_pc", arguments={}),
        ctx,
    )
    assert result.status == "COMPLETED"
    answer = validate_answer(
        AnswerCandidate(
            text="not financial truth",
            build_ref=result.result.build_ref,
            evidence_refs=result.evidence_refs,
        ),
        [result],
        ctx,
        [CompletionRequirement(kind="BUILD", constraints_revision=canonical.revision)],
    )
    assert "MOUSE: Canonical gear" in answer.text
    total = result.result.receipt.build.total_price + 500_000
    assert f"{total:,} VND" in answer.text


@pytest.mark.parametrize("pinned", [False, True])
def test_chosen_accessory_allocation_is_not_global_infeasibility_proof(
    realistic_catalog, pinned
):
    from ai_service.capabilities.assistant.requirements.search_report import (
        search_report,
    )
    from ai_service.capabilities.pc_builder.optimizer import DeterministicPCOptimizer

    catalog = [
        row for row in rows(realistic_catalog) if row.component.category != "CPU"
    ]
    mouse = accessory(catalog)
    catalog.append(mouse)
    canonical = state(
        requirement("SETUP", "scope", TextValue(text="FULL_SETUP")),
        requirement(
            "SETUP",
            "accessory_budget",
            QuantityValue(amount=1_000_000, unit="VND"),
            operator="LTE",
        ),
        requirement(
            "MOUSE",
            "identity" if pinned else "selection",
            ReferenceValue(variant_id=mouse.variant_id)
            if pinned
            else TextValue(text="RECOMMEND"),
        ),
    )
    service = PCBuildApplicationService(DeterministicPCOptimizer(enable_pruning=False))
    receipt = optimize_requirements(canonical, catalog, service)
    assert receipt.outcome == "INFEASIBLE"
    result = search_report(canonical, catalog, service, receipt, build_ref=uuid4())
    assert result.outcome == ("INFEASIBLE" if pinned else "SEARCH_INCOMPLETE")
