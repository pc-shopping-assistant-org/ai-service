from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from ai_service.capabilities.assistant.provenance import CatalogRowV1
from ai_service.capabilities.assistant.requirements.compiler import compile_requirements
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
    Evidence,
    QuantityValue,
    ReferenceValue,
    Requirement,
    RequirementPatch,
    Resource,
    TextValue,
)
from ai_service.capabilities.assistant.requirements.merge import merge_requirements
from ai_service.capabilities.assistant.requirements.resolver import (
    Draft,
    DraftItem,
    resolve_draft,
)
from ai_service.capabilities.assistant.requirements.verifier import verify_satisfaction
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)

MESSAGE = "Build 30 triệu gaming, RAM ít nhất 32GB."
MESSAGE_ID = uuid4()


def requirement(subject, predicate, value, *, operator="EQ", strength="HARD"):
    return Requirement(
        id=uuid4(),
        subject=subject,
        predicate=predicate,
        operator=operator,
        value=value,
        strength=strength,
        evidence=Evidence(
            message_id=MESSAGE_ID, start=0, end=len(MESSAGE), quote=MESSAGE
        ),
    )


def state(*extra):
    requirements = [
        requirement(
            "SETUP",
            "budget",
            QuantityValue(amount=30_000_000, unit="VND"),
            operator="LTE",
        ),
        *extra,
    ]
    goal = requirement("WORKLOAD", "use_case", TextValue(text="GAMING_1080P"))
    return CanonicalRequirementState(
        requirements={r.id: r for r in requirements},
        goals={goal.id: goal},
    )


def rows(catalog):
    return [
        CatalogRowV1(
            variant_id=(identifier := uuid4()),
            sku=f"SKU-{index}",
            product_status="ACTIVE",
            variant_status="ACTIVE",
            quantity=1,
            component=part.model_copy(update={"id": identifier}),
            retrieved_at=datetime.now(UTC),
            evidence_refs=["backend/catalog"],
        )
        for index, part in enumerate(catalog)
    ]


def test_atomic_merge_rejects_unbound_evidence_and_preserves_original():
    original = CanonicalRequirementState()
    valid = requirement(
        "RAM", "capacity", QuantityValue(amount=32, unit="GB"), operator="GTE"
    )
    forged = valid.model_copy(
        update={
            "id": uuid4(),
            "evidence": valid.evidence.model_copy(update={"quote": "invented"}),
        }
    )
    patch = RequirementPatch(set_requirements=[valid, forged])
    with pytest.raises(ValueError, match="evidence"):
        merge_requirements(original, patch, {MESSAGE_ID: MESSAGE})
    assert original.requirements == {} and original.revision == 0
    merged = merge_requirements(
        original, RequirementPatch(set_requirements=[valid]), {MESSAGE_ID: MESSAGE}
    )
    assert merged.requirements[valid.id] == valid
    assert merged.revision == 1
    assert merge_requirements(merged, RequirementPatch(), {}).revision == 1
    cleared = merge_requirements(
        merged,
        RequirementPatch(
            clear_requirements=[valid.id], clear_evidence={valid.id: valid.evidence}
        ),
        {MESSAGE_ID: MESSAGE},
    )
    assert cleared.requirements == {} and cleared.revision == 2


def test_quantity_and_state_reject_loose_values():
    with pytest.raises(ValidationError):
        QuantityValue(amount=True, unit="GB")
    with pytest.raises(ValidationError):
        CanonicalRequirementState(
            requirements={
                uuid4(): requirement(
                    "RAM", "capacity", QuantityValue(amount=32, unit="GB")
                )
            }
        )


def test_hard_capacity_filters_before_optimizer_and_report_reads_original(
    realistic_catalog,
):
    catalog = rows(realistic_catalog)
    capacity = requirement(
        "RAM", "capacity", QuantityValue(amount=32, unit="GB"), operator="GTE"
    )
    canonical = state(capacity)
    compiled = compile_requirements(canonical, catalog)
    assert compiled.ready and compiled.optimizer_input is not None
    normalized = compiled.optimizer_input
    assert all(
        p.capacity_gb >= 32 for p in normalized.candidates if p.category == "RAM"
    )
    result = (
        PCBuildApplicationService()
        .build_pc(normalized.constraints, normalized.candidates)
        .optimization
    )
    for build in result.builds.values():
        assert verify_satisfaction(
            canonical, compiled, build, catalog, observed_use_case=result.use_case
        ).hard_satisfied
    # Deliberately faulty compiler omits the RAM filter. Verifier must not trust it.
    faulty = compiled.model_copy(deep=True)
    faulty.optimizer_input.candidates = [
        row.component
        for row in catalog
        if row.component.category != "RAM" or row.component.capacity_gb == 16
    ]
    wrong_result = (
        PCBuildApplicationService()
        .build_pc(faulty.optimizer_input.constraints, faulty.optimizer_input.candidates)
        .optimization
    )
    for build in wrong_result.builds.values():
        report = verify_satisfaction(
            canonical, faulty, build, catalog, observed_use_case=wrong_result.use_case
        )
        assert "COMPILED_STATE_MISMATCH" not in report.errors
        assert "MAPPING_COVERAGE_MISMATCH" not in report.errors
        assert not next(
            item for item in report.results if item.requirement_id == capacity.id
        ).satisfied
        assert not report.hard_satisfied


def test_owned_resource_is_not_implicitly_selected_or_free(realistic_catalog):
    catalog = rows(realistic_catalog)
    gpu = next(row for row in catalog if row.component.category == "GPU")
    canonical = state()
    resource = Resource(
        id=uuid4(),
        variant_id=gpu.variant_id,
        evidence=requirement(
            "GPU", "identity", ReferenceValue(variant_id=gpu.variant_id)
        ).evidence,
    )
    canonical.available_resources[resource.id] = resource
    compiled = compile_requirements(canonical, catalog)
    assert compiled.optimizer_input.constraints.owned_parts == []
    assert (
        compiled.optimizer_input.effective_spending_vnd[gpu.variant_id]
        == gpu.component.price
    )


def test_unsupported_soft_model_preference_is_not_silently_dropped(realistic_catalog):
    preferred = requirement(
        "GPU", "model", TextValue(text="RTX 5070 Ti"), strength="SOFT"
    )
    compiled = compile_requirements(state(preferred), rows(realistic_catalog))
    assert not compiled.ready and compiled.optimizer_input is None
    assert (
        next(m for m in compiled.mapping if m.requirement_id == preferred.id).status
        == "UNSUPPORTED"
    )


def test_conflicting_identity_and_capacity_never_substitutes_pin(realistic_catalog):
    catalog = rows(realistic_catalog)
    ram = next(
        row
        for row in catalog
        if row.component.category == "RAM" and row.component.capacity_gb == 16
    )
    pin = requirement("RAM", "identity", ReferenceValue(variant_id=ram.variant_id))
    capacity = requirement(
        "RAM", "capacity", QuantityValue(amount=32, unit="GB"), operator="GTE"
    )
    compiled = compile_requirements(state(pin, capacity), catalog)
    assert not compiled.ready
    assert any(m.status == "CONFLICT" for m in compiled.mapping)


def test_units_are_explicit_and_missing_metadata_does_not_become_infeasible(
    realistic_catalog,
):
    catalog = rows(realistic_catalog)
    capacity = requirement(
        "STORAGE", "capacity", QuantityValue(amount=1, unit="TB"), operator="GTE"
    )
    compiled = compile_requirements(state(capacity), catalog)
    assert compiled.ready
    assert all(
        p.capacity_gb >= 1000
        for p in compiled.optimizer_input.candidates
        if p.category == "STORAGE"
    )
    for row in catalog:
        if row.component.category == "STORAGE":
            row.component.capacity_gb = None
    unavailable = compile_requirements(state(capacity), catalog)
    assert not unavailable.ready
    assert any(m.reason == "METADATA_INSUFFICIENT" for m in unavailable.mapping)


def test_resolver_never_guesses_variant_and_does_not_turn_preference_into_pin(
    realistic_catalog,
):
    catalog = rows(realistic_catalog)
    duplicate = catalog[0].model_copy(deep=True)
    duplicate.variant_id = uuid4()
    duplicate.component.id = duplicate.variant_id
    duplicate.sku = "DUPLICATE-SKU"
    name = catalog[0].component.name
    item = DraftItem(
        subject="CPU",
        predicate="identity",
        value=TextValue(text=name),
        start=0,
        end=len(name),
    )
    draft = Draft(items=[item])
    resolution = resolve_draft(
        draft,
        message_id=uuid4(),
        message=name,
        state=CanonicalRequirementState(),
        catalog=[*catalog, duplicate],
    )
    assert resolution.patch is None
    assert resolution.pending[0].reason == "AMBIGUOUS_VARIANT"
    single = resolve_draft(
        draft,
        message_id=uuid4(),
        message=name,
        state=CanonicalRequirementState(),
        catalog=catalog,
    )
    assert single.patch is not None
    assert single.patch.set_requirements[0].value.variant_id == catalog[0].variant_id
    preference = DraftItem(
        subject="GPU",
        predicate="model",
        strength="SOFT",
        value=TextValue(text="RTX 5070 Ti"),
        start=0,
        end=11,
    )
    preferred = resolve_draft(
        Draft(items=[preference]),
        message_id=uuid4(),
        message="RTX 5070 Ti",
        state=CanonicalRequirementState(),
        catalog=catalog,
    )
    assert preferred.patch.set_requirements[0].predicate == "model"
    assert preferred.patch.set_requirements[0].strength == "SOFT"
    assert isinstance(preferred.patch.set_requirements[0].value, TextValue)


def test_ambiguous_one_item_blocks_entire_patch_but_keeps_draft(realistic_catalog):
    message = "30 triệu và giữ unknown GPU"
    draft = Draft(
        items=[
            DraftItem(
                subject="SETUP",
                predicate="budget",
                operator="LTE",
                value=QuantityValue(amount=30_000_000, unit="VND"),
                start=0,
                end=8,
            ),
            DraftItem(
                subject="GPU",
                predicate="identity",
                value=TextValue(text="unknown GPU"),
                start=16,
                end=len(message),
            ),
        ]
    )
    resolution = resolve_draft(
        draft,
        message_id=uuid4(),
        message=message,
        state=CanonicalRequirementState(),
        catalog=rows(realistic_catalog),
    )
    assert resolution.patch is None and resolution.draft == draft
    assert resolution.pending[0].reason == "VARIANT_NOT_FOUND"


def test_verifier_rejects_missing_parts_and_false_spending(realistic_catalog):
    catalog = rows(realistic_catalog)
    canonical = state()
    compiled = compile_requirements(canonical, catalog)
    result = (
        PCBuildApplicationService()
        .build_pc(
            compiled.optimizer_input.constraints, compiled.optimizer_input.candidates
        )
        .optimization
    )
    build = next(iter(result.builds.values())).model_copy(deep=True)
    build.parts.pop("RAM")
    assert not verify_satisfaction(
        canonical, compiled, build, catalog, observed_use_case=result.use_case
    ).hard_satisfied
    # A malicious output cannot claim an owned resource was selected/free.
    build = next(iter(result.builds.values())).model_copy(deep=True)
    gpu = build.parts["GPU"]
    build.spending_prices["GPU"] = 0
    build.total_price -= gpu.price
    assert not verify_satisfaction(
        canonical, compiled, build, catalog, observed_use_case=result.use_case
    ).hard_satisfied


def test_verifier_accepts_parent_grounded_stock_cooler(realistic_catalog):
    catalog = rows(realistic_catalog)
    cpu = next(row for row in catalog if row.component.name == "Intel Core i5-12400F")
    canonical = state(
        requirement("CPU", "identity", ReferenceValue(variant_id=cpu.variant_id))
    )
    compiled = compile_requirements(canonical, catalog)
    result = (
        PCBuildApplicationService()
        .build_pc(
            compiled.optimizer_input.constraints, compiled.optimizer_input.candidates
        )
        .optimization
    )
    assert any(
        build.parts["COOLER"].id not in {row.variant_id for row in catalog}
        for build in result.builds.values()
    )
    assert all(
        verify_satisfaction(
            canonical, compiled, build, catalog, observed_use_case=result.use_case
        ).hard_satisfied
        for build in result.builds.values()
    )


def test_owned_reference_is_resolved_only_when_explicitly_requested(realistic_catalog):
    catalog = rows(realistic_catalog)
    row = next(row for row in catalog if row.component.category == "GPU")
    canonical = state()
    resource_id = uuid4()
    canonical.available_resources[resource_id] = Resource(
        id=resource_id,
        variant_id=row.variant_id,
        evidence=requirement(
            "GPU", "identity", ReferenceValue(variant_id=row.variant_id)
        ).evidence,
    )
    name = row.component.name
    draft = Draft(
        items=[
            DraftItem(
                subject="GPU",
                predicate="use_resource",
                value=TextValue(text=name),
                start=0,
                end=len(name),
            )
        ]
    )
    resolution = resolve_draft(
        draft,
        message_id=(message_id := uuid4()),
        message=name,
        state=canonical,
        catalog=catalog,
    )
    assert resolution.patch is not None
    merged = merge_requirements(canonical, resolution.patch, {message_id: name})
    compiled = compile_requirements(merged, catalog)
    assert (
        compiled.optimizer_input.constraints.owned_parts[0].component.id
        == row.variant_id
    )
    assert compiled.optimizer_input.effective_spending_vnd[row.variant_id] == 0


def test_followup_requires_explicit_replacement_target(realistic_catalog):
    existing = requirement("RAM", "capacity", QuantityValue(amount=16, unit="GB"))
    canonical = state(existing)
    message = "32 GB"
    item = DraftItem(
        subject="RAM",
        predicate="capacity",
        value=QuantityValue(amount=32, unit="GB"),
        start=0,
        end=len(message),
    )
    unresolved = resolve_draft(
        Draft(items=[item]),
        message_id=uuid4(),
        message=message,
        state=canonical,
        catalog=rows(realistic_catalog),
    )
    assert (
        unresolved.patch is None
        and unresolved.pending[0].reason == "REPLACEMENT_TARGET_REQUIRED"
    )
    item.target_id = existing.id
    resolved = resolve_draft(
        Draft(items=[item]),
        message_id=uuid4(),
        message=message,
        state=canonical,
        catalog=rows(realistic_catalog),
    )
    assert resolved.patch.set_requirements[0].id == existing.id


def test_numeric_requirement_must_match_source_quantity_and_unit():
    message = "25 triệu"
    draft = Draft(
        items=[
            DraftItem(
                subject="SETUP",
                predicate="budget",
                operator="LTE",
                value=QuantityValue(amount=30_000_000, unit="VND"),
                start=0,
                end=len(message),
            )
        ]
    )
    result = resolve_draft(
        draft,
        message_id=uuid4(),
        message=message,
        state=CanonicalRequirementState(),
        catalog=[],
    )
    assert (
        result.patch is None
        and result.pending[0].reason == "QUANTITY_EVIDENCE_MISMATCH"
    )


def test_decimal_nominal_capacity_normalizes_without_conflating_binary_units():
    message = "SSD 1,5 TB"
    draft = Draft(
        items=[
            DraftItem(
                subject="STORAGE",
                predicate="capacity",
                value=QuantityValue(amount=1500, unit="GB"),
                start=0,
                end=len(message),
            )
        ]
    )
    result = resolve_draft(
        draft,
        message_id=uuid4(),
        message=message,
        state=CanonicalRequirementState(),
        catalog=[],
    )
    assert result.patch.set_requirements[0].value == QuantityValue(
        amount=1500, unit="GB"
    )
    message = "SSD 1 TiB"
    draft.items[0].end = len(message)
    result = resolve_draft(
        draft,
        message_id=uuid4(),
        message=message,
        state=CanonicalRequirementState(),
        catalog=[],
    )
    assert (
        result.patch is None
        and result.pending[0].reason == "QUANTITY_EVIDENCE_MISMATCH"
    )


def test_capacity_evidence_with_ambiguous_units_requires_narrower_span():
    message = "RAM 32 GB và SSD 1 TB"
    draft = Draft(
        items=[
            DraftItem(
                subject="RAM",
                predicate="capacity",
                value=QuantityValue(amount=32, unit="GB"),
                start=0,
                end=len(message),
            )
        ]
    )
    result = resolve_draft(
        draft,
        message_id=uuid4(),
        message=message,
        state=CanonicalRequirementState(),
        catalog=[],
    )
    assert (
        result.patch is None
        and result.pending[0].reason == "QUANTITY_EVIDENCE_AMBIGUOUS"
    )


def test_state_and_patch_ids_are_unique_across_requirements_goals_resources():
    constraint = requirement("RAM", "capacity", QuantityValue(amount=32, unit="GB"))
    resource = Resource(
        id=constraint.id, variant_id=uuid4(), evidence=constraint.evidence
    )
    with pytest.raises(ValidationError, match="identities must be unique"):
        CanonicalRequirementState(
            requirements={constraint.id: constraint},
            available_resources={resource.id: resource},
        )
    with pytest.raises(ValidationError, match="Duplicate patch identity"):
        RequirementPatch(set_requirements=[constraint], set_resources=[resource])


def test_binary_alternative_is_not_filtered_out_before_capacity_ambiguity_check():
    from ai_service.capabilities.assistant.requirements.units import verify_quantity

    normalized, reason = verify_quantity(
        "32 GB hoặc 64 GiB", QuantityValue(amount=32, unit="GB")
    )
    assert normalized is None and reason == "QUANTITY_EVIDENCE_AMBIGUOUS"
