"""Public projection of verified requirements/build; never a second state authority."""

from ai_service.application.ports.hardware import ComponentCategory
from ai_service.capabilities.assistant.conversation_core import ConsultationStateV1
from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.public_contracts import (
    PublicBuildPartV1,
    PublicBuildV1,
)
from ai_service.capabilities.assistant.stateful_contracts import (
    OwnedPartRefV1,
    PinnedPartRefV1,
)
from ai_service.capabilities.pc_builder.schemas import (
    ConstraintSource,
    ConstraintValue,
    UseCaseProfile,
)

from .build import BuildReceipt
from .compiler import Compilation
from .contracts import CanonicalRequirementState, QuantityValue, TextValue


def public_projection(
    state: CanonicalRequirementState,
    receipt: BuildReceipt | None,
    previous: PublicBuildV1 | None,
    *,
    compilation: Compilation | None = None,
) -> tuple[ConsultationStateV1, PublicBuildV1 | None]:
    consultation = ConsultationStateV1(constraints_revision=state.revision)
    requirements = [*state.goals.values(), *state.requirements.values()]
    budgets = [
        r.value.amount
        for r in requirements
        if r.subject == "SETUP"
        and r.predicate == "budget"
        and r.operator == "LTE"
        and r.strength == "HARD"
        and isinstance(r.value, QuantityValue)
        and r.value.unit == "VND"
    ]
    if budgets:
        consultation.constraints.target_budget_vnd = ConstraintValue(
            value=min(budgets), source=ConstraintSource.USER
        )
    workloads = {
        r.value.text
        for r in requirements
        if r.subject == "WORKLOAD"
        and r.predicate == "use_case"
        and r.operator == "EQ"
        and r.strength == "HARD"
        and isinstance(r.value, TextValue)
    }
    if len(workloads) == 1:
        value = next(iter(workloads))
        if value in set(UseCaseProfile):
            consultation.constraints.use_case = ConstraintValue(
                value=UseCaseProfile(value), source=ConstraintSource.USER
            )
    if compilation is not None and compilation.state_hash != canonical_hash(state):
        raise ValueError("Projection compilation belongs to another requirement state")
    normalized = (
        receipt.compilation.optimizer_input
        if receipt
        else compilation.optimizer_input
        if compilation
        else None
    )
    if normalized is not None:
        consultation.owned_parts = {
            part.category: OwnedPartRefV1(variant_id=part.component.id)
            for part in normalized.constraints.owned_parts
            if part.component is not None and part.component.id is not None
        }
        consultation.pinned_parts = {
            slot: PinnedPartRefV1(variant_id=identity)
            for slot, identity in normalized.constraints.pinned_parts.items()
        }
    elif compilation is not None:
        # Partial requirements need no invented optimizer receipt. Only bindings
        # independently admitted by the compiler can enter this legacy view.
        conflicted = {
            item.subject for item in compilation.mapping if item.status == "CONFLICT"
        }
        for item in compilation.mapping:
            if (
                item.status != "APPLIED"
                or item.subject not in set(ComponentCategory)
                or item.subject in conflicted
                or len(item.target_bindings) != 1
            ):
                continue
            slot = ComponentCategory(item.subject)
            if item.predicate == "identity":
                consultation.pinned_parts[slot] = PinnedPartRefV1(
                    variant_id=item.target_bindings[0]
                )
            elif item.predicate == "use_resource":
                consultation.owned_parts[slot] = OwnedPartRefV1(
                    variant_id=item.target_bindings[0]
                )
    if receipt is None or receipt.outcome != "BUILT":
        consultation.build_stale = previous is not None
        historical = previous.model_copy(update={"stale": True}) if previous else None
        consultation.published_build_hash = (
            canonical_hash(historical) if historical else None
        )
        return ConsultationStateV1.model_validate(consultation.model_dump()), historical
    if normalized is None or receipt.build is None:
        raise ValueError("Built receipt has no verified optimizer input/result")
    build = receipt.build
    canonical_ids = {part.id for part in normalized.candidates}
    parts = []
    for slot, part in build.parts.items():
        synthetic = part.id not in canonical_ids
        if synthetic and slot not in {ComponentCategory.GPU, ComponentCategory.COOLER}:
            raise ValueError("Build has an ungrounded component")
        parts.append(
            PublicBuildPartV1(
                slot=slot,
                variant_id=None if synthetic else part.id,
                synthetic_kind=(
                    "INTEGRATED_GPU"
                    if slot == ComponentCategory.GPU
                    else "STOCK_COOLER"
                )
                if synthetic
                else None,
                name=part.name,
                spending_vnd=build.spending_prices[slot],
                is_owned=slot in consultation.owned_parts,
                is_pinned=slot in consultation.pinned_parts,
            )
        )
    published = PublicBuildV1(
        objective=build.objective,
        parts=parts,
        total_spending_vnd=build.total_price,
        stale=False,
    )
    consultation.published_build_hash = canonical_hash(published)
    return ConsultationStateV1.model_validate(consultation.model_dump()), published
