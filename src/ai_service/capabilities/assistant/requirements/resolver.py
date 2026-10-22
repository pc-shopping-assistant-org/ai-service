"""Evidence-bound draft resolution. Only explicit identity mentions bind a SKU.

The caller supplies revalidated canonical candidates; search rank is not a
uniqueness proof. An exact name with two variants still requires clarification.
"""

from typing import Annotated, Literal
from uuid import UUID, uuid5

from pydantic import Field, StrictInt

from ai_service.capabilities.assistant.provenance import CatalogRowV1
from ai_service.capabilities.assistant.stateful_contracts import (
    ClarificationReplyV1,
    ClarificationSelectionV1,
    Contract,
    NonNegativeInt,
    Text,
)

from .contracts import (
    CanonicalRequirementState,
    Evidence,
    QuantityValue,
    ReferenceValue,
    Requirement,
    RequirementPatch,
    Resource,
    Subject,
    TextValue,
)
from .tasks import TaskDraft
from .units import (
    UNIT_ALIASES,
    QuantityConfirmation,
    UnresolvedQuantityValue,
    confirmed_quantity,
    unresolved_amount,
    verify_quantity,
)


class DraftItem(Contract):
    target: Literal["REQUIREMENT", "GOAL", "RESOURCE"] = "REQUIREMENT"
    operation: Literal["SET", "CLEAR"] = "SET"
    target_id: UUID | None = None
    subject: Subject
    predicate: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
    operator: Literal["EQ", "GTE", "LTE"] = "EQ"
    strength: Literal["HARD", "SOFT"] = "HARD"
    value: (
        Annotated[
            QuantityValue | UnresolvedQuantityValue | TextValue,
            Field(discriminator="kind"),
        ]
        | None
    ) = None
    start: StrictInt = Field(ge=0)
    end: StrictInt = Field(gt=0)


class Draft(Contract):
    """Legacy combined payload; retained only for persisted state/old graph reads."""

    schema_version: Literal[1] = 1
    items: list[DraftItem] = Field(max_length=100)
    tasks: list[TaskDraft] = Field(default_factory=list, max_length=8)


class RequirementDraft(Contract):
    """Model proposal containing requirement items only."""

    schema_version: Literal[2] = 2
    items: list[DraftItem] = Field(max_length=100)


# Historical checkpoints may still contain the V1 combined shape. Current
# requirement/application ports accept RequirementDraft only.
type PersistedRequirementDraft = RequirementDraft | Draft


class Pending(Contract):
    item_index: int
    reason: Text
    candidate_ids: list[UUID] = Field(default_factory=list, max_length=20)


class Resolution(Contract):
    # Draft is accepted only to reopen historical V1 persisted artifacts.
    draft: PersistedRequirementDraft
    patch: RequirementPatch | None
    pending: list[Pending] = Field(default_factory=list)


class PendingDraft(Contract):
    message_id: UUID
    message: Text = Field(max_length=4000)
    constraints_revision: NonNegativeInt
    resolution: Resolution
    selections: dict[int, UUID] = Field(default_factory=dict)
    quantity_confirmations: dict[int, QuantityConfirmation] = Field(
        default_factory=dict, max_length=100
    )
    item_sources: dict[int, "DraftSource"] = Field(
        default_factory=dict, max_length=100, exclude_if=lambda value: not value
    )
    discarded_items: list[NonNegativeInt] = Field(
        default_factory=list, max_length=100, exclude_if=lambda value: not value
    )

    def source_for(self, item_index: int) -> "DraftSource":
        return self.item_sources.get(
            item_index, DraftSource(message_id=self.message_id, message=self.message)
        )

    def source_messages(self) -> dict[UUID, str]:
        messages = {self.message_id: self.message}
        for source in [
            *self.item_sources.values(),
            *(
                DraftSource(message_id=item.evidence.message_id, message=item.message)
                for item in self.quantity_confirmations.values()
            ),
        ]:
            if (
                source.message_id in messages
                and messages[source.message_id] != source.message
            ):
                raise ValueError("Conflicting draft source messages")
            messages[source.message_id] = source.message
        return messages


class DraftSource(Contract):
    message_id: UUID
    message: Text = Field(max_length=4000)


class ChoiceSelection(ClarificationSelectionV1):
    start: StrictInt = Field(ge=0)
    end: StrictInt = Field(gt=0)
    quote: Text = Field(max_length=4000)


class ChoiceDraft(Contract):
    selections: list[ChoiceSelection] = Field(default_factory=list, max_length=100)


def choice_reply(
    choice: ChoiceDraft, message: str, pending: PendingDraft
) -> ClarificationReplyV1 | None:
    choice = ChoiceDraft.model_validate(choice.model_dump())
    for item in choice.selections:
        if (
            item.end <= item.start
            or item.end > len(message)
            or message[item.start : item.end] != item.quote
        ):
            raise ValueError("Unbound clarification choice evidence")
    if not choice.selections:
        return None
    return ClarificationReplyV1(
        pending_message_id=pending.message_id,
        selections=[
            ClarificationSelectionV1(
                item_index=item.item_index, variant_id=item.variant_id
            )
            for item in choice.selections
        ],
    )


def selection_for_pending(
    pending: PendingDraft, reply: ClarificationReplyV1, state: CanonicalRequirementState
) -> dict[int, UUID]:
    if (
        pending.message_id != reply.pending_message_id
        or pending.constraints_revision != state.revision
    ):
        raise ValueError("Stale clarification source/revision")
    choices = {item.item_index: item for item in pending.resolution.pending}
    selected = dict(pending.selections)
    for selection in reply.selections:
        item = choices.get(selection.item_index)
        if item is None or selection.variant_id not in item.candidate_ids:
            raise ValueError("Invalid clarification candidate")
        selected[selection.item_index] = selection.variant_id
    return selected


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def resolve_draft(
    draft: PersistedRequirementDraft,
    *,
    message_id: UUID,
    message: str,
    state: CanonicalRequirementState,
    catalog: list[CatalogRowV1],
    selections: dict[int, UUID] | None = None,
    quantity_confirmations: dict[int, QuantityConfirmation] | None = None,
    item_sources: dict[int, DraftSource] | None = None,
    discarded_items: list[int] | None = None,
) -> Resolution:
    if isinstance(draft, RequirementDraft):
        draft = RequirementDraft.model_validate(draft.model_dump())
    else:
        draft = Draft.model_validate(draft.model_dump())
        for task in draft.tasks:
            task.validate_source(message)
    state = CanonicalRequirementState.model_validate(state.model_dump())
    catalog = [CatalogRowV1.model_validate(row.model_dump()) for row in catalog]
    if len({row.variant_id for row in catalog}) != len(catalog):
        raise ValueError("Duplicate canonical variant identity")
    patch = RequirementPatch()
    pending: list[Pending] = []
    discarded = set(discarded_items or [])
    if any(
        index < 0 or index >= len(draft.items)
        for index in discarded | set(item_sources or {})
    ):
        raise ValueError("Invalid draft item source/discard index")
    for index, item in enumerate(draft.items):
        if index in discarded:
            continue
        source = (item_sources or {}).get(
            index, DraftSource(message_id=message_id, message=message)
        )
        if item.end <= item.start or item.end > len(source.message):
            raise ValueError("Invalid extraction evidence span")
        quote = source.message[item.start : item.end]
        evidence = Evidence(
            message_id=source.message_id, start=item.start, end=item.end, quote=quote
        )
        identifier = item.target_id or uuid5(message_id, f"{item.target}:{index}")
        existing_ids = {
            "REQUIREMENT": set(state.requirements),
            "GOAL": set(state.goals),
            "RESOURCE": set(state.available_resources),
        }[item.target]
        if item.target_id is not None and item.target_id not in existing_ids:
            raise ValueError("Extraction cannot invent an existing target identity")
        if item.operation == "CLEAR":
            if item.value is not None or item.target_id is None:
                raise ValueError("CLEAR requires existing target and no value")
            patch.clear_evidence[identifier] = evidence
            if item.target == "REQUIREMENT":
                patch.clear_requirements.append(identifier)
            elif item.target == "GOAL":
                patch.clear_goals.append(identifier)
            else:
                patch.clear_resources.append(identifier)
            continue
        if item.value is None:
            raise ValueError("SET requires a value")
        value: QuantityValue | UnresolvedQuantityValue | TextValue | ReferenceValue = (
            item.value
        )
        if isinstance(value, UnresolvedQuantityValue):
            if unresolved_amount(quote) != value.amount:
                pending.append(
                    Pending(item_index=index, reason="QUANTITY_EVIDENCE_MISMATCH")
                )
                continue
            confirmation = (quantity_confirmations or {}).get(index)
            if confirmation is None:
                pending.append(
                    Pending(item_index=index, reason="QUANTITY_UNIT_REQUIRED")
                )
                continue
            alias = UNIT_ALIASES.get(confirmation.evidence.quote.strip().casefold())
            if alias is None:
                raise ValueError("Invalid quantity confirmation")
            confirmed = confirmed_quantity(
                value.amount, confirmation.evidence.quote, alias[0]
            )
            if confirmed != confirmation.value:
                raise ValueError("Invalid quantity confirmation value")
            evidence.supporting_evidence = [confirmation.evidence]
            value = confirmed
        elif isinstance(value, QuantityValue):
            normalized, reason = verify_quantity(quote, value)
            if normalized is None:
                pending.append(Pending(item_index=index, reason=reason))
                continue
            value = normalized
        if (
            item.target_id is None
            and item.target in ("REQUIREMENT", "GOAL")
            and any(
                existing.subject == item.subject
                and existing.predicate == item.predicate
                for existing in (
                    state.goals.values()
                    if item.target == "GOAL"
                    else state.requirements.values()
                )
            )
        ):
            pending.append(
                Pending(item_index=index, reason="REPLACEMENT_TARGET_REQUIRED")
            )
            continue
        if item.predicate in ("identity", "use_resource") or item.target == "RESOURCE":
            if (
                not isinstance(value, TextValue)
                or item.operator != "EQ"
                or item.strength != "HARD"
            ):
                pending.append(
                    Pending(item_index=index, reason="IDENTITY_SEMANTICS_REQUIRED")
                )
                continue
            text = _normalized(value.text)
            if text not in _normalized(quote):
                raise ValueError("Identity mention absent from source evidence")
            candidates = [
                row
                for row in catalog
                if row.component.category == item.subject
                and (
                    text in _normalized(row.component.name)
                    or text == _normalized(row.sku)
                )
            ]
            exact = [
                row
                for row in candidates
                if text in (_normalized(row.component.name), _normalized(row.sku))
            ]
            selected = (selections or {}).get(index)
            if selected is not None:
                if selected not in {row.variant_id for row in candidates}:
                    raise ValueError("Unavailable clarification candidate")
                value = ReferenceValue(variant_id=selected)
            elif len(candidates) == 1 and len(exact) == 1:
                value = ReferenceValue(variant_id=exact[0].variant_id)
            else:
                pending.append(
                    Pending(
                        item_index=index,
                        reason="AMBIGUOUS_VARIANT"
                        if candidates
                        else "VARIANT_NOT_FOUND",
                        candidate_ids=[row.variant_id for row in candidates[:20]],
                    )
                )
                continue
        if item.target == "RESOURCE":
            assert isinstance(value, ReferenceValue)
            patch.set_resources.append(
                Resource(id=identifier, variant_id=value.variant_id, evidence=evidence)
            )
        else:
            requirement = Requirement(
                id=identifier,
                subject=item.subject,
                predicate=item.predicate,
                operator=item.operator,
                value=value,
                strength=item.strength,
                evidence=evidence,
            )
            (
                patch.set_goals if item.target == "GOAL" else patch.set_requirements
            ).append(requirement)
    return Resolution(
        draft=draft,
        patch=None if pending else RequirementPatch.model_validate(patch.model_dump()),
        pending=pending,
    )
