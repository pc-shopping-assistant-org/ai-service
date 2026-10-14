"""Owned public V2 projections, not serialized framework/application state."""

from typing import Literal
from uuid import UUID

from pydantic import Field

from ai_service.capabilities.assistant.conversation_core import Digest
from ai_service.capabilities.assistant.limits import MAX_PUBLIC_PRODUCTS
from ai_service.capabilities.assistant.provenance import CatalogRowV1
from ai_service.capabilities.assistant.public_contracts import (
    ClarificationItemV1,
    ClarificationOptionV1,
    ConversationViewV1,
    PublicClarificationV1,
    PublicTaskClarificationV1,
    RunCompletedV1,
    TurnResultV1,
)
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
    Subject,
    Value,
)
from ai_service.capabilities.assistant.requirements.resolver import PendingDraft
from ai_service.capabilities.assistant.requirements.units import UnresolvedQuantityValue
from ai_service.capabilities.assistant.schemas import ProductCard
from ai_service.capabilities.assistant.stateful_contracts import (
    Contract,
    NonNegativeInt,
    Text,
)

from .actions import FinalOutcome
from .contracts import CatalogResult, ComparisonResult, DetailsResult, ToolRecord
from .outcomes import AnswerReason
from .published_state import FlexiblePublishedState
from .task_resolution import public_task_options


class PublicTaskOutcomeV2(Contract):
    kind: Literal["BUILD", "COMPARE", "ANSWER", "EXPLAIN_SAVED_BUILD"]
    status: Literal["COMPLETED", "UNRESOLVED"]
    outcome: FinalOutcome | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    reason: AnswerReason | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class PublicProposalChangeV2(Contract):
    target: Literal["REQUIREMENT", "GOAL", "RESOURCE"]
    operation: Literal["SET", "CLEAR"]
    item_id: UUID
    subject: Subject | None = None
    predicate: Text | None = None
    operator: Literal["EQ", "GTE", "LTE"] | None = None
    strength: Literal["HARD", "SOFT"] | None = None
    value: Value | None = None
    variant_id: UUID | None = None


class PublicProposalV2(Contract):
    proposal_id: UUID
    proposal_hash: Digest
    target_revision: NonNegativeInt
    base_requirements_hash: Digest
    changes: list[PublicProposalChangeV2] = Field(min_length=1, max_length=180)


class PublicFlexibleStateV2(Contract):
    status: Literal["COMPLETED", "UNRESOLVED", "WAITING_INPUT", "DISCARDED"]
    task_message: Text
    task_message_id: UUID
    constraints_revision: NonNegativeInt
    answer: Text
    task_outcomes: list[PublicTaskOutcomeV2] = Field(max_length=8)
    proposal: PublicProposalV2 | None = None
    clarification: PublicClarificationV1 | None = None
    task_clarification: PublicTaskClarificationV1 | None = None
    missing_fields: list[Text] = Field(default_factory=list, max_length=20)


class ConversationViewV2(ConversationViewV1):
    schema_version: Literal[2] = 2
    flexible: PublicFlexibleStateV2 | None = None


class TurnResultV2(TurnResultV1):
    schema_version: Literal[2] = 2
    flexible: PublicFlexibleStateV2 | None = None
    products: list[ProductCard] = Field(
        default_factory=list, max_length=MAX_PUBLIC_PRODUCTS
    )


def public_tool_products(
    records: list[ToolRecord], evidence_refs: list[str]
) -> list[ProductCard]:
    """Only products actually selected by the validated final answer."""
    selected = set(evidence_refs)
    products = {}
    for record in records:
        result = record.result
        if result is None or result.status != "COMPLETED":
            continue
        payload = result.result
        items = (
            [payload.product]
            if isinstance(payload, DetailsResult)
            else payload.products
            if isinstance(payload, (CatalogResult, ComparisonResult))
            else []
        )
        for item in items:
            if item.evidence_ref in selected:
                products[item.product_id] = ProductCard(
                    id=item.product_id,
                    name=item.name,
                    list_price=item.price_vnd,
                    specifications=item.specifications,
                    status="ACTIVE",
                )
    return list(products.values())[:20]


class RunCompletedV2(RunCompletedV1):
    result: TurnResultV2


def public_resolution_options(
    pending: PendingDraft, catalog: list[CatalogRowV1]
) -> PublicClarificationV1:
    by_id = {row.variant_id: row for row in catalog}
    if any(
        identity not in by_id
        for item in pending.resolution.pending
        for identity in item.candidate_ids
    ):
        raise ValueError("Clarification labels require canonical catalog candidates")
    return PublicClarificationV1(
        pending_message_id=pending.message_id,
        constraints_revision=pending.constraints_revision,
        items=[
            ClarificationItemV1(
                item_index=item.item_index,
                reason=item.reason,
                quantity_amount=value.amount
                if isinstance(
                    value := pending.resolution.draft.items[item.item_index].value,
                    UnresolvedQuantityValue,
                )
                else None,
                unit_options=["VND", "GB", "TB", "GiB", "TiB", "W", "FPS"]
                if item.reason == "QUANTITY_UNIT_REQUIRED"
                else [],
                options=[
                    ClarificationOptionV1(
                        variant_id=identity,
                        sku=by_id[identity].sku,
                        name=by_id[identity].component.name,
                    )
                    for identity in item.candidate_ids
                ],
            )
            for item in pending.resolution.pending
        ],
    )


def to_public_flexible(
    state: FlexiblePublishedState, requirements: CanonicalRequirementState
) -> PublicFlexibleStateV2:
    state = FlexiblePublishedState.model_validate(state.model_dump())
    proposal = None
    if state.pending_proposal:
        pending = state.pending_proposal
        changes = []
        for target, setting, clearing, current in (
            (
                "REQUIREMENT",
                pending.patch.set_requirements,
                pending.patch.clear_requirements,
                requirements.requirements,
            ),
            (
                "GOAL",
                pending.patch.set_goals,
                pending.patch.clear_goals,
                requirements.goals,
            ),
        ):
            for operation, items in (
                ("SET", setting),
                ("CLEAR", [current[identity] for identity in clearing]),
            ):
                for item in items:
                    changes.append(
                        PublicProposalChangeV2.model_validate(
                            {
                                "target": target,
                                "operation": operation,
                                "item_id": item.id,
                                "subject": item.subject,
                                "predicate": item.predicate,
                                "operator": item.operator,
                                "strength": item.strength,
                                "value": item.value,
                            }
                        )
                    )
        for operation, resources in (
            ("SET", pending.patch.set_resources),
            (
                "CLEAR",
                [
                    requirements.available_resources[identity]
                    for identity in pending.patch.clear_resources
                ],
            ),
        ):
            changes.extend(
                PublicProposalChangeV2.model_validate(
                    {
                        "target": "RESOURCE",
                        "operation": operation,
                        "item_id": item.id,
                        "variant_id": item.variant_id,
                    }
                )
                for item in resources
            )
        proposal = PublicProposalV2(
            proposal_id=pending.proposal_id,
            proposal_hash=pending.proposal_hash,
            target_revision=pending.target_revision,
            base_requirements_hash=pending.base_requirements_hash,
            changes=changes,
        )
    task_outcomes = [
        PublicTaskOutcomeV2(
            kind=item.task.kind,
            status=item.status,
            outcome=item.outcome,
            reason=item.reason,
        )
        for item in state.task_outcomes
    ]
    if state.pending_task_resolution:
        task_outcomes = [
            PublicTaskOutcomeV2(kind=item.kind, status="UNRESOLVED")
            for item in state.pending_task_resolution.tasks
        ]
    return PublicFlexibleStateV2(
        status=state.status,
        task_message=state.task_message,
        task_message_id=state.task_message_id,
        constraints_revision=requirements.revision,
        answer=state.assistant_message,
        task_outcomes=task_outcomes,
        proposal=proposal,
        clarification=state.resolution_options
        or public_resolution_options(state.pending_resolution.draft, [])
        if state.pending_resolution
        else None,
        task_clarification=public_task_options(state.pending_task_resolution)
        if state.pending_task_resolution
        else None,
        missing_fields=state.clarification.missing_fields
        if state.clarification
        else [],
    )
