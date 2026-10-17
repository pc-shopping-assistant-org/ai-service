"""Versioned, bounded utility bonus; never a feasibility constraint."""

from uuid import UUID

from ai_service.application.ports.hardware import ComponentCategory, ComponentSpec
from ai_service.capabilities.pc_builder.schemas import ScoreBreakdown, ScoreDimension

PREFERENCE_POLICY_VERSION = "equal-soft-utility-v1"
PREFERENCE_WEIGHT = 0.10


def apply_preferences(
    breakdown: ScoreBreakdown,
    parts: dict[ComponentCategory, ComponentSpec],
    scores: dict[UUID, float],
) -> ScoreBreakdown:
    if not scores:
        return breakdown
    utility = sum(
        scores.get(part.id, 0.0) if part.id else 0.0 for part in parts.values()
    )
    if not 0 <= utility <= 1:
        raise ValueError("Selected preference utility must be in [0, 1]")
    base_weight = 1 - PREFERENCE_WEIGHT
    return ScoreBreakdown(
        total_score=breakdown.total_score * base_weight
        + utility * 100 * PREFERENCE_WEIGHT,
        dimensions=[
            dimension.model_copy(
                update={
                    "weight": dimension.weight * base_weight,
                    "contribution": dimension.contribution * base_weight,
                }
            )
            for dimension in breakdown.dimensions
        ]
        + [
            ScoreDimension(
                name="soft_preferences",
                raw_value=utility,
                normalized_score=utility * 100,
                weight=PREFERENCE_WEIGHT,
                contribution=utility * 100 * PREFERENCE_WEIGHT,
            )
        ],
    )
