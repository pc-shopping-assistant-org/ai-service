"""Factual serialization layer for LLM explanation context.

Strictly copies and formats grounded facts and evidence produced by the core optimizer.
NEVER recalculates, synthesizes, or invents scores, FPS, or benchmark values.
"""

from __future__ import annotations

from typing import Any

from ai_service.capabilities.pc_builder.application.service import (
    BuildApplicationResult,
)


def build_explanation_context(
    result: BuildApplicationResult,
) -> dict[str, Any]:
    """Serialize the application result into a factual explanation context for the LLM.

    The returned context contains only:
    - Recommended build parts and exact prices.
    - Power calculation snapshot (sustained, transient, minimum, recommended).
    - Factual MetricEvidence directly emitted by the optimizer.
    - Factual compatibility status.
    - High-level comparison summary across all evaluated objective builds (price, score).

    INVARIANT:
    Zero score recalculation. All scores and evidence items are passed as emitted by the Core.
    """
    recommendation = result.recommendation
    build = recommendation.build

    return {
        "recommended_objective": recommendation.objective.value,
        "reason_code": recommendation.reason_code.value,
        "policy_version": recommendation.policy_version,
        "total_price": build.total_price,
        "objective_score": build.objective_score,
        "compatibility_status": build.compatibility_status.value,
        "parts": {
            category.value: {
                "name": part.name,
                "price": part.price,
                "spending_price": build.spending_prices.get(category, part.price),
                "is_owned": category in build.owned_categories,
                "brand": part.brand,
                "is_integrated": part.is_integrated,
            }
            for category, part in build.parts.items()
        },
        "power": build.power_estimate.model_dump(),
        "evidence": [item.model_dump() for item in build.evidence],
        "objective_builds": {
            objective.value: {
                "total_price": candidate.total_price,
                "objective_score": candidate.objective_score,
            }
            for objective, candidate in result.optimization.builds.items()
        },
    }
