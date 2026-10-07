"""PC Builder Application Layer.

Contains recommendation policies, application orchestration services, and factual explanation serialization.
"""

from ai_service.capabilities.pc_builder.application.explanation_context import (
    build_explanation_context,
)
from ai_service.capabilities.pc_builder.application.recommendation_policy import (
    DEFAULT_OBJECTIVE_BY_PROFILE,
    FALLBACK_OBJECTIVE_ORDER,
    RECOMMENDATION_POLICY_VERSION,
    RecommendationDecision,
    RecommendationPolicyError,
    RecommendationReason,
    choose_recommended_build,
)
from ai_service.capabilities.pc_builder.application.service import (
    BuildApplicationResult,
    PCBuildApplicationService,
)

__all__ = [
    "DEFAULT_OBJECTIVE_BY_PROFILE",
    "FALLBACK_OBJECTIVE_ORDER",
    "RECOMMENDATION_POLICY_VERSION",
    "BuildApplicationResult",
    "PCBuildApplicationService",
    "RecommendationDecision",
    "RecommendationPolicyError",
    "RecommendationReason",
    "build_explanation_context",
    "choose_recommended_build",
]
