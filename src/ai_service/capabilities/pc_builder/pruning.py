"""Pareto pruning and adaptive budget envelopes for the Multiple-Choice Knapsack Problem with Pairwise Compatibility Constraints.

Prunes candidates that are provably dominated across all objective-relevant dimensions
while strictly preserving compatibility feasibility and exact search coverage.

INVARIANT:
Every feature used by score_build(...) or check_compatibility(...) must either:
1. participate in dominance dimensions for that category and objective, or
2. be proven equivalent/irrelevant prior to pruning.
Otherwise, dominance cannot be asserted and dominates() must return False.

Pure Python, 100% deterministic, no external network or LLM dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_service.application.ports.hardware import ComponentCategory, ComponentSpec
from ai_service.capabilities.pc_builder.schemas import BuildObjective, UseCaseProfile


@dataclass(frozen=True, slots=True)
class EnvelopeRange:
    preferred_min: float
    preferred_max: float
    absolute_min: float
    absolute_max: float


@dataclass(frozen=True, slots=True)
class CandidatePool:
    """Two-tier candidate pool for search ordering without excluding affordable options."""

    category: ComponentCategory
    preferred: list[ComponentSpec]
    all_affordable: list[ComponentSpec]


# Configured allocation envelopes per use-case profile
DEFAULT_ENVELOPES: dict[UseCaseProfile, dict[ComponentCategory, EnvelopeRange]] = {
    UseCaseProfile.GAMING_1080P: {
        ComponentCategory.GPU: EnvelopeRange(0.30, 0.48, 0.18, 0.60),
        ComponentCategory.CPU: EnvelopeRange(0.18, 0.28, 0.10, 0.38),
        ComponentCategory.MAINBOARD: EnvelopeRange(0.08, 0.16, 0.05, 0.22),
        ComponentCategory.RAM: EnvelopeRange(0.05, 0.12, 0.03, 0.18),
        ComponentCategory.STORAGE: EnvelopeRange(0.04, 0.10, 0.03, 0.15),
        ComponentCategory.PSU: EnvelopeRange(0.05, 0.10, 0.03, 0.15),
        ComponentCategory.CASE: EnvelopeRange(0.03, 0.08, 0.02, 0.12),
        ComponentCategory.COOLER: EnvelopeRange(0.02, 0.06, 0.00, 0.10),
    },
    UseCaseProfile.GAMING_1440P: {
        ComponentCategory.GPU: EnvelopeRange(0.35, 0.55, 0.22, 0.65),
        ComponentCategory.CPU: EnvelopeRange(0.15, 0.25, 0.10, 0.35),
        ComponentCategory.MAINBOARD: EnvelopeRange(0.08, 0.16, 0.05, 0.22),
        ComponentCategory.RAM: EnvelopeRange(0.05, 0.12, 0.03, 0.18),
        ComponentCategory.STORAGE: EnvelopeRange(0.04, 0.10, 0.03, 0.15),
        ComponentCategory.PSU: EnvelopeRange(0.05, 0.10, 0.03, 0.15),
        ComponentCategory.CASE: EnvelopeRange(0.03, 0.08, 0.02, 0.12),
        ComponentCategory.COOLER: EnvelopeRange(0.02, 0.07, 0.00, 0.12),
    },
    UseCaseProfile.GAMING_4K: {
        ComponentCategory.GPU: EnvelopeRange(0.42, 0.62, 0.30, 0.72),
        ComponentCategory.CPU: EnvelopeRange(0.14, 0.22, 0.08, 0.30),
        ComponentCategory.MAINBOARD: EnvelopeRange(0.08, 0.15, 0.05, 0.20),
        ComponentCategory.RAM: EnvelopeRange(0.05, 0.12, 0.03, 0.18),
        ComponentCategory.STORAGE: EnvelopeRange(0.04, 0.10, 0.03, 0.15),
        ComponentCategory.PSU: EnvelopeRange(0.06, 0.12, 0.04, 0.18),
        ComponentCategory.CASE: EnvelopeRange(0.04, 0.09, 0.02, 0.14),
        ComponentCategory.COOLER: EnvelopeRange(0.03, 0.08, 0.01, 0.14),
    },
    UseCaseProfile.CONTENT_CREATION_3D: {
        ComponentCategory.CPU: EnvelopeRange(0.22, 0.35, 0.15, 0.45),
        ComponentCategory.GPU: EnvelopeRange(0.28, 0.45, 0.18, 0.55),
        ComponentCategory.RAM: EnvelopeRange(0.08, 0.18, 0.05, 0.25),
        ComponentCategory.STORAGE: EnvelopeRange(0.06, 0.14, 0.04, 0.20),
        ComponentCategory.MAINBOARD: EnvelopeRange(0.08, 0.16, 0.05, 0.22),
        ComponentCategory.PSU: EnvelopeRange(0.05, 0.11, 0.03, 0.16),
        ComponentCategory.COOLER: EnvelopeRange(0.03, 0.09, 0.01, 0.15),
        ComponentCategory.CASE: EnvelopeRange(0.03, 0.08, 0.02, 0.12),
    },
    UseCaseProfile.AI_DATA_SCIENCE: {
        ComponentCategory.GPU: EnvelopeRange(0.40, 0.65, 0.25, 0.75),
        ComponentCategory.CPU: EnvelopeRange(0.12, 0.22, 0.08, 0.30),
        ComponentCategory.RAM: EnvelopeRange(0.08, 0.18, 0.05, 0.25),
        ComponentCategory.STORAGE: EnvelopeRange(0.05, 0.12, 0.03, 0.18),
        ComponentCategory.MAINBOARD: EnvelopeRange(0.07, 0.15, 0.04, 0.20),
        ComponentCategory.PSU: EnvelopeRange(0.06, 0.12, 0.04, 0.18),
        ComponentCategory.COOLER: EnvelopeRange(0.02, 0.07, 0.01, 0.12),
        ComponentCategory.CASE: EnvelopeRange(0.03, 0.08, 0.02, 0.12),
    },
    UseCaseProfile.OFFICE_BUDGET: {
        ComponentCategory.CPU: EnvelopeRange(0.22, 0.38, 0.12, 0.50),
        ComponentCategory.MAINBOARD: EnvelopeRange(0.12, 0.24, 0.08, 0.32),
        ComponentCategory.RAM: EnvelopeRange(0.08, 0.18, 0.05, 0.25),
        ComponentCategory.STORAGE: EnvelopeRange(0.08, 0.18, 0.04, 0.25),
        ComponentCategory.PSU: EnvelopeRange(0.08, 0.16, 0.04, 0.22),
        ComponentCategory.CASE: EnvelopeRange(0.05, 0.14, 0.03, 0.20),
        ComponentCategory.COOLER: EnvelopeRange(0.00, 0.08, 0.00, 0.12),
        ComponentCategory.GPU: EnvelopeRange(0.00, 0.25, 0.00, 0.40),
    },
}


def get_candidate_pool(
    candidates: list[ComponentSpec],
    budget: int,
    category: ComponentCategory,
    profile: UseCaseProfile,
    min_viable_preferred: int = 2,
) -> CandidatePool:
    """Partition candidates into preferred envelope (for prioritized exploration) and all affordable.

    Guarantees that exact search retains full access to all affordable candidates so
    no globally optimal combination is excluded.
    """
    affordable = [c for c in candidates if c.price <= budget]
    if not affordable:
        return CandidatePool(category=category, preferred=[], all_affordable=[])

    profile_envelopes = DEFAULT_ENVELOPES.get(profile, DEFAULT_ENVELOPES[UseCaseProfile.GAMING_1080P])
    envelope = profile_envelopes.get(category, EnvelopeRange(0.0, 1.0, 0.0, 1.0))

    pref_min = int(budget * envelope.preferred_min)
    pref_max = int(budget * envelope.preferred_max)
    preferred = [c for c in affordable if pref_min <= c.price <= pref_max]

    if len(preferred) >= min_viable_preferred:
        return CandidatePool(category=category, preferred=preferred, all_affordable=affordable)

    # Adaptive expansion to absolute envelope
    abs_min = int(budget * envelope.absolute_min)
    abs_max = int(budget * envelope.absolute_max)
    expanded = [c for c in affordable if abs_min <= c.price <= abs_max]

    if len(expanded) >= min_viable_preferred:
        return CandidatePool(category=category, preferred=expanded, all_affordable=affordable)

    # Fallback preferred to all affordable
    return CandidatePool(category=category, preferred=list(affordable), all_affordable=affordable)


def _norm_str(val: str | None) -> str:
    """Normalize string by uppercasing and stripping whitespace and hyphens."""
    return (val or "").strip().upper().replace(" ", "").replace("-", "")


def preserves_cpu_optional_capabilities(better: ComponentSpec, worse: ComponentSpec) -> bool:
    """Ensure CPU 'better' does not lose feasibility-enabling capabilities provided by 'worse'.

    A CPU with iGPU or stock cooler cannot be dominated by a CPU without them, as
    losing them requires adding separate components (GPU or cooler), altering feasibility.
    """
    if worse.has_integrated_graphics:
        if not better.has_integrated_graphics:
            return False
        # If both have iGPU, better must not be worse in benchmark or power
        if worse.integrated_graphics_score is not None:
            if better.integrated_graphics_score is None:
                return False
            if better.integrated_graphics_score < worse.integrated_graphics_score:
                return False
        if worse.integrated_graphics_power_watts is not None:
            if better.integrated_graphics_power_watts is None:
                return False
            if better.integrated_graphics_power_watts > worse.integrated_graphics_power_watts:
                return False

    if worse.includes_stock_cooler:
        if not better.includes_stock_cooler:
            return False
        # If both include stock cooler, better must not have worse clearance, cooling TDP or score
        if worse.stock_cooler_height_mm is not None:
            if better.stock_cooler_height_mm is None:
                return False
            if better.stock_cooler_height_mm > worse.stock_cooler_height_mm:
                return False
        if worse.stock_cooler_tdp_watts is not None:
            if better.stock_cooler_tdp_watts is None:
                return False
            if better.stock_cooler_tdp_watts < worse.stock_cooler_tdp_watts:
                return False
        if worse.stock_cooler_score is not None:
            if better.stock_cooler_score is None:
                return False
            if better.stock_cooler_score < worse.stock_cooler_score:
                return False

    return True


def stable_component_key(c: ComponentSpec) -> tuple[Any, ...]:
    """Stable tuple identity for deduplication and deterministic sorting."""
    if c.id is not None:
        return (str(c.id),)
    return (
        c.category.value,
        c.brand or "",
        c.name,
        c.price,
        c.socket or "",
        c.ram_type or "",
        c.ram_slots or 0,
        c.form_factor or "",
        tuple(sorted(c.supported_form_factors or ())),
        c.performance_score or 0.0,
        c.tdp_watts or 0,
        c.wattage or 0,
        c.vram_gb or 0,
        c.capacity_gb or 0,
        c.psu_tier or "",
        c.efficiency_rating or "",
        c.gpu_length_mm or 0,
        c.max_gpu_length_mm or 0,
        c.cooler_height_mm or 0,
        c.max_cooler_height_mm or 0,
        bool(c.has_integrated_graphics),
        c.integrated_graphics_score or 0.0,
        c.integrated_graphics_power_watts or 0,
        bool(c.includes_stock_cooler),
        c.stock_cooler_height_mm or 0,
        c.stock_cooler_tdp_watts or 0,
        c.stock_cooler_score or 0.0,
        bool(c.is_integrated),
        tuple(sorted(c.supported_sockets or ())),
    )


def resolve_component_metric(comp: ComponentSpec, key: str) -> float | None:
    """Resolve metric value for Pareto dominance and scoring using canonical definitions."""
    if key == "psu_quality":
        if comp.psu_tier:
            tier_map = {"A": 100.0, "B": 85.0, "C": 70.0, "D": 50.0}
            return tier_map.get(comp.psu_tier.upper())
        if comp.performance_score is not None:
            return float(comp.performance_score)
        return None

    if key == "performance_score":
        return float(comp.performance_score) if comp.performance_score is not None else None

    # Standard numeric fields
    val = getattr(comp, key, None)
    if val is not None and isinstance(val, (int, float)):
        return float(val)

    extra = comp.extra_specs.get(key)
    if extra is not None and isinstance(extra, (int, float)):
        return float(extra)

    return None


def get_dominance_dimensions(
    category: ComponentCategory,
    profile: UseCaseProfile,
    objective: BuildObjective,
) -> list[str]:
    """Return relevant comparison dimensions for a category under specific profile and objective.

    Prefix with '-' indicates lower-is-better (e.g. '-tdp_watts', '-gpu_length_mm').
    Positive indicates higher-is-better (e.g. 'performance_score', 'vram_gb').
    """
    if category == ComponentCategory.GPU:
        dims = ["performance_score", "-tdp_watts", "-gpu_length_mm"]
        if profile in {UseCaseProfile.AI_DATA_SCIENCE, UseCaseProfile.CONTENT_CREATION_3D, UseCaseProfile.GAMING_4K}:
            dims.append("vram_gb")
        return dims

    if category == ComponentCategory.CPU:
        return ["performance_score", "-tdp_watts"]

    if category == ComponentCategory.MAINBOARD:
        # Conservative policy: do not prune motherboards purely on price without comprehensive
        # VRM/PCIe/M.2/connectivity specifications.
        return []

    if category == ComponentCategory.RAM:
        dims = ["capacity_gb"]
        if objective in {BuildObjective.PERFORMANCE, BuildObjective.BALANCED}:
            dims.append("performance_score")
        return dims

    if category == ComponentCategory.STORAGE:
        dims = ["capacity_gb"]
        if objective in {BuildObjective.PERFORMANCE, BuildObjective.BALANCED}:
            dims.append("performance_score")
        return dims

    if category == ComponentCategory.PSU:
        # Minimum quality requirement across all objectives: wattage and canonical psu_quality
        return ["wattage", "psu_quality"]

    if category == ComponentCategory.COOLER:
        return ["performance_score", "-cooler_height_mm"]

    if category == ComponentCategory.CASE:
        return ["max_gpu_length_mm", "max_cooler_height_mm"]

    return []


def dominates(
    better: ComponentSpec,
    worse: ComponentSpec,
    profile: UseCaseProfile,
    objective: BuildObjective,
) -> bool:
    """Check if component 'better' Pareto-dominates 'worse' under a specific profile and objective.

    Conservative Pareto rules:
    1. 'better' cannot be more expensive than 'worse'.
    2. Must share identical interface constraints (socket, RAM standard, form factor).
    3. In ALL profile- and objective-relevant dimensions: 'better' is NOT worse than 'worse'.
    4. In AT LEAST ONE dimension (price or a metric): 'better' is STRICTLY superior.
    5. Fail-Safe: If any relevant metric is None on either component, returns False (never prune on missing data).
    6. Non-empty dimensions: If no quality dimensions exist for this category, returns False (never prune purely on price).
    """
    if better.category != worse.category:
        return False

    # 1. Price constraint: better must not be more expensive
    if better.price > worse.price:
        return False

    # 2. Interface matching: socket
    if _norm_str(better.socket) != _norm_str(worse.socket):
        return False

    # CPU feasibility capability preservation (iGPU, stock cooler)
    if better.category == ComponentCategory.CPU and not preserves_cpu_optional_capabilities(better, worse):
        return False

    # RAM standard matching
    if _norm_str(better.ram_type) != _norm_str(worse.ram_type):
        return False

    # Mainboard form factor matching
    if better.category == ComponentCategory.MAINBOARD and _norm_str(better.form_factor) != _norm_str(worse.form_factor):
        return False

    # GPU integrated vs dedicated matching: dedicated cannot be dominated by iGPU and vice versa
    if better.category == ComponentCategory.GPU and better.is_integrated != worse.is_integrated:
        return False

    # Cooler supported sockets matching: better must support at least all sockets that worse supports
    if better.category == ComponentCategory.COOLER and worse.supported_sockets:
        w_sockets = {_norm_str(s) for s in worse.supported_sockets}
        b_sockets = {_norm_str(s) for s in better.supported_sockets}
        if not w_sockets.issubset(b_sockets):
            return False

    # Case supported form factors: better must support at least all form factors that worse supports
    if better.category == ComponentCategory.CASE:
        w_supported = {_norm_str(s) for s in (worse.supported_form_factors or [worse.form_factor or ""])}
        b_supported = {_norm_str(s) for s in (better.supported_form_factors or [better.form_factor or ""])}
        if not w_supported.issubset(b_supported):
            return False

    # 3. Dimensions evaluation
    dimensions = get_dominance_dimensions(better.category, profile, objective)
    if not dimensions:
        # Conservative fail-safe: cannot assert dominance without quality dimensions
        return False

    strictly_better_count = 0
    if better.price < worse.price:
        strictly_better_count += 1

    for dim in dimensions:
        is_lower_better = dim.startswith("-")
        key = dim[1:] if is_lower_better else dim

        val_better = resolve_component_metric(better, key)
        val_worse = resolve_component_metric(worse, key)

        # Fail-safe: missing data on either component strictly invalidates dominance proof
        if val_better is None or val_worse is None:
            return False

        if is_lower_better:
            # Lower is better (e.g. TDP, physical length)
            if val_better > val_worse:
                return False  # 'better' has worse power/clearance
            if val_better < val_worse:
                strictly_better_count += 1
        else:
            # Higher is better (e.g. benchmark score, VRAM, capacity)
            if val_better < val_worse:
                return False  # 'better' has worse score/capacity
            if val_better > val_worse:
                strictly_better_count += 1

    return strictly_better_count > 0


def pareto_prune(
    candidates: list[ComponentSpec],
    profile: UseCaseProfile,
    objective: BuildObjective,
) -> list[ComponentSpec]:
    """Prune dominated candidates for a specific objective.

    Maintains deterministic identity-based comparisons and ordering.
    """
    if len(candidates) <= 1:
        return list(candidates)

    non_dominated: list[ComponentSpec] = []

    for candidate in candidates:
        is_dominated = False
        for other in candidates:
            # Check identity: don't compare a component with itself
            if candidate is other or stable_component_key(candidate) == stable_component_key(other):
                continue

            if dominates(other, candidate, profile, objective):
                is_dominated = True
                break

        if not is_dominated:
            non_dominated.append(candidate)

    return non_dominated


def pareto_prune_all_objectives(
    candidates: list[ComponentSpec],
    profile: UseCaseProfile,
) -> list[ComponentSpec]:
    """Retain candidates that are non-dominated under AT LEAST ONE objective.

    Preserves viable candidates across PERFORMANCE, BALANCED, and UPGRADE_FRIENDLY.
    """
    if len(candidates) <= 1:
        return list(candidates)

    retained_keys: set[tuple[Any, ...]] = set()
    retained_candidates: list[ComponentSpec] = []

    for obj in BuildObjective:
        pruned = pareto_prune(candidates, profile, obj)
        for c in pruned:
            c_key = stable_component_key(c)
            if c_key not in retained_keys:
                retained_keys.add(c_key)
                retained_candidates.append(c)

    # Sort deterministically by price, then stable component key
    retained_candidates.sort(key=lambda c: (c.price, stable_component_key(c)))
    return retained_candidates


__all__ = [
    "DEFAULT_ENVELOPES",
    "CandidatePool",
    "EnvelopeRange",
    "dominates",
    "get_candidate_pool",
    "get_dominance_dimensions",
    "pareto_prune",
    "pareto_prune_all_objectives",
    "preserves_cpu_optional_capabilities",
    "resolve_component_metric",
    "stable_component_key",
]
