"""Deterministic Combinatorial PC Optimizer (Multiple-Choice Knapsack with Compatibility Constraints).

Pure mathematical and constraint optimization logic.
No LLM, no agent dependencies, 100% reproducible and testable.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid5

from ai_service.application.ports.hardware import ComponentCategory, ComponentSpec
from ai_service.capabilities.pc_builder.pruning import (
    get_candidate_pool,
    pareto_prune_all_objectives,
    resolve_component_metric,
)
from ai_service.capabilities.pc_builder.schemas import (
    BuildObjective,
    CompatibilityStatus,
    MetricEvidence,
    OptimizationResult,
    PCBuildConstraints,
    PowerEstimate,
    RankedBuild,
    ScoreBreakdown,
    ScoreDimension,
    UseCaseProfile,
)

SYNTHETIC_NAMESPACE = UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

STANDARD_PSU_SIZES = [450, 500, 550, 600, 650, 750, 850, 1000, 1200]

FORM_FACTOR_RANKS: dict[str, int] = {
    "MINIITX": 1,
    "MINI-ITX": 1,
    "ITX": 1,
    "MICROATX": 2,
    "MICRO-ATX": 2,
    "MATX": 2,
    "ATX": 3,
    "EATX": 4,
    "E-ATX": 4,
}

DEFAULT_PLATFORM_POLICY: dict[str, float] = {
    "AM5": 100.0,
    "LGA1851": 90.0,
    "AM4": 40.0,
    "LGA1700": 40.0,
}

FIXED_CATEGORY_ORDER = (
    ComponentCategory.CPU,
    ComponentCategory.MAINBOARD,
    ComponentCategory.RAM,
    ComponentCategory.GPU,
    ComponentCategory.CASE,
    ComponentCategory.COOLER,
    ComponentCategory.STORAGE,
    ComponentCategory.PSU,
)

# Configured scoring policies: (UseCaseProfile, BuildObjective) -> dict[metric_name, weight]
# Sum of weights strictly equals 1.0 for each policy
SCORING_POLICIES: dict[tuple[UseCaseProfile, BuildObjective], dict[str, float]] = {
    # GAMING 1440P
    (UseCaseProfile.GAMING_1440P, BuildObjective.PERFORMANCE): {
        "gpu_perf": 0.50,
        "cpu_perf": 0.25,
        "cost_efficiency": 0.15,
        "ram_capacity": 0.10,
    },
    (UseCaseProfile.GAMING_1440P, BuildObjective.BALANCED): {
        "gpu_perf": 0.35,
        "cpu_perf": 0.20,
        "psu_quality": 0.15,
        "platform_longevity": 0.15,
        "ram_capacity": 0.10,
        "psu_headroom": 0.05,
    },
    (UseCaseProfile.GAMING_1440P, BuildObjective.UPGRADE_FRIENDLY): {
        "platform_longevity": 0.35,
        "ram_slots": 0.20,
        "psu_headroom": 0.20,
        "gpu_perf": 0.15,
        "cpu_perf": 0.10,
    },
    # GAMING 1080P
    (UseCaseProfile.GAMING_1080P, BuildObjective.PERFORMANCE): {
        "gpu_perf": 0.42,
        "cpu_perf": 0.30,
        "cost_efficiency": 0.18,
        "ram_capacity": 0.10,
    },
    (UseCaseProfile.GAMING_1080P, BuildObjective.BALANCED): {
        "gpu_perf": 0.32,
        "cpu_perf": 0.24,
        "psu_quality": 0.15,
        "platform_longevity": 0.15,
        "ram_capacity": 0.10,
        "psu_headroom": 0.04,
    },
    (UseCaseProfile.GAMING_1080P, BuildObjective.UPGRADE_FRIENDLY): {
        "platform_longevity": 0.35,
        "ram_slots": 0.20,
        "psu_headroom": 0.20,
        "cpu_perf": 0.15,
        "gpu_perf": 0.10,
    },
    # GAMING 4K
    (UseCaseProfile.GAMING_4K, BuildObjective.PERFORMANCE): {
        "gpu_perf": 0.55,
        "gpu_vram": 0.15,
        "cpu_perf": 0.15,
        "cost_efficiency": 0.10,
        "ram_capacity": 0.05,
    },
    (UseCaseProfile.GAMING_4K, BuildObjective.BALANCED): {
        "gpu_perf": 0.40,
        "gpu_vram": 0.15,
        "cpu_perf": 0.15,
        "psu_quality": 0.15,
        "platform_longevity": 0.10,
        "psu_headroom": 0.05,
    },
    (UseCaseProfile.GAMING_4K, BuildObjective.UPGRADE_FRIENDLY): {
        "platform_longevity": 0.30,
        "psu_headroom": 0.25,
        "gpu_vram": 0.15,
        "ram_slots": 0.15,
        "gpu_perf": 0.15,
    },
    # AI & DATA SCIENCE
    (UseCaseProfile.AI_DATA_SCIENCE, BuildObjective.PERFORMANCE): {
        "gpu_vram": 0.38,
        "gpu_perf": 0.28,
        "cpu_perf": 0.14,
        "ram_capacity": 0.10,
        "cost_efficiency": 0.10,
    },
    (UseCaseProfile.AI_DATA_SCIENCE, BuildObjective.BALANCED): {
        "gpu_vram": 0.30,
        "gpu_perf": 0.25,
        "ram_capacity": 0.15,
        "psu_quality": 0.15,
        "cpu_perf": 0.10,
        "psu_headroom": 0.05,
    },
    (UseCaseProfile.AI_DATA_SCIENCE, BuildObjective.UPGRADE_FRIENDLY): {
        "platform_longevity": 0.30,
        "psu_headroom": 0.25,
        "ram_slots": 0.20,
        "gpu_vram": 0.15,
        "gpu_perf": 0.10,
    },
    # CONTENT CREATION 3D
    (UseCaseProfile.CONTENT_CREATION_3D, BuildObjective.PERFORMANCE): {
        "cpu_perf": 0.35,
        "gpu_perf": 0.30,
        "ram_capacity": 0.15,
        "storage_capacity": 0.10,
        "cost_efficiency": 0.10,
    },
    (UseCaseProfile.CONTENT_CREATION_3D, BuildObjective.BALANCED): {
        "cpu_perf": 0.28,
        "gpu_perf": 0.24,
        "ram_capacity": 0.16,
        "psu_quality": 0.12,
        "platform_longevity": 0.10,
        "storage_capacity": 0.10,
    },
    (UseCaseProfile.CONTENT_CREATION_3D, BuildObjective.UPGRADE_FRIENDLY): {
        "platform_longevity": 0.30,
        "ram_slots": 0.25,
        "psu_headroom": 0.20,
        "cpu_perf": 0.15,
        "gpu_perf": 0.10,
    },
    # OFFICE BUDGET
    (UseCaseProfile.OFFICE_BUDGET, BuildObjective.PERFORMANCE): {
        "cpu_perf": 0.45,
        "cost_efficiency": 0.30,
        "ram_capacity": 0.15,
        "storage_capacity": 0.10,
    },
    (UseCaseProfile.OFFICE_BUDGET, BuildObjective.BALANCED): {
        "cpu_perf": 0.35,
        "cost_efficiency": 0.25,
        "psu_quality": 0.15,
        "platform_longevity": 0.15,
        "ram_capacity": 0.10,
    },
    (UseCaseProfile.OFFICE_BUDGET, BuildObjective.UPGRADE_FRIENDLY): {
        "platform_longevity": 0.35,
        "ram_slots": 0.25,
        "psu_headroom": 0.20,
        "cpu_perf": 0.20,
    },
}

# Startup assertion: Every (UseCaseProfile, BuildObjective) pair must be defined,
# and each policy's weights must sum to 1.0 within floating point tolerance.
for _prof in UseCaseProfile:
    for _obj in BuildObjective:
        if (_prof, _obj) not in SCORING_POLICIES:
            raise AssertionError(f"Missing scoring policy for ({_prof}, {_obj})")
        _policy_sum = sum(SCORING_POLICIES[(_prof, _obj)].values())
        if abs(_policy_sum - 1.0) > 1e-6:
            raise AssertionError(
                f"Scoring policy ({_prof}, {_obj}) weights sum to {_policy_sum}, must equal 1.0"
            )


def _norm(val: str | None) -> str:
    return (val or "").strip().upper().replace(" ", "").replace("-", "")


def stable_component_id(comp: ComponentSpec) -> str:
    """Return a deterministic string identity for tie-breaking and sorting."""
    if comp.id is not None:
        return str(comp.id)
    fingerprint = (
        comp.category.value,
        comp.brand or "",
        comp.name,
        comp.price,
        comp.socket or "",
        comp.ram_type or "",
        comp.ram_slots or 0,
        comp.form_factor or "",
        ",".join(sorted(comp.supported_form_factors or ())),
        comp.performance_score or 0.0,
        comp.tdp_watts or 0,
        comp.wattage or 0,
        comp.vram_gb or 0,
        comp.capacity_gb or 0,
        comp.psu_tier or "",
        comp.efficiency_rating or "",
        comp.gpu_length_mm or 0,
        comp.max_gpu_length_mm or 0,
        comp.cooler_height_mm or 0,
        comp.max_cooler_height_mm or 0,
        int(bool(comp.has_integrated_graphics)),
        comp.integrated_graphics_score or 0.0,
        comp.integrated_graphics_power_watts or 0,
        int(bool(comp.includes_stock_cooler)),
        comp.stock_cooler_height_mm or 0,
        comp.stock_cooler_tdp_watts or 0,
        comp.stock_cooler_score or 0.0,
        int(bool(comp.is_integrated)),
        ",".join(sorted(comp.supported_sockets or ())),
    )
    return "|".join(str(x) for x in fingerprint)


def build_tie_key(parts: dict[ComponentCategory, ComponentSpec], total_price: int) -> tuple[Any, ...]:
    """Deterministic tie-breaking key: lower price first, then lexicographical component identity."""
    return (
        total_price,
        tuple(stable_component_id(parts[cat]) for cat in FIXED_CATEGORY_ORDER if cat in parts),
    )


def round_up_psu(target: float) -> int:
    """Find the smallest standard commercial PSU capacity >= target load.

    Raises:
        ValueError: If target exceeds the maximum supported commercial size (1200W).
    """
    for size in STANDARD_PSU_SIZES:
        if size >= target:
            return size
    raise ValueError(
        f"Required PSU capacity {target:.0f}W exceeds maximum supported standard commercial size ({STANDARD_PSU_SIZES[-1]}W)."
    )


def calculate_power_estimate(
    parts: dict[ComponentCategory, ComponentSpec],
    profile: UseCaseProfile,
) -> PowerEstimate:
    """Calculate sustained peak power and commercial PSU requirements without fake fallback defaults."""
    cpu = parts.get(ComponentCategory.CPU)
    gpu = parts.get(ComponentCategory.GPU)

    if cpu is not None and cpu.tdp_watts is None:
        raise ValueError(f"CPU '{cpu.name}' is missing required specification 'tdp_watts'.")
    if gpu is not None and gpu.tdp_watts is None:
        raise ValueError(f"GPU '{gpu.name}' is missing required specification 'tdp_watts'.")

    cpu_tdp = float(cpu.tdp_watts) if (cpu and cpu.tdp_watts is not None) else 65.0
    cpu_peak = cpu_tdp * (1.20 if cpu_tdp >= 105 else 1.10)

    gpu_tdp = float(gpu.tdp_watts) if (gpu and gpu.tdp_watts is not None) else 20.0
    gpu_peak = gpu_tdp * 1.0

    platform_watts = 65.0
    sustained_peak = cpu_peak + gpu_peak + platform_watts

    if gpu_tdp >= 280:
        transient_factor = 0.30
    elif gpu_tdp >= 180:
        transient_factor = 0.20
    else:
        transient_factor = 0.12

    transient_allowance = gpu_peak * transient_factor

    # Minimum safe capacity incorporates 10% operational margin above sustained load
    min_target = sustained_peak * 1.10
    min_psu = round_up_psu(min_target)

    rec_target = sustained_peak + transient_allowance + 50.0
    rec_psu = round_up_psu(rec_target)

    rationale = (
        f"Sustained peak load: {sustained_peak:.0f}W (CPU {cpu_peak:.0f}W, GPU {gpu_peak:.0f}W, Platform {platform_watts:.0f}W). "
        f"Transient allowance: +{transient_allowance:.0f}W. "
        f"Minimum capacity: {min_psu}W, Recommended capacity: {rec_psu}W."
    )

    return PowerEstimate(
        cpu_estimated_peak_watts=round(cpu_peak, 1),
        gpu_estimated_peak_watts=round(gpu_peak, 1),
        platform_estimated_watts=round(platform_watts, 1),
        estimated_sustained_peak_watts=round(sustained_peak, 1),
        transient_allowance_watts=round(transient_allowance, 1),
        minimum_capacity_watts=min_psu,
        recommended_capacity_watts=rec_psu,
        sizing_rationale=rationale,
    )


def check_compatibility_status(
    parts: dict[ComponentCategory, ComponentSpec],
    min_psu_watts: int,
) -> CompatibilityStatus:
    """Enforce strict hard physical and electrical compatibility constraints.

    Returns:
        COMPATIBLE: All required specs are known and strictly compatible.
        INCOMPATIBLE: A known mismatch or physical clearance/power violation occurs.
        UNKNOWN: A critical specification is missing.
    """
    cpu = parts.get(ComponentCategory.CPU)
    mb = parts.get(ComponentCategory.MAINBOARD)
    ram = parts.get(ComponentCategory.RAM)
    gpu = parts.get(ComponentCategory.GPU)
    case = parts.get(ComponentCategory.CASE)
    cooler = parts.get(ComponentCategory.COOLER)
    psu = parts.get(ComponentCategory.PSU)

    # 1. CPU vs Mainboard Socket: must be known and matching
    if cpu and mb:
        if not cpu.socket or not mb.socket:
            return CompatibilityStatus.UNKNOWN
        if _norm(cpu.socket) != _norm(mb.socket):
            return CompatibilityStatus.INCOMPATIBLE

    # 2. Mainboard vs RAM Type: must be known and matching
    if mb and ram:
        if not mb.ram_type or not ram.ram_type:
            return CompatibilityStatus.UNKNOWN
        if _norm(mb.ram_type) != _norm(ram.ram_type):
            return CompatibilityStatus.INCOMPATIBLE

    # 3. Mainboard Form Factor vs Case: must be known and supported
    if mb and case:
        if not mb.form_factor:
            return CompatibilityStatus.UNKNOWN
        mb_ff = _norm(mb.form_factor)
        if case.supported_form_factors:
            supported = [_norm(s) for s in case.supported_form_factors]
            if mb_ff not in supported:
                return CompatibilityStatus.INCOMPATIBLE
        elif case.form_factor:
            mb_rank = FORM_FACTOR_RANKS.get(mb_ff)
            case_rank = FORM_FACTOR_RANKS.get(_norm(case.form_factor))
            if mb_rank is None or case_rank is None:
                return CompatibilityStatus.UNKNOWN
            if mb_rank > case_rank:
                return CompatibilityStatus.INCOMPATIBLE
        else:
            return CompatibilityStatus.UNKNOWN

    # 4. GPU Length vs Case Max GPU Length (dedicated GPU only)
    if gpu and case and not gpu.is_integrated:
        if gpu.gpu_length_mm is not None and case.max_gpu_length_mm is not None:
            if gpu.gpu_length_mm > case.max_gpu_length_mm:
                return CompatibilityStatus.INCOMPATIBLE
        elif gpu.gpu_length_mm is None or case.max_gpu_length_mm is None:
            return CompatibilityStatus.UNKNOWN

    # 5. Cooler Height vs Case Max Cooler Height
    if cooler and case:
        if cooler.cooler_height_mm is None or case.max_cooler_height_mm is None:
            return CompatibilityStatus.UNKNOWN
        if cooler.cooler_height_mm > case.max_cooler_height_mm:
            return CompatibilityStatus.INCOMPATIBLE

    # 6. PSU Rated Wattage vs Minimum System Required Wattage
    if psu:
        if psu.wattage is None:
            return CompatibilityStatus.UNKNOWN
        if psu.wattage < min_psu_watts:
            return CompatibilityStatus.INCOMPATIBLE

    # 7. Cooler vs CPU Socket (if cooler specifies supported sockets or socket)
    if cooler and cpu:
        if cooler.supported_sockets:
            if not cpu.socket:
                return CompatibilityStatus.UNKNOWN
            if _norm(cpu.socket) not in [_norm(s) for s in cooler.supported_sockets]:
                return CompatibilityStatus.INCOMPATIBLE
        elif cooler.socket:
            if not cpu.socket:
                return CompatibilityStatus.UNKNOWN
            if _norm(cpu.socket) != _norm(cooler.socket):
                return CompatibilityStatus.INCOMPATIBLE

    return CompatibilityStatus.COMPATIBLE


def check_compatibility(
    parts: dict[ComponentCategory, ComponentSpec],
    min_psu_watts: int,
) -> bool:
    """Convenience predicate for hard compatibility. Only COMPATIBLE returns True."""
    return check_compatibility_status(parts, min_psu_watts) == CompatibilityStatus.COMPATIBLE


def score_build(
    parts: dict[ComponentCategory, ComponentSpec],
    objective: BuildObjective,
    profile: UseCaseProfile,
    budget: int,
    power: PowerEstimate,
    platform_policy: dict[str, float] | None = None,
    total_spending: int | None = None,
) -> ScoreBreakdown:
    """Multi-attribute utility scoring strictly normalized to [0.0, 100.0] under (profile, objective).

    Policy-driven validation: If any dimension with weight > 0 lacks underlying data, returns -1.0 breakdown.
    """
    spending = total_spending if total_spending is not None else sum(p.price for p in parts.values())
    if spending > budget:
        return ScoreBreakdown(total_score=-1.0, dimensions=[])

    policy_map = platform_policy or DEFAULT_PLATFORM_POLICY
    if (profile, objective) not in SCORING_POLICIES:
        raise ValueError(f"No scoring policy registered for ({profile}, {objective}).")
    weights = SCORING_POLICIES[(profile, objective)]

    gpu = parts.get(ComponentCategory.GPU)
    cpu = parts.get(ComponentCategory.CPU)
    ram = parts.get(ComponentCategory.RAM)
    storage = parts.get(ComponentCategory.STORAGE)
    mb = parts.get(ComponentCategory.MAINBOARD)
    psu = parts.get(ComponentCategory.PSU)

    # 1. Policy-driven validation: fail-safe rejection if any required metric is missing
    for dim, weight in weights.items():
        if weight <= 0:
            continue
        if dim == "gpu_perf" and (gpu is None or gpu.performance_score is None):
            return ScoreBreakdown(total_score=-1.0, dimensions=[])
        if dim == "gpu_vram" and (gpu is None or gpu.vram_gb is None):
            return ScoreBreakdown(total_score=-1.0, dimensions=[])
        if dim == "cpu_perf" and (cpu is None or cpu.performance_score is None):
            return ScoreBreakdown(total_score=-1.0, dimensions=[])
        if dim == "ram_capacity" and (ram is None or ram.capacity_gb is None):
            return ScoreBreakdown(total_score=-1.0, dimensions=[])
        if dim == "storage_capacity" and (storage is None or storage.capacity_gb is None):
            return ScoreBreakdown(total_score=-1.0, dimensions=[])
        if dim == "platform_longevity" and (mb is None or not mb.socket or _norm(mb.socket) not in policy_map):
            return ScoreBreakdown(total_score=-1.0, dimensions=[])
        if dim == "ram_slots" and (mb is None or mb.ram_slots is None):
            return ScoreBreakdown(total_score=-1.0, dimensions=[])
        if dim == "psu_quality" and (psu is None or resolve_component_metric(psu, "psu_quality") is None):
            return ScoreBreakdown(total_score=-1.0, dimensions=[])
        if dim == "psu_headroom" and (psu is None or psu.wattage is None):
            return ScoreBreakdown(total_score=-1.0, dimensions=[])
        if dim == "cost_efficiency" and (
            gpu is None or gpu.performance_score is None or cpu is None or cpu.performance_score is None
        ):
            return ScoreBreakdown(total_score=-1.0, dimensions=[])

    # 2. Compute exact normalized dimension scores [0.0, 100.0]
    dimension_scores: dict[str, float] = {}
    raw_values: dict[str, float | int | str] = {}

    if gpu and gpu.performance_score is not None:
        dimension_scores["gpu_perf"] = max(0.0, min(100.0, float(gpu.performance_score)))
        raw_values["gpu_perf"] = float(gpu.performance_score)

    if gpu and gpu.vram_gb is not None:
        dimension_scores["gpu_vram"] = max(0.0, min(100.0, (float(gpu.vram_gb) / 24.0) * 100.0))
        raw_values["gpu_vram"] = int(gpu.vram_gb)

    if cpu and cpu.performance_score is not None:
        dimension_scores["cpu_perf"] = max(0.0, min(100.0, float(cpu.performance_score)))
        raw_values["cpu_perf"] = float(cpu.performance_score)

    if ram and ram.capacity_gb is not None:
        dimension_scores["ram_capacity"] = max(0.0, min(100.0, (float(ram.capacity_gb) / 64.0) * 100.0))
        raw_values["ram_capacity"] = int(ram.capacity_gb)

    if storage and storage.capacity_gb is not None:
        dimension_scores["storage_capacity"] = max(0.0, min(100.0, (float(storage.capacity_gb) / 2000.0) * 100.0))
        raw_values["storage_capacity"] = int(storage.capacity_gb)

    if mb and mb.socket:
        dimension_scores["platform_longevity"] = policy_map.get(_norm(mb.socket), 50.0)
        raw_values["platform_longevity"] = mb.socket

    if mb and mb.ram_slots is not None:
        dimension_scores["ram_slots"] = 100.0 if mb.ram_slots >= 4 else 50.0
        raw_values["ram_slots"] = int(mb.ram_slots)

    if psu:
        pq = resolve_component_metric(psu, "psu_quality")
        if pq is not None:
            dimension_scores["psu_quality"] = pq
            raw_values["psu_quality"] = pq
        if psu.wattage:
            psu_watts = float(psu.wattage)
            headroom_ratio = max(0.0, (psu_watts - power.estimated_sustained_peak_watts) / psu_watts)
            dimension_scores["psu_headroom"] = max(0.0, min(100.0, (headroom_ratio / 0.50) * 100.0))
            raw_values["psu_headroom"] = round(psu_watts - power.estimated_sustained_peak_watts, 1)

    if "gpu_perf" in dimension_scores and "cpu_perf" in dimension_scores:
        price_millions = max(1.0, spending / 1_000_000.0)
        raw_efficiency = (dimension_scores["gpu_perf"] + dimension_scores["cpu_perf"]) / price_millions
        dimension_scores["cost_efficiency"] = max(0.0, min(100.0, (raw_efficiency / 12.0) * 100.0))
        raw_values["cost_efficiency"] = round(raw_efficiency, 2)

    # 3. Transparent ScoreBreakdown with individual dimensions and contributions
    dimensions: list[ScoreDimension] = []
    total_score = 0.0

    for dim, weight in weights.items():
        if dim in dimension_scores:
            norm_s = round(dimension_scores[dim], 4)
            contrib = round(norm_s * weight, 4)
            raw_v = raw_values.get(dim, norm_s)
            dimensions.append(
                ScoreDimension(
                    name=dim,
                    raw_value=raw_v,
                    normalized_score=norm_s,
                    weight=weight,
                    contribution=contrib,
                )
            )
            total_score += contrib

    return ScoreBreakdown(
        total_score=round(total_score, 4),
        dimensions=dimensions,
    )


def _generate_factual_evidence(
    parts: dict[ComponentCategory, ComponentSpec],
    objective: BuildObjective,
    power: PowerEstimate,
    score: float,
    breakdown: ScoreBreakdown | None = None,
) -> list[MetricEvidence]:
    """Generate factual metric evidence strictly computed from component specifications and sizing formulas."""
    gpu = parts.get(ComponentCategory.GPU)
    cpu = parts.get(ComponentCategory.CPU)
    mb = parts.get(ComponentCategory.MAINBOARD)
    psu = parts.get(ComponentCategory.PSU)

    evidence: list[MetricEvidence] = [
        MetricEvidence(
            metric="objective_score",
            raw_value=score,
            normalized_score=score,
            unit="/100",
            label=f"Điểm tối ưu ({objective})",
            source="scoring_engine",
        ),
    ]

    if breakdown:
        for dim in breakdown.dimensions:
            evidence.append(
                MetricEvidence(
                    metric=f"score_dim_{dim.name}",
                    raw_value=dim.raw_value,
                    normalized_score=dim.normalized_score,
                    weight=dim.weight,
                    contribution=dim.contribution,
                    label=f"Thành phần điểm: {dim.name}",
                    source="scoring_breakdown",
                )
            )

    evidence.extend([
        MetricEvidence(
            metric="sustained_peak_watts",
            raw_value=power.estimated_sustained_peak_watts,
            unit="W",
            label="Công suất tiêu thụ tối đa (Full load)",
            source="power_calculator",
        ),
        MetricEvidence(
            metric="psu_capacity_watts",
            raw_value=psu.wattage if psu and psu.wattage else power.minimum_capacity_watts,
            unit="W",
            label="Công suất bộ nguồn",
            source="component_spec",
        ),
    ])

    if gpu:
        evidence.append(MetricEvidence(metric="gpu_name", raw_value=gpu.name, label="Card đồ họa", source="catalog"))
        if gpu.vram_gb:
            evidence.append(MetricEvidence(metric="gpu_vram", raw_value=gpu.vram_gb, unit="GB", label="Bộ nhớ VRAM", source="spec"))
        if gpu.performance_score:
            evidence.append(MetricEvidence(metric="gpu_score", raw_value=gpu.performance_score, unit="/100", label="Điểm Benchmark GPU", source="benchmark"))

    if cpu:
        evidence.append(MetricEvidence(metric="cpu_name", raw_value=cpu.name, label="Bộ vi xử lý", source="catalog"))
        if cpu.socket:
            evidence.append(MetricEvidence(metric="cpu_socket", raw_value=cpu.socket, label="Chuẩn Socket", source="spec"))
        if cpu.performance_score:
            evidence.append(MetricEvidence(metric="cpu_score", raw_value=cpu.performance_score, unit="/100", label="Điểm Benchmark CPU", source="benchmark"))

    if mb and mb.socket:
        evidence.append(MetricEvidence(metric="mainboard_socket", raw_value=mb.socket, label="Socket Bo mạch chủ", source="spec"))

    return evidence


class DeterministicPCOptimizer:
    """Pure mathematical and constraint optimization engine for PC component builds."""

    def __init__(
        self,
        platform_policy: dict[str, float] | None = None,
        enable_pruning: bool = True,
    ) -> None:
        self.platform_policy = platform_policy or DEFAULT_PLATFORM_POLICY
        self.enable_pruning = enable_pruning

    def optimize(
        self,
        constraints: PCBuildConstraints,
        catalog: list[ComponentSpec],
    ) -> OptimizationResult:
        """Run deterministic constrained enumeration with early feasibility pruning."""
        budget = constraints.target_budget_vnd.value
        profile = constraints.use_case.value

        owned_map: dict[ComponentCategory, Any] = {
            op.category: op for op in constraints.owned_parts
        }

        def get_spending_price(cat: ComponentCategory, comp: ComponentSpec) -> int:
            if cat in owned_map and owned_map[cat].exclude_from_budget:
                return 0
            return comp.price

        # 1. Partition catalog by category and filter by locked user brand and form factor preferences
        categorized: dict[ComponentCategory, list[ComponentSpec]] = {cat: [] for cat in ComponentCategory}
        for item in catalog:
            if item.category in categorized:
                if (
                    constraints.preferred_gpu_brand.locked
                    and constraints.preferred_gpu_brand.value
                    and item.category == ComponentCategory.GPU
                ):
                    pref_b = constraints.preferred_gpu_brand.value.upper()
                    if pref_b not in (item.brand or "").upper() and pref_b not in item.name.upper():
                        continue
                if (
                    constraints.preferred_cpu_brand.locked
                    and constraints.preferred_cpu_brand.value
                    and item.category == ComponentCategory.CPU
                ):
                    pref_c = constraints.preferred_cpu_brand.value.upper()
                    if pref_c not in (item.brand or "").upper() and pref_c not in item.name.upper():
                        continue
                if (
                    constraints.form_factor_preference.locked
                    and constraints.form_factor_preference.value
                ):
                    pref_ff = _norm(constraints.form_factor_preference.value)
                    if (
                        item.category == ComponentCategory.MAINBOARD
                        and item.form_factor
                        and _norm(item.form_factor) != pref_ff
                    ):
                        continue
                    if item.category == ComponentCategory.CASE:
                        supported = [_norm(s) for s in (item.supported_form_factors or [item.form_factor or ""])]
                        if supported and pref_ff not in supported:
                            continue

                categorized[item.category].append(item)

        # 2. Candidate retrieval: partition into preferred envelope and all affordable
        # Apply profile- and objective-aware Pareto pruning across all objectives
        pruned_pools: dict[ComponentCategory, list[ComponentSpec]] = {}
        total_pruned_candidates = 0

        for cat, items in categorized.items():
            if cat in owned_map:
                owned_comp = owned_map[cat].component
                if owned_comp is not None:
                    pruned_pools[cat] = [owned_comp]
                    continue
                else:
                    raise ValueError(f"Owned part for category '{cat}' must specify a valid ComponentSpec.")

            pool = get_candidate_pool(items, budget, cat, profile)
            if self.enable_pruning:
                non_dominated = pareto_prune_all_objectives(pool.all_affordable, profile)
                total_pruned_candidates += len(pool.all_affordable) - len(non_dominated)
            else:
                non_dominated = list(pool.all_affordable)

            # Sort globally by (spending_price asc, stable_id asc)
            non_dominated.sort(key=lambda c: (get_spending_price(cat, c), stable_component_id(c)))
            pruned_pools[cat] = non_dominated

        cpus = pruned_pools.get(ComponentCategory.CPU, [])
        mainboards = pruned_pools.get(ComponentCategory.MAINBOARD, [])
        rams = pruned_pools.get(ComponentCategory.RAM, [])
        catalog_gpus = pruned_pools.get(ComponentCategory.GPU, [])
        cases = pruned_pools.get(ComponentCategory.CASE, [])
        catalog_coolers = pruned_pools.get(ComponentCategory.COOLER, [])
        storages = pruned_pools.get(ComponentCategory.STORAGE, [])
        psus = pruned_pools.get(ComponentCategory.PSU, [])

        best_builds: dict[
            BuildObjective,
            tuple[float, tuple[Any, ...], dict[ComponentCategory, ComponentSpec], PowerEstimate, ScoreBreakdown] | None,
        ] = {
            BuildObjective.PERFORMANCE: None,
            BuildObjective.BALANCED: None,
            BuildObjective.UPGRADE_FRIENDLY: None,
        }

        evaluated_combinations = 0

        # Deterministic constrained enumeration with early budget and compatibility pruning
        for cpu in cpus:
            # Build GPU options: dedicated catalog GPUs plus iGPU only if CPU has integrated graphics
            if ComponentCategory.GPU in owned_map:
                available_gpus = list(catalog_gpus)
            else:
                available_gpus = list(catalog_gpus)
                if (
                    cpu.has_integrated_graphics
                    and cpu.integrated_graphics_score is not None
                    and cpu.integrated_graphics_power_watts is not None
                ):
                    igpu_spec = ComponentSpec(
                        id=uuid5(SYNTHETIC_NAMESPACE, f"{stable_component_id(cpu)}:igpu"),
                        name=f"Đồ họa tích hợp {cpu.name} (iGPU)",
                        category=ComponentCategory.GPU,
                        price=0,
                        tdp_watts=cpu.integrated_graphics_power_watts,
                        performance_score=cpu.integrated_graphics_score,
                        is_integrated=True,
                        brand=cpu.brand,
                    )
                    available_gpus.append(igpu_spec)

            if not available_gpus:
                continue

            # CRITICAL FIX 1: Sort available_gpus by spending price ascending to prevent break bug
            available_gpus.sort(key=lambda c: (get_spending_price(ComponentCategory.GPU, c), stable_component_id(c)))

            # Build Cooler options: aftermarket catalog coolers plus stock cooler only if CPU includes one
            if ComponentCategory.COOLER in owned_map:
                available_coolers = list(catalog_coolers)
            else:
                available_coolers = list(catalog_coolers)
                if (
                    cpu.includes_stock_cooler
                    and cpu.stock_cooler_height_mm is not None
                    and cpu.stock_cooler_score is not None
                    and (cpu.stock_cooler_tdp_watts is None or cpu.stock_cooler_tdp_watts >= (cpu.tdp_watts or 65))
                ):
                    stock_cooler = ComponentSpec(
                        id=uuid5(SYNTHETIC_NAMESPACE, f"{stable_component_id(cpu)}:stock_cooler"),
                        name=f"Tản nhiệt kèm theo CPU {cpu.name} (Stock Cooler)",
                        category=ComponentCategory.COOLER,
                        price=0,
                        cooler_height_mm=cpu.stock_cooler_height_mm,
                        performance_score=cpu.stock_cooler_score,
                        tdp_watts=cpu.stock_cooler_tdp_watts,
                        supported_sockets=[cpu.socket] if cpu.socket else [],
                        brand=cpu.brand,
                    )
                    available_coolers.append(stock_cooler)

            if not available_coolers:
                continue

            # CRITICAL FIX 1: Sort available_coolers by spending price ascending to prevent break bug
            available_coolers.sort(key=lambda c: (get_spending_price(ComponentCategory.COOLER, c), stable_component_id(c)))

            for mb in mainboards:
                # Socket check
                if not cpu.socket or not mb.socket or _norm(cpu.socket) != _norm(mb.socket):
                    continue
                sub_1 = get_spending_price(ComponentCategory.CPU, cpu) + get_spending_price(ComponentCategory.MAINBOARD, mb)
                if sub_1 > budget:
                    break

                for ram in rams:
                    # RAM type check
                    if not mb.ram_type or not ram.ram_type or _norm(mb.ram_type) != _norm(ram.ram_type):
                        continue
                    sub_2 = sub_1 + get_spending_price(ComponentCategory.RAM, ram)
                    if sub_2 > budget:
                        break

                    for gpu in available_gpus:
                        sub_3 = sub_2 + get_spending_price(ComponentCategory.GPU, gpu)
                        if sub_3 > budget:
                            break

                        temp_parts = {
                            ComponentCategory.CPU: cpu,
                            ComponentCategory.MAINBOARD: mb,
                            ComponentCategory.RAM: ram,
                            ComponentCategory.GPU: gpu,
                        }
                        power = calculate_power_estimate(temp_parts, profile)

                        for case in cases:
                            # Form factor check: strict membership if supported_form_factors present, else ranking
                            if case.supported_form_factors:
                                if _norm(mb.form_factor) not in [_norm(s) for s in case.supported_form_factors]:
                                    continue
                            elif case.form_factor:
                                mb_rank = FORM_FACTOR_RANKS.get(_norm(mb.form_factor))
                                case_rank = FORM_FACTOR_RANKS.get(_norm(case.form_factor))
                                if mb_rank is None or case_rank is None or mb_rank > case_rank:
                                    continue
                            else:
                                continue

                            # Dedicated GPU length check
                            if (
                                not gpu.is_integrated
                                and gpu.gpu_length_mm
                                and case.max_gpu_length_mm
                                and gpu.gpu_length_mm > case.max_gpu_length_mm
                            ):
                                continue

                            sub_4 = sub_3 + get_spending_price(ComponentCategory.CASE, case)
                            if sub_4 > budget:
                                break

                            for cooler in available_coolers:
                                # Cooler height check
                                if cooler.cooler_height_mm and case.max_cooler_height_mm and cooler.cooler_height_mm > case.max_cooler_height_mm:
                                    continue

                                # Cooler socket check
                                if cooler.supported_sockets:
                                    if not cpu.socket or _norm(cpu.socket) not in [_norm(s) for s in cooler.supported_sockets]:
                                        continue
                                elif cooler.socket and (
                                    not cpu.socket or _norm(cpu.socket) != _norm(cooler.socket)
                                ):
                                    continue

                                sub_5 = sub_4 + get_spending_price(ComponentCategory.COOLER, cooler)
                                if sub_5 > budget:
                                    break

                                for storage in storages:
                                    sub_6 = sub_5 + get_spending_price(ComponentCategory.STORAGE, storage)
                                    if sub_6 > budget:
                                        break

                                    for psu in psus:
                                        if psu.wattage and psu.wattage < power.minimum_capacity_watts:
                                            continue
                                        total_spending = sub_6 + get_spending_price(ComponentCategory.PSU, psu)
                                        if total_spending > budget:
                                            break

                                        full_parts = {
                                            ComponentCategory.CPU: cpu,
                                            ComponentCategory.MAINBOARD: mb,
                                            ComponentCategory.RAM: ram,
                                            ComponentCategory.GPU: gpu,
                                            ComponentCategory.CASE: case,
                                            ComponentCategory.COOLER: cooler,
                                            ComponentCategory.STORAGE: storage,
                                            ComponentCategory.PSU: psu,
                                        }

                                        # Enforce complete physical and electrical compatibility
                                        compat_status = check_compatibility_status(
                                            full_parts, power.minimum_capacity_watts
                                        )
                                        if compat_status != CompatibilityStatus.COMPATIBLE:
                                            continue

                                        evaluated_combinations += 1
                                        tie_key = build_tie_key(full_parts, total_spending)

                                        # Score for all three objectives
                                        for obj in BuildObjective:
                                            breakdown = score_build(
                                                full_parts,
                                                obj,
                                                profile,
                                                budget,
                                                power,
                                                self.platform_policy,
                                                total_spending=total_spending,
                                            )
                                            if breakdown.total_score < 0.0:
                                                # Reject build with missing required scoring metric
                                                continue

                                            score_val = breakdown.total_score
                                            current_record = best_builds[obj]
                                            if current_record is None or score_val > current_record[0]:
                                                best_builds[obj] = (score_val, tie_key, full_parts, power, breakdown)
                                            elif abs(score_val - current_record[0]) < 1e-6 and tie_key < current_record[1]:
                                                # Deterministic tie-breaking: lower price first, then lexicographical ID
                                                best_builds[obj] = (score_val, tie_key, full_parts, power, breakdown)

        # Build output structures
        ranked_builds: dict[BuildObjective, RankedBuild] = {}
        for obj in BuildObjective:
            record = best_builds[obj]
            if record is not None:
                score_val, _, parts, p_est, breakdown = record
                evidence = _generate_factual_evidence(parts, obj, p_est, score_val, breakdown)
                spending = sum(get_spending_price(cat, p) for cat, p in parts.items())
                ranked_builds[obj] = RankedBuild(
                    objective=obj,
                    objective_score=score_val,
                    total_price=spending,
                    parts=parts,
                    power_estimate=p_est,
                    evidence=evidence,
                    compatibility_status=CompatibilityStatus.COMPATIBLE,
                )

        if not ranked_builds:
            raise ValueError(f"No compatible build configuration found within budget {budget:,} VND.")

        return OptimizationResult(
            target_budget_vnd=budget,
            use_case=profile,
            builds=ranked_builds,
            candidates_evaluated=evaluated_combinations,
            pruned_count=total_pruned_candidates,
        )


__all__ = [
    "DEFAULT_PLATFORM_POLICY",
    "FIXED_CATEGORY_ORDER",
    "FORM_FACTOR_RANKS",
    "SCORING_POLICIES",
    "SYNTHETIC_NAMESPACE",
    "CompatibilityStatus",
    "DeterministicPCOptimizer",
    "build_tie_key",
    "calculate_power_estimate",
    "check_compatibility",
    "check_compatibility_status",
    "round_up_psu",
    "score_build",
    "stable_component_id",
]
