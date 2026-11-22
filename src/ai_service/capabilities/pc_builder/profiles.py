"""JSON-backed registry of PC builder use-case profiles.

Loads profile envelopes, objective scoring weights, saturation thresholds, and
workload guardrails from ``profiles.json`` (or an override path configured via
``AI_PC_BUILDER_PROFILES_PATH``), validating every profile at load time.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_service.application.ports.hardware import ComponentCategory
from ai_service.capabilities.pc_builder.schemas import BuildObjective, UseCaseProfile

DEFAULT_PROFILES_PATH = Path(__file__).with_name("profiles.json")
PROFILES_PATH_ENV = "AI_PC_BUILDER_PROFILES_PATH"


@dataclass(frozen=True, slots=True)
class EnvelopeRange:
    preferred_min: float
    preferred_max: float
    absolute_min: float
    absolute_max: float


@dataclass(frozen=True, slots=True)
class ProfileSpec:
    """Single source of truth for one UseCaseProfile's optimization behavior."""

    profile: UseCaseProfile
    default_objective: BuildObjective
    envelopes: dict[ComponentCategory, EnvelopeRange]
    scoring_weights: dict[BuildObjective, dict[str, float]]
    ram_saturation_gb: float = 32.0
    storage_saturation_gb: float = 1000.0
    allow_igpu_only: bool = False
    igpu_bonus: float = 0.0
    bottleneck_threshold: float | None = None
    bottleneck_slope: float = 0.4
    bottleneck_max_penalty: float = 12.0
    cuda_bonus: float = 0.0
    multicore_max_bonus: float = 0.0

    @property
    def uses_vram(self) -> bool:
        return any(
            weights.get("gpu_vram", 0.0) > 0.0
            for weights in self.scoring_weights.values()
        )


def _parse_envelope(raw: Any) -> EnvelopeRange:
    if isinstance(raw, list) and len(raw) == 4:
        return EnvelopeRange(float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
    if isinstance(raw, dict):
        return EnvelopeRange(
            float(raw["preferred_min"]),
            float(raw["preferred_max"]),
            float(raw["absolute_min"]),
            float(raw["absolute_max"]),
        )
    raise ValueError(f"Invalid envelope range: {raw!r}")


def load_profile_registry(path: Path | str | None = None) -> dict[UseCaseProfile, ProfileSpec]:
    """Load and validate all UseCaseProfile specifications from a JSON file."""
    env_override = os.environ.get(PROFILES_PATH_ENV, "").strip()
    source = Path(path) if path is not None else (Path(env_override) if env_override else DEFAULT_PROFILES_PATH)
    raw_data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw_data, dict):
        raise TypeError("Profiles JSON root must be an object mapping UseCaseProfile -> spec")

    registry: dict[UseCaseProfile, ProfileSpec] = {}
    for profile_key, entry in raw_data.items():
        profile = UseCaseProfile(profile_key)
        envelopes = {
            ComponentCategory(cat_key): _parse_envelope(bounds)
            for cat_key, bounds in entry["envelopes"].items()
        }
        scoring_weights = {
            BuildObjective(obj_key): {str(metric): float(weight) for metric, weight in metrics.items()}
            for obj_key, metrics in entry["scoring_weights"].items()
        }
        registry[profile] = ProfileSpec(
            profile=profile,
            default_objective=BuildObjective(entry["default_objective"]),
            envelopes=envelopes,
            scoring_weights=scoring_weights,
            ram_saturation_gb=float(entry.get("ram_saturation_gb", 32.0)),
            storage_saturation_gb=float(entry.get("storage_saturation_gb", 1000.0)),
            allow_igpu_only=bool(entry.get("allow_igpu_only", False)),
            igpu_bonus=float(entry.get("igpu_bonus", 0.0)),
            bottleneck_threshold=(
                float(entry["bottleneck_threshold"])
                if entry.get("bottleneck_threshold") is not None
                else None
            ),
            bottleneck_slope=float(entry.get("bottleneck_slope", 0.4)),
            bottleneck_max_penalty=float(entry.get("bottleneck_max_penalty", 12.0)),
            cuda_bonus=float(entry.get("cuda_bonus", 0.0)),
            multicore_max_bonus=float(entry.get("multicore_max_bonus", 0.0)),
        )

    for required_profile in UseCaseProfile:
        if required_profile not in registry:
            raise ValueError(f"Missing ProfileSpec for {required_profile}")
        spec = registry[required_profile]
        for required_objective in BuildObjective:
            if required_objective not in spec.scoring_weights:
                raise ValueError(f"Missing scoring policy for ({required_profile}, {required_objective})")
            weight_sum = sum(spec.scoring_weights[required_objective].values())
            if abs(weight_sum - 1.0) > 1e-6:
                raise ValueError(
                    f"Scoring policy ({required_profile}, {required_objective}) weights sum to {weight_sum}, must equal 1.0"
                )
    return registry


PROFILE_REGISTRY: dict[UseCaseProfile, ProfileSpec] = load_profile_registry()


def get_profile_spec(profile: UseCaseProfile) -> ProfileSpec:
    spec = PROFILE_REGISTRY.get(profile)
    if spec is None:
        raise ValueError(f"No profile specification registered for {profile}.")
    return spec


def blend_profile_specs(
    primary: ProfileSpec,
    secondary: ProfileSpec,
    secondary_weight: float = 0.30,
) -> ProfileSpec:
    """Linearly blend two ProfileSpecs (e.g. 70% Gaming + 30% Software Dev).

    Preserves weight sum == 1.0 for every BuildObjective and conservatively
    requires both profiles to allow iGPU-only builds before enabling allow_igpu_only.
    """
    if not (0.0 <= secondary_weight <= 0.5):
        raise ValueError("secondary_weight must be between 0.0 and 0.5")
    if secondary_weight == 0.0 or primary.profile == secondary.profile:
        return primary

    w1 = 1.0 - secondary_weight
    w2 = secondary_weight

    # Blend envelopes
    envelopes: dict[ComponentCategory, EnvelopeRange] = {}
    all_cats = set(primary.envelopes) | set(secondary.envelopes)
    default_env = EnvelopeRange(0.0, 1.0, 0.0, 1.0)
    for cat in all_cats:
        e1 = primary.envelopes.get(cat, default_env)
        e2 = secondary.envelopes.get(cat, default_env)
        envelopes[cat] = EnvelopeRange(
            preferred_min=round(e1.preferred_min * w1 + e2.preferred_min * w2, 4),
            preferred_max=round(e1.preferred_max * w1 + e2.preferred_max * w2, 4),
            absolute_min=round(e1.absolute_min * w1 + e2.absolute_min * w2, 4),
            absolute_max=round(e1.absolute_max * w1 + e2.absolute_max * w2, 4),
        )

    # Blend scoring weights per objective and normalize strictly to 1.0
    scoring_weights: dict[BuildObjective, dict[str, float]] = {}
    for obj in BuildObjective:
        m1 = primary.scoring_weights[obj]
        m2 = secondary.scoring_weights[obj]
        merged = {
            key: m1.get(key, 0.0) * w1 + m2.get(key, 0.0) * w2
            for key in (set(m1) | set(m2))
        }
        total = sum(merged.values())
        scoring_weights[obj] = {
            key: val / total for key, val in merged.items() if val > 0.0
        }

    # Blend bottleneck guardrail
    if primary.bottleneck_threshold is not None and secondary.bottleneck_threshold is not None:
        bn_threshold: float | None = round(primary.bottleneck_threshold * w1 + secondary.bottleneck_threshold * w2, 4)
        bn_slope = round(primary.bottleneck_slope * w1 + secondary.bottleneck_slope * w2, 4)
        bn_max = round(primary.bottleneck_max_penalty * w1 + secondary.bottleneck_max_penalty * w2, 4)
    elif primary.bottleneck_threshold is not None:
        bn_threshold = primary.bottleneck_threshold
        bn_slope = primary.bottleneck_slope
        bn_max = round(primary.bottleneck_max_penalty * w1, 4)
    elif secondary.bottleneck_threshold is not None:
        bn_threshold = secondary.bottleneck_threshold
        bn_slope = secondary.bottleneck_slope
        bn_max = round(secondary.bottleneck_max_penalty * w2, 4)
    else:
        bn_threshold = None
        bn_slope = primary.bottleneck_slope
        bn_max = primary.bottleneck_max_penalty

    allow_igpu = primary.allow_igpu_only and secondary.allow_igpu_only

    return ProfileSpec(
        profile=primary.profile,
        default_objective=primary.default_objective,
        envelopes=envelopes,
        scoring_weights=scoring_weights,
        ram_saturation_gb=round(primary.ram_saturation_gb * w1 + secondary.ram_saturation_gb * w2, 2),
        storage_saturation_gb=round(primary.storage_saturation_gb * w1 + secondary.storage_saturation_gb * w2, 2),
        allow_igpu_only=allow_igpu,
        igpu_bonus=round(primary.igpu_bonus * w1 + secondary.igpu_bonus * w2, 4) if allow_igpu else 0.0,
        bottleneck_threshold=bn_threshold,
        bottleneck_slope=bn_slope,
        bottleneck_max_penalty=bn_max,
        cuda_bonus=round(primary.cuda_bonus * w1 + secondary.cuda_bonus * w2, 4),
        multicore_max_bonus=round(primary.multicore_max_bonus * w1 + secondary.multicore_max_bonus * w2, 4),
    )


def resolve_effective_profile_spec(
    primary: UseCaseProfile,
    secondary: UseCaseProfile | None = None,
    secondary_weight: float = 0.30,
) -> ProfileSpec:
    """Resolve a single or blended ProfileSpec for optimization."""
    primary_spec = get_profile_spec(primary)
    if secondary is None or secondary == primary or secondary_weight <= 0.0:
        return primary_spec
    return blend_profile_specs(primary_spec, get_profile_spec(secondary), secondary_weight)


# Backwards-compatible derived views
DEFAULT_ENVELOPES: dict[UseCaseProfile, dict[ComponentCategory, EnvelopeRange]] = {
    profile: spec.envelopes for profile, spec in PROFILE_REGISTRY.items()
}

SCORING_POLICIES: dict[tuple[UseCaseProfile, BuildObjective], dict[str, float]] = {
    (profile, objective): weights
    for profile, spec in PROFILE_REGISTRY.items()
    for objective, weights in spec.scoring_weights.items()
}

DEFAULT_OBJECTIVE_BY_PROFILE: dict[UseCaseProfile, BuildObjective] = {
    profile: spec.default_objective for profile, spec in PROFILE_REGISTRY.items()
}

__all__ = [
    "DEFAULT_ENVELOPES",
    "DEFAULT_OBJECTIVE_BY_PROFILE",
    "DEFAULT_PROFILES_PATH",
    "PROFILES_PATH_ENV",
    "PROFILE_REGISTRY",
    "SCORING_POLICIES",
    "EnvelopeRange",
    "ProfileSpec",
    "blend_profile_specs",
    "get_profile_spec",
    "load_profile_registry",
    "resolve_effective_profile_spec",
]
