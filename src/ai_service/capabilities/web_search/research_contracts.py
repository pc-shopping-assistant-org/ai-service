"""Strict research requests. Names and conditions are search subjects, not facts."""

from enum import StrEnum
from typing import Annotated
from uuid import UUID

from pydantic import Field, StrictInt, model_validator

from ai_service.application.ports.hardware import ComponentCategory
from ai_service.capabilities.assistant.stateful_contracts import Contract, Text

MAX_RESEARCH_SOURCES = 2
RESEARCH_PAGE_CHARS = 2000
MAX_BUILD_RESEARCH_PARTS = len(ComponentCategory)
MAX_WORKLOAD_HARDWARE = 3
MAX_RESEARCH_SUBJECTS = MAX_WORKLOAD_HARDWARE + 1
MAX_RESEARCH_SEARCH_RESULTS = 5
MAX_RESEARCH_SOURCE_TIERS = 3

type ResearchSubject = Annotated[Text, Field(max_length=300)]
type ResearchQuestion = Annotated[Text, Field(max_length=500)]
type ResearchCondition = Annotated[Text, Field(max_length=100)]


class ResearchSourceStatus(StrEnum):
    READABLE = "READABLE"
    NO_RESULTS = "NO_RESULTS"
    UNREADABLE = "UNREADABLE"
    UNAVAILABLE = "UNAVAILABLE"
    SOURCE_UNRESOLVED = "SOURCE_UNRESOLVED"


class ResearchMode(StrEnum):
    BENCHMARK = "BENCHMARK"
    REQUIREMENTS = "REQUIREMENTS"


class WorkloadBenchmarkKind(StrEnum):
    GAMING = "GAMING"
    PRODUCTIVITY = "PRODUCTIVITY"


class SoftwareType(StrEnum):
    VGA_DRIVER = "VGA_DRIVER"
    BIOS_UPDATE = "BIOS_UPDATE"
    CHIPSET_DRIVER = "CHIPSET_DRIVER"
    RGB_CONTROL = "RGB_CONTROL"


class BenchmarkCondition(StrEnum):
    HARDWARE = "hardware"
    RESOLUTION = "resolution"
    PRESET = "preset"
    UPSCALING = "upscaling"


class HardwareResearchFocus(StrEnum):
    GENERAL = "GENERAL"
    PERFORMANCE = "PERFORMANCE"
    PRODUCTIVITY = "PRODUCTIVITY"
    THERMAL_NOISE = "THERMAL_NOISE"
    MOTHERBOARD = "MOTHERBOARD"
    MEMORY = "MEMORY"
    STORAGE = "STORAGE"
    PSU_QUALITY = "PSU_QUALITY"
    PSU_TIER = "PSU_TIER"
    CASE_AIRFLOW = "CASE_AIRFLOW"
    KNOWN_ISSUES = "KNOWN_ISSUES"
    OFFICIAL_SPECS = "OFFICIAL_SPECS"


class ResearchHardwareArgs(Contract):
    """Search and read reviews/specifications of an exact model, focused on its role."""

    model: ResearchSubject
    focus: HardwareResearchFocus = HardwareResearchFocus.GENERAL
    question: ResearchQuestion | None = None
    max_sources: StrictInt = Field(
        default=MAX_RESEARCH_SOURCES,
        ge=1,
        le=MAX_RESEARCH_SOURCES,
        description="Maximum pages per source tier (official, independent review).",
    )


class ResearchCompatibilityArgs(Contract):
    """Read sources about a pair: BIOS, RAM clearance, fit or power connectors."""

    part_a: ResearchSubject
    part_b: ResearchSubject
    concern: ResearchQuestion | None = None
    max_sources: StrictInt = Field(
        default=MAX_RESEARCH_SOURCES,
        ge=1,
        le=MAX_RESEARCH_SOURCES,
        description="Maximum pages per source tier.",
    )


class ResearchWorkloadArgs(Contract):
    """Read game/application requirements or benchmarks; omit unknown test conditions."""

    workload: ResearchSubject
    mode: ResearchMode = ResearchMode.BENCHMARK
    benchmark_kind: WorkloadBenchmarkKind = WorkloadBenchmarkKind.GAMING
    hardware: list[ResearchSubject] = Field(
        default_factory=list, max_length=MAX_WORKLOAD_HARDWARE
    )
    resolution: ResearchCondition | None = None
    preset: ResearchCondition | None = None
    upscaling: ResearchCondition | None = None
    max_sources: StrictInt = Field(
        default=MAX_RESEARCH_SOURCES,
        ge=1,
        le=MAX_RESEARCH_SOURCES,
        description="Maximum pages per source tier.",
    )


class ResearchPCBuildArgs(Contract):
    """Research each selected PC part. Select a current build_ref, or omit for the saved build."""

    build_ref: UUID | None = None
    slots: list[ComponentCategory] | None = Field(
        default=None, min_length=1, max_length=MAX_BUILD_RESEARCH_PARTS
    )
    question: ResearchQuestion | None = None

    @model_validator(mode="after")
    def distinct_slots(self) -> "ResearchPCBuildArgs":
        if self.slots and len(set(self.slots)) != len(self.slots):
            raise ValueError("Research slots must be distinct")
        return self


class ResearchDriverArgs(Contract):
    """Read official manufacturer driver/BIOS/support pages. Never download or install software."""

    component_name: ResearchSubject
    software_type: SoftwareType = SoftwareType.VGA_DRIVER
    operating_system: ResearchCondition | None = None
    version: ResearchCondition | None = None
    max_sources: StrictInt = Field(
        default=MAX_RESEARCH_SOURCES, ge=1, le=MAX_RESEARCH_SOURCES
    )
