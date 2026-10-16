"""Application-owned source selection; domains are configuration, not model arguments."""

import re
from enum import StrEnum
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.web_search.research_contracts import HardwareResearchFocus


class SourceTier(StrEnum):
    OFFICIAL = "OFFICIAL"
    INDEPENDENT_REVIEW = "INDEPENDENT_REVIEW"
    GENERAL_WEB = "GENERAL_WEB"
    STORE_LISTING = "STORE_LISTING"


class OfficialSourceStatus(StrEnum):
    RESOLVED = "RESOLVED"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"


# Initial coverage, not an exhaustive or immutable list of trustworthy websites.
DEFAULT_MANUFACTURERS: dict[str, list[str]] = {
    "amd": ["amd.com"],
    "nvidia": ["nvidia.com"],
    "intel": ["intel.com"],
    "asus": ["asus.com"],
    "msi": ["msi.com"],
    "gigabyte": ["gigabyte.com"],
    "asrock": ["asrock.com"],
    "corsair": ["corsair.com"],
    "kingston": ["kingston.com"],
    "crucial": ["crucial.com"],
    "samsung": ["samsung.com"],
    "western digital": ["westerndigital.com"],
    "seagate": ["seagate.com"],
    "deepcool": ["deepcool.com"],
    "thermalright": ["thermalright.com"],
    "cooler master": ["coolermaster.com"],
    "be quiet": ["bequiet.com"],
    "noctua": ["noctua.at"],
    "seasonic": ["seasonic.com"],
    "nzxt": ["nzxt.com"],
    "lian li": ["lian-li.com"],
    "fractal design": ["fractal-design.com"],
    "g skill": ["gskill.com"],
    "blender": ["blender.org"],
    "adobe": ["adobe.com"],
}
DEFAULT_ALIASES = {
    "ryzen": "amd",
    "radeon": "amd",
    "geforce": "nvidia",
    "rtx": "nvidia",
    "rog": "asus",
    "aorus": "gigabyte",
    "wd": "western digital",
    "gskill": "g skill",
    "fractal": "fractal design",
}
DEFAULT_PUBLISHERS = {
    "blender": ["blender.org"],
    "premiere pro": ["helpx.adobe.com"],
    "adobe premiere": ["helpx.adobe.com"],
    "photoshop": ["helpx.adobe.com"],
    "after effects": ["helpx.adobe.com"],
    "autocad": ["autodesk.com"],
}
DEFAULT_REVIEWS: dict[str, list[str]] = {
    HardwareResearchFocus.GENERAL: [
        "techpowerup.com",
        "tomshardware.com",
        "gamersnexus.net",
        "techspot.com",
    ],
    HardwareResearchFocus.PERFORMANCE: [
        "techpowerup.com",
        "techspot.com",
        "computerbase.de",
        "gamersnexus.net",
        "tomshardware.com",
    ],
    HardwareResearchFocus.PRODUCTIVITY: [
        "pugetsystems.com",
        "phoronix.com",
        "techpowerup.com",
    ],
    HardwareResearchFocus.THERMAL_NOISE: [
        "gamersnexus.net",
        "techpowerup.com",
        "tomshardware.com",
        "kitguru.net",
    ],
    HardwareResearchFocus.MOTHERBOARD: [
        "techpowerup.com",
        "tomshardware.com",
        "techspot.com",
        "kitguru.net",
    ],
    HardwareResearchFocus.MEMORY: [
        "techpowerup.com",
        "tomshardware.com",
        "techspot.com",
    ],
    HardwareResearchFocus.PSU_QUALITY: [
        "hwbusters.com",
        "cybenetics.com",
        "tomshardware.com",
        "techpowerup.com",
    ],
    HardwareResearchFocus.PSU_TIER: ["cultists.network", "hwbusters.com"],
    HardwareResearchFocus.STORAGE: [
        "storagereview.com",
        "techpowerup.com",
        "tomshardware.com",
    ],
    HardwareResearchFocus.CASE_AIRFLOW: [
        "gamersnexus.net",
        "techpowerup.com",
        "tomshardware.com",
        "kitguru.net",
    ],
}


def normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def within_domains(url: str, domains: list[str]) -> bool:
    host = urlsplit(url).hostname or ""
    return any(host == domain or host.endswith("." + domain) for domain in domains)


class ResearchSourcePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = Field(default="manufacturer-first-v1", min_length=1)
    manufacturers: dict[str, list[str]] = Field(
        default_factory=lambda: {
            key: list(value) for key, value in DEFAULT_MANUFACTURERS.items()
        }
    )
    aliases: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_ALIASES))
    chip_designers: list[str] = Field(
        default_factory=lambda: ["amd", "intel", "nvidia"]
    )
    reviews: dict[str, list[str]] = Field(
        default_factory=lambda: {
            key: list(value) for key, value in DEFAULT_REVIEWS.items()
        }
    )
    allow_general_fallback: bool = False
    publishers: dict[str, list[str]] = Field(
        default_factory=lambda: {
            key: list(value) for key, value in DEFAULT_PUBLISHERS.items()
        }
    )
    requirement_stores: list[str] = Field(
        default_factory=lambda: ["store.steampowered.com"]
    )

    @model_validator(mode="after")
    def valid_registry(self) -> "ResearchSourcePolicy":
        for registry in (
            self.manufacturers,
            self.reviews,
            self.publishers,
            {"stores": self.requirement_stores},
        ):
            for domains in registry.values():
                if not domains or any(
                    not re.fullmatch(r"[a-z0-9]+(?:[.-][a-z0-9]+)*\.[a-z]{2,}", domain)
                    for domain in domains
                ):
                    raise ValueError(
                        "Source registry requires bare lowercase domain names"
                    )
        return self

    @property
    def digest(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))

    def official_domains(self, subject: str, brand: str | None = None) -> list[str]:
        if brand:
            key = normalized(brand)
            return self.manufacturers.get(self.aliases.get(key, key), [])
        subject = f" {normalized(subject)} "
        # Explicit board/part manufacturer beats a chipset family alias (ASUS RTX).
        direct = [key for key in self.manufacturers if f" {key} " in subject]
        board_partners = [key for key in direct if key not in self.chip_designers]
        if len(board_partners) == 1:
            return self.manufacturers[board_partners[0]]
        if len(direct) == 1:
            return self.manufacturers[direct[0]]
        if len(direct) > 1:
            return []
        inferred = {
            brand for alias, brand in self.aliases.items() if f" {alias} " in subject
        }
        return (
            self.manufacturers.get(next(iter(inferred)), [])
            if len(inferred) == 1
            else []
        )

    def review_domains(self, focus: str) -> list[str]:
        return self.reviews.get(
            focus, self.reviews.get(HardwareResearchFocus.GENERAL, [])
        )

    def publisher_domains(self, workload: str) -> list[str]:
        subject = f" {normalized(workload)} "
        matches = [key for key in self.publishers if f" {normalized(key)} " in subject]
        return self.publishers[max(matches, key=len)] if matches else []
