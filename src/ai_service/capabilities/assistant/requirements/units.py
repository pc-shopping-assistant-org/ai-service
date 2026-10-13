"""Finite quantity normalization from source spans, not an expression DSL.

Nominal TB = 1000 GB. Binary units remain distinct/unsupported by the compiler.
Ambiguous separators or multiple different quantities require narrower evidence.
"""

import re
from decimal import Decimal
from typing import Literal

from pydantic import Field, StrictInt

from ai_service.capabilities.assistant.stateful_contracts import (
    Contract,
    QuantityUnit,
    Text,
)

from .contracts import QuantityValue, SourceEvidence


class UnresolvedQuantityValue(Contract):
    """Extraction-only quantity; never enters canonical requirements."""

    kind: Literal["UNRESOLVED_QUANTITY"] = "UNRESOLVED_QUANTITY"
    amount: StrictInt = Field(ge=0)


class QuantityConfirmation(Contract):
    value: QuantityValue
    evidence: SourceEvidence
    message: Text = Field(max_length=4000)


UNIT_ALIASES: dict[str, tuple[QuantityUnit, int]] = {
    "vnd": ("VND", 1),
    "vnđ": ("VND", 1),
    "đ": ("VND", 1),
    "triệu": ("VND", 1_000_000),
    "trieu": ("VND", 1_000_000),
    "million": ("VND", 1_000_000),
    "tr": ("VND", 1_000_000),
    "gb": ("GB", 1),
    "tb": ("TB", 1),
    "gib": ("GiB", 1),
    "tib": ("TiB", 1),
    "w": ("W", 1),
    "fps": ("FPS", 1),
}


def unresolved_amount(quote: str) -> int | None:
    if MEASUREMENT.search(quote):
        return None
    numbers = re.findall(r"(?<![\w.,-])\d+(?![\w.,])", quote)
    if len(numbers) != 1 or len(numbers[0]) > 32:
        return None
    return int(numbers[0])


def confirmed_quantity(amount: int, quote: str, unit: QuantityUnit) -> QuantityValue:
    alias = UNIT_ALIASES.get(quote.strip().casefold())
    if alias is None or alias[0] != unit:
        raise ValueError("Unbound quantity unit confirmation")
    return normalize_quantity(QuantityValue(amount=amount * alias[1], unit=unit))


MEASUREMENT = re.compile(
    r"(?<![\w.-])(\d+(?:[.,]\d+)*)\s*(triệu|trieu|million|vnđ|vnd|gib|tib|gb|tb|fps|w|đ|tr)(?!\w)",
    re.IGNORECASE,
)


def normalize_quantity(value: QuantityValue) -> QuantityValue:
    if value.unit == "TB":
        return QuantityValue(amount=value.amount * 1000, unit="GB")
    return value


def verify_quantity(
    quote: str, value: QuantityValue
) -> tuple[QuantityValue | None, str]:
    expected = normalize_quantity(value)
    measured: set[tuple[int, str]] = set()
    ambiguous = False
    for match in MEASUREMENT.finditer(quote):
        number, alias = match.groups()
        alias = alias.casefold()
        money = alias in {"triệu", "trieu", "million", "vnđ", "vnd", "đ", "tr"}
        if (expected.unit == "VND") != money:
            continue
        unit = (
            "VND" if money else {"gib": "GiB", "tib": "TiB"}.get(alias, alias.upper())
        )
        capacity_units = {"GB", "TB", "GiB", "TiB"}
        if unit != expected.unit and not (
            unit in capacity_units and expected.unit in capacity_units
        ):
            continue
        if len(number) > 32:
            return None, "QUANTITY_EVIDENCE_MISMATCH"
        if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", number):
            if not money or alias in {"triệu", "trieu", "million", "tr"}:
                ambiguous = True
                continue
            number = number.replace(".", "").replace(",", "")
        elif number.count(".") + number.count(",") > 1:
            ambiguous = True
            continue
        multiplier = (
            1_000_000
            if alias in {"triệu", "trieu", "million", "tr"}
            else 1000
            if unit == "TB"
            else 1
        )
        amount = Decimal(number.replace(",", ".")) * multiplier
        if amount != amount.to_integral_value():
            ambiguous = True
            continue
        measured.add((int(amount), "GB" if unit == "TB" else unit))
    if ambiguous or len(measured) > 1:
        return None, "QUANTITY_EVIDENCE_AMBIGUOUS"
    if measured != {(expected.amount, expected.unit)}:
        return None, "QUANTITY_EVIDENCE_MISMATCH"
    return expected, "QUANTITY_NORMALIZED"
