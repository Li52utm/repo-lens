from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class ECBSeriesDefinition:
    """
    Define one approved ECB series used by RepoLens.
    """

    name: str
    dataflow: str
    series_key: str
    unit: str
    classification: str
    description: str


ESTR_RATE: Final[ECBSeriesDefinition] = ECBSeriesDefinition(
    name="Euro short-term rate",
    dataflow="EST",
    series_key="B.EU000A2X2A25.WT",
    unit="Percent per annum",
    classification="Official",
    description=(
        "Wholesale unsecured overnight euro "
        "borrowing costs of euro-area banks."
    ),
)