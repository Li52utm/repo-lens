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
        "Wholesale unsecured overnight euro borrowing "
        "costs of euro-area banks."
    ),
)


ECB_DEPOSIT_FACILITY_RATE: Final[ECBSeriesDefinition] = (
    ECBSeriesDefinition(
        name="ECB deposit facility rate",
        dataflow="FM",
        series_key="B.U2.EUR.4F.KR.DFR.LEV",
        unit="Percent per annum",
        classification="Official",
        description=(
            "ECB deposit facility policy rate, recorded "
            "on official rate-change dates."
        ),
    )
)


APPROVED_ECB_SERIES: Final[
    tuple[ECBSeriesDefinition, ...]
] = (
    ESTR_RATE,
    ECB_DEPOSIT_FACILITY_RATE,
)