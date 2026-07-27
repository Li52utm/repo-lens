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
    output_filename: str


ESTR_RATE: Final[ECBSeriesDefinition] = ECBSeriesDefinition(
    name="Euro short-term rate",
    dataflow="EST",
    series_key="B.EU000A2X2A25.WT",
    unit="Percent per annum",
    classification="Official",
    description=(
        "Volume-weighted trimmed mean rate representing "
        "wholesale unsecured overnight euro borrowing costs."
    ),
    output_filename="estr_rate.csv",
)


ECB_DEPOSIT_FACILITY_RATE: Final[ECBSeriesDefinition] = (
    ECBSeriesDefinition(
        name="ECB deposit facility rate",
        dataflow="FM",
        series_key="B.U2.EUR.4F.KR.DFR.LEV",
        unit="Percent per annum",
        classification="Official",
        description=(
            "Official ECB deposit facility policy rate."
        ),
        output_filename="deposit_facility_rate.csv",
    )
)


ESTR_TOTAL_VOLUME: Final[ECBSeriesDefinition] = (
    ECBSeriesDefinition(
        name="€STR total volume",
        dataflow="EST",
        series_key="B.EU000A2X2A25.TT",
        unit="Millions of euro",
        classification="Official",
        description=(
            "Total eligible transaction volume underlying "
            "the daily €STR calculation."
        ),
        output_filename="estr_total_volume.csv",
    )
)


ESTR_RATE_25TH_PERCENTILE: Final[ECBSeriesDefinition] = (
    ECBSeriesDefinition(
        name="€STR rate at 25th percentile of volume",
        dataflow="EST",
        series_key="B.EU000A2X2A25.R25",
        unit="Percent per annum",
        classification="Official",
        description=(
            "Transaction rate at the 25th percentile "
            "of cumulative eligible volume."
        ),
        output_filename="estr_rate_25th_percentile.csv",
    )
)


ESTR_RATE_75TH_PERCENTILE: Final[ECBSeriesDefinition] = (
    ECBSeriesDefinition(
        name="€STR rate at 75th percentile of volume",
        dataflow="EST",
        series_key="B.EU000A2X2A25.R75",
        unit="Percent per annum",
        classification="Official",
        description=(
            "Transaction rate at the 75th percentile "
            "of cumulative eligible volume."
        ),
        output_filename="estr_rate_75th_percentile.csv",
    )
)


ESTR_ACTIVE_BANKS: Final[ECBSeriesDefinition] = (
    ECBSeriesDefinition(
        name="€STR number of active banks",
        dataflow="EST",
        series_key="B.EU000A2X2A25.NB",
        unit="Count",
        classification="Official",
        description=(
            "Number of reporting banks contributing "
            "eligible transactions to the daily €STR."
        ),
        output_filename="estr_active_banks.csv",
    )
)


ESTR_TRANSACTION_COUNT: Final[ECBSeriesDefinition] = (
    ECBSeriesDefinition(
        name="€STR number of transactions",
        dataflow="EST",
        series_key="B.EU000A2X2A25.NT",
        unit="Count",
        classification="Official",
        description=(
            "Number of eligible transactions underlying "
            "the daily €STR calculation."
        ),
        output_filename="estr_transaction_count.csv",
    )
)


ESTR_MARKET_QUALITY_SERIES: Final[
    tuple[ECBSeriesDefinition, ...]
] = (
    ESTR_TOTAL_VOLUME,
    ESTR_RATE_25TH_PERCENTILE,
    ESTR_RATE_75TH_PERCENTILE,
    ESTR_ACTIVE_BANKS,
    ESTR_TRANSACTION_COUNT,
)


APPROVED_ECB_SERIES: Final[
    tuple[ECBSeriesDefinition, ...]
] = (
    ESTR_RATE,
    ECB_DEPOSIT_FACILITY_RATE,
    *ESTR_MARKET_QUALITY_SERIES,
)