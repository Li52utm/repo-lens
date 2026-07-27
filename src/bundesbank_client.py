from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Final

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.sovereign_market_data import (
    DataStatus,
    SovereignBenchmarkDefinition,
    SovereignDataValidationError,
    business_days_between,
    classify_data_status,
    validate_benchmark_observations,
)


BUNDESBANK_API_BASE_URL: Final[str] = (
    "https://api.statistiken.bundesbank.de/rest/data"
)

SDMX_CSV_ACCEPT_HEADER: Final[str] = (
    "application/vnd.sdmx.data+csv;version=1.0.0"
)

DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0


class BundesbankClientError(RuntimeError):
    """
    Base exception for the Bundesbank API client.
    """


class BundesbankRequestError(BundesbankClientError):
    """
    Raised when a Bundesbank HTTP request fails.
    """


class BundesbankResponseError(BundesbankClientError):
    """
    Raised when a Bundesbank response cannot be validated.
    """


@dataclass(frozen=True)
class BundesbankSeriesRequest:
    """
    Define one Bundesbank data request.
    """

    dataflow: str
    series_key: str
    start_period: date | None = None
    end_period: date | None = None
    language: str = "en"

    def __post_init__(self) -> None:
        if not self.dataflow.strip():
            raise ValueError(
                "dataflow must not be empty."
            )

        if not self.series_key.strip():
            raise ValueError(
                "series_key must not be empty."
            )

        if (
            self.start_period is not None
            and self.end_period is not None
            and self.start_period
            > self.end_period
        ):
            raise ValueError(
                "start_period must not be after end_period."
            )


@dataclass(frozen=True)
class BundesbankSeriesResult:
    """
    Store one validated Bundesbank time series response.
    """

    request: BundesbankSeriesRequest
    observations: pd.DataFrame
    retrieved_at: pd.Timestamp
    request_url: str


class BundesbankClient:
    """
    Download official Bundesbank statistics through its SDMX API.
    """

    def __init__(
        self,
        base_url: str = BUNDESBANK_API_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        session: requests.Session | None = None,
    ) -> None:
        if timeout_seconds <= 0.0:
            raise ValueError(
                "timeout_seconds must be positive."
            )

        self.base_url = base_url.rstrip(
            "/"
        )

        self.timeout_seconds = timeout_seconds

        self.session = (
            session
            if session is not None
            else self._build_session()
        )

    @staticmethod
    def _build_session() -> requests.Session:
        """
        Build a requests session with conservative retries.
        """
        retry_policy = Retry(
            total=4,
            connect=4,
            read=4,
            status=4,
            backoff_factor=0.6,
            status_forcelist=(
                429,
                500,
                502,
                503,
                504,
            ),
            allowed_methods=frozenset(
                {
                    "GET",
                }
            ),
            raise_on_status=False,
        )

        adapter = HTTPAdapter(
            max_retries=retry_policy
        )

        session = requests.Session()

        session.mount(
            "https://",
            adapter,
        )

        session.mount(
            "http://",
            adapter,
        )

        session.headers.update(
            {
                "User-Agent": (
                    "RepoLens/1.0 "
                    "(sovereign research dashboard)"
                ),
                "Accept": SDMX_CSV_ACCEPT_HEADER,
            }
        )

        return session

    def build_url(
        self,
        request: BundesbankSeriesRequest,
    ) -> str:
        """
        Build the Bundesbank series endpoint URL.
        """
        return (
            f"{self.base_url}/"
            f"{request.dataflow}/"
            f"{request.series_key}"
        )

    @staticmethod
    def build_params(
        request: BundesbankSeriesRequest,
    ) -> dict[str, str]:
        """
        Build supported Bundesbank API query parameters.
        """
        parameters = {
            "format": "sdmx_csv",
            "lang": request.language,
        }

        if request.start_period is not None:
            parameters["startPeriod"] = (
                request.start_period.isoformat()
            )

        if request.end_period is not None:
            parameters["endPeriod"] = (
                request.end_period.isoformat()
            )

        return parameters

    def fetch_series(
        self,
        request: BundesbankSeriesRequest,
    ) -> BundesbankSeriesResult:
        """
        Download and parse one Bundesbank SDMX-CSV series.
        """
        request_url = self.build_url(
            request
        )

        try:
            response = self.session.get(
                request_url,
                params=self.build_params(
                    request
                ),
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as error:
            raise BundesbankRequestError(
                "Bundesbank request failed before a response "
                f"was received: {error}"
            ) from error

        if response.status_code != 200:
            response_preview = response.text[
                :500
            ]

            raise BundesbankRequestError(
                "Bundesbank returned HTTP "
                f"{response.status_code} for {response.url}. "
                f"Response preview: {response_preview}"
            )

        observations = self.parse_sdmx_csv(
            response.text
        )

        return BundesbankSeriesResult(
            request=request,
            observations=observations,
            retrieved_at=pd.Timestamp.now(
                tz="UTC"
            ),
            request_url=str(
                response.url
            ),
        )

    @staticmethod
    def parse_sdmx_csv(
        csv_text: str,
    ) -> pd.DataFrame:
        """
        Parse a Bundesbank SDMX-CSV response.

        The parser deliberately requires the standard SDMX observation
        fields rather than relying on column order.
        """
        if not csv_text.strip():
            raise BundesbankResponseError(
                "Bundesbank returned an empty response."
            )

        try:
            raw_data = pd.read_csv(
                StringIO(
                    csv_text
                )
            )
        except pd.errors.ParserError as error:
            raise BundesbankResponseError(
                "Bundesbank response was not valid CSV."
            ) from error

        required_columns = {
            "TIME_PERIOD",
            "OBS_VALUE",
        }

        missing_columns = (
            required_columns
            - set(
                raw_data.columns
            )
        )

        if missing_columns:
            raise BundesbankResponseError(
                "Bundesbank SDMX response is missing required "
                f"columns: {sorted(missing_columns)}."
            )

        observations = pd.DataFrame(
            {
                "observation_date": pd.to_datetime(
                    raw_data[
                        "TIME_PERIOD"
                    ],
                    errors="coerce",
                ),
                "value": pd.to_numeric(
                    raw_data[
                        "OBS_VALUE"
                    ],
                    errors="coerce",
                ),
            }
        )

        observations = observations.dropna(
            subset=[
                "observation_date",
                "value",
            ]
        )

        if observations.empty:
            raise BundesbankResponseError(
                "Bundesbank response contains no valid observations."
            )

        observations = observations.sort_values(
            "observation_date"
        )

        observations = observations.drop_duplicates(
            subset="observation_date",
            keep="last",
        )

        observations = observations.reset_index(
            drop=True
        )

        return observations

    def fetch_benchmark(
        self,
        definition: SovereignBenchmarkDefinition,
        start_period: date | None = None,
        end_period: date | None = None,
        as_of_date: date | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Download one benchmark and apply the RepoLens data contract.
        """
        request = BundesbankSeriesRequest(
            dataflow=definition.dataflow,
            series_key=definition.series_key,
            start_period=start_period,
            end_period=end_period,
        )

        result = self.fetch_series(
            request
        )

        effective_as_of_date = (
            pd.Timestamp.now(
                tz="UTC"
            ).tz_localize(
                None
            )
            if as_of_date is None
            else pd.Timestamp(
                as_of_date
            )
        )

        observations = result.observations.copy()

        observations[
            "business_days_stale"
        ] = observations[
            "observation_date"
        ].apply(
            lambda observation_date: business_days_between(
                observation_date=observation_date,
                as_of_date=effective_as_of_date,
            )
        )

        observations["data_status"] = observations[
            "business_days_stale"
        ].apply(
            lambda days_stale: classify_data_status(
                original_status=definition.data_status,
                business_days_stale=int(
                    days_stale
                ),
            ).value
        )

        observations = observations.rename(
            columns={
                "value": "yield_percent",
            }
        )

        observations["country"] = (
            definition.country_name
        )

        observations["country_code"] = (
            definition.country_code
        )

        observations["tenor_years"] = (
            definition.tenor_years
        )

        observations["benchmark_name"] = (
            definition.benchmark_name
        )

        observations["source_name"] = (
            definition.provider_name
        )

        observations["source_series"] = (
            f"{definition.dataflow}."
            f"{definition.series_key}"
        )

        observations["source_timestamp"] = (
            result.retrieved_at
        )

        output_columns = [
            "observation_date",
            "country",
            "country_code",
            "tenor_years",
            "benchmark_name",
            "yield_percent",
            "source_name",
            "source_series",
            "source_timestamp",
            "data_status",
            "business_days_stale",
        ]

        return validate_benchmark_observations(
            observations[
                output_columns
            ]
        )

    @staticmethod
    def save_csv(
        observations: pd.DataFrame,
        output_path: Path,
    ) -> Path:
        """
        Save validated sovereign benchmark observations.
        """
        if output_path.suffix.lower() != ".csv":
            raise ValueError(
                "output_path must use the .csv extension."
            )

        validated = validate_benchmark_observations(
            observations
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        validated.to_csv(
            output_path,
            index=False,
        )

        return output_path


def combine_benchmark_series(
    benchmark_frames: tuple[
        pd.DataFrame,
        ...,
    ],
) -> pd.DataFrame:
    """
    Combine independently downloaded sovereign benchmark series.
    """
    if not benchmark_frames:
        raise SovereignDataValidationError(
            "benchmark_frames must not be empty."
        )

    combined = pd.concat(
        benchmark_frames,
        ignore_index=True,
    )

    return validate_benchmark_observations(
        combined
    )