from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import StringIO
from pathlib import Path
from time import sleep
from typing import Final

import pandas as pd
import requests


ECB_API_BASE_URL: Final[str] = (
    "https://data-api.ecb.europa.eu/service/data"
)

DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0
DEFAULT_MAX_ATTEMPTS: Final[int] = 3
DEFAULT_RETRY_DELAY_SECONDS: Final[float] = 1.0


class ECBClientError(RuntimeError):
    """
    Base exception raised by the ECB data client.
    """


class ECBRequestError(ECBClientError):
    """
    Raised when an ECB API request cannot be completed successfully.
    """


class ECBDataValidationError(ECBClientError):
    """
    Raised when an ECB response does not contain valid observations.
    """


@dataclass(frozen=True)
class ECBSeriesRequest:
    """
    Define one ECB Data Portal time-series request.
    """

    dataflow: str
    series_key: str
    start_period: date | None = None
    end_period: date | None = None
    last_n_observations: int | None = None

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
            and self.start_period > self.end_period
        ):
            raise ValueError(
                "start_period must not be after end_period."
            )

        if (
            self.last_n_observations is not None
            and self.last_n_observations <= 0
        ):
            raise ValueError(
                "last_n_observations must be positive."
            )


@dataclass(frozen=True)
class ECBSeriesResult:
    """
    Store validated ECB observations and request metadata.
    """

    dataflow: str
    series_key: str
    source_url: str
    retrieved_at_utc: pd.Timestamp
    observations: pd.DataFrame


class ECBClient:
    """
    Download and validate time-series data from the ECB Data Portal.
    """

    def __init__(
        self,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
        session: requests.Session | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be positive."
            )

        if max_attempts <= 0:
            raise ValueError(
                "max_attempts must be positive."
            )

        if retry_delay_seconds < 0:
            raise ValueError(
                "retry_delay_seconds must not be negative."
            )

        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.session = session or requests.Session()

        self.session.headers.update(
            {
                "User-Agent": (
                    "RepoLens/0.1 "
                    "European rates research project"
                ),
                "Accept": "text/csv",
            }
        )

    def build_url(
        self,
        request: ECBSeriesRequest,
    ) -> str:
        """
        Build the ECB API endpoint for one dataflow and series key.
        """
        dataflow = request.dataflow.strip()
        series_key = request.series_key.strip()

        return (
            f"{ECB_API_BASE_URL}/"
            f"{dataflow}/"
            f"{series_key}"
        )

    def build_parameters(
        self,
        request: ECBSeriesRequest,
    ) -> dict[str, str | int]:
        """
        Build validated ECB query parameters.
        """
        parameters: dict[str, str | int] = {
            "format": "csvdata",
            "detail": "full",
        }

        if request.start_period is not None:
            parameters["startPeriod"] = (
                request.start_period.isoformat()
            )

        if request.end_period is not None:
            parameters["endPeriod"] = (
                request.end_period.isoformat()
            )

        if request.last_n_observations is not None:
            parameters["lastNObservations"] = (
                request.last_n_observations
            )

        return parameters

    def fetch_series(
        self,
        request: ECBSeriesRequest,
    ) -> ECBSeriesResult:
        """
        Download, parse and validate one ECB time series.
        """
        url = self.build_url(request)
        parameters = self.build_parameters(request)

        response = self._request_with_retries(
            url=url,
            parameters=parameters,
        )

        observations = self.parse_csv_response(
            csv_text=response.text,
        )

        return ECBSeriesResult(
            dataflow=request.dataflow,
            series_key=request.series_key,
            source_url=response.url,
            retrieved_at_utc=pd.Timestamp.now(
                tz="UTC"
            ),
            observations=observations,
        )

    def _request_with_retries(
        self,
        url: str,
        parameters: dict[str, str | int],
    ) -> requests.Response:
        """
        Make an ECB request with bounded retries.
        """
        last_error: Exception | None = None

        for attempt_number in range(
            1,
            self.max_attempts + 1,
        ):
            try:
                response = self.session.get(
                    url,
                    params=parameters,
                    timeout=self.timeout_seconds,
                )

                if response.status_code == 200:
                    return response

                if response.status_code in {
                    429,
                    500,
                    502,
                    503,
                    504,
                }:
                    raise ECBRequestError(
                        "ECB API returned retryable status "
                        f"{response.status_code}."
                    )

                raise ECBRequestError(
                    "ECB API request failed with status "
                    f"{response.status_code}: "
                    f"{response.text[:300]}"
                )

            except (
                requests.RequestException,
                ECBRequestError,
            ) as error:
                last_error = error

                if attempt_number == self.max_attempts:
                    break

                sleep(
                    self.retry_delay_seconds
                    * attempt_number
                )

        raise ECBRequestError(
            "ECB API request failed after "
            f"{self.max_attempts} attempts."
        ) from last_error

    @staticmethod
    def parse_csv_response(
        csv_text: str,
    ) -> pd.DataFrame:
        """
        Convert ECB CSV output into a validated observation table.
        """
        if not csv_text.strip():
            raise ECBDataValidationError(
                "ECB response was empty."
            )

        try:
            raw_data = pd.read_csv(
                StringIO(csv_text)
            )
        except Exception as error:
            raise ECBDataValidationError(
                "ECB response could not be parsed as CSV."
            ) from error

        required_columns = {
            "TIME_PERIOD",
            "OBS_VALUE",
        }

        missing_columns = (
            required_columns
            - set(raw_data.columns)
        )

        if missing_columns:
            raise ECBDataValidationError(
                "ECB response is missing required columns: "
                f"{sorted(missing_columns)}."
            )

        observations = raw_data.copy()

        observations["TIME_PERIOD"] = pd.to_datetime(
            observations["TIME_PERIOD"],
            errors="coerce",
        )

        observations["OBS_VALUE"] = pd.to_numeric(
            observations["OBS_VALUE"],
            errors="coerce",
        )

        observations = observations.dropna(
            subset=[
                "TIME_PERIOD",
                "OBS_VALUE",
            ]
        )

        if observations.empty:
            raise ECBDataValidationError(
                "ECB response contained no valid observations."
            )

        observations = observations.sort_values(
            "TIME_PERIOD"
        )

        observations = observations.drop_duplicates(
            subset="TIME_PERIOD",
            keep="last",
        )

        observations = observations.reset_index(
            drop=True
        )

        observations = observations.rename(
            columns={
                "TIME_PERIOD": "observation_date",
                "OBS_VALUE": "value",
            }
        )

        preferred_columns = [
            "observation_date",
            "value",
        ]

        optional_columns = [
            column
            for column in observations.columns
            if column not in preferred_columns
        ]

        return observations[
            preferred_columns
            + optional_columns
        ]

    @staticmethod
    def save_csv(
        result: ECBSeriesResult,
        output_path: Path,
    ) -> Path:
        """
        Save validated observations to a local CSV file.
        """
        if output_path.suffix.lower() != ".csv":
            raise ValueError(
                "output_path must use the .csv extension."
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        result.observations.to_csv(
            output_path,
            index=False,
        )

        return output_path