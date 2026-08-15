from datetime import date

import pytest

from src.sovereign_money_market import (
    GERMANY_BUBILL_FEB_2027,
    GERMANY_BUBILL_MAY_2027,
    GERMANY_BUBILL_NOV_2026,
    MONEY_MARKET_INSTRUMENTS,
    MoneyMarketInstrumentValidationError,
    RemainingMaturityBucket,
    get_money_market_instrument,
    validate_money_market_universe,
)


def test_initial_money_market_universe_has_three_verified_bubills() -> None:
    assert len(MONEY_MARKET_INSTRUMENTS) == 3
    assert {i.isin for i in MONEY_MARKET_INSTRUMENTS} == {
        "DE000BU0E352",
        "DE000BU0E386",
        "DE000BU0E410",
    }


def test_bubills_are_zero_coupon_actual_360() -> None:
    for instrument in MONEY_MARKET_INSTRUMENTS:
        assert instrument.is_zero_coupon
        assert instrument.interest_day_count_basis == 360
        assert instrument.redemption_value_per_100 == pytest.approx(100.0)


def test_remaining_maturity_buckets_on_15_august_2026() -> None:
    as_of_date = date(2026, 8, 15)

    assert GERMANY_BUBILL_NOV_2026.remaining_maturity_bucket(
        as_of_date
    ) == RemainingMaturityBucket.M3_6

    assert GERMANY_BUBILL_FEB_2027.remaining_maturity_bucket(
        as_of_date
    ) == RemainingMaturityBucket.M6_12

    assert GERMANY_BUBILL_MAY_2027.remaining_maturity_bucket(
        as_of_date
    ) == RemainingMaturityBucket.M6_12


def test_market_value_and_pull_to_par() -> None:
    instrument = GERMANY_BUBILL_NOV_2026

    assert instrument.market_value_eur(
        face_value_eur=10_000_000.0,
        price_per_100=99.40,
    ) == pytest.approx(9_940_000.0)

    assert instrument.redemption_value_eur(
        face_value_eur=10_000_000.0,
    ) == pytest.approx(10_000_000.0)

    assert instrument.pull_to_par_eur(
        face_value_eur=10_000_000.0,
        price_per_100=99.40,
    ) == pytest.approx(60_000.0)


def test_get_money_market_instrument() -> None:
    assert get_money_market_instrument(
        "de000bu0e386"
    ).isin == "DE000BU0E386"


def test_unknown_isin_is_rejected() -> None:
    with pytest.raises(
        MoneyMarketInstrumentValidationError,
        match="Unknown money-market instrument",
    ):
        get_money_market_instrument("DE0000000000")


def test_duplicate_isin_is_rejected() -> None:
    with pytest.raises(
        MoneyMarketInstrumentValidationError,
        match="duplicate ISINs",
    ):
        validate_money_market_universe(
            (
                GERMANY_BUBILL_NOV_2026,
                GERMANY_BUBILL_NOV_2026,
            )
        )


def test_as_of_date_after_maturity_is_rejected() -> None:
    with pytest.raises(
        MoneyMarketInstrumentValidationError,
        match="after maturity_date",
    ):
        GERMANY_BUBILL_NOV_2026.days_to_maturity(
            date(2026, 11, 19)
        )