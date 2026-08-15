from datetime import date

import pytest

from src.sovereign_money_market import (
    FRANCE_BTF_FEB_2027,
    FRANCE_BTF_JUL_2027,
    FRANCE_BTF_OCT_2026,
    FRENCH_BTFS,
    GERMAN_BUBILLS,
    GERMANY_BUBILL_FEB_2027,
    GERMANY_BUBILL_MAY_2027,
    GERMANY_BUBILL_NOV_2026,
    MONEY_MARKET_INSTRUMENTS,
    MoneyMarketInstrumentValidationError,
    MoneyMarketSecurityType,
    RemainingMaturityBucket,
    get_money_market_instrument,
    validate_money_market_universe,
)


def test_money_market_universe_has_six_verified_securities() -> None:
    assert len(MONEY_MARKET_INSTRUMENTS) == 6

    assert {i.isin for i in MONEY_MARKET_INSTRUMENTS} == {
        "DE000BU0E352",
        "DE000BU0E386",
        "DE000BU0E410",
        "FR0129704088",
        "FR0129704146",
        "FR0129704179",
    }


def test_country_money_market_sets() -> None:
    assert len(GERMAN_BUBILLS) == 3
    assert len(FRENCH_BTFS) == 3

    assert {
        instrument.country
        for instrument in GERMAN_BUBILLS
    } == {
        "Germany"
    }

    assert {
        instrument.country
        for instrument in FRENCH_BTFS
    } == {
        "France"
    }


def test_discount_securities_are_zero_coupon_actual_360() -> None:
    for instrument in MONEY_MARKET_INSTRUMENTS:
        assert instrument.is_zero_coupon
        assert instrument.interest_day_count_basis == 360
        assert instrument.redemption_value_per_100 == pytest.approx(
            100.0
        )


def test_french_securities_are_btfs() -> None:
    assert all(
        instrument.security_type
        == MoneyMarketSecurityType.BTF
        for instrument in FRENCH_BTFS
    )


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

    assert FRANCE_BTF_OCT_2026.remaining_maturity_bucket(
        as_of_date
    ) == RemainingMaturityBucket.M1_3

    assert FRANCE_BTF_FEB_2027.remaining_maturity_bucket(
        as_of_date
    ) == RemainingMaturityBucket.M3_6

    assert FRANCE_BTF_JUL_2027.remaining_maturity_bucket(
        as_of_date
    ) == RemainingMaturityBucket.M6_12


def test_market_value_and_pull_to_par() -> None:
    instrument = GERMANY_BUBILL_NOV_2026

    assert instrument.market_value_eur(
        face_value_eur=10_000_000.0,
        price_per_100=99.40,
    ) == pytest.approx(
        9_940_000.0
    )

    assert instrument.redemption_value_eur(
        face_value_eur=10_000_000.0,
    ) == pytest.approx(
        10_000_000.0
    )

    assert instrument.pull_to_par_eur(
        face_value_eur=10_000_000.0,
        price_per_100=99.40,
    ) == pytest.approx(
        60_000.0
    )


def test_french_btf_market_value_and_pull_to_par() -> None:
    instrument = FRANCE_BTF_OCT_2026

    assert instrument.market_value_eur(
        face_value_eur=25_000_000.0,
        price_per_100=99.50,
    ) == pytest.approx(
        24_875_000.0
    )

    assert instrument.redemption_value_eur(
        face_value_eur=25_000_000.0,
    ) == pytest.approx(
        25_000_000.0
    )

    assert instrument.pull_to_par_eur(
        face_value_eur=25_000_000.0,
        price_per_100=99.50,
    ) == pytest.approx(
        125_000.0
    )


def test_get_money_market_instrument() -> None:
    assert get_money_market_instrument(
        "de000bu0e386"
    ).isin == "DE000BU0E386"

    assert get_money_market_instrument(
        "fr0129704088"
    ).isin == "FR0129704088"


def test_unknown_isin_is_rejected() -> None:
    with pytest.raises(
        MoneyMarketInstrumentValidationError,
        match="Unknown money-market instrument",
    ):
        get_money_market_instrument(
            "DE0000000000"
        )


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
            date(
                2026,
                11,
                19,
            )
        )