import pytest

from src.repo_adjusted_relative_value import (
    RepoAdjustedRelativeValueValidationError,
    RepoFundingLegInput,
    analyse_repo_adjusted_relative_value,
    analyse_repo_funding_leg,
)
from src.sovereign_relative_value import PositionDirection


def make_leg(
    *,
    isin: str,
    direction: PositionDirection,
    face_value_eur: float = 10_000_000.0,
    specific_repo_rate_percent: float = 1.50,
    gc_repo_rate_percent: float = 2.00,
    repo_days: int = 30,
    day_count_basis: int = 360,
) -> RepoFundingLegInput:
    return RepoFundingLegInput(
        isin=isin,
        direction=direction,
        face_value_eur=face_value_eur,
        dirty_price_per_100=100.0,
        haircut_percent=0.0,
        specific_repo_rate_percent=specific_repo_rate_percent,
        gc_repo_rate_percent=gc_repo_rate_percent,
        repo_days=repo_days,
        day_count_basis=day_count_basis,
    )


def test_long_special_collateral_has_positive_signed_funding_impact() -> None:
    result = analyse_repo_funding_leg(
        make_leg(
            isin="DE0000000001",
            direction=PositionDirection.LONG,
        )
    )

    assert result.specialness_bp == pytest.approx(
        50.0
    )

    assert (
        result.unsigned_financing_edge_vs_gc_eur
        > 0.0
    )

    assert (
        result.signed_financing_impact_vs_gc_eur
        > 0.0
    )


def test_short_special_collateral_reverses_the_funding_sign() -> None:
    result = analyse_repo_funding_leg(
        make_leg(
            isin="DE0000000002",
            direction=PositionDirection.SHORT,
        )
    )

    assert (
        result.unsigned_financing_edge_vs_gc_eur
        > 0.0
    )

    assert (
        result.signed_financing_impact_vs_gc_eur
        < 0.0
    )

    assert (
        result.signed_financing_impact_vs_gc_eur
        == pytest.approx(
            -result.unsigned_financing_edge_vs_gc_eur
        )
    )


def test_long_special_anchor_vs_gc_hedge_has_positive_net_overlay() -> None:
    analysis = analyse_repo_adjusted_relative_value(
        anchor=make_leg(
            isin="IT0000000001",
            direction=PositionDirection.LONG,
            specific_repo_rate_percent=1.00,
            gc_repo_rate_percent=2.00,
        ),
        hedge=make_leg(
            isin="DE0000000001",
            direction=PositionDirection.SHORT,
            specific_repo_rate_percent=2.00,
            gc_repo_rate_percent=2.00,
        ),
    )

    assert (
        analysis.anchor.specialness_bp
        == pytest.approx(
            100.0
        )
    )

    assert (
        analysis.hedge.specialness_bp
        == pytest.approx(
            0.0
        )
    )

    assert (
        analysis.net_signed_financing_impact_vs_gc_eur
        > 0.0
    )

    assert (
        analysis.anchor_minus_hedge_specialness_bp
        == pytest.approx(
            100.0
        )
    )


def test_shorting_more_special_anchor_produces_negative_net_overlay() -> None:
    analysis = analyse_repo_adjusted_relative_value(
        anchor=make_leg(
            isin="IT0000000001",
            direction=PositionDirection.SHORT,
            specific_repo_rate_percent=1.00,
            gc_repo_rate_percent=2.00,
        ),
        hedge=make_leg(
            isin="DE0000000001",
            direction=PositionDirection.LONG,
            specific_repo_rate_percent=2.00,
            gc_repo_rate_percent=2.00,
        ),
    )

    assert (
        analysis.net_signed_financing_impact_vs_gc_eur
        < 0.0
    )


def test_equal_specialness_and_equal_notional_offset_for_opposite_directions() -> None:
    analysis = analyse_repo_adjusted_relative_value(
        anchor=make_leg(
            isin="IT0000000001",
            direction=PositionDirection.LONG,
            specific_repo_rate_percent=1.50,
            gc_repo_rate_percent=2.00,
        ),
        hedge=make_leg(
            isin="DE0000000001",
            direction=PositionDirection.SHORT,
            specific_repo_rate_percent=1.50,
            gc_repo_rate_percent=2.00,
        ),
    )

    assert (
        analysis.net_signed_financing_impact_vs_gc_eur
        == pytest.approx(
            0.0,
            abs=1e-9,
        )
    )

    assert (
        analysis.anchor_minus_hedge_specialness_bp
        == pytest.approx(
            0.0
        )
    )


def test_dv01_sized_unequal_notionals_are_preserved() -> None:
    analysis = analyse_repo_adjusted_relative_value(
        anchor=make_leg(
            isin="IT0000000001",
            direction=PositionDirection.LONG,
            face_value_eur=8_000_000.0,
            specific_repo_rate_percent=1.25,
            gc_repo_rate_percent=2.00,
        ),
        hedge=make_leg(
            isin="DE0000000001",
            direction=PositionDirection.SHORT,
            face_value_eur=10_000_000.0,
            specific_repo_rate_percent=1.90,
            gc_repo_rate_percent=2.00,
        ),
    )

    assert analysis.anchor.face_value_eur == pytest.approx(
        8_000_000.0
    )

    assert analysis.hedge.face_value_eur == pytest.approx(
        10_000_000.0
    )

    expected_net = (
        analysis.anchor.signed_financing_impact_vs_gc_eur
        + analysis.hedge.signed_financing_impact_vs_gc_eur
    )

    assert (
        analysis.net_signed_financing_impact_vs_gc_eur
        == pytest.approx(
            expected_net
        )
    )


def test_repo_term_mismatch_is_rejected() -> None:
    with pytest.raises(
        RepoAdjustedRelativeValueValidationError,
        match="repo_days must match",
    ):
        analyse_repo_adjusted_relative_value(
            anchor=make_leg(
                isin="IT0000000001",
                direction=PositionDirection.LONG,
                repo_days=7,
            ),
            hedge=make_leg(
                isin="DE0000000001",
                direction=PositionDirection.SHORT,
                repo_days=30,
            ),
        )


def test_day_count_mismatch_is_rejected() -> None:
    with pytest.raises(
        RepoAdjustedRelativeValueValidationError,
        match="day_count_basis must match",
    ):
        analyse_repo_adjusted_relative_value(
            anchor=make_leg(
                isin="IT0000000001",
                direction=PositionDirection.LONG,
                day_count_basis=360,
            ),
            hedge=make_leg(
                isin="DE0000000001",
                direction=PositionDirection.SHORT,
                day_count_basis=365,
            ),
        )


def test_same_direction_is_rejected() -> None:
    with pytest.raises(
        RepoAdjustedRelativeValueValidationError,
        match="directions must be opposite",
    ):
        analyse_repo_adjusted_relative_value(
            anchor=make_leg(
                isin="IT0000000001",
                direction=PositionDirection.LONG,
            ),
            hedge=make_leg(
                isin="DE0000000001",
                direction=PositionDirection.LONG,
            ),
        )


def test_same_instrument_is_rejected() -> None:
    with pytest.raises(
        RepoAdjustedRelativeValueValidationError,
        match="different instruments",
    ):
        analyse_repo_adjusted_relative_value(
            anchor=make_leg(
                isin="IT0000000001",
                direction=PositionDirection.LONG,
            ),
            hedge=make_leg(
                isin="IT0000000001",
                direction=PositionDirection.SHORT,
            ),
        )