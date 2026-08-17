"""Lag feature offsets must be explicit and horizon-safe."""

from __future__ import annotations

import pandas as pd
import pytest

from gridup.features.temporal import add_lag_features


def _frame(size: int = 130) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tarih": pd.date_range("2025-01-01", periods=size, freq="D"),
            "deger": [float(index) for index in range(size)],
        }
    )


def test_shifts_are_absolute_row_offsets() -> None:
    result = add_lag_features(
        _frame(),
        "deger",
        shifts=[31, 62, 93],
        time_column="tarih",
        horizon=31,
    )

    assert result.loc[31, "deger_shift31"] == pytest.approx(0.0)
    assert result.loc[62, "deger_shift62"] == pytest.approx(0.0)
    assert result.loc[93, "deger_shift93"] == pytest.approx(0.0)


def test_shift_shorter_than_forecast_horizon_is_rejected() -> None:
    with pytest.raises(ValueError, match="horizon"):
        add_lag_features(_frame(), "deger", shifts=[30], time_column="tarih", horizon=31)


def test_legacy_lags_warn_and_keep_origin_relative_semantics() -> None:
    with pytest.warns(DeprecationWarning, match="shifts"):
        result = add_lag_features(_frame(), "deger", [31], time_column="tarih", horizon=31)

    # Legacy lag=31 means horizon + lag - 1 = an absolute shift of 61.
    assert result.loc[61, "deger_ufuk31_lag31"] == pytest.approx(0.0)


def test_lags_and_shifts_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="birlikte"):
        add_lag_features(
            _frame(),
            "deger",
            [1],
            shifts=[31],
            time_column="tarih",
            horizon=31,
        )
