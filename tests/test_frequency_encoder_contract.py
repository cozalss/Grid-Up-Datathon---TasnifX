"""Frequency encoding must learn exactly once from the training distribution."""

from __future__ import annotations

import pandas as pd
import pytest

from gridup.features.categorical import FrequencyEncoder


def test_transform_reuses_training_frequencies_on_test() -> None:
    train = pd.DataFrame({"kategori": ["a"] * 9 + ["b"]})
    test = pd.DataFrame({"kategori": ["a"] + ["b"] * 9})

    encoder = FrequencyEncoder(["kategori"]).fit(train)
    transformed = encoder.transform(test)

    assert transformed.loc[0, "kategori_frekans"] == pytest.approx(0.9)
    assert transformed.loc[1, "kategori_frekans"] == pytest.approx(0.1)


def test_test_distribution_cannot_change_fitted_mapping() -> None:
    train = pd.DataFrame({"kategori": ["a", "a", "a", "b"]})
    one_test = pd.DataFrame({"kategori": ["a", "b"]})
    repeated_test = pd.DataFrame({"kategori": ["a"] * 50 + ["b"]})

    encoder = FrequencyEncoder(["kategori"]).fit(train)

    one = encoder.transform(one_test)["kategori_frekans"]
    repeated = encoder.transform(repeated_test)["kategori_frekans"]
    assert one.iloc[0] == pytest.approx(repeated.iloc[0])
    assert one.iloc[1] == pytest.approx(repeated.iloc[-1])


def test_unseen_category_is_zero_but_missing_value_stays_missing() -> None:
    train = pd.DataFrame({"kategori": ["a", "a", None]})
    test = pd.DataFrame({"kategori": ["yeni", None]})

    transformed = FrequencyEncoder(["kategori"]).fit(train).transform(test)

    assert transformed.loc[0, "kategori_frekans"] == pytest.approx(0.0)
    assert pd.isna(transformed.loc[1, "kategori_frekans"])


def test_transform_before_fit_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        FrequencyEncoder(["kategori"]).transform(pd.DataFrame({"kategori": ["a"]}))
