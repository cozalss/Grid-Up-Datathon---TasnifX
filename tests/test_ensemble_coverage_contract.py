"""OOF kapsam maskesinin ansambl ogrenmesine gercekten uygulanma kontrati."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.ensemble import (
    correlation_matrix,
    greedy_forward_selection,
    hill_climb_weights,
    prune_by_correlation,
)


def _case() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    y = np.array([100.0, 100.0, 0.0, 1.0, 2.0, 3.0])
    predictions = {
        "covered_winner": np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0]),
        "padding_winner": np.array([100.0, 100.0, 8.0, 8.0, 8.0, 8.0]),
        "correlated": np.array([1000.0, -1000.0, 0.0, 2.0, 4.0, 6.0]),
    }
    covered = np.array([False, False, True, True, True, True])
    return predictions, y, covered


def test_hill_climb_applies_covered_mask_in_weight_learning() -> None:
    predictions, y, covered = _case()
    masked = hill_climb_weights(predictions, y, metric="mae", covered=covered, verbose=False)
    sliced = hill_climb_weights(
        {name: values[covered] for name, values in predictions.items()},
        y[covered],
        metric="mae",
        verbose=False,
    )
    assert masked == pytest.approx(sliced)
    assert masked["covered_winner"] == pytest.approx(1.0)


def test_greedy_and_prune_apply_covered_mask() -> None:
    predictions, y, covered = _case()
    greedy = greedy_forward_selection(
        predictions,
        y,
        metric="mae",
        covered=covered,
        max_models=3,
        verbose=False,
    )
    kept = prune_by_correlation(
        predictions,
        y,
        metric="mae",
        covered=covered,
        max_members=1,
    )
    assert greedy == {"covered_winner": 1.0}
    assert kept == ["covered_winner"]


def test_correlation_matrix_applies_covered_mask() -> None:
    predictions, _, covered = _case()
    actual = correlation_matrix(predictions, covered=covered)
    expected = pd.DataFrame({name: values[covered] for name, values in predictions.items()}).corr()
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "call",
    [
        lambda p, y, c: hill_climb_weights(p, y, covered=c, verbose=False),
        lambda p, y, c: greedy_forward_selection(p, y, covered=c, verbose=False),
        lambda p, y, c: prune_by_correlation(p, y, covered=c),
    ],
)
def test_general_ensemble_apis_reject_empty_coverage(call) -> None:
    predictions, y, _ = _case()
    with pytest.raises(ValueError, match="kalmadi"):
        call(predictions, y, np.zeros(len(y), dtype=bool))
