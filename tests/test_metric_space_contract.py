"""CV hedef/erken-durdurma uzayi ve metrik kenar durumlari kontratlari."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import gridup.models as model_module
from gridup.metrics import rmsle, smape
from gridup.models import cross_validate


class _MeanModel:
    def __init__(self, mean: float, n_features: int) -> None:
        self.mean = mean
        self.feature_importances_ = np.ones(n_features, dtype="float64")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.mean, dtype="float64")


def _folds() -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (np.array([3, 4, 5]), np.array([0, 1, 2])),
        (np.array([0, 1, 2]), np.array([3, 4, 5])),
    ]


def test_cross_validate_target_transform_scores_and_predicts_in_raw_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_targets: list[np.ndarray] = []

    def fake_fit(
        kind: str,
        params: dict[str, Any],
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> _MeanModel:
        seen_targets.append(y_train.copy())
        return _MeanModel(float(np.mean(y_train)), x_train.shape[1])

    monkeypatch.setattr(model_module, "_fit_one_fold", fake_fit)
    train = pd.DataFrame({"x": np.arange(6, dtype="float64")})
    test = pd.DataFrame({"x": [10.0, 11.0]})
    raw_y = np.array([0.0, 3.0, 8.0, 15.0, 24.0, 35.0])

    result = cross_validate(
        train,
        raw_y,
        _folds(),
        kind="lightgbm",
        metric="mae",
        params={"objective": "regression"},
        test=test,
        target_transform="log1p",
        early_stopping_rounds=0,
        verbose=False,
    )

    transformed = np.log1p(raw_y)
    np.testing.assert_allclose(seen_targets[0], transformed[[3, 4, 5]])
    np.testing.assert_allclose(seen_targets[1], transformed[[0, 1, 2]])
    first_raw = np.expm1(np.mean(transformed[[3, 4, 5]]))
    second_raw = np.expm1(np.mean(transformed[[0, 1, 2]]))
    np.testing.assert_allclose(result.oof_predictions[:3], first_raw)
    np.testing.assert_allclose(result.oof_predictions[3:], second_raw)
    np.testing.assert_allclose(result.test_predictions, (first_raw + second_raw) / 2)
    assert result.fold_scores == pytest.approx(
        [
            np.mean(np.abs(raw_y[:3] - first_raw)),
            np.mean(np.abs(raw_y[3:] - second_raw)),
        ]
    )
    assert result.overall_score == pytest.approx(np.mean(np.abs(raw_y - result.oof_predictions)))
    assert result.score_space == "raw"
    assert result.target_transform == "log1p"


@pytest.mark.parametrize(
    ("kind", "params", "expected"),
    [
        ("lightgbm", {"objective": "regression"}, "mae"),
        ("xgboost", {"objective": "reg:squarederror"}, "mae"),
        ("catboost", {"loss_function": "RMSE"}, "MAE"),
    ],
)
def test_early_stopping_metric_is_mapped_at_the_top_level(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    params: dict[str, Any],
    expected: str,
) -> None:
    seen: list[dict[str, Any]] = []

    def fake_fit(
        model_kind: str,
        model_params: dict[str, Any],
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> _MeanModel:
        seen.append(dict(model_params))
        return _MeanModel(float(np.mean(y_train)), x_train.shape[1])

    monkeypatch.setattr(model_module, "_fit_one_fold", fake_fit)
    frame = pd.DataFrame({"x": np.arange(6, dtype="float64")})
    y = np.arange(6, dtype="float64")

    result = cross_validate(
        frame,
        y,
        _folds(),
        kind=kind,
        params=params,
        metric="mae",
        early_stopping_metric="mae",
        verbose=False,
    )

    assert seen and all(item["eval_metric"] == expected for item in seen)
    assert result.early_stopping_metric == expected


def test_early_stopping_metric_conflict_fails_closed() -> None:
    frame = pd.DataFrame({"x": np.arange(6, dtype="float64")})
    with pytest.raises(ValueError, match="celisiyor|çelişiyor"):
        cross_validate(
            frame,
            np.arange(6, dtype="float64"),
            _folds(),
            kind="lightgbm",
            params={"objective": "regression", "eval_metric": "rmse"},
            early_stopping_metric="mae",
            verbose=False,
        )


@pytest.mark.parametrize(
    ("kind", "params", "expected"),
    [
        ("lightgbm", {"objective": "regression"}, "rmse"),
        ("xgboost", {"objective": "reg:squarederror"}, "rmse"),
        ("catboost", {"loss_function": "RMSE"}, "RMSE"),
    ],
)
def test_rmsle_early_stopping_is_rmse_when_fit_target_is_log1p(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    params: dict[str, Any],
    expected: str,
) -> None:
    seen: list[dict[str, Any]] = []

    def fake_fit(
        model_kind: str,
        model_params: dict[str, Any],
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> _MeanModel:
        seen.append(dict(model_params))
        return _MeanModel(float(np.mean(y_train)), x_train.shape[1])

    monkeypatch.setattr(model_module, "_fit_one_fold", fake_fit)
    frame = pd.DataFrame({"x": np.arange(6, dtype="float64")})
    y = np.arange(6, dtype="float64")

    result = cross_validate(
        frame,
        y,
        _folds(),
        kind=kind,
        params=params,
        metric="rmsle",
        target_transform="log1p",
        early_stopping_metric="rmsle",
        verbose=False,
    )

    assert seen and all(item["eval_metric"] == expected for item in seen)
    assert result.early_stopping_metric == expected


@pytest.mark.parametrize("metric", ["mae", "rmse"])
def test_transformed_target_rejects_non_equivalent_raw_early_metric(metric: str) -> None:
    frame = pd.DataFrame({"x": np.arange(6, dtype="float64")})

    with pytest.raises(ValueError, match="matematiksel|denk|uzay"):
        cross_validate(
            frame,
            np.arange(6, dtype="float64"),
            _folds(),
            kind="lightgbm",
            params={"objective": "regression"},
            metric=metric,
            target_transform="log1p",
            early_stopping_metric=metric,
            verbose=False,
        )


def test_transformed_target_requires_explicit_safe_early_metric() -> None:
    frame = pd.DataFrame({"x": np.arange(6, dtype="float64")})

    with pytest.raises(ValueError, match="early_stopping_metric"):
        cross_validate(
            frame,
            np.arange(6, dtype="float64"),
            _folds(),
            kind="lightgbm",
            params={"objective": "regression"},
            metric="rmsle",
            target_transform="log1p",
            verbose=False,
        )


def test_untransformed_rmsle_early_stopping_is_rejected() -> None:
    frame = pd.DataFrame({"x": np.arange(6, dtype="float64")})

    with pytest.raises(ValueError, match="log1p|denk"):
        cross_validate(
            frame,
            np.arange(6, dtype="float64"),
            _folds(),
            kind="lightgbm",
            params={"objective": "regression"},
            metric="rmsle",
            early_stopping_metric="rmsle",
            verbose=False,
        )


def test_smape_zero_zero_contribution_is_zero() -> None:
    assert smape(np.array([0.0, 0.0]), np.array([0.0, 0.0])) == 0.0
    assert smape(np.array([0.0, 10.0]), np.array([0.0, 20.0])) == pytest.approx(100 / 3)


def test_rmsle_rejects_negative_ground_truth() -> None:
    with pytest.raises(ValueError, match="negatif"):
        rmsle(np.array([1.0, -0.1]), np.array([1.0, 0.0]))
