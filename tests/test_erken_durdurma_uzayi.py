"""Donusumlu hedefte erken durdurma UZAYI sozlesmesi (P1-4, 2026-08-18 denetimi).

Olculdu: sqrt donusumu icin ham-uzay esdegeri olmadigi icin guard erken
durdurmayi kapatiyor, 2000 sabit agac kosuluyordu -> benchmark 393,00 MAE
("sqrt kotu" artefakti). Artik ``early_stopping_space="fit"`` ile fit
uzayinda (or. l2) durdurulur ve bu UYARI ile belgelenir; ham uzay
("raw") sozlesmesi eskisi gibi kapali kalir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.models import _resolve_early_stopping_metric, cross_validate


def _veri(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = np.square(1.0 + 0.6 * x["a"].to_numpy() + rng.normal(scale=0.3, size=n))
    idx = np.arange(n)
    folds = [(idx[:200], idx[200:250]), (idx[:250], idx[250:])]
    return x, y, folds


def test_ham_uzayda_sqrt_ile_mae_hala_reddedilir() -> None:
    with pytest.raises(ValueError, match="denk degildir"):
        _resolve_early_stopping_metric(
            "lightgbm", {}, "mae", target_transform="sqrt", early_stopping_rounds=50
        )


def test_fit_uzayi_secenegi_uyari_verir_ve_backend_metrigi_cozer() -> None:
    with pytest.warns(UserWarning, match="FIT uzayinda"):
        params, backend = _resolve_early_stopping_metric(
            "lightgbm",
            {},
            "rmse",
            target_transform="sqrt",
            early_stopping_rounds=50,
            early_stopping_space="fit",
        )
    assert backend == "rmse" and params["eval_metric"] == "rmse"
    with pytest.raises(ValueError, match="early_stopping_space"):
        _resolve_early_stopping_metric(
            "lightgbm", {}, "rmse", target_transform="sqrt", early_stopping_space="x"
        )
    with pytest.raises(ValueError, match="zorunlu"):
        _resolve_early_stopping_metric(
            "lightgbm", {}, None, target_transform="sqrt", early_stopping_space="fit"
        )


def test_sqrt_fit_uzayi_erken_durdurma_uctan_uca_ve_ham_skor() -> None:
    """Skor HAM uzayda (MAE), erken durdurma sqrt uzayinda -- tur sayisi butcenin altinda."""
    x, y, folds = _veri()
    with pytest.warns(UserWarning, match="FIT uzayinda"):
        sonuc = cross_validate(
            x,
            y,
            folds,
            kind="lightgbm",
            metric="mae",
            params={"n_estimators": 400, "learning_rate": 0.1, "verbose": -1},
            target_transform="sqrt",
            early_stopping_metric="rmse",
            early_stopping_space="fit",
            early_stopping_rounds=20,
            verbose=False,
        )
    assert np.isfinite(sonuc.overall_score)
    # OOF ham uzayda: negatif olmayan hedefe karsi tahminler de negatif degil
    assert (sonuc.oof_predictions[sonuc.oof_covered] >= 0).all()
    # Modeller erken durmus olmali: her fold modelinin agac sayisi butcenin altinda
    for model in sonuc.models:
        best = getattr(model, "best_iteration_", None)
        assert best is not None and 0 < best < 400, best
    assert sonuc.early_stopping_metric == "rmse"
