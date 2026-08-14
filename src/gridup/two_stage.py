"""Iki asamali (hurdle) model: sifir-siskin sayim hedefleri icin.

NE ZAMAN KULLANILIR
-------------------
Hedefin buyuk kismi sifirsa -- gunluk ilce bazinda kesinti sayisi tipik olarak
%70-80 sifirdir -- tek bir regresyon iki isi ayni anda ogrenmek zorunda kalir:

  1. "Bugun olay olacak mi?"    (sinifllandirma problemi)
  2. "Olacaksa kac tane?"       (sayim problemi)

Bunlar FARKLI sureclerdir. Kesinti olup olmamasi hava/ekipman kosuluna baglidir;
kac tane olacagi ise ilcenin buyuklugune ve sebeke yogunluguna baglidir. Tek
model her ikisinde de tavizli kalir.

Hurdle modeli ikisini ayirir::

    E[y] = P(y > 0) x E[y | y > 0]

MAE METRIGINDE DIKKAT
---------------------
Yukaridaki carpim, hedefin **beklenen degerini** verir. Ama MAE'de optimal
tahmin beklenen deger DEGIL, kosullu **medyandir**.

Sifir-siskin bir dagilimda medyan cogu zaman 0'dir. Yani MAE altinda "her zaman
0 tahmin et" sasirtici derecede guclu bir baseline'dir ve beklenen deger tahmini
ondan DAHA KOTU olabilir.

Bu yuzden ``predict`` iki mod sunar:

  ``expected``   -> P x E[y|y>0].  RMSE ve benzeri kare hatali metrikler icin.
  ``thresholded``-> P < esik ise 0, degilse E[y|y>0]. MAE icin genellikle daha iyi.

Esik ELLE SECILMEZ; ``tune_threshold`` ile fold-disi tahminler uzerinde
optimize edilir. 0,5 varsayilani neredeyse her zaman yanlistir.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from .metrics import get_metric
from .models import CVResult, ModelKind, cross_validate, starter_params
from .validation import assert_folds_align

__all__ = ["TwoStageResult", "fit_two_stage", "tune_threshold", "zero_baseline_score"]

PredictMode = Literal["expected", "thresholded"]


def zero_baseline_score(y_true: np.ndarray, *, metric: str = "mae") -> float:
    """"Her zaman 0 tahmin et" baseline'inin skoru.

    BUNU HER ZAMAN HESAPLA. Sifir-siskin veride bu baseline sasirtici derecede
    gucludur; modelin onu gectigini DOGRULAMADAN ilerleme. Gecmiyorsa problem
    modelde degil, yaklasimdadir.
    """
    metric_fn, _, _ = get_metric(metric)
    return float(metric_fn(y_true, np.zeros_like(np.asarray(y_true, dtype="float64"))))


@dataclass
class TwoStageResult:
    """Iki asamali modelin egitim ciktisi."""

    classifier: CVResult
    regressor: CVResult
    oof_probability: np.ndarray
    oof_magnitude: np.ndarray
    positive_mask: np.ndarray
    best_threshold: float | None = None
    metric_name: str = "mae"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def predict_oof(self, *, mode: PredictMode = "thresholded") -> np.ndarray:
        """Fold-disi birlesik tahmin uretir -- esik optimizasyonu icin."""
        return _combine(
            self.oof_probability, self.oof_magnitude,
            mode=mode, threshold=self.best_threshold,
        )

    def summary(self) -> str:
        lines = [
            f"1. asama (olay var mi):  AUC={self.classifier.overall_score:.6f}",
            f"2. asama (kac tane):     {self.regressor.metric_name}="
            f"{self.regressor.overall_score:.6f}  "
            f"({int(self.positive_mask.sum()):,} pozitif ornekle egitildi)",
        ]
        if self.best_threshold is not None:
            lines.append(f"Optimum esik: {self.best_threshold:.4f}")
        for key, value in self.diagnostics.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)


def _combine(
    probability: np.ndarray,
    magnitude: np.ndarray,
    *,
    mode: PredictMode,
    threshold: float | None,
) -> np.ndarray:
    """Iki asamayi tek tahmine birlestirir."""
    probability = np.asarray(probability, dtype="float64")
    magnitude = np.clip(np.asarray(magnitude, dtype="float64"), 0, None)

    if mode == "expected":
        return probability * magnitude

    if mode == "thresholded":
        if threshold is None:
            raise ValueError(
                "thresholded mod icin esik gerekli. tune_threshold ile bul -- "
                "0.5 varsayilani sifir-siskin veride neredeyse her zaman yanlistir."
            )
        return np.where(probability >= threshold, magnitude, 0.0)

    raise ValueError(f"Bilinmeyen mod '{mode}'. 'expected' veya 'thresholded'.")


def tune_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
    magnitude: np.ndarray,
    *,
    metric: str = "mae",
    n_steps: int = 200,
) -> dict[str, float]:
    """Esigi FOLD-DISI tahminler uzerinde optimize eder.

    KRITIK: ``probability`` ve ``magnitude`` OOF olmali. Egitim tahminleri
    uzerinde optimize edilen bir esik de asiri uyum yapar ve leaderboard'da
    tutmaz.

    Returns:
        ``best_threshold``, ``best_score``, ``score_at_half``,
        ``score_expected`` (carpim modu), ``score_all_zero`` (baseline).
    """
    metric_fn, greater_is_better, _ = get_metric(metric)
    y_true = np.asarray(y_true, dtype="float64")

    thresholds = np.linspace(0.01, 0.99, n_steps)
    scores = np.array(
        [
            float(metric_fn(y_true, _combine(probability, magnitude,
                                             mode="thresholded", threshold=value)))
            for value in thresholds
        ]
    )
    best_index = int(np.argmax(scores) if greater_is_better else np.argmin(scores))

    return {
        "best_threshold": float(thresholds[best_index]),
        "best_score": float(scores[best_index]),
        "score_at_half": float(
            metric_fn(y_true, _combine(probability, magnitude,
                                       mode="thresholded", threshold=0.5))
        ),
        "score_expected": float(
            metric_fn(y_true, _combine(probability, magnitude, mode="expected", threshold=None))
        ),
        "score_all_zero": zero_baseline_score(y_true, metric=metric),
    }


def _stage2_predictions_everywhere(
    regressor: CVResult,
    train: pd.DataFrame,
    fold_list: Sequence[tuple[np.ndarray, np.ndarray]],
    source_folds: Sequence[int],
    *,
    kind: ModelKind,
) -> np.ndarray:
    """2. asama modelini SIFIR satirlarinda da calistirir -- fold-disi.

    NEDEN SABIT MEDYAN YANLISTI
    ---------------------------
    Onceki surum sifir satirlarina sabit ``median(y[y>0])`` yaziyordu ve
    gerekce olarak "esik zaten o satirlari sifirlayacak" deniyordu. Bu gerekce
    DAIRESELDIR: esik tam da bu degerler kullanilarak ayarlaniyor.

    Somut zarar: dusuk bir esik denendiginde sifir satirlari gercek model
    tahminlerini degil, sabit medyani alir. Medyan bu satirlar icin genellikle
    fazla yuksektir, dolayisiyla dusuk esiklerin maliyeti YAPAY olarak siser ve
    ``tune_threshold`` esigi gereginden yukari iter. Model gereginden fazla
    sifir tahmin eder.

    SIZINTI YOK: 2. asama modeli yalnizca kendi fold'unun POZITIF egitim
    satirlarinda egitildi; sifir satirlarinin hedefini hic gormedi. Onlarin
    uzerinde tahmin uretmek, herhangi bir gorulmemis satirda tahmin uretmekle
    aynidir.
    """
    from .models import _predict, _prepare_categoricals

    prepared, _, _ = _prepare_categoricals(train, None, kind)
    predictions = np.full(len(train), np.nan, dtype="float64")

    for model, fold_position in zip(regressor.models, source_folds, strict=True):
        _, valid_idx = fold_list[fold_position]
        predictions[valid_idx] = _predict(
            model, prepared.iloc[valid_idx], needs_proba=False
        )

    # Hicbir fold'un dogrulamadigi satirlar kalabilir -- TimeSeriesSplit ilk
    # donemi hic valid yapmaz. Bu satirlarda 1. asama da (cross_validate) 0
    # birakir, yani esik ne olursa olsun tahmin 0 cikar. Sabit bir ceza olarak
    # her esikte AYNI sekilde davranirlar ve esik SECIMINI saptirmazlar.
    uncovered = np.isnan(predictions)
    if uncovered.any():
        predictions[uncovered] = 0.0

    return np.clip(predictions, 0, None)


def fit_two_stage(
    train: pd.DataFrame,
    target: np.ndarray | pd.Series,
    folds: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    test: pd.DataFrame | None = None,
    kind: ModelKind = "lightgbm",
    metric: str = "mae",
    classifier_params: dict[str, Any] | None = None,
    regressor_params: dict[str, Any] | None = None,
    magnitude_metric: str = "mae",
    early_stopping_rounds: int = 200,
    verbose: bool = True,
) -> TwoStageResult:
    """Iki asamali modeli egitir ve esigi OOF uzerinde optimize eder.

    1. asama tum veriyle egitilir: ``y > 0`` ikili hedefi.
    2. asama YALNIZCA pozitif orneklerle egitilir: ``y | y > 0``.

    Args:
        folds: CV bolmeleri. 2. asama icin, her fold'un pozitif alt kumesine
            **yeniden indekslenir** -- bu, sizintisiz kalmasinin sarti.

    Returns:
        ``TwoStageResult``.

    Raises:
        ValueError: Hedefte hic pozitif yoksa veya hic sifir yoksa (o durumda
            iki asamali modelin anlami yok).
    """
    fold_list = list(folds)
    y = np.asarray(target, dtype="float64").ravel()
    assert_folds_align(len(train), fold_list)

    positive = y > 0
    if not positive.any():
        raise ValueError("Hedefte hic pozitif deger yok -- iki asamali model anlamsiz.")
    if positive.all():
        raise ValueError(
            "Hedefte hic sifir yok -- iki asamali model gereksiz. Duz regresyon kullan."
        )

    zero_share = float((~positive).mean())
    if verbose:
        print(f"Sifir orani: %{zero_share * 100:.1f}  ({int(positive.sum()):,} pozitif ornek)")
        if zero_share < 0.4:
            print(
                "  NOT: sifir orani dusuk. Iki asamali modelin duz regresyondan iyi "
                "olmasi beklenmez -- ikisini de dene ve CV ile karsilastir."
            )

    # --- 1. asama: olay var mi -------------------------------------------
    if verbose:
        print("\n1. asama -- olay var mi (siniflandirma)")

    classifier_config = classifier_params or starter_params(kind, "binary")
    classifier = cross_validate(
        train, positive.astype(int), fold_list,
        kind=kind, task_type="binary", metric="auc",
        params=classifier_config, test=test,
        early_stopping_rounds=early_stopping_rounds, verbose=verbose,
    )

    # --- 2. asama: kac tane ------------------------------------------------
    if verbose:
        print("\n2. asama -- kac tane (yalnizca pozitif ornekler)")

    positive_index = np.flatnonzero(positive)
    # Konumsal indeks eslemesi: orijinal satir -> pozitif alt kumedeki konum.
    remap = np.full(len(train), -1, dtype="int64")
    remap[positive_index] = np.arange(len(positive_index))

    positive_folds: list[tuple[np.ndarray, np.ndarray]] = []
    # Hangi pozitif-fold'un hangi ORIJINAL fold'dan geldigini izle: 2. asama
    # modelini o fold'un TUM validation satirlarinda (sifirlar dahil) kullanmak
    # icin gerekli.
    source_folds: list[int] = []
    for index, (train_idx, valid_idx) in enumerate(fold_list):
        mapped_train = remap[train_idx][remap[train_idx] >= 0]
        mapped_valid = remap[valid_idx][remap[valid_idx] >= 0]
        if mapped_train.size and mapped_valid.size:
            positive_folds.append((mapped_train, mapped_valid))
            source_folds.append(index)

    if not positive_folds:
        raise ValueError(
            "Pozitif orneklerle hicbir fold kurulamadi -- pozitif ornek sayisi cok az."
        )

    regressor_config = regressor_params or starter_params(kind, "regression", objective="mae")
    regressor = cross_validate(
        train.iloc[positive_index], y[positive_index], positive_folds,
        kind=kind, task_type="regression", metric=magnitude_metric,
        params=regressor_config, test=test,
        early_stopping_rounds=early_stopping_rounds, verbose=verbose,
    )

    magnitude_oof = _stage2_predictions_everywhere(
        regressor, train, fold_list, source_folds, kind=kind
    )

    tuning = tune_threshold(y, classifier.oof_predictions, magnitude_oof, metric=metric)

    diagnostics = {
        "sifir_orani": round(zero_share, 4),
        f"hep_sifir_baseline_{metric}": round(tuning["score_all_zero"], 6),
        f"carpim_modu_{metric}": round(tuning["score_expected"], 6),
        f"esikli_mod_{metric}": round(tuning["best_score"], 6),
        f"esik_0.5_{metric}": round(tuning["score_at_half"], 6),
    }

    if verbose:
        print("\n--- Iki asamali karsilastirma ---")
        for key, value in diagnostics.items():
            print(f"  {key:<32} {value}")
        if tuning["best_score"] >= tuning["score_all_zero"]:
            print(
                "  UYARI: model 'hep sifir' baseline'ini GECEMEDI. Sifir-siskin veride "
                "bu olur -- duz regresyonu ve farkli objective'leri de dene."
            )

    return TwoStageResult(
        classifier=classifier,
        regressor=regressor,
        oof_probability=classifier.oof_predictions,
        oof_magnitude=magnitude_oof,
        positive_mask=positive,
        best_threshold=tuning["best_threshold"],
        metric_name=metric,
        diagnostics=diagnostics,
    )
