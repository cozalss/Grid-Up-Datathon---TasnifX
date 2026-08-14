"""Metrikler ve metrik-spesifik optimizasyon.

Kaggle'da metrik bir detay degil, STRATEJIDIR. Ayni model, metrik dogru ele
alinmadiginda 100 sira asagida biter. Bu modul her yaygin metrigi ve onun
"hilesini" bir arada tutar.

METRIK -> HILE TABLOSU
----------------------
    RMSLE  -> hedefi log1p ile donustur, RMSE ile egit, tahminde expm1 uygula
    MAE    -> L2 degil L1 objective kullan (LightGBM: objective="mae")
    MAPE   -> kucuk gercek degerler metrigi patlatir; log donusum veya
              agirliklandirma dusun
    AUC    -> esik SECME, olasilik siralamasi yeterli; kalibrasyon gereksiz
    F1     -> esik CV uzerinden optimize edilmeli; 0.5 varsayilani neredeyse
              her zaman yanlistir
    LogLoss-> kalibrasyon SART (isotonic / Platt)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

__all__ = [
    "rmse",
    "rmsle",
    "mape",
    "smape",
    "get_metric",
    "METRIC_REGISTRY",
    "optimize_threshold",
    "log_transform_target",
    "inverse_log_transform",
]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Kok ortalama kare hata."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Kok ortalama kare logaritmik hata.

    Negatif tahminler log'u tanimsiz kilar; 0'a kirpiyoruz. Bu kirpma
    SESSIZ DEGIL -- cok sayida negatif tahmin varsa modelin yanlis olcekte
    calistiginin isaretidir.
    """
    y_pred_clipped = np.clip(y_pred, 0, None)
    y_true_clipped = np.clip(y_true, 0, None)
    return float(
        np.sqrt(np.mean((np.log1p(y_pred_clipped) - np.log1p(y_true_clipped)) ** 2))
    )


def mape(y_true: np.ndarray, y_pred: np.ndarray, *, epsilon: float = 1e-9) -> float:
    """Ortalama mutlak yuzde hata (%).

    UYARI: gercek deger 0'a yaklastikca patlar. Elektrik tuketiminde gece
    saatlerinde veya kapali abonelerde bu sik olur. MAPE metrikse, sifir
    yakini satirlarin metrigi domine ettigini kontrol et.
    """
    y_true = np.asarray(y_true, dtype="float64")
    denominator = np.where(np.abs(y_true) < epsilon, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denominator)) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray, *, epsilon: float = 1e-9) -> float:
    """Simetrik ortalama mutlak yuzde hata (%). MAPE'nin sifir-dayanikli surumu."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator < epsilon, np.nan, denominator)
    return float(np.nanmean(np.abs(y_true - y_pred) / denominator) * 100)


METRIC_REGISTRY: dict[str, dict[str, object]] = {
    "rmse": {"fn": rmse, "greater_is_better": False, "needs_proba": False},
    "rmsle": {"fn": rmsle, "greater_is_better": False, "needs_proba": False},
    "mae": {"fn": mean_absolute_error, "greater_is_better": False, "needs_proba": False},
    "mape": {"fn": mape, "greater_is_better": False, "needs_proba": False},
    "smape": {"fn": smape, "greater_is_better": False, "needs_proba": False},
    "r2": {"fn": r2_score, "greater_is_better": True, "needs_proba": False},
    "auc": {"fn": roc_auc_score, "greater_is_better": True, "needs_proba": True},
    "logloss": {"fn": log_loss, "greater_is_better": False, "needs_proba": True},
    "f1": {"fn": f1_score, "greater_is_better": True, "needs_proba": False},
    "accuracy": {"fn": accuracy_score, "greater_is_better": True, "needs_proba": False},
}


def get_metric(name: str) -> tuple[Callable[..., float], bool, bool]:
    """Metrik adindan ``(fonksiyon, buyuk_daha_iyi, olasilik_gerekli)`` dondurur."""
    key = name.lower()
    if key not in METRIC_REGISTRY:
        raise ValueError(f"Bilinmeyen metrik '{name}'. Secenekler: {sorted(METRIC_REGISTRY)}")
    entry = METRIC_REGISTRY[key]
    return entry["fn"], entry["greater_is_better"], entry["needs_proba"]  # type: ignore[return-value]


def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    metric: str = "f1",
    n_steps: int = 200,
) -> dict[str, float]:
    """Siniflandirma esigini CV tahminleri uzerinde optimize eder.

    NEDEN: 0.5 esigi yalnizca siniflar dengeli VE model kalibreyse dogrudur.
    Dengesiz veride (or. arizali trafo orani %2) optimum esik cogu zaman
    0.1'in altindadir. Bu tek satirlik degisiklik F1'i ikiye katlayabilir.

    KRITIK: Esigi FOLD-DISI (OOF) tahminler uzerinde optimize et, egitim
    tahminleri uzerinde DEGIL. Aksi halde esik de asiri uyum yapar.

    Returns:
        ``best_threshold``, ``best_score``, ``score_at_half`` iceren sozluk.
    """
    metric_fn, greater_is_better, _ = get_metric(metric)

    thresholds = np.linspace(0.01, 0.99, n_steps)
    scores = np.array(
        [float(metric_fn(y_true, (y_proba >= threshold).astype(int))) for threshold in thresholds]
    )

    best_index = int(np.argmax(scores) if greater_is_better else np.argmin(scores))

    return {
        "best_threshold": float(thresholds[best_index]),
        "best_score": float(scores[best_index]),
        "score_at_half": float(metric_fn(y_true, (y_proba >= 0.5).astype(int))),
    }


def log_transform_target(y: np.ndarray) -> np.ndarray:
    """``log1p`` donusumu -- RMSLE metrigi icin standart hamle.

    RMSLE ile puanlanan bir yarismada hedefi ``log1p`` ile donusturup RMSE
    optimize etmek, RMSLE'yi dogrudan optimize etmeye esdegerdir ve cok daha
    kararli egitim verir. Elektrik tuketimi gibi saga carpik dagilimlarda
    ayrica hedefi normallestirdigi icin metrik RMSE olsa BILE denemeye deger.
    """
    y = np.asarray(y, dtype="float64")
    if np.any(y < -1):
        raise ValueError("log1p, -1'den kucuk degerlerde tanimsiz. Hedefi kontrol et.")
    return np.log1p(y)


def inverse_log_transform(y_log: np.ndarray, *, clip_negative: bool = True) -> np.ndarray:
    """``log1p`` donusumunu geri alir.

    ``clip_negative``: elektrik tuketimi/kesinti suresi negatif olamaz; ters
    donusum sonrasi kirpmak hem fiziksel olarak dogru hem skoru iyilestirir.
    """
    result = np.expm1(np.asarray(y_log, dtype="float64"))
    return np.clip(result, 0, None) if clip_negative else result
