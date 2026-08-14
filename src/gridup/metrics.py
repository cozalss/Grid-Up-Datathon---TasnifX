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

import warnings
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
    "mape_coverage",
    "MAPE_ZERO_WARN_RATIO",
    "smape",
    "get_metric",
    "METRIC_REGISTRY",
    "optimize_threshold",
    "log_transform_target",
    "inverse_log_transform",
    "postprocess_predictions",
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


#: MAPE'de disarida birakilan sifir satirlarinin orani bunu asarsa uyaririz.
MAPE_ZERO_WARN_RATIO = 0.01


def mape_coverage(y_true: np.ndarray, *, epsilon: float = 1e-9) -> float:
    """MAPE'nin gercekte olctugu satirlarin orani (0..1).

    1.0'dan kucukse metrik verinin TAMAMINI olcmuyor demektir.
    """
    values = np.asarray(y_true, dtype="float64")
    if values.size == 0:
        return 0.0
    return float(np.mean(np.abs(values) >= epsilon))


def mape(y_true: np.ndarray, y_pred: np.ndarray, *, epsilon: float = 1e-9) -> float:
    """Ortalama mutlak yuzde hata (%).

    Sifir (veya sifira cok yakin) gercek degerli satirlar **dislanir** --
    aksi halde bolme patlar. Ama bu sessiz bir daralmadir ve tehlikelidir:
    satirlarin yarisi sifirsa MAPE yalnizca diger yariyi olcer ve sayi gayet
    normal gorunur. Bu yuzden dislama orani anlamli oldugunda UYARIRIZ.

    2023 GDZ Datathon'unda resmi metrik MAPE'ydi ve orada sorun cikmadi:
    hedef "Dagitilan Enerji (MWh)" sifirdan cok uzakta calisiyordu. **2026
    Grid Up icin bu garanti DEGILDIR.** Hedef kesinti suresi / ariza sayisi
    gibi sifir-siskin bir buyuklukse:

      * MAPE satirlarin cogunda tanimsizdir -> ``smape`` veya ``mae`` kullan
      * CatBoost/LightGBM'de ``eval_metric="MAPE"`` bu satirlarda anlamsiz
        gradyan uretir -> erken durdurma gurultuye gore karar verir

    Raises:
        Uyari degil, ``UserWarning`` -- kosmayi durdurmaz ama loga duser.
    """
    y_true = np.asarray(y_true, dtype="float64")
    covered = np.abs(y_true) >= epsilon
    excluded = 1.0 - (float(np.mean(covered)) if y_true.size else 0.0)
    if excluded > MAPE_ZERO_WARN_RATIO:
        warnings.warn(
            f"MAPE satirlarin %{excluded * 100:.1f}'ini DISLIYOR (gercek deger ~0). "
            f"Metrik yalnizca kalan %{(1 - excluded) * 100:.1f}'i olcuyor. "
            "Sifir-siskin hedefte 'smape' veya 'mae' kullanmayi dusun.",
            UserWarning,
            stacklevel=2,
        )
    denominator = np.where(covered, y_true, np.nan)
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


def postprocess_predictions(
    predictions: np.ndarray,
    *,
    round_to_integer: bool = False,
    clip_min: float | None = 0.0,
    clip_max: float | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Tahminleri fiziksel kisitlara ve metrige gore duzeltir. YENI dizi dondurur.

    ÜÇ UCUZ KAZANÇ, hepsi kanitli:

    1. **Negatif kirpma.** Kesinti sayisi, sure, tuketim negatif OLAMAZ.
       Kirpma tek basina skor kazandirir.

    2. **Yuvarlama (sayim hedefi + MAE).** Hedef tam sayiysa ve metrik MAE ise,
       ``2.4`` yerine ``2`` tahmin etmek hatayi dogrudan azaltir. 2024 GDZ
       birincisinin final mimarisi: 25 seed full-data + mean blend +
       **round** + **clip**.
       DIKKAT: metrik RMSE ise yuvarlama genellikle ZARAR verir -- RMSE'de
       optimal tahmin kosullu ORTALAMADIR ve o tam sayi olmak zorunda degildir.

    3. **Fiziksel ust sinir.** Bir arastirma, modellerin sehirlerin %19,8'inde
       musteri sayisindan FAZLA kesinti tahmin ettigini olcmus (5,2 kat asiri
       tahmin). ``clip_max`` ile gercekci bir tavan koymak bu ucu keser.

    Args:
        round_to_integer: Metrik MAE **ve** hedef sayim ise ``True``.
        clip_min: Alt sinir. Fiziksel buyukluklerde ``0.0`` birak.
        clip_max: Ust sinir (or. ilcedeki abone sayisi). ``None`` = sinirsiz.
    """
    values = np.asarray(predictions, dtype="float64").copy()
    report: list[str] = []

    if clip_min is not None:
        below = int((values < clip_min).sum())
        if below:
            report.append(f"{below:,} tahmin alt sinira ({clip_min}) cekildi")
        values = np.maximum(values, clip_min)

    if clip_max is not None:
        above = int((values > clip_max).sum())
        if above:
            report.append(f"{above:,} tahmin ust sinira ({clip_max}) cekildi")
        values = np.minimum(values, clip_max)

    if round_to_integer:
        values = np.round(values)
        report.append("tam sayiya yuvarlandi")

    if verbose and report:
        print("[postprocess] " + " · ".join(report))

    return values


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
