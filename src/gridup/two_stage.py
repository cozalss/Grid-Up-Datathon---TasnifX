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

__all__ = [
    "TwoStageResult",
    "fit_two_stage",
    "fit_quantile_ladder",
    "fit_conditional_quantile_ladder",
    "CONDITIONAL_LADDER_LEVELS",
    "mae_optimal_quantile",
    "conditional_quantile_from_hurdle",
    "tune_threshold",
    "zero_baseline_score",
]

#: ``_combine``in GERCEKTEN destekledigi modlar.
#:
#: Eskiden burada bir ucuncu deger vardi: ``"mae_optimal"``. ``_combine`` onu
#: reddediyordu, yani tip denetleyicisi cagriyi GECERLI sayarken calisma
#: zamaninda ``ValueError: Bilinmeyen mod 'mae_optimal'`` aliniyordu (olculdu).
#: MAE-optimal tahmin iki asamadan degil, KUANTIL MERDIVENINDEN uretilir --
#: bkz. ``conditional_quantile_from_hurdle``; tek bir olasilik+buyukluk
#: ciftinden hesaplanamaz, bu yuzden ``_combine``in imzasina ait degildir.
PredictMode = Literal["expected", "thresholded"]


def mae_optimal_quantile(positive_probability: np.ndarray) -> np.ndarray:
    """MAE altinda optimal kosullu kuantil seviyesini hesaplar.

    TUREV
    -----
    Hedefin dagilimi bir KARISIM: ``(1-p)`` agirlikla tam 0'da kutle, ``p``
    agirlikla ``F(y | y>0)`` kosullu dagilimi.

    MAE'yi minimize eden tahmin, karisimin MEDYANIDIR. Medyan ``q``,
    ``CDF(q) = 0.5`` kosulunu saglar::

        CDF(q) = (1 - p) + p * F(q | y>0) = 0.5
        =>  p * F = p - 0.5
        =>  F = 1 - 0.5 / p

    Yani 2. asamadan istenmesi gereken kuantil ``q* = 1 - 0.5/p``dir.

    NE ANLAMA GELIYOR
    -----------------
    ``p = 0.50``  ->  q* = 0.00   sinirda; medyan tam 0
    ``p = 0.60``  ->  q* = 0.167  kosullu dagilimin ALT ucundan
    ``p = 0.80``  ->  q* = 0.375
    ``p = 1.00``  ->  q* = 0.50   klasik medyan

    ``p <= 0.5`` ise karisimin medyani 0'dir -- tahmin 0 olmalidir. Fonksiyon
    bu satirlar icin ``NaN`` dondurur; cagiran taraf onlari sifirlar.

    BU NEDEN ONEMLI
    ---------------
    ``expected`` modu (``p * E[y|y>0]``) BEKLENEN degeri verir -- kare hatali
    metrikler icin dogru, MAE icin degil. ``thresholded`` modu ise kosullu
    dagilimin MEDYANINI (q=0.5) kullanir; oysa ``p < 1`` iken dogru kuantil
    her zaman 0.5'in ALTINDADIR. Ikisi de MAE altinda suboptimaldir.
    """
    probability = np.asarray(positive_probability, dtype="float64")
    # divide : p == 0        -> 0.5/0 = inf
    # over   : p ~ 1e-320    -> 0.5/p float64 araligini asar (hypothesis buldu)
    # invalid: inf - inf     -> NaN
    # Ucu de BEKLENEN durumlardir ve asagidaki ``quantile > 0`` kontrolu
    # hepsini NaN'a cevirir. Uyariyi susturmazsak kullaniciya bir sey
    # bozulmus gibi gorunur -- sifir-siskin veride p=0 son derece olagandir.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        quantile = 1.0 - 0.5 / probability
    return np.where(quantile > 0, quantile, np.nan)


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
    #: Hangi satirlar GERCEKTEN bir fold tarafindan dogrulandi. Kapsanmayan
    #: satirlarda ``oof_probability`` ve ``oof_magnitude`` 0'dir -- tahmin
    #: degil, DOLGUDUR.
    oof_covered: np.ndarray | None = None

    @property
    def coverage(self) -> float:
        """Fold'larin dogruladigi satir orani."""
        if self.oof_covered is None:
            return 1.0
        return float(self.oof_covered.mean())

    def covered(self) -> np.ndarray:
        """Kapsam maskesi; yoksa hepsi True."""
        if self.oof_covered is None:
            return np.ones(len(self.oof_probability), dtype=bool)
        return self.oof_covered

    def predict_oof(self, *, mode: PredictMode = "thresholded") -> np.ndarray:
        """Fold-disi birlesik tahmin uretir -- esik optimizasyonu icin.

        DIKKAT: Kapsanmayan satirlar 0 doner. Skorlamada ``covered()`` maskesi
        ile filtrele -- aksi halde hic dogrulanmamis satirlari "sifir tahmin
        edildi" diye puanlarsin.
        """
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


def fit_quantile_ladder(
    train: pd.DataFrame,
    target: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    quantiles: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    test: pd.DataFrame | None = None,
    params: dict[str, Any] | None = None,
    early_stopping_rounds: int = 100,
    verbose: bool = True,
) -> dict[float, CVResult]:
    """Birden fazla kuantil seviyesinde regresyon egitir -- "kuantil merdiveni".

    ``mae_optimal_quantile`` her SATIR icin farkli bir kuantil seviyesi ister.
    Tek bir modelle bu saglanamaz; bir merdiven egitip aradaki degerleri
    interpolasyonla elde ederiz.

    Args:
        quantiles: Egitilecek seviyeler. Sik aralik daha iyi interpolasyon
            verir ama her seviye AYRI bir CV kosusu demektir -- 10 seviye x
            5 fold = 50 model egitimi.

    Returns:
        ``{kuantil: CVResult}``.

    NOT: LightGBM'de ``objective="quantile"`` ve ``alpha`` kullanilir. Bu
    surumde (4.6.0) ``alpha=0.5`` sorunsuz calisiyor -- olculdu, bir donem
    bildirilen best_iteration=1 hatasi burada YOK.
    """
    ladder: dict[float, CVResult] = {}
    base = dict(params) if params else starter_params("lightgbm", "regression")

    for level in quantiles:
        level_params = dict(base)
        level_params["objective"] = "quantile"
        level_params["alpha"] = float(level)

        if verbose:
            print(f"  kuantil {level:.2f} egitiliyor...")

        ladder[float(level)] = cross_validate(
            train, target, folds,
            kind="lightgbm", task_type="regression", metric="mae",
            params=level_params, test=test,
            early_stopping_rounds=early_stopping_rounds, verbose=False,
        )

    return ladder


#: ``q* = 1 - 0.5/p`` her zaman ``(0, 0.5]`` araligindadir (p=1 -> 0.5).
#: 0.5 USTU seviyeler MAE-optimal yolda HIC kullanilmaz -- egitilirlerse
#: CV kosusu bosa gider. Varsayilan merdiven bu araligi sikca orneklemeli.
CONDITIONAL_LADDER_LEVELS: tuple[float, ...] = (
    0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
)


def fit_conditional_quantile_ladder(
    train: pd.DataFrame,
    target: np.ndarray | pd.Series,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    quantiles: Sequence[float] = CONDITIONAL_LADDER_LEVELS,
    params: dict[str, Any] | None = None,
    kind: ModelKind = "lightgbm",
    early_stopping_rounds: int = 100,
    verbose: bool = True,
) -> dict[float, np.ndarray]:
    """POZITIF satirlarla egitilmis kuantil merdiveni; TUM satirlara tahmin.

    ``conditional_quantile_from_hurdle``in ihtiyac duydugu sey budur ve
    onceki surumde **public API ile uretilemiyordu.**

    NEDEN ``fit_quantile_ladder`` YETMIYOR (olculdu)
    -----------------------------------------------
    ``fit_quantile_ladder`` hedefin TAMAMIYLA (sifirlar dahil) egitilir,
    yani MARJINAL kuantilleri ogrenir. Oysa ``q* = 1 - 0.5/p`` turetimi
    ``F(y | y>0)`` KOSULLU dagilimin kuantilini ister. Ikisi ayni sey degildir:

        sifir orani %50 olan bir hurdle veride
          marjinal merdivenin q=0.50 tahmini : 3.700
          gercek KOSULLU medyan (y>0)        : 9.328     -> 2.5 KAT fark
          marjinal merdivenin q=0.05 tahmini : 0.000     -> tamamen sifir

    Sonuc pratikte de gorulur: marjinal merdivenle MAE 5.2678, oysa duz
    ``thresholded`` modu 5.0914 verir. Yani "MAE-optimal" diye sunulan yol,
    basit yoldan DAHA KOTUYDU.

    Args:
        quantiles: Kuantil seviyeleri. Varsayilan ``(0, 0.5]`` araligini
            orneklemektedir; q* asla 0.5'i asmaz.

    Returns:
        ``{kuantil: tahmin dizisi}`` -- her dizi ``len(train)`` uzunlugunda ve
        SIFIR satirlarini da icerir (model onlarin hedefini hic gormedi, bu
        yuzden sizinti degildir -- ``_stage2_predictions_everywhere`` ile ayni
        gerekce).
    """
    y = np.asarray(target).ravel()
    fold_list = list(folds)
    positive = y > 0
    positive_index = np.flatnonzero(positive)
    if positive_index.size == 0:
        raise ValueError("Pozitif satir yok; kosullu merdiven egitilemez.")

    # Fold'lari POZITIF alt kumesinin konumsal indekslerine cevir.
    yeniden: dict[int, int] = {
        int(eski): yeni for yeni, eski in enumerate(positive_index)
    }
    positive_folds: list[tuple[np.ndarray, np.ndarray]] = []
    source_folds: list[int] = []
    for konum, (train_idx, valid_idx) in enumerate(fold_list):
        tr = np.array([yeniden[i] for i in train_idx if i in yeniden], dtype=int)
        va = np.array([yeniden[i] for i in valid_idx if i in yeniden], dtype=int)
        if tr.size == 0 or va.size == 0:
            continue
        positive_folds.append((tr, va))
        source_folds.append(konum)
    if not positive_folds:
        raise ValueError(
            "Hicbir fold'da hem egitim hem dogrulama pozitifi yok; "
            "kosullu merdiven kurulamaz."
        )

    base = dict(params) if params else starter_params(kind, "regression")
    ladder: dict[float, np.ndarray] = {}

    for level in quantiles:
        level_params = dict(base)
        level_params["objective"] = "quantile"
        level_params["alpha"] = float(level)
        if verbose:
            print(f"  kosullu kuantil {level:.2f} egitiliyor...")

        result = cross_validate(
            train.iloc[positive_index], y[positive_index], positive_folds,
            kind=kind, task_type="regression", metric="mae",
            params=level_params, early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )
        ladder[float(level)] = _stage2_predictions_everywhere(
            result, train, fold_list, source_folds, kind=kind
        )

    return ladder


def conditional_quantile_from_hurdle(
    positive_probability: np.ndarray,
    ladder_predictions: dict[float, np.ndarray],
    *,
    verbose: bool = True,
) -> np.ndarray:
    """Kuantil merdivenini kullanarak MAE-optimal tahmini uretir.

    Her satir icin:
      1. ``q* = 1 - 0.5/p`` hesapla
      2. ``q* <= 0`` ise (yani ``p <= 0.5``) tahmin **0**
      3. Degilse merdivendeki komsu iki seviye arasinda dogrusal interpolasyon

    MERDIVEN **KOSULLU** OLMALIDIR
    ------------------------------
    ``q*`` turetimi ``F(y | y>0)`` kuantilini ister. ``fit_quantile_ladder``
    hedefin TAMAMIYLA egitilir ve MARJINAL kuantil ogrenir; ikisini
    karistirmak sessizce yanlis tahmin uretir (olculdu: marjinal q=0.50 ->
    3.700, gercek kosullu medyan -> 9.328).

    **``fit_conditional_quantile_ladder`` kullan.** Marjinal bir merdiven
    verildiginde bu fonksiyon uyarir ama durduramaz -- kesin ayrimi yalnizca
    egitim verisi bilir.

    Args:
        positive_probability: 1. asamanin OOF olasiliklari.
        ladder_predictions: ``{kuantil: tahmin dizisi}`` -- her dizi tum
            satirlar icin o kuantil seviyesindeki **kosullu** tahmin.
        verbose: Kirpma / bosa seviye / marjinal-merdiven uyarilarini basar.

    Returns:
        MAE-optimal tahminler.

    Raises:
        ValueError: Merdiven bossa veya dizi uzunluklari uyusmuyorsa.
    """
    if not ladder_predictions:
        raise ValueError("Kuantil merdiveni bos.")

    levels = np.array(sorted(ladder_predictions), dtype="float64")
    matrix = np.column_stack([ladder_predictions[level] for level in levels])

    probability = np.asarray(positive_probability, dtype="float64")
    if len(probability) != matrix.shape[0]:
        raise ValueError(
            f"Olasilik ({len(probability)}) ve tahmin ({matrix.shape[0]}) "
            "uzunluklari farkli."
        )

    _merdiven_uyarilari(levels, matrix, probability, verbose=verbose)

    targets = mae_optimal_quantile(probability)
    result = np.zeros(len(probability), dtype="float64")

    needs_quantile = ~np.isnan(targets)
    if needs_quantile.any():
        wanted = np.clip(targets[needs_quantile], levels.min(), levels.max())
        rows = np.flatnonzero(needs_quantile)
        # Satir bazinda interpolasyon: her satirin kendi merdiven degerleri
        # uzerinde, kendi hedef kuantiline gore.
        for row, level in zip(rows, wanted, strict=True):
            result[row] = float(np.interp(level, levels, matrix[row]))

    return np.clip(result, 0, None)


def _merdiven_uyarilari(
    levels: np.ndarray,
    matrix: np.ndarray,
    probability: np.ndarray,
    *,
    verbose: bool,
) -> None:
    """Merdiven MARJINAL mi, q* kirpiliyor mu? Ikisi de sessiz kalmamali."""
    if not verbose:
        return

    hedefler = mae_optimal_quantile(probability)
    gecerli = ~np.isnan(hedefler)

    # 1. Kirpma: q* merdivenin ALTINA duserse en dusuk seviyeye yapisir ve
    #    sistematik ASIRI TAHMIN uretir. OLCULDU: [0.05, 0.5] merdiveninde
    #    q* min 0.0003, 646 satirin 56'si (%8.7) kirpildi.
    if gecerli.any():
        alt_asan = int((hedefler[gecerli] < levels.min()).sum())
        if alt_asan:
            print(
                f"[conditional_quantile] {alt_asan:,}/{int(gecerli.sum()):,} "
                f"satirda q* merdivenin altinda (min q*={hedefler[gecerli].min():.4f} "
                f"< {levels.min()}); en dusuk seviyeye kirpildi -> ASIRI TAHMIN. "
                "Merdivene daha dusuk bir seviye ekle."
            )
    # 2. Bosa egitilmis seviyeler: q* asla 0.5'i asmaz.
    bosa = levels[levels > 0.5]
    if bosa.size:
        print(
            f"[conditional_quantile] {list(np.round(bosa, 3))} seviyeleri HIC "
            "kullanilmaz (q* <= 0.5). Bunlari egitmek CV kosusunu bosa harciyor."
        )
    # 3. Merdiven MARJINAL gorunuyor mu? Kosullu merdivende en dusuk seviye
    #    pozitif buyuklukleri tahmin eder; marjinal merdivende sifir-siskin
    #    veride TAM SIFIRA cokerdi. OLCULDU: marjinal q=0.05 ortalamasi 0.0000.
    en_dusuk = matrix[:, 0]
    sifir_payi = float(np.mean(np.isclose(en_dusuk, 0.0)))
    if sifir_payi > 0.5:
        print(
            f"[conditional_quantile] UYARI: en dusuk seviyenin (q={levels.min()}) "
            f"tahminlerinin %{sifir_payi * 100:.0f}'i sifir. Merdiven MARJINAL "
            "kuantilleri ogrenmis olabilir (fit_quantile_ladder tum hedefle "
            "egitir). Bu fonksiyon KOSULLU kuantil ister -- "
            "fit_conditional_quantile_ladder kullan."
        )


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


def _teshis_tablosu(
    tuning: dict[str, float],
    *,
    zero_share: float,
    metric: str,
    covered: np.ndarray,
    n_rows: int,
    verbose: bool,
) -> dict[str, Any]:
    """Karsilastirma tablosunu kurar ve gerekiyorsa basar."""
    diagnostics: dict[str, Any] = {
        "sifir_orani": round(zero_share, 4),
        "oof_kapsami": round(float(covered.mean()), 4),
        f"hep_sifir_baseline_{metric}": round(tuning["score_all_zero"], 6),
        f"carpim_modu_{metric}": round(tuning["score_expected"], 6),
        f"esikli_mod_{metric}": round(tuning["best_score"], 6),
        f"esik_0.5_{metric}": round(tuning["score_at_half"], 6),
    }
    if covered.mean() < 1.0:
        diagnostics["not"] = (
            f"skorlar {int(covered.sum()):,}/{n_rows:,} kapsanan satirda "
            "hesaplandi (kapsanmayanlar dogrulanmis tahmin tasimaz)"
        )

    if verbose:
        print("\n--- Iki asamali karsilastirma ---")
        for key, value in diagnostics.items():
            print(f"  {key:<32} {value}")
        if tuning["best_score"] >= tuning["score_all_zero"]:
            print(
                "  UYARI: model 'hep sifir' baseline'ini GECEMEDI. Sifir-siskin veride "
                "bu olur -- duz regresyonu ve farkli objective'leri de dene."
            )
        elif tuning["score_at_half"] >= tuning["score_all_zero"]:
            # SECILMIS esik baseline'i geciyor ama SABIT esik gecemiyor:
            # kazanc 200 noktali izgara aramasindan gelmis olabilir. Ayni
            # veride hem esik secip hem skor raporlamak iyimserdir.
            print(
                "  DIKKAT: baseline yalnizca SECILMIS esikle gecildi "
                f"({tuning['best_score']:.6f}); sabit esik 0.5 gecemedi "
                f"({tuning['score_at_half']:.6f}). Kazanc 200 noktali izgara "
                "aramasindan geliyor olabilir -- esigi ayri bir kumede dogrula."
            )
    return diagnostics


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

    # Skorlar ve esik YALNIZCA kapsanan satirlarda hesaplanir.
    #
    # NEDEN: TimeSeriesSplit/purged bolme ilk donemi HIC dogrulamaz. O
    # satirlarda oof_probability tam olarak 0.0'dir -- bir tahmin degil,
    # dolgudur. Onlari skora katmak, gercek pozitif satirlari "sifir tahmin
    # edildi" diye puanlamaktir.
    #
    # OLCULDU (600 satir, kapsam %75): kapsanmayan 150 satirin gercek hedef
    # ortalamasi 6.086 iken oof_probability'leri tek deger: 0.0.
    #   TUM satirlarda   -> mae 5.4283
    #   yalniz kapsanan  -> mae 5.2092   (%4.2 sisme)
    # Esik SECIMI sapmaz (o satirlar her esikte ayni davranir) ama RAPORLANAN
    # skor siser ve fold sayisi degisince karsilastirilamaz hale gelir.
    covered = classifier.oof_covered
    if covered is None:
        covered = np.ones(len(y), dtype=bool)
    covered = np.asarray(covered, dtype=bool)

    tuning = tune_threshold(
        y[covered],
        classifier.oof_predictions[covered],
        magnitude_oof[covered],
        metric=metric,
    )

    diagnostics = _teshis_tablosu(
        tuning, zero_share=zero_share, metric=metric,
        covered=covered, n_rows=len(y), verbose=verbose,
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
        oof_covered=covered,
    )
