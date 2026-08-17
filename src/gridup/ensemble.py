"""Model harmanlama: tepe tirmanma (hill climbing) ve sira ortalamasi.

Kaggle'da ilk 10 ile ilk 100 arasindaki fark genellikle tek bir model degil,
BIRDEN FAZLA CESITLI modelin harmanidir. Ama harmanlama agirliklari elle
secilmez -- OOF tahminleri uzerinde OGRENILIR.

TEMEL KURAL: Cesitlilik (diversity) tek tek performanstan onemlidir.
Korelasyonu 0.99 olan iki mukemmel model, korelasyonu 0.85 olan iki iyi
modelden daha kotu harmanlanir.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .metrics import get_metric

__all__ = [
    "hill_climb_weights",
    "rank_average",
    "correlation_matrix",
    "greedy_forward_selection",
    "stack_oof",
    "prune_by_correlation",
    "power_mean_blend",
    "tune_power_mean",
    "POWER_MEAN_EPSILON",
    "POWER_MEAN_GRID",
]


def _dolgu_satirlari(oof_predictions: dict[str, np.ndarray]) -> np.ndarray:
    """TUM uyelerde ayni anda TAM SIFIR olan satirlarin maskesi.

    Bu kalip tesadufen olusmaz: gercek tahminlerin hepsinin ayni satirda
    bit-birebir 0.0 olmasi pratikte imkansizdir. Yani bu satirlar temel
    modellerin OOF kapsami disidir -- tahmin degil DOLGU.
    """
    diziler = [np.asarray(v, dtype="float64") for v in oof_predictions.values()]
    dolgu = np.ones(len(diziler[0]), dtype=bool)
    for dizi in diziler:
        dolgu &= dizi == 0.0
    return dolgu


def _uyar_kapsam_disi(
    oof_predictions: dict[str, np.ndarray],
    fonksiyon: str,
    covered: np.ndarray | None = None,
) -> None:
    """Dolgu satiri varsa SESSIZ GECMEZ -- raporlanan skor yanlis olur.

    purged_time_series_split / TimeSeriesSplit ilk donemi hicbir fold'un valid
    tarafina koymaz. O satirlar harmana girdiginde metrige sabit bir terim
    ekler; agirliklarin argmin'i degismez ama RAPORLANAN SAYI yanlistir ve
    deney gunlugu, juri slaydi ve LB karsilastirmasi bunun uzerine kurulur.

    OLCULDU (3 uye, TimeSeriesSplit(4), N=3000, kapsam %80):
      maskesiz hill climbing rmse 2.754756 / maskeli 2.213196  -> %24.5 sapma

    ``covered`` VERILIRSE OLCUM SEZGININ ONUNE GECER
    -----------------------------------------------
    "Tum uyeler tam sifir" bir SEZGIDIR ve sifir-siskin hedefte YANILIR --
    ki bu yarismanin gercek hedef profili odur. OLCULDU (sifir orani %86.8,
    KFold(4), yani GERCEK dolgu satiri YOK, 3 uye MAE objective):

        olculen kapsam            : 1.0  (tam)
        tum uyeler tam sifir olan : 3.918/4.000  (%98.0)
        sezgiye dayali uyari      : 1 adet  -> TAMAMEN YANLIS ALARM

    Kullaniciya verisinin %98'ini maskelemesini soyluyordu. Olculmus maske
    varsa ona bakiyoruz; sezgi yalnizca olcum YOKKEN ve o zaman da "olabilir"
    diliyle konusuyor.
    """
    if covered is not None:
        maske = np.asarray(covered, dtype=bool)
        adet = int((~maske).sum())
        if adet == 0:
            return
        warnings.warn(
            f"{fonksiyon}: {adet:,} satir (%{(~maske).mean() * 100:.1f}) hicbir "
            "fold tarafindan dogrulanmadi; bu satirlardaki tahminler DOLGUDUR ve "
            "raporlanan skoru sisirir (olculdu: rmse 2.213196 -> 2.754756). "
            "covered_oof_matrix() ile maskele ve hedefi ayni indeksle kirp.",
            UserWarning,
            stacklevel=3,
        )
        return

    dolgu = _dolgu_satirlari(oof_predictions)
    adet = int(dolgu.sum())
    if adet == 0:
        return
    warnings.warn(
        f"{fonksiyon}: {adet:,} satirda (%{dolgu.mean() * 100:.1f}) TUM uyeler tam "
        "sifir. Bu OOF kapsami disi DOLGU satiri OLABILIR (raporlanan skoru "
        "sisirir; olculdu: rmse 2.213196 -> 2.754756) ama sifir-siskin hedefte "
        "MESRU de olabilir. Emin olmak icin olculmus kapsam maskesini ver: "
        "covered=zoo.oof_covered (ya da ZooResult.covered_oof_matrix()).",
        UserWarning,
        stacklevel=3,
    )


def _prediction_matrix(
    predictions: dict[str, np.ndarray], function: str
) -> tuple[list[str], np.ndarray]:
    """Harman girdilerini tek, uzunlugu dogrulanmis matrise cevirir."""
    if not predictions:
        raise ValueError(f"{function}: en az bir tahmin dizisi gerekli.")
    names = list(predictions)
    arrays = [np.asarray(predictions[name], dtype="float64").ravel() for name in names]
    lengths = {len(values) for values in arrays}
    if len(lengths) != 1:
        raise ValueError(f"{function}: tahmin dizilerinin uzunluklari ayni olmali.")
    return names, np.column_stack(arrays)


def _coverage_mask(
    predictions: dict[str, np.ndarray],
    covered: np.ndarray | None,
    function: str,
    n_rows: int,
) -> np.ndarray:
    """Olculmus OOF kapsam maskesini dogrular; yoksa eski uyariyi korur."""
    if covered is None:
        _uyar_kapsam_disi(predictions, function)
        return np.ones(n_rows, dtype=bool)
    mask = np.asarray(covered, dtype=bool).ravel()
    if len(mask) != n_rows:
        raise ValueError(
            f"{function}: covered ({len(mask)}) ve tahminler ({n_rows}) uzunluklari farkli."
        )
    if not mask.any():
        raise ValueError(f"{function}: covered uygulandiktan sonra hic satir kalmadi.")
    return mask


def correlation_matrix(
    predictions: dict[str, np.ndarray], *, covered: np.ndarray | None = None
) -> pd.DataFrame:
    """Model tahminleri arasindaki korelasyon matrisi.

    OKUMA KILAVUZU:
      > 0.99  -> modeller aslinda ayni; harmanlamak kazanc getirmez
      0.90-0.98 -> saglikli cesitlilik, harmanlama ise yarar
      < 0.85  -> cok farkli; biri belirgin kotuyse harmanlamak zarar verebilir
    """
    names, matrix = _prediction_matrix(predictions, "correlation_matrix")
    mask = _coverage_mask(predictions, covered, "correlation_matrix", len(matrix))
    frame = pd.DataFrame(matrix[mask], columns=names)
    return frame.corr()


def _fold_dilimlerini_dogrula(
    fold_slices: Sequence[np.ndarray] | None, stability_penalty: float, n_rows: int
) -> list[np.ndarray] | None:
    """Kararlilik cezasinin girdilerini fit'ten ONCE dogrular.

    Ceza fold-bazli skor ister; dilim verilmeden ceza istemek sessizce eski
    davranisa dusmek olurdu -- kullanici cezanin uygulandigini sanirdi.
    """
    if stability_penalty < 0:
        raise ValueError(f"stability_penalty >= 0 olmali, verilen: {stability_penalty}")
    if stability_penalty == 0:
        return None
    if fold_slices is None:
        raise ValueError(
            "stability_penalty > 0 icin fold_slices ZORUNLU: ceza fold'lar arasi "
            "std uzerinden tanimli. purged_time_series_split'in valid indekslerini "
            "(OOF dizisine gore konumsal) liste olarak ver."
        )
    dilimler = [np.asarray(dilim, dtype=np.int64).ravel() for dilim in fold_slices]
    if len(dilimler) < 2:
        raise ValueError("fold_slices en az 2 dilim icermeli -- tek dilimde std anlamsiz.")
    for sira, dilim in enumerate(dilimler):
        if dilim.size == 0:
            raise ValueError(f"fold_slices[{sira}] bos -- her dilim en az bir satir icermeli.")
        if dilim.min() < 0 or dilim.max() >= n_rows:
            raise ValueError(
                f"fold_slices[{sira}] OOF dizisinin disina tasiyor "
                f"(uzunluk {n_rows}, gorulen aralik [{dilim.min()}, {dilim.max()}])."
            )
    return dilimler


def _covered_fold_slices(
    fold_slices: Sequence[np.ndarray] | None,
    stability_penalty: float,
    n_rows: int,
    covered: np.ndarray,
) -> list[np.ndarray] | None:
    """Fold dilimlerini dogrular ve OOF kapsami ile kesistirir."""
    slices = _fold_dilimlerini_dogrula(fold_slices, stability_penalty, n_rows)
    if slices is None:
        return None
    covered_slices = [fold_slice[covered[fold_slice]] for fold_slice in slices]
    if any(fold_slice.size == 0 for fold_slice in covered_slices):
        raise ValueError(
            "hill_climb_weights: covered uygulandiktan sonra en az bir fold'da hic satir kalmadi."
        )
    return covered_slices


def hill_climb_weights(
    oof_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    *,
    metric: str = "rmse",
    n_iterations: int = 200,
    step: float = 0.01,
    covered: np.ndarray | None = None,
    stability_penalty: float = 0.0,
    fold_slices: Sequence[np.ndarray] | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """OOF tahminleri uzerinde harmanlama agirliklarini ogrenir.

    YONTEM (tepe tirmanma): Bos bir harmandan basla; her adimda hangi modelden
    ``step`` kadar EKLEMEK metrigi en cok iyilestiriyorsa onu ekle. Iyilesme
    kalmayinca dur.

    Neden dogrusal regresyon degil: tepe tirmanma agirliklari dogal olarak
    negatif olmayan tutar ve toplamı 1'e normalize eder; asiri uyum riski
    cok daha dusuktur. Kaggle'da yillardir standart yontem budur.

    KRITIK: Agirliklar OOF uzerinde ogrenilir, egitim tahminleri uzerinde
    DEGIL. Aksi halde harman da asiri uyum yapar.

    KARARLILIK CEZASI (Home Credit 2024 + M5 1.si, docs/10 bolum 2)
    ---------------------------------------------------------------
    ``stability_penalty > 0`` verildiginde tirmanmanin amaci degisir:
    kucuk-iyi metrikte ``ortalama(fold skorlari) + ceza * std(fold skorlari)``,
    buyuk-iyi metrikte ``ortalama - ceza * std``. Home Credit 2024'un LB
    kaymasi dersi ve M5 1.sinin "fold std'yi minimize et" ilkesiyle cifte
    kaynak: OOF ortalamasini tek fold'un hediyesiyle parlatan agirlik,
    leaderboard'da geri teper. ``fold_slices`` her fold'un valid satirlarinin
    OOF dizisine gore KONUMSAL indeks dizileridir.

    Varsayilan ``0.0`` eski davranisin BIREBIR aynisidir (tum-OOF tek skor);
    ``fold_slices`` verilse bile ceza sifirsa dilimler kullanilmaz -- ortalama
    fold skoru, tum-OOF skoruyla ayni sey degildir ve sessizce degistirilmez.

    Returns:
        ``{model_adi: agirlik}`` -- toplami 1.0.

    Raises:
        ValueError: ``stability_penalty < 0``; ceza pozitifken ``fold_slices``
            verilmemis, bos, tek dilimli veya indeksleri dizinin disindaysa.
    """
    metric_fn, greater_is_better, _ = get_metric(metric)

    names, matrix = _prediction_matrix(oof_predictions, "hill_climb_weights")
    y_values = np.asarray(y_true).ravel()
    if len(y_values) != len(matrix):
        raise ValueError("hill_climb_weights: y_true ve tahmin uzunluklari ayni olmali.")
    mask = _coverage_mask(oof_predictions, covered, "hill_climb_weights", len(matrix))
    weights = np.zeros(len(names))

    dilimler = _covered_fold_slices(fold_slices, stability_penalty, len(matrix), mask)

    def score_of(weight_vector: np.ndarray) -> float:
        total = weight_vector.sum()
        if total == 0:
            return -np.inf if greater_is_better else np.inf
        blended = matrix @ (weight_vector / total)
        if dilimler is None:
            return float(metric_fn(y_values[mask], blended[mask]))
        fold_skorlari = [float(metric_fn(y_values[dilim], blended[dilim])) for dilim in dilimler]
        ortalama = float(np.mean(fold_skorlari))
        sapma = float(np.std(fold_skorlari))
        # Kucuk-iyi metrikte ceza EKLENIR; buyuk-iyi metrikte CIKARILIR --
        # iki durumda da "oynak harman" objektifte kotulesir.
        if greater_is_better:
            return ortalama - stability_penalty * sapma
        return ortalama + stability_penalty * sapma

    # Ilk adim: en iyi tekil modelden basla. Ceza ACIKSA tekil skor da AYNI
    # cezali objektifle hesaplanir -- baslangic ve tirmanma ayni sayiyi
    # kovalamazsa ilk secim tirmanmanin amacina yabanci kalir. Ceza kapaliyken
    # eski hesap BIREBIR korunur (bit-esit geriye uyumluluk).
    def _tekil(index: int) -> float:
        birim = np.zeros(len(names))
        birim[index] = 1.0
        return score_of(birim)

    if dilimler is None:
        single_scores = [
            float(metric_fn(y_values[mask], matrix[mask, index])) for index in range(len(names))
        ]
    else:
        single_scores = [_tekil(index) for index in range(len(names))]
    best_index = int(np.argmax(single_scores) if greater_is_better else np.argmin(single_scores))
    weights[best_index] = 1.0
    best_score = single_scores[best_index]

    if verbose:
        print(f"Baslangic: {names[best_index]}  {metric}={best_score:.6f}")

    for iteration in range(n_iterations):
        candidate_scores = []
        for index in range(len(names)):
            trial = weights.copy()
            trial[index] += step
            candidate_scores.append(score_of(trial))

        candidate_index = int(
            np.argmax(candidate_scores) if greater_is_better else np.argmin(candidate_scores)
        )
        candidate_score = candidate_scores[candidate_index]

        improved = (
            candidate_score > best_score if greater_is_better else candidate_score < best_score
        )
        if not improved:
            if verbose:
                print(f"{iteration} adimda yakinsadi.")
            break

        weights[candidate_index] += step
        best_score = candidate_score

    normalized = weights / weights.sum()
    # strict=True: uzunluklar yapisal olarak esit, ama esit DEGILSE zip sessizce
    # kisa olana kirpar ve bir modeli harmandan gorunmez sekilde dusurur.
    result = {name: float(weight) for name, weight in zip(names, normalized, strict=True)}

    if verbose:
        print(f"Harman {metric}={best_score:.6f}")
        for name, weight in sorted(result.items(), key=lambda pair: pair[1], reverse=True):
            if weight > 0.001:
                print(f"  {name:<28} {weight:.4f}")

    return result


#: Kuvvet ortalamasinda log/negatif-us korumasi icin taban deger.
POWER_MEAN_EPSILON = 1e-9

#: ``tune_power_mean`` varsayilan taramasi. ASHRAE 1.-2.sinin araligi bu
#: civardaydi; 1.0 (aritmetik ortalama) HER ZAMAN izgaraya dahil edilir.
POWER_MEAN_GRID = (0.5, 1.0, 2.0)


def power_mean_blend(
    predictions: dict[str, np.ndarray],
    weights: dict[str, float],
    *,
    p: float = 1.0,
) -> np.ndarray:
    """Agirlikli kuvvet ortalamasi harmani (ASHRAE 1.-2., docs/10 bolum 2).

    Formul::

        harman = (sum_i w_i * x_i^p) ^ (1/p)        (p != 0)
        harman = exp(sum_i w_i * log(x_i))          (p == 0, geometrik)

    ``p = 1`` agirlikli aritmetik ortalamaya BIREBIR indirgenir (kirpma da
    yapilmaz -- dogrusal harmanla bit-esit). ``p > 1`` buyuk tahminlere,
    ``p < 1`` kucuk tahminlere agirlik verir; ASHRAE kazananlari dogrusal
    yerine kuvvet ortalamasiyla kucuk ama tutarli kazanc olctu.

    NEGATIF VE SIFIR KORUMASI: ``p != 1`` icin kesirli us negatif sayida
    tanimsizdir; tahminler 0'a kirpilir (sayim/sure hedefinde negatif tahmin
    zaten anlamsizdir). ``p <= 0`` icin sifirlar ``POWER_MEAN_EPSILON``
    tabanina cekilir ki log/negatif us patlamasin.

    Args:
        predictions: ``{model_adi: tahmin dizisi}`` -- hepsi ayni uzunlukta.
        weights: ``{model_adi: agirlik}`` -- ``hill_climb_weights`` ciktisi.
            Kume predictions ile AYNI olmali; eksik/fazla ad sessiz gecmez.
        p: Kuvvet. 1 = aritmetik, 0 = geometrik, 2 = karesel ortalama.

    Returns:
        Harmanlanmis tahmin dizisi (float64).

    Raises:
        ValueError: predictions bos; agirlik kumesi uyusmuyor; negatif
            agirlik; agirlik toplami sifir; dizi uzunluklari farkli.
    """
    if not predictions:
        raise ValueError("Bos tahmin sozlugu.")
    names = list(predictions)
    eksik = sorted(set(names) - set(weights))
    fazla = sorted(set(weights) - set(names))
    if eksik or fazla:
        raise ValueError(
            f"weights ve predictions ayni adlari tasimali. Eksik agirlik: {eksik}, "
            f"fazla agirlik: {fazla}."
        )
    w = np.array([float(weights[name]) for name in names], dtype="float64")
    if (w < 0).any():
        raise ValueError("Negatif agirlik: kuvvet ortalamasi negatif agirlikla tanimsiz.")
    toplam = w.sum()
    if toplam <= 0:
        raise ValueError("Agirlik toplami sifir -- en az bir pozitif agirlik gerekli.")
    w = w / toplam

    boylar = {name: len(np.asarray(predictions[name]).ravel()) for name in names}
    if len(set(boylar.values())) != 1:
        raise ValueError(f"Tahmin dizileri ayni uzunlukta degil: {boylar}")
    matrix = np.column_stack(
        [np.asarray(predictions[name], dtype="float64").ravel() for name in names]
    )

    if p == 1.0:
        # Eleman-eleman birikim (matris carpimi DEGIL): BLAS dot'un ara
        # yuvarlamasi, elle yazilmis `w1*x1 + w2*x2` harmanindan son bitte
        # sapabiliyor. Dogrusal harmanla BIT-ESIT indirgeme vaadi bunu ister.
        toplam_dizi = np.zeros(matrix.shape[0], dtype="float64")
        for sutun, agirlik in enumerate(w):
            toplam_dizi = toplam_dizi + agirlik * matrix[:, sutun]
        return toplam_dizi

    kirpik = np.clip(matrix, 0.0, None)
    if p == 0.0:
        loglar = np.log(np.maximum(kirpik, POWER_MEAN_EPSILON))
        return np.exp(loglar @ w)
    if p < 0:
        kirpik = np.maximum(kirpik, POWER_MEAN_EPSILON)
    return np.power(np.power(kirpik, p) @ w, 1.0 / p)


def tune_power_mean(
    predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    *,
    weights: dict[str, float],
    p_grid: Sequence[float] = POWER_MEAN_GRID,
    metric: str = "mae",
    covered: np.ndarray | None = None,
) -> tuple[float, float, pd.DataFrame]:
    """Kuvvet ortalamasinin ``p`` degerini OOF uzerinde tarar.

    BILINCLI SADELIK: agirliklar SABITTIR (``hill_climb_weights`` ciktisi) ve
    yalnizca ``p`` taranir. Agirlik ve p'yi birlikte aramak arama uzayini
    buyutur ve OOF gurultusune asiri uyum riskini artirir -- ASHRAE deseni de
    "once agirlik, sonra p" idi. ``p = 1.0`` izgarada yoksa OTOMATIK eklenir:
    "dogrusal harman" secenegi hic denenmezse kazancin kaynagi olculemez.

    Args:
        predictions: ``{model_adi: OOF tahmin dizisi}``.
        y_true: Gercek degerler.
        weights: Sabit harman agirliklari (``hill_climb_weights`` ciktisi).
        p_grid: Taranacak kuvvetler.
        metric: ``METRIC_REGISTRY`` icindeki bir ad.
        covered: OOF kapsam maskesi; verilirse yalnizca kapsanan satirlar
            skorlanir (purged bolmede ilk donem dolgudur).

    Returns:
        ``(best_p, best_score, table)`` -- table ``p``/``skor`` kolonlu.

    Raises:
        ValueError: Uzunluklar uyumsuz veya skorlanacak satir kalmadiysa.
    """
    y = np.asarray(y_true, dtype="float64").ravel()
    n = len(next(iter(predictions.values()))) if predictions else 0
    if len(y) != n:
        raise ValueError(f"y_true ({len(y)}) ve tahminler ({n}) uzunluklari farkli.")

    maske = np.ones(len(y), dtype=bool)
    if covered is not None:
        maske = np.asarray(covered, dtype=bool).ravel()
        if len(maske) != len(y):
            raise ValueError(f"covered ({len(maske)}) ve y_true ({len(y)}) uzunluklari farkli.")
    if not maske.any():
        raise ValueError("Skorlanacak satir kalmadi (bos kapsam maskesi).")

    kapsanan = {
        name: np.asarray(dizi, dtype="float64").ravel()[maske] for name, dizi in predictions.items()
    }
    metric_fn, greater_is_better, _ = get_metric(metric)

    # Yuvarlama float artiklarini temizler; 1.0 daima izgaraya girer.
    kuvvetler = np.unique(
        np.round(np.concatenate([np.asarray(p_grid, dtype="float64").ravel(), [1.0]]), 6)
    )
    skorlar = np.array(
        [
            float(metric_fn(y[maske], power_mean_blend(kapsanan, weights, p=float(p))))
            for p in kuvvetler
        ]
    )
    tablo = pd.DataFrame({"p": kuvvetler, "skor": skorlar})

    best_index = int(np.argmax(skorlar) if greater_is_better else np.argmin(skorlar))
    return float(kuvvetler[best_index]), float(skorlar[best_index]), tablo


def greedy_forward_selection(
    oof_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    *,
    metric: str = "rmse",
    max_models: int = 30,
    with_replacement: bool = True,
    covered: np.ndarray | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """Tekrarli acgozlu secim (Caruana yontemi) ile harman kurar.

    Tepe tirmanmanin akrabasi: her turda harmana EKLENECEK en iyi modeli sec
    (ayni model birden fazla kez secilebilir -- bu ona daha yuksek agirlik verir).

    Az sayida modelle (3-6) tepe tirmanma yeterlidir; 10+ modelde bu yontem
    daha kararli sonuc verir.
    """
    metric_fn, greater_is_better, _ = get_metric(metric)

    names, matrix = _prediction_matrix(oof_predictions, "greedy_forward_selection")
    y_values = np.asarray(y_true).ravel()
    if len(y_values) != len(matrix):
        raise ValueError("greedy_forward_selection: y_true ve tahmin uzunluklari ayni olmali.")
    mask = _coverage_mask(oof_predictions, covered, "greedy_forward_selection", len(matrix))
    covered_predictions = {name: matrix[mask, index] for index, name in enumerate(names)}
    covered_y = y_values[mask]
    selected: list[str] = []
    running_sum = np.zeros_like(covered_y, dtype="float64")
    best_score = -np.inf if greater_is_better else np.inf

    for step in range(max_models):
        candidates = names if with_replacement else [n for n in names if n not in selected]
        if not candidates:
            break

        scores = []
        for name in candidates:
            trial = (running_sum + covered_predictions[name]) / (len(selected) + 1)
            scores.append(float(metric_fn(covered_y, trial)))

        index = int(np.argmax(scores) if greater_is_better else np.argmin(scores))
        improved = scores[index] > best_score if greater_is_better else scores[index] < best_score
        if not improved:
            if verbose:
                print(f"{step} modelde yakinsadi.")
            break

        selected.append(candidates[index])
        running_sum = running_sum + covered_predictions[candidates[index]]
        best_score = scores[index]

    if not selected:
        raise ValueError("Hicbir model secilmedi -- tahminleri kontrol et.")

    counts = {name: selected.count(name) / len(selected) for name in set(selected)}

    if verbose:
        print(f"Harman {metric}={best_score:.6f}  ({len(selected)} adimda)")
        for name, weight in sorted(counts.items(), key=lambda pair: pair[1], reverse=True):
            print(f"  {name:<28} {weight:.4f}")

    return counts


def prune_by_correlation(
    oof_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    *,
    metric: str = "rmse",
    max_correlation: float = 0.99,
    max_members: int = 8,
    covered: np.ndarray | None = None,
) -> list[str]:
    """Birbirine cok benzeyen modelleri eler, cesitliligi korur.

    YONTEM: En iyi modelden basla. Sirayla ekle; ama eklenecek model zaten
    secilmis bir modelle ``max_correlation`` uzerinde korelasyona sahipse ATLA.

    NEDEN: Korelasyonu 0,995 olan iki model harmanlandiginda kazanc yaklasik
    sifirdir -- ama her ikisi de tahmin suresini, model boyutunu ve
    aciklanabilirlik maliyetini iki katina cikarir. Juri notebook'u okuyacagi
    bu yarismada bu maliyet gercektir.

    Args:
        max_correlation: Bu esigin ustunde korelasyon varsa ikincisi elenir.
        max_members: Harmanda en fazla kac model kalsin.

    Returns:
        Tutulacak model adlari, en iyiden baslayarak.
    """
    metric_fn, greater_is_better, _ = get_metric(metric)

    names, matrix = _prediction_matrix(oof_predictions, "prune_by_correlation")
    y_values = np.asarray(y_true).ravel()
    if len(y_values) != len(matrix):
        raise ValueError("prune_by_correlation: y_true ve tahmin uzunluklari ayni olmali.")
    mask = _coverage_mask(oof_predictions, covered, "prune_by_correlation", len(matrix))
    covered_y = y_values[mask]
    scores = {
        name: float(metric_fn(covered_y, matrix[mask, index])) for index, name in enumerate(names)
    }
    order = sorted(scores, key=lambda name: scores[name], reverse=greater_is_better)

    frame = pd.DataFrame(matrix[mask], columns=names)
    correlations = frame.corr()

    kept: list[str] = []
    for name in order:
        if len(kept) >= max_members:
            break
        if any(abs(correlations.loc[name, other]) > max_correlation for other in kept):
            continue
        kept.append(name)

    return kept


#: Meta-model bu kadar satirdan az ile egitilirse degenere kabul edilir.
MIN_META_TRAIN_ROWS = 20


def _resolve_base_coverage(
    oof_predictions: dict[str, np.ndarray],
    base_covered: np.ndarray | None,
    *,
    verbose: bool,
) -> np.ndarray:
    """Temel modellerin GERCEK tahmin urettigi satirlarin maskesi.

    ``base_covered`` verilmezse tespit ederiz: **tum uyelerde ayni anda tam
    sifir** olan satirlar dolgudur. Bu kalip tesadufen olusmaz -- gercek
    tahminlerin hepsinin ayni satirda bit-birebir 0.0 olmasi pratikte
    imkansizdir.

    Tespit YETERLI DEGILDIR, sadece son savunmadir: dogrusu
    ``CVResult.oof_covered`` maskelerinin kesisimini VERMEKTIR.
    """
    n = len(next(iter(oof_predictions.values())))

    if base_covered is not None:
        maske = np.asarray(base_covered, dtype=bool).ravel()
        if len(maske) != n:
            raise ValueError(
                f"base_covered uzunlugu ({len(maske)}) OOF uzunluguyla ({n}) uyusmuyor."
            )
        return maske

    dolgu = _dolgu_satirlari(oof_predictions)
    maske = ~dolgu

    oran = float(dolgu.mean())
    if oran > 0 and verbose:
        print(
            f"  NOT: {int(dolgu.sum()):,} satirda ({oran:.1%}) TUM uyeler tam sifir -- "
            "temel OOF kapsami disi sayildi ve meta-egitimden cikarildi.\n"
            "  Daha guvenlisi: base_covered=zoo.oof_covered ver (ZooResult bu "
            "kesisimi OLCEREK tasir, biz burada sadece tahmin ediyoruz)."
        )
    return maske


def stack_oof(
    oof_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    test_predictions: dict[str, np.ndarray] | None = None,
    base_covered: np.ndarray | None = None,
    meta: str = "ridge",
    metric: str = "rmse",
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, object]:
    """Ikinci seviye model (stacking) ile harmanlar.

    Hill climbing agirliklari OGRENIR ama hepsi POZITIF ve toplami 1'dir.
    Stacking bu kisiti kaldirir: bir model sistematik olarak fazla tahmin
    ediyorsa meta-model ona NEGATIF agirlik verip duzeltebilir.

    Bedeli aciklanabilirliktir. Bu yarismada notebook juri tarafindan
    okunuyor; stacking kazanci kucukse hill climbing tercih edilmelidir.
    ``vs_hill_climbing`` alani bu karari vermen icin ikisini de raporluyor.

    KRITIK -- SIZINTI: Meta-model, birinci seviye OOF tahminleri uzerinde
    AYNI FOLD'LARLA capraz dogrulanir. Tum OOF uzerinde tek seferde egitmek,
    meta-modelin birinci seviyenin hatalarina asiri uymasina yol acar.

    Args:
        meta: ``ridge`` (dogrusal, aciklanabilir) veya ``lgbm`` (dogrusal
            olmayan, daha guclu ama daha opak).

    Returns:
        ``oof`` (meta-modelin OOF tahmini), ``test``, ``score``,
        ``vs_hill_climbing``, ``coefficients`` iceren sozluk.
    """
    metric_fn, greater_is_better, _ = get_metric(metric)

    names = list(oof_predictions)
    features = pd.DataFrame(oof_predictions)
    y = np.asarray(y_true, dtype="float64")

    # TEMEL MODELLERIN KAPSAMI -- bu blogun varlik sebebi
    # ---------------------------------------------------
    # Birinci seviye OOF dizilerinde, hicbir fold'un valid tarafinda olmayan
    # satirlar SIFIRDIR (gercek tahmin degil, dolgu). Meta-modeli o satirlarla
    # egitmek, sifir feature'lari GERCEK hedefle eslestirmek demektir.
    #
    # TimeSeriesSplit'te bu felakete donusur: ilk fold'un egitim kumesi en
    # ESKI bloktur ve o blok hicbir temel fold'un valid tarafinda degildir --
    # yani meta-egitim verisinin %100'u sifirdir.
    #
    # OLCULDU (3 uye, TimeSeriesSplit(4), N=3000):
    #   fold-1 meta katsayilari : [0. 0. 0.]   (hepsi TAM sifir)
    #   fold-1 test tahmin std  : 5.55e-17     (TAM SABIT model)
    #   test RMSE (maskesiz)    : 1.0791
    #   test RMSE (maskeli)     : 0.5352       <- IKI KATI fark
    # Sabit model test harmanina girdigi icin harman 3/4 oraninda BUZUSUYOR.
    base_mask = _resolve_base_coverage(oof_predictions, base_covered, verbose=verbose)

    meta_oof = np.zeros(len(y), dtype="float64")
    covered = np.zeros(len(y), dtype=bool)
    coefficient_rows: list[np.ndarray] = []
    fold_models: list[Any] = []
    atlanan: list[int] = []

    for fold_index, (train_idx, valid_idx) in enumerate(folds, start=1):
        egitim = train_idx[base_mask[train_idx]]
        if len(egitim) < MIN_META_TRAIN_ROWS:
            # Bu fold'un meta-modeli DEGENERE olurdu. Modeli listeye
            # EKLEMIYORUZ: sabit bir model test harmanini buzusturur.
            atlanan.append(fold_index)
            continue
        model = _build_meta(meta, seed)
        model.fit(features.iloc[egitim], y[egitim])
        uygula = valid_idx[base_mask[valid_idx]]
        if len(uygula) == 0:
            atlanan.append(fold_index)
            continue
        meta_oof[uygula] = model.predict(features.iloc[uygula])
        covered[uygula] = True
        fold_models.append(model)
        if hasattr(model, "coef_"):
            coefficient_rows.append(np.asarray(model.coef_, dtype="float64").ravel())

    if atlanan:
        warnings.warn(
            f"Stacking: {len(atlanan)} fold ATLANDI (fold {atlanan}) -- temel "
            f"modellerin OOF kapsami disinda kaldiklari icin meta-modelleri "
            "degenere olurdu. Kalan fold'larla devam ediliyor.\n"
            "Bu normaldir: TimeSeriesSplit'te ilk donem hicbir fold'un valid "
            "tarafinda olmaz.",
            UserWarning,
            stacklevel=2,
        )
    if not fold_models:
        raise ValueError(
            "Stacking icin kullanilabilir fold kalmadi: temel modellerin OOF "
            "kapsami cok dar. Daha az fold veya daha kucuk test_span dene."
        )

    score = float(metric_fn(y[covered], meta_oof[covered])) if covered.any() else float("nan")

    # Karsilastirma: hill climbing ne veriyordu?
    weights = hill_climb_weights(
        {name: oof_predictions[name][covered] for name in names},
        y[covered],
        metric=metric,
        verbose=False,
    )
    hill_blend = sum(weights[name] * oof_predictions[name][covered] for name in names)
    hill_score = float(metric_fn(y[covered], hill_blend))

    stacking_wins = score > hill_score if greater_is_better else score < hill_score

    test_blend = None
    if test_predictions:
        test_features = pd.DataFrame({name: test_predictions[name] for name in names})
        # Fold modellerinin ortalamasi -- tek model yerine, varyansi dusurur.
        test_blend = np.mean([model.predict(test_features) for model in fold_models], axis=0)

    coefficients = (
        dict(zip(names, np.mean(coefficient_rows, axis=0), strict=True)) if coefficient_rows else {}
    )

    if verbose:
        print(f"Stacking ({meta}) {metric}: {score:.6f}")
        print(f"Hill climbing  {metric}: {hill_score:.6f}")
        print(
            "  -> "
            + (
                "stacking kazandi"
                if stacking_wins
                else "hill climbing kazandi (daha aciklanabilir, onu tercih et)"
            )
        )
        if coefficients:
            print("  Meta-model katsayilari:")
            for name, value in sorted(coefficients.items(), key=lambda p: -abs(p[1])):
                marker = "  (negatif -- duzeltici)" if value < 0 else ""
                print(f"    {name:<28} {value:+.4f}{marker}")

    return {
        "oof": meta_oof,
        "test": test_blend,
        "score": score,
        "coverage": float(covered.mean()),
        "coefficients": coefficients,
        "vs_hill_climbing": {
            "stacking": score,
            "hill_climbing": hill_score,
            "stacking_wins": bool(stacking_wins),
            "hill_weights": weights,
        },
    }


def _build_meta(meta: str, seed: int) -> Any:
    """Meta-model orneği kurar."""
    if meta == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=1.0, random_state=seed)

    if meta == "lgbm":
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=7,
            min_child_samples=50,
            verbose=-1,
            random_state=seed,
        )

    raise ValueError(f"Bilinmeyen meta model '{meta}'. 'ridge' veya 'lgbm'.")


def rank_average(
    predictions: Sequence[np.ndarray], weights: Sequence[float] | None = None
) -> np.ndarray:
    """Sira bazli agirlikli ortalama.

    NE ZAMAN: metrik siralamaya duyarliysa (AUC, MAP, NDCG) ve modellerin
    cikti olcekleri farkliysa. Bir model 0-1 olasilik, digeri -3..+7 skor
    uretiyorsa deger ortalamasi ikincisine haksiz agirlik verir; sira
    ortalamasi bu sorunu tamamen ortadan kaldirir.
    """
    if not predictions:
        raise ValueError("Bos tahmin listesi.")

    ranked = [pd.Series(prediction).rank(pct=True).to_numpy() for prediction in predictions]

    if weights is None:
        weights = [1.0 / len(ranked)] * len(ranked)
    if len(weights) != len(ranked):
        raise ValueError("Agirlik sayisi tahmin sayisiyla uyusmuyor.")

    total = sum(weights)
    return sum((weight / total) * column for weight, column in zip(weights, ranked, strict=True))
