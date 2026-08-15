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
    "sqrt_transform_target",
    "inverse_sqrt_transform",
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


#: Esik optimizasyonu bu skorun uzerine cikarsa uyaririz. Gercek bir
#: yarismada fold-disi tahminlerle 0.99 ustu F1/accuracy pratikte GORULMEZ;
#: gorulduyse neredeyse her zaman EGITIM tahminleri verilmistir.
SUSPICIOUS_SCORE = 0.99

#: AYNI sezgi, KUCUK-DAHA-IYI metrikler icin. logloss/mae/rmse'de "supheli
#: mukemmellik" yukari degil ASAGI dogrudur: skor 0'a yaklasir. Esik simetrik
#: secildi (1 - 0.99). Hard 0/1 tahminlerde bu metrikler ancak siniflar
#: neredeyse hatasiz ayrildiginda buranin altina iner -- logloss'ta tek bir
#: hatali satir bile ~36 katki verir, yani fold-disi bir dizide 0.01'in altina
#: inmek pratikte imkansizdir.
SUSPICIOUS_LOW_SCORE = 0.01


#: Secilen esikte pozitif tahmin orani bu araligin DISINDAysa tahmin
#: DEJENEREDIR (herkese ayni etiket). Kucuk-daha-iyi metriklerde mukemmel
#: gorunen skor o zaman sizintidan degil, metrigin kendisinden gelir.
DEGENERATE_POSITIVE_RATE = (0.02, 0.98)


def _skor_supheli(
    best_score: float, greater_is_better: bool, positive_rate: float | None = None
) -> bool:
    """Esik skoru "fazla iyi" mi? Metrigin YONUNE ve DEJENERELIGE gore bakar.

    Buyuk-daha-iyi metrikte mukemmellik yukaridan (``> 0.99``), kucuk-daha-iyi
    metrikte asagidan (``< 0.01``) gelir. Tek yone bakmak, metrigin yarisinda
    sezgiyi sessizce kapatir.

    DEJENERE TAHMIN ELENIR (olculdu)
    --------------------------------
    Kucuk-daha-iyi tarafta tek bir mutlak sabit, olcekleri tamamen farkli
    metriklere (mae, logloss 0..34.5, mape 0..sonsuz) uygulaniyordu ve MUMKUN
    OLAN EN KOTU modele "sizinti var" diyordu:

        mape + "her seye POZITIF de" modeli, n=5000
          best_threshold=0.0100  best_score(mape)=0.000000
          pozitif tahmin orani  = 1.0000   <- mumkun olan EN KOTU tahmin
          ayni tahminin accuracy'si = 0.2094
          -> SUPHELI UYARI: 1  (tamamen yanlis)

    Mekanizma deterministik: ikili hedefte hard tahminle MAPE, pozitifler
    arasindaki yanlis-negatif oranidir; hicbir pozitif kacmayinca TAM 0 olur.
    Bu bir sizinti belirtisi degil, metrigin dejenere halidir. Pozitif tahmin
    orani ucta ise sezgiyi susturuyoruz.
    """
    if greater_is_better:
        return best_score > SUSPICIOUS_SCORE
    if best_score >= SUSPICIOUS_LOW_SCORE:
        return False
    if positive_rate is None:
        return True
    alt, ust = DEGENERATE_POSITIVE_RATE
    return alt <= positive_rate <= ust


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

    # 0.5 IZGARAYA ACIKCA EKLENIR.
    #
    # np.linspace(0.01, 0.99, 200) adimi 0.004924'tur ve 0.5'i ISKALAR
    # (en yakin nokta 0.4975). Bu fonksiyonun tum amaci 0.5'i yenmekken,
    # 0.5'i hic denemedigi icin ONDAN KOTU bir esik dondurebiliyordu.
    # OLCULDU: dengesiz veride en iyi f1=0.7356 dondu, oysa 0.5 esiginde
    # f1=0.7429 -- yani "optimizasyon" skoru DUSURDU.
    thresholds = np.unique(np.concatenate([np.linspace(0.01, 0.99, n_steps), [0.5]]))
    scores = np.array(
        [float(metric_fn(y_true, (y_proba >= threshold).astype(int))) for threshold in thresholds]
    )

    best_index = int(np.argmax(scores) if greater_is_better else np.argmin(scores))
    best_score = float(scores[best_index])

    # SIZINTI SEZGISI
    # ---------------
    # Bu fonksiyon aldigi dizinin fold-disi mi yoksa EGITIM tahmini mi
    # oldugunu BILEMEZ -- imzasinda fold yok ve olamaz da (dogal kullanim
    # zaten fold'lardan uretilmis bir OOF dizisidir).
    #
    # Ama belirtisini yakalayabilir. OLCULDU: ayni hedefte
    #   egitim tahminiyle optimize -> f1 = 1.000
    #   OOF tahminiyle optimize    -> f1 = 0.612
    # Gercek bir yarismada fold-disi tahminlerle 0.99 ustu skor pratikte
    # gorulmez. Gorulduyse ya sizinti vardir ya problem trivialdir; ikisi de
    # kullanicinin BILMESI gereken seylerdir.
    #
    # SEZGI IKI YONE DE BAKAR. Onceki surumdeki ``greater_is_better and ...``
    # kapisi, kucuk-daha-iyi metriklerde mekanizmayi TAMAMEN devre disi
    # birakiyordu -- oysa oralarda supheli mukemmellik yukari degil ASAGI
    # dogrudur. OLCULDU (y_proba = y, yani tam sizintili dizi, n=800):
    #   f1       best_score=1.000000 -> uyari 1   (yakalaniyordu)
    #   accuracy best_score=1.000000 -> uyari 1   (yakalaniyordu)
    #   logloss  best_score=0.000000 -> uyari 0   (SESSIZ)
    #   mae      best_score=0.000000 -> uyari 0   (SESSIZ)
    #   rmse     best_score=0.000000 -> uyari 0   (SESSIZ)
    pozitif_oran = float((np.asarray(y_proba) >= thresholds[best_index]).mean())
    if _skor_supheli(best_score, greater_is_better, pozitif_oran):
        yon = "yuksek" if greater_is_better else "dusuk"
        sinir = SUSPICIOUS_SCORE if greater_is_better else SUSPICIOUS_LOW_SCORE
        warnings.warn(
            f"Esik optimizasyonu {metric}={best_score:.4f} buldu -- fold-disi "
            f"tahminlerde bu deger supheli derecede {yon} (sinir {sinir}).\n"
            "Kontrol et: y_proba GERCEKTEN fold-disi mi? Egitim tahminlerinde "
            "optimize edilen esik, gercek veride cok daha kotu calisir.\n"
            "Dogru kullanim: CVResult.covered_predictions() ile OOF dizisini al.",
            UserWarning,
            stacklevel=2,
        )

    return {
        "best_threshold": float(thresholds[best_index]),
        "best_score": best_score,
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
       birincisinin (Pikachow) final mimarisi: 25 seed full-data +
       mean blend + **round** + **clip** (final sunumu s.24).
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


def sqrt_transform_target(y: np.ndarray) -> np.ndarray:
    """``sqrt`` donusumu -- carpik sayim hedefleri icin log1p'den yumusak alternatif.

    NEREDEN GELDI (2024-2025 yarisma kaniti)
    ----------------------------------------
    Rohlik Sales v2'de hem 2. hem 3. BIRBIRINDEN BAGIMSIZ olarak ayni sonucu
    raporladi: carpik, sifir agirlikli satis hedefinde ``sqrt(y)`` + L2
    objective, hem ham MAE objective'ini hem LightGBM'in yerli Tweedie'sini
    gecti. Ikincinin gerekcesi: "hedef Tweedie dagilimli; sqrt onu MSE'nin
    sevdigi sekle sokar." Karsi ornek de kayitli: Rohlik Orders 3.'sunde
    ``log1p`` CV'yi KOTULESTIRDI (ayni yarismada 2. log1p'den kazanc
    raporlarken). Yani donusum secimi TEORIDEN OKUNAMAZ -- log1p, sqrt ve
    ham hedef ayni benchmark'ta yan yana OLCULUR, kazanan veriye gore secilir.

    log1p ile fark: sqrt buyuk degerleri log kadar sert ezmez (sqrt(100)=10,
    log1p(100)=4.6) -- uzun ama COK uzun olmayan kuyruklarda log fazla
    baskilar, sqrt dengeyi korur.
    """
    y = np.asarray(y, dtype="float64")
    if np.any(y < 0):
        raise ValueError("sqrt negatif degerde tanimsiz. Hedefi kontrol et.")
    return np.sqrt(y)


def inverse_sqrt_transform(y_sqrt: np.ndarray, *, clip_negative: bool = True) -> np.ndarray:
    """``sqrt`` donusumunu geri alir (kare).

    ``clip_negative`` KRITIKTIR ve kareden ONCE uygulanir: model sqrt
    uzayinda ``-0.5`` tahmin edebilir; dogrudan kare almak onu ``+0.25``e
    cevirir -- isaret hatasi sessizce pozitif tahmine donusur. Once 0'a
    kirpip sonra kare aliyoruz.
    """
    values = np.asarray(y_sqrt, dtype="float64")
    if clip_negative:
        values = np.clip(values, 0, None)
    return np.square(values)
