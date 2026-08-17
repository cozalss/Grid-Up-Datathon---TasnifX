"""Capraz dogrulamali egitim dongusu: LightGBM / XGBoost / CatBoost.

Tek bir ``cross_validate`` fonksiyonu tum modelleri ayni sozlesmeyle egitir ve
her zaman ayni seyi dondurur: fold-disi (OOF) tahminler, test tahminleri, fold
skorlari ve feature onemleri.

NEDEN OOF TAHMINLER MERKEZDE
----------------------------
OOF vektoru yarismanin para birimidir:
  * gercek CV skorunu verir (fold ortalamasi degil, TUM veri uzerinde tek skor)
  * esik optimizasyonu OOF uzerinde yapilir
  * harmanlama (blending) agirliklari OOF uzerinde ogrenilir
  * hata analizi OOF uzerinde yapilir

Bu yuzden ``cross_validate`` OOF'u her zaman dondurur ve diske yazmayi onerir.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from .compat import MISSING_CATEGORY, categorical_columns, safe_str
from .metrics import (
    get_metric,
    inverse_log_transform,
    inverse_sqrt_transform,
    log_transform_target,
    sqrt_transform_target,
)
from .validation import assert_folds_align

__all__ = [
    "CVResult",
    "cross_validate",
    "fit_without_validation",
    "LGB_DEFAULTS",
    "XGB_DEFAULTS",
    "CAT_DEFAULTS",
    "EARLY_STOPPING_BIAS_NOTE",
    "INFRASTRUCTURE_KEYS",
    "MIN_CATEGORY_OVERLAP",
    "LGB_LOG_LINK_OBJECTIVES",
    "assert_finite_target",
    "merge_infrastructure_params",
    "monotone_constraints_for",
    "starter_params",
]

ModelKind = Literal["lightgbm", "xgboost", "catboost"]
TargetTransform = (
    str | tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]] | None
)


_EARLY_STOPPING_METRICS: dict[str, dict[str, str]] = {
    "lightgbm": {
        "rmse": "rmse",
        "rmsle": "rmse",
        "mae": "mae",
        "mape": "mape",
        "r2": "l2",
        "auc": "auc",
        "logloss": "binary_logloss",
        "accuracy": "binary_error",
    },
    "xgboost": {
        "rmse": "rmse",
        "rmsle": "rmsle",
        "mae": "mae",
        "mape": "mape",
        "r2": "rmse",
        "auc": "auc",
        "logloss": "logloss",
        "accuracy": "error",
    },
    "catboost": {
        "rmse": "RMSE",
        "rmsle": "MSLE",
        "mae": "MAE",
        "mape": "MAPE",
        "smape": "SMAPE",
        "r2": "R2",
        "auc": "AUC",
        "logloss": "Logloss",
        "f1": "F1",
        "accuracy": "Accuracy",
    },
}


def _resolve_early_stopping_metric(
    kind: ModelKind,
    params: dict[str, Any],
    metric: str | None,
    *,
    target_transform: str | None = None,
    early_stopping_rounds: int = 200,
) -> tuple[dict[str, Any], str | None]:
    """Resmi metrik adini backend ``eval_metric`` degerine cevirir.

    Ayni ayarin hem ust-seviye API'de hem ``params`` icinde farkli verilmesi
    erken durdurmayi sessizce baska objektife baglardi; bu nedenle celiskide
    hangisinin kazanacagini tahmin etmek yerine kapali hata veririz.
    """
    resolved = dict(params)
    existing = resolved.get("eval_metric")
    if early_stopping_rounds <= 0:
        return resolved, str(existing) if existing is not None else None

    if target_transform is not None:
        if metric is None:
            raise ValueError(
                "target_transform ile early stopping kullanirken matematiksel olarak "
                "denk early_stopping_metric acikca verilmelidir."
            )
        if not (target_transform == "log1p" and metric.lower() == "rmsle"):
            raise ValueError(
                f"target_transform={target_transform!r} ile ham uzaydaki "
                f"early_stopping_metric={metric!r} matematiksel olarak denk degildir. "
                "Early stopping'i kapatip sabit tur kullanin veya log1p+RMSLE secin."
            )
    elif metric is not None and metric.lower() == "rmsle":
        raise ValueError(
            "Donusumsuz RMSLE, backend RMSE'ye denk degildir. "
            "target_transform='log1p' kullanin veya early stopping'i kapatin."
        )

    if metric is None:
        return resolved, str(existing) if existing is not None else None

    key = metric.lower()
    # Model log1p(y) uzerinde egitiliyorsa ham uzaydaki RMSLE, fit uzayinda
    # RMSE'ye denktir. XGBoost'un ``rmsle``si veya CatBoost'un ``MSLE``si bu
    # donusturulmus hedefe bir KEZ DAHA log uygular ve erken durdurmayi yanlis
    # uzaya baglar. Kullanici resmi metriği RMSLE diye belirtmeye devam eder;
    # backend'e matematiksel olarak denk olan RMSE gider.
    if target_transform == "log1p" and key == "rmsle":
        key = "rmse"
    mapping = _EARLY_STOPPING_METRICS[kind]
    if key not in mapping:
        raise ValueError(
            f"early_stopping_metric={metric!r}, {kind} icin desteklenmiyor. "
            f"Desteklenen genel adlar: {sorted(mapping)}"
        )
    backend_metric = mapping[key]
    if existing is not None and str(existing).lower() != backend_metric.lower():
        raise ValueError(
            "early_stopping_metric ile params['eval_metric'] celisiyor: "
            f"{metric!r} -> {backend_metric!r}, params={existing!r}."
        )
    resolved["eval_metric"] = backend_metric
    return resolved, backend_metric


def _resolve_target_transform(
    target: np.ndarray,
    transform: TargetTransform,
    task_type: str,
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray], str | None]:
    """Ham hedefi fit uzayina, tahmini yeniden ham skor uzayina baglar."""
    if transform is None:
        return target, lambda values: np.asarray(values, dtype="float64"), None
    if task_type != "regression":
        raise ValueError("target_transform yalnizca regression gorevlerinde kullanilabilir.")

    if isinstance(transform, str):
        key = transform.lower()
        transforms: dict[
            str,
            tuple[
                Callable[[np.ndarray], np.ndarray],
                Callable[[np.ndarray], np.ndarray],
            ],
        ] = {
            "log1p": (log_transform_target, inverse_log_transform),
            "sqrt": (sqrt_transform_target, inverse_sqrt_transform),
        }
        if key not in transforms:
            raise ValueError(
                f"Bilinmeyen target_transform {transform!r}. Secenekler: {sorted(transforms)}"
            )
        forward, inverse = transforms[key]
        name = key
    else:
        if len(transform) != 2 or not all(callable(fn) for fn in transform):
            raise ValueError("target_transform (forward, inverse) callable cifti olmali.")
        forward, inverse = transform
        name = "custom"

    transformed = np.asarray(forward(target), dtype="float64").ravel()
    if transformed.shape != target.shape:
        raise ValueError("target_transform hedef uzunlugunu/boyutunu degistiremez.")
    assert_finite_target(transformed)
    return transformed, inverse, name


def _inverse_predictions(
    predictions: np.ndarray,
    inverse: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Backend tahminini resmi (ham) skor uzayina guvenle geri cevirir."""
    model_space = np.asarray(predictions, dtype="float64").ravel()
    raw = np.asarray(inverse(model_space), dtype="float64").ravel()
    if raw.shape != model_space.shape:
        raise ValueError("target_transform inverse tahmin uzunlugunu/boyutunu degistiremez.")
    if not np.isfinite(raw).all():
        raise ValueError("target_transform inverse NaN/sonsuz tahmin uretti.")
    return raw


# --- Baslangic parametreleri -------------------------------------------------
# Bunlar "iyi ilk deneme" degerleridir, optimum degil. Once bunlarla bir baseline
# kur, sonra Optuna ile ara. Erken durdurma (early stopping) ile birlikte
# n_estimators yuksek birakilir -- model kendi duracagi yeri bulur.

LGB_DEFAULTS: dict[str, Any] = {
    "n_estimators": 5000,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 40,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": -1,
    # ONEM OLCUSU: 'split' DEGIL 'gain'.
    #
    # LightGBM varsayilani 'split' -- bir feature'in kac kez bolmede
    # kullanildigi. Cok agacli bir modelde bu sayi DOYAR: gurultu
    # feature'lari da defalarca bolmeye girer ve gercek sinyalden ayirt
    # edilemez hale gelir.
    #
    # OLCULDU (3 gercek sinyal + 40 saf gurultu, 5000 agac, N=4000):
    #   split -> gercek/gurultu onem orani =  0.97x   (sinyal gurultuden
    #                                                  DAHA ONEMSIZ gorunuyor)
    #   gain  -> gercek/gurultu onem orani = 12.80x
    # 'gain' bolmenin sagladigi GERCEK kayip azalmasini olcer ve doymaz.
    "importance_type": "gain",
}

XGB_DEFAULTS: dict[str, Any] = {
    "n_estimators": 5000,
    "learning_rate": 0.03,
    "max_depth": 7,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "enable_categorical": True,
    "n_jobs": -1,
}

CAT_DEFAULTS: dict[str, Any] = {
    "iterations": 5000,
    "learning_rate": 0.03,
    "depth": 7,
    "l2_leaf_reg": 3.0,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.85,
    "verbose": 0,
    "allow_writing_files": False,
}


#: ALTYAPI anahtarlari -- ogrenme hiperparametresi DEGIL, calisma sarti.
#: Kullanici ``params`` verdiginde bunlar KAYBOLMAMALI:
#:
#:   enable_categorical yoksa -> XGBoost kategorik kolonda COKER
#:   verbose yoksa            -> konsol binlerce satir cikti ile dolar
#:   allow_writing_files yoksa-> CatBoost her kosuda catboost_info/ klasoru yazar
#:   tree_method yoksa        -> XGBoost yavas 'exact' moda duser
#:   n_jobs yoksa             -> tek cekirdek kullanilir
#:
#: OLCULDU: params={'n_estimators':200} vermek enable_categorical'i dusuruyor
#: ve "DataFrame.dtypes for data must be int, float, bool or category" hatasi
#: aliniyordu. Kullanici sadece agac sayisini degistirmek istemisti.
#:
#: Ogrenme hiperparametrelerini (learning_rate, num_leaves, reg_alpha ...)
#: BILEREK birlestirmiyoruz: kullanici params verdiyse modelin ogrenme
#: davranisini tam olarak o belirlemeli -- yoksa gizli varsayilanlar
#: tekrarlanabilirligi bozar.
INFRASTRUCTURE_KEYS: dict[str, tuple[str, ...]] = {
    "lightgbm": ("verbose", "n_jobs", "importance_type"),
    "xgboost": ("tree_method", "enable_categorical", "n_jobs"),
    "catboost": ("verbose", "allow_writing_files"),
}

_DEFAULTS_BY_KIND: dict[str, dict[str, Any]] = {
    "lightgbm": LGB_DEFAULTS,
    "xgboost": XGB_DEFAULTS,
    "catboost": CAT_DEFAULTS,
}


def assert_finite_target(target: np.ndarray) -> None:
    """Hedefte NaN/sonsuz varsa hata firlatir.

    Hedefte NaN/inf SESSIZCE GECIYORDU ve bu en tehlikeli hata bicimidir:
    LightGBM bu satirlari egitimde yok sayar, skor MAKUL GORUNEN bir sayi
    cikar ve neyin yanlis oldugu anlasilmaz.

    OLCULDU: 400 satirin BIRINDE NaN olan bir hedefte skor 0.527 dondu --
    tamamen inandirici. inf durumunda 4.1e+35 dondu; absurt ama yine
    HATASIZ. Ikisi de sessizce yanlis CV skoru uretir.

    Siniflandirmada olasilik metrikleri icin de gecerli: NaN etiketle
    egitilen model, hicbir zaman ogrenmedigi bir sinifi tahmin eder.

    Raises:
        ValueError: Sayisal hedefte NaN veya sonsuz deger varsa.
    """
    y = np.asarray(target).ravel()
    if len(y) == 0 or not np.issubdtype(y.dtype, np.number):
        return
    if np.isfinite(y).all():
        return
    bozuk = int((~np.isfinite(y)).sum())
    raise ValueError(
        f"Hedefte {bozuk} adet NaN/sonsuz deger var (%{bozuk / len(y) * 100:.1f}). "
        "Model bunlari sessizce yok sayar ve skor MAKUL GORUNEN ama "
        "YANLIS bir sayi olur.\n"
        "Temizle: maske = np.isfinite(y); train=train[maske]; y=y[maske]\n"
        "SONRA fold'lari YENIDEN uret -- konumsal indeksler kaydi."
    )


def merge_infrastructure_params(kind: str, params: dict[str, Any]) -> dict[str, Any]:
    """Kullanici parametrelerine eksik ALTYAPI anahtarlarini ekler.

    Kullanicinin acikca verdigi hicbir deger EZILMEZ; yalnizca eksik olan
    altyapi anahtarlari varsayilandan tamamlanir.
    """
    tamamlanmis = dict(params)
    varsayilan = _DEFAULTS_BY_KIND.get(kind, {})
    for anahtar in INFRASTRUCTURE_KEYS.get(kind, ()):
        if anahtar not in tamamlanmis and anahtar in varsayilan:
            tamamlanmis[anahtar] = varsayilan[anahtar]
    return tamamlanmis


def starter_params(
    kind: ModelKind, task_type: str, *, objective: str | None = None
) -> dict[str, Any]:
    """Model ve gorev tipine gore baslangic parametreleri dondurur.

    CatBoost yuksek kardinaliteli kategorik kolonlarda (trafo_id, ilce)
    genellikle en iyi tek modeldir cunku sirali hedef kodlamayi kendi icinde,
    sizintisiz yapar. Once onu dene.

    Args:
        objective: Varsayilani ezer. Sayim (count) hedefi icin ``COUNT_OBJECTIVES``
            anahtarlarindan birini kullan: ``poisson``, ``tweedie``, ``mae``.

    OBJECTIVE SECIMI TAHMINLE DEGIL ARAMAYLA YAPILIR
    -------------------------------------------------
    Hedef bir SAYIM ise (gunluk kesinti adedi gibi) dagilim saga carpik ve
    sifir-siskindir. Bu durumda ``l2`` cogu zaman en iyi secim DEGILDIR:

      * ``poisson``  -- saf sayim, asiri yayilim yoksa
      * ``tweedie``  -- sifir kutlesi olan negatif olmayan veri (M5 kazanani
                        bunu kullandi); ``tweedie_variance_power`` 1.1-1.5 arasi aranir
      * ``mae``/``l1`` -- resmi metrik MAE ise dogrudan onu optimize et

    2024 GDZ Datathon birincisi (Pikachow) objective'i **Optuna arama
    uzayina koydu** (final sunumu s.23: TPESampler, parametre listesinin
    basinda ``objective``) -- yani hangisinin kazandigi veriye baglidir ve
    deneyle bulunur. 2023 birincisi ise parametreleri elle sabitlemisti;
    iki yil iki farkli yol, aramak guvenli olandir.
    """
    # Genel objective anahtarini kutuphaneye ozgu ada CEVIR.
    #
    # Docstring "COUNT_OBJECTIVES anahtarlarindan birini kullan: poisson,
    # tweedie, mae" diyor -- ama onceki surum degeri OLDUGU GIBI geciriyordu.
    # Sonuc: objective="mae" LightGBM'de calisiyor, XGBoost'ta
    # "Unknown objective function", CatBoost'ta "mae loss is not supported"
    # ile cokuyordu. fit_two_stage bu yolu sabit kodladigi icin iki
    # kutuphanede de tamamen kullanilamazdi (olculdu).
    objective = _resolve_objective(kind, objective)

    if kind == "lightgbm":
        params = dict(LGB_DEFAULTS)
        params["objective"] = (
            objective
            or {
                "regression": "regression",
                "binary": "binary",
                "multiclass": "multiclass",
            }[task_type]
        )
        return params

    if kind == "xgboost":
        params = dict(XGB_DEFAULTS)
        params["objective"] = (
            objective
            or {
                "regression": "reg:squarederror",
                "binary": "binary:logistic",
                "multiclass": "multi:softprob",
            }[task_type]
        )
        return params

    if kind == "catboost":
        params = dict(CAT_DEFAULTS)
        params["loss_function"] = (
            objective
            or {
                "regression": "RMSE",
                "binary": "Logloss",
                "multiclass": "MultiClass",
            }[task_type]
        )
        return params

    raise ValueError(f"Bilinmeyen model tipi '{kind}'.")


def _resolve_objective(kind: ModelKind, objective: str | None) -> str | None:
    """Genel objective anahtarini kutuphaneye ozgu ada cevirir.

    ``poisson``/``tweedie``/``mae``/``l2`` gibi genel anahtarlar
    ``COUNT_OBJECTIVES`` uzerinden cevrilir. Zaten kutuphaneye ozgu bir ad
    verilmisse (``reg:tweedie``, ``MAE``, ``count:poisson``) oldugu gibi
    birakilir -- kullanicinin acik tercihini ezmeyiz.
    """
    if objective is None:
        return None
    esleme = COUNT_OBJECTIVES.get(kind, {})
    return esleme.get(objective, objective)


# Sayim hedefi icin denenmesi gereken objective'ler, kutuphane bazinda.
# Hepsini ayni fold'lar uzerinde calistirip CV ile sec -- tahmin etme.
COUNT_OBJECTIVES: dict[str, dict[str, str]] = {
    "lightgbm": {"poisson": "poisson", "tweedie": "tweedie", "mae": "mae", "l2": "regression"},
    "xgboost": {
        "poisson": "count:poisson",
        "tweedie": "reg:tweedie",
        "mae": "reg:absoluteerror",
        "l2": "reg:squarederror",
    },
    "catboost": {
        "poisson": "Poisson",
        "tweedie": "Tweedie:variance_power=1.5",
        "mae": "MAE",
        "l2": "RMSE",
    },
}


def monotone_constraints_for(
    feature_columns: Sequence[str],
    increasing: Sequence[str] = (),
    decreasing: Sequence[str] = (),
) -> dict[str, Any]:
    """Uc kutuphane icin monotonik kisit parametresi uretir -- OTOMATIK UYGULAMAZ.

    NEREDEN GELDI (docs/10 bolum 3): kredi-PD benchmark'i (arXiv 2512.17945)
    monotonik kisitlarin performans maliyetini ~%0-2.9 olarak olctu --
    neredeyse bedava bir asiri-uyum sigortasi. Seyrek veride (az gozlemli
    ilce, uc hava degerleri) agac, fiziksel olarak tek yonlu olmasi gereken
    iliskiyi (ruzgar arttikca kesinti AZALMAZ) gurultuden ters cevirebilir;
    kisit bunu yapisal olarak engeller.

    BILINCLI OLARAK hicbir model fonksiyonu bunu kendiliginden uygulamaz:
    kisit bir alan-bilgisi iddiasidir ve yanlis yonlu bir kisit modele
    dogrudan zarar verir. Kullanan, yonu kendisi secer ve parametreye koyar::

        kisit = monotone_constraints_for(kolonlar, increasing=["ruzgar_max"])
        lgb_params["monotone_constraints"] = kisit["lightgbm"]
        xgb_params["monotone_constraints"] = kisit["xgboost"]
        cat_params["monotone_constraints"] = kisit["catboost"]

    Args:
        feature_columns: Modele girecek kolonlar, MODEL SIRASINDA. Sira
            yanlissa kisit yanlis kolona uygulanir ve hicbir hata firlamaz.
        increasing: Hedefle artan iliskiye zorlanacak kolonlar (+1).
        decreasing: Azalan iliskiye zorlanacak kolonlar (-1).

    Returns:
        ``{"lightgbm": [-1/0/1, ...], "xgboost": "(1,0,-1,...)",
        "catboost": [-1/0/1, ...]}`` -- LightGBM/CatBoost kolon sirasinda
        tamsayi listesi, XGBoost ayni bilginin parantezli metin hali.

    Raises:
        ValueError: Kisit listesinde ``feature_columns``ta olmayan ad varsa
            (yazim hatasi sessizce "kisit yok" olurdu) veya ayni kolon iki
            yonde birden istendiyse.
    """
    kolonlar = list(feature_columns)
    artan = set(increasing)
    azalan = set(decreasing)

    cakisan = sorted(artan & azalan)
    if cakisan:
        raise ValueError(f"Ayni kolon iki yonde birden kisitlanamaz: {cakisan}")
    bilinmeyen = sorted((artan | azalan) - set(kolonlar))
    if bilinmeyen:
        raise ValueError(
            f"Kisit listesindeki su kolonlar feature_columns icinde yok: {bilinmeyen}. "
            "Yazim hatasi sessizce 'kisit yok' olurdu -- bu yuzden hata."
        )

    vektor = [1 if kolon in artan else (-1 if kolon in azalan else 0) for kolon in kolonlar]
    return {
        "lightgbm": vektor,
        "xgboost": "(" + ",".join(str(deger) for deger in vektor) + ")",
        "catboost": list(vektor),
    }


#: ``cross_validate`` erken durdurma ile kosarken skora eklenen bilinen
#: yanlilik. Sayi tek basina dolasmasin diye ``CVResult.warnings``a yazilir.
#: Gerekce ve olcum tablosu ``cross_validate`` docstring'inde.
EARLY_STOPPING_BIAS_NOTE = (
    "Agac sayisi, skorun hesaplandigi AYNI valid fold'da secildi (erken "
    "durdurma) -- bu skor hafif IYIMSER. OLCULDU (N=3000, KFold(4), lr=0.05, "
    "esr=200): erken durdurmali CV 2.118068, ayni fold'larda sabit 87 agac "
    "2.121460 -- fark 0.003392 (%0.16). Tek basina kucuk; ama Optuna (%0.3) ve "
    "geri eleme (+0.0137) yanliliklariyla AYNI YONDE toplanir. Nihai karari "
    "ayrilmis bir holdout veya LB ile dogrula."
)


@dataclass
class CVResult:
    """Bir capraz dogrulama kosusunun tum ciktisi."""

    oof_predictions: np.ndarray
    test_predictions: np.ndarray | None
    fold_scores: list[float]
    overall_score: float
    feature_importance: pd.DataFrame
    models: list[Any] = field(default_factory=list, repr=False)
    elapsed_seconds: float = 0.0
    metric_name: str = ""
    model_kind: str = ""
    #: Hangi satirlarin GERCEKTEN bir fold'un valid tarafinda oldugu.
    #: Kapsanmayan satirlarda ``oof_predictions`` SIFIRDIR -- gercek bir
    #: tahmin degil, dolgu. Bu maskeyi kullanmadan tum diziyi harmanlamaya
    #: veya korelasyona sokmak sonucu bozar (olculdu: gercek korelasyon 0.93
    #: iken tum diziyle 0.47). ``covered_predictions()`` kisayolunu kullan.
    oof_covered: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    #: Bu skorun NASIL uretildigine dair, sayiyi okuyan herkesin bilmesi
    #: gereken cekinceler. Bos degilse ``summary()`` bunlari basar. Sessiz
    #: kalmamak icin var: skor tek basina dolasima girdiginde (deney gunlugu,
    #: juri slaydi) yaniyla birlikte tasinsin.
    warnings: list[str] = field(default_factory=list)
    #: Fold ve genel skorlarin hesaplandigi hedef uzayi. Donusum kullanilsa
    #: bile resmi metrik her zaman ham hedefte hesaplanir.
    score_space: str = "raw"
    #: Fit sirasinda kullanilan hedef donusumu (``None`` / ``log1p`` / ``sqrt``).
    target_transform: str | None = None
    #: Ust-seviye metrik adinin backend'e cevrilmis erken durdurma karsiligi.
    early_stopping_metric: str | None = None

    def covered_predictions(self) -> tuple[np.ndarray, np.ndarray]:
        """``(kapsanan_satir_indeksleri, o_satirlarin_tahminleri)``.

        Harmanlama ve korelasyon HER ZAMAN bu ciftle yapilmalidir. Purged
        TimeSeriesSplit'te ilk donem hicbir fold'un valid tarafinda olmaz;
        o satirlarin OOF degeri 0'dir ve gercek bir tahmin degildir.
        """
        if self.oof_covered.size == 0:
            # Eski kayitlar/elle kurulmus CVResult -- hepsini kapsanmis say.
            return np.arange(len(self.oof_predictions)), self.oof_predictions
        indeks = np.flatnonzero(self.oof_covered)
        return indeks, self.oof_predictions[indeks]

    @property
    def coverage(self) -> float:
        """OOF kapsam orani (0..1)."""
        if self.oof_covered.size == 0:
            return 1.0
        return float(self.oof_covered.mean())

    @property
    def fold_std(self) -> float:
        """Fold'lar arasi standart sapma.

        YORUM: ``fold_std`` genel skorun %10'undan buyukse CV gurultuludur.
        O durumda kucuk iyilesmelere guvenme -- ya fold sayisini artir ya da
        tekrarli CV (repeated CV) kullan. Aksi halde gurultuyu kovalarsin.
        """
        return float(np.std(self.fold_scores)) if self.fold_scores else 0.0

    @property
    def is_stable(self) -> bool:
        """CV guvenilir mi? Skorun %10'undan kucuk sapma stabil sayilir.

        ``overall_score`` sifira cok yakinsa oransal karsilastirma anlamsizdir
        (bazi metriklerde 0 MUKEMMEL skordur). O durumda mutlak sapmaya bakariz
        -- kosulsuz "gurultulu" demek yanlis yonde bir sinyaldir.
        """
        if not self.fold_scores:
            return False
        scale = abs(self.overall_score)
        if scale < 1e-12:
            return self.fold_std < 1e-6
        return self.fold_std / scale < 0.10

    def summary(self) -> str:
        stability = "STABIL" if self.is_stable else "GURULTULU -- dikkat"
        lines = [
            f"Model: {self.model_kind}   Metrik: {self.metric_name}",
            f"CV skoru (OOF, tum veri): {self.overall_score:.6f}",
            f"Fold ortalamasi: {np.mean(self.fold_scores):.6f} "
            f"+/- {self.fold_std:.6f}  [{stability}]",
            f"Fold skorlari: {[round(score, 5) for score in self.fold_scores]}",
            f"Sure: {self.elapsed_seconds:.1f} sn",
        ]
        for uyari in self.warnings:
            lines.append(f"UYARI: {uyari}")
        top = self.feature_importance.head(15)
        lines.append("\nEn onemli 15 feature:")
        for _, row in top.iterrows():
            lines.append(f"  {row['feature']:<45} {row['importance']:>12.1f}")
        return "\n".join(lines)


#: Test satirlarinin en az bu kadari train'de GORULMUS bir kategoriye dusmeli.
#: Altina inince sessiz gecmeyiz -- bkz. ``_warn_on_low_category_overlap``.
MIN_CATEGORY_OVERLAP = 0.50


def _warn_on_low_category_overlap(
    column: str, train_text: pd.Series, test_text: pd.Series
) -> float:
    """Test satirlarinin kacinin kategorisi train'de gorulmus -- olcer ve dondurur.

    NEDEN OLCUYORUZ: birlesik kategori kumesi kurdugumuz icin ortusme SIFIR
    olsa bile hicbir sey patlamaz, NaN bile olusmaz -- test tarafi sadece
    train'de hic ornegi olmayan kodlara duser ve model onlar icin ogrendigi
    hicbir seyi kullanamaz. Kusur CV'de GORUNMEZ cunku CV yalnizca train
    icindedir.

    OLCULDU (30 ilce, test tarafinda basta TEK BOSLUK -> ortak kategori 0):
      CV rmse (train ici)        :  3.0139
      gercek test RMSE           : 28.5162
      hep-ortalama baseline RMSE : 25.1324
    Yani model CV'de 8 kat iyi gorunurken gercekte baseline'in ALTINDA.
    Baska hicbir katman bunu yakalamiyor: profile() ve leakage_report()
    yalnizca kolon ADLARINI karsilastirir, check_train_test_overlap ise
    gun-1 zincirinde hic cagrilmaz.

    Returns:
        Satir agirlikli ortusme orani (0..1). Test bossa 1.0.
    """
    train_seen = set(train_text.dropna().unique())
    test_values = test_text.dropna()
    if len(test_values) == 0:
        return 1.0

    row_overlap = float(test_values.isin(train_seen).mean())
    if row_overlap >= MIN_CATEGORY_OVERLAP:
        return row_overlap

    test_seen = set(test_values.unique())
    ortak = len(train_seen & test_seen)
    print(
        f"[cross_validate] UYARI: '{column}' kolonunda train/test kategori "
        f"ortusmesi COK DUSUK -- test satirlarinin yalnizca %{row_overlap * 100:.1f}'i "
        f"train'de gorulmus bir degere dusuyor "
        f"(ortak kategori {ortak}; train {len(train_seen)}, test {len(test_seen)}).\n"
        "  Model bu satirlar icin ogrendigi hicbir seyi kullanamaz ve CV skoru "
        "bunu GOSTERMEZ (CV yalnizca train icindedir).\n"
        "  Once nedenini bul: bosluk/buyuk-kucuk harf farki, farkli kod semasi "
        "veya gercekten yeni kategoriler. Sonra ya normalize et ya da bu kolonu "
        "frekans/hedef kodlamasiyla degistir."
    )
    return row_overlap


def _prepare_categoricals(
    train: pd.DataFrame, test: pd.DataFrame | None, kind: ModelKind
) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
    """Kategorik kolonlari modele uygun tipe cevirir.

    Train ve test AYNI kategori kumesini paylasmak zorundadir; aksi halde
    LightGBM/XGBoost farkli kodlamalar uretir ve tahminler sessizce bozulur.

    Ortak kategori kumesi kurmak dtype'lari hizalar ama ORTUSMEYI garanti
    ETMEZ: sifir ortusme sessizce gecerdi (olculdu: 0 uyari). Artik her
    kategorik kolonun ortusmesi olculur ve
    ``MIN_CATEGORY_OVERLAP``in altinda kalirsa kullaniciya SOYLENIR.
    """
    # Surumden bagimsiz: pandas 3.0'da metin 'str' dtype'indadir ve
    # is_object_dtype onu gormez -- bkz. compat.is_categorical_like.
    categorical = categorical_columns(train)
    if not categorical:
        return train, test, []

    train_out, test_out = train.copy(), (test.copy() if test is not None else None)

    # safe_str kullaniyoruz, .astype(str) DEGIL: pandas 2.x'te .astype(str)
    # NaN'i literal "None" stringine cevirir ve GBDT'lerin yerli eksik-deger
    # islemesini sessizce devre disi birakir. Bu makinede (pandas 3.0) NaN
    # korunur -- yani hata yalnizca Kaggle'in eski imajinda ortaya cikar.
    for column in categorical:
        if test_out is not None and column in test_out.columns:
            train_text = safe_str(train_out[column])
            test_text = safe_str(test_out[column])
            # Kodlamadan ONCE olc: birlesik kategori kumesi kurulduktan sonra
            # ortusme kaybi NaN birakmaz, yani geriye izi kalmaz.
            _warn_on_low_category_overlap(column, train_text, test_text)
            combined = pd.concat([train_text, test_text], ignore_index=True)
            categories = pd.Index(combined.dropna().unique())
            dtype = pd.CategoricalDtype(categories=categories)
            train_out[column] = train_text.astype(dtype)
            test_out[column] = test_text.astype(dtype)
        else:
            train_out[column] = safe_str(train_out[column]).astype("category")

    if kind == "catboost":
        # CatBoost kategorikleri string olarak ister ve NaN KABUL ETMEZ.
        # fillna, astype(str)'den ONCE uygulanmali -- safe_str bunu garanti eder.
        for column in categorical:
            train_out[column] = safe_str(train_out[column], missing=MISSING_CATEGORY)
            if test_out is not None and column in test_out.columns:
                test_out[column] = safe_str(test_out[column], missing=MISSING_CATEGORY)

    return train_out, test_out, categorical


#: LightGBM'de LOG-LINK kullanan regresyon objective'leri: ham skor log(mu),
#: tahmin exp(ham skor). Offset (init_score) bu uzayda eklenir; tahmine geri
#: eklerken exp uygulanmali. Bkz. ``_predict_with_offset``.
LGB_LOG_LINK_OBJECTIVES = frozenset({"poisson", "gamma", "tweedie"})

#: LightGBM'de KIMLIK-LINK regresyon objective'leri: tahmin = ham skor.
_LGB_IDENTITY_LINK_OBJECTIVES = frozenset(
    {
        "",
        "regression",
        "regression_l1",
        "regression_l2",
        "l1",
        "l2",
        "mae",
        "mse",
        "rmse",
        "mean_absolute_error",
        "mean_squared_error",
        "huber",
        "fair",
        "quantile",
        "mape",
    }
)


def _offset_link(kind: ModelKind, params: dict[str, Any]) -> str:
    """Offset'in tahmine nasil geri eklenecegini belirler.

    Bilinmeyen objective'de TAHMIN ETMEYIZ: yanlis link, tum tahminleri
    sessizce yanlis olcege tasir (exp'lenmemis poisson tahmini log uzayinda
    kalir ve MAE makul gorunen ama anlamsiz bir sayi olur).
    """
    objective = str(params.get("objective", ""))
    if kind == "xgboost":
        if objective.startswith(("binary", "multi", "rank")):
            raise NotImplementedError(
                f"init_score simdilik yalnizca regresyon objective'leri icin: '{objective}'."
            )
        # XGBoost base_margin'i predict'te KENDISI ekler ve link'i uygular.
        return "internal"
    if objective in LGB_LOG_LINK_OBJECTIVES:
        return "log"
    if objective in _LGB_IDENTITY_LINK_OBJECTIVES:
        return "identity"
    raise NotImplementedError(
        f"init_score icin '{objective}' objective'inin link'i bilinmiyor. "
        "Log-link (poisson/gamma/tweedie) veya kimlik-link regresyon kullan -- "
        "aksi halde offset tahmine YANLIS uzayda eklenir ve hata firlamazdi."
    )


def _validate_init_score(
    values: np.ndarray | Sequence[float], n_rows: int, name: str
) -> np.ndarray:
    """Offset dizisini fit'ten ONCE dogrular (boy + NaN/inf).

    ``sample_weight``in aksine NEGATIF deger MESRUDUR: log(maruziyet) 1'in
    altindaki maruziyette negatiftir. NaN/inf ise modeli sessizce bozar.
    """
    dizi = np.asarray(values, dtype="float64").ravel()
    if len(dizi) != n_rows:
        raise ValueError(f"{name} ({len(dizi)}) ve veri ({n_rows}) uzunluklari farkli.")
    if not np.isfinite(dizi).all():
        bozuk = int((~np.isfinite(dizi)).sum())
        raise ValueError(
            f"{name} icinde {bozuk} adet NaN/sonsuz deger var. log(maruziyet) "
            "kullaniyorsan sifir maruziyetli satirlari once ele al "
            "(np.log(np.maximum(maruziyet, 1)) gibi bilincli bir kuralla)."
        )
    return dizi


def _resolve_offsets(
    kind: ModelKind,
    model_params: dict[str, Any],
    init_score: np.ndarray | Sequence[float] | None,
    test_init_score: np.ndarray | Sequence[float] | None,
    n_train: int,
    test: pd.DataFrame | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """``init_score``/``test_init_score`` ciftini dogrular ve dizilere cevirir.

    Offset ya HER dilime (train + valid + test) birden uygulanir ya hic:
    tek tarafli offset, tahmin olcegini sessizce kaydirir.
    """
    if init_score is None:
        if test_init_score is not None:
            raise ValueError(
                "test_init_score verildi ama init_score yok -- offset ya train ve "
                "test'e birden uygulanir ya hic."
            )
        return None, None
    if kind == "catboost":
        raise NotImplementedError(
            "init_score CatBoost'a baglanmadi: baseline yalnizca Pool(baseline=...) "
            "API'siyle tasinabiliyor ve Poisson'da prediction_type='Exponent' "
            "carpitmasi gerekiyor -- sessizce yanlis baglamak yerine acikca "
            "reddediyoruz. Offset icin lightgbm veya xgboost kullan."
        )
    _offset_link(kind, model_params)  # bilinmeyen objective linkinde ERKEN hata
    offsets = _validate_init_score(init_score, n_train, "init_score")
    if test is None:
        if test_init_score is not None:
            raise ValueError("test_init_score verildi ama test frame'i yok.")
        return offsets, None
    if test_init_score is None:
        raise ValueError(
            "test verildi ama test_init_score yok: offset'siz test tahmini olcegi "
            "sessizce kaydirir. Test satirlari icin AYNI tanimli offset'i ver."
        )
    return offsets, _validate_init_score(test_init_score, len(test), "test_init_score")


def _predict_with_offset(
    kind: ModelKind,
    model: Any,
    features: pd.DataFrame,
    offset: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    """Offset'li (init_score/base_margin) tahmin -- kutuphane farkini kapatir.

    XGBoost ``base_margin``i predict'te kendisi ekler ve link'i uygular.
    LightGBM ise init_score'u tahmin aninda BILMEZ: ham skoru alip offset'i
    biz ekleriz ve objective'in link'ini biz uygulariz
    (poisson/gamma/tweedie -> exp, kimlik-link regresyon -> oldugu gibi).
    """
    if kind == "xgboost":
        return np.asarray(model.predict(features, base_margin=offset)).ravel()
    raw = np.asarray(model.predict(features, raw_score=True)).ravel()
    total = raw + np.asarray(offset, dtype="float64").ravel()
    if _offset_link(kind, params) == "log":
        return np.exp(total)
    return total


def _fit_one_fold(
    kind: ModelKind,
    params: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    categorical: list[str],
    early_stopping_rounds: int,
    sample_weight: np.ndarray | None = None,
    init_train: np.ndarray | None = None,
    init_valid: np.ndarray | None = None,
) -> Any:
    """Tek bir fold egitir ve modeli dondurur.

    ``sample_weight`` yalnizca TRAIN dilimini olcekler; eval_set agirliksiz
    kalir -- erken durdurma ve fold skoru boylece agirlikli/agirliksiz
    kosular arasinda karsilastirilabilir olur (bilincli tercih).

    ``init_train``/``init_valid`` (offset) ise IKI dilime birden gecirilir:
    erken durdurmanin izledigi valid kaybi da offset'li uzayda olculmeli,
    yoksa agac sayisi yanlis modele gore secilir.
    """
    if kind == "lightgbm":
        import lightgbm as lgb

        is_classification = params.get("objective", "").startswith(("binary", "multiclass"))
        model_class = lgb.LGBMClassifier if is_classification else lgb.LGBMRegressor
        # ``eval_metric`` fit-parametresidir; constructor'a da gecirilirse
        # LightGBM onu bilinmeyen booster parametresi olarak yorumlayabilir.
        fit_metric = params.get("eval_metric")
        constructor_params = {key: value for key, value in params.items() if key != "eval_metric"}
        model = model_class(**constructor_params)
        model.fit(
            x_train,
            y_train,
            sample_weight=sample_weight,
            init_score=init_train,
            eval_set=[(x_valid, y_valid)],
            eval_init_score=[init_valid] if init_valid is not None else None,
            eval_metric=fit_metric,
            categorical_feature=categorical or "auto",
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        return model

    if kind == "xgboost":
        import xgboost as xgb

        is_classification = params.get("objective", "").startswith(("binary", "multi"))
        model_class = xgb.XGBClassifier if is_classification else xgb.XGBRegressor
        model = model_class(**params, early_stopping_rounds=early_stopping_rounds)
        model.fit(
            x_train,
            y_train,
            sample_weight=sample_weight,
            base_margin=init_train,
            eval_set=[(x_valid, y_valid)],
            base_margin_eval_set=[init_valid] if init_valid is not None else None,
            verbose=False,
        )
        return model

    if kind == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        if init_train is not None:  # savunma; asil kapi _resolve_offsets'te
            raise NotImplementedError("init_score CatBoost'a baglanmadi -- bkz. cross_validate.")
        is_classification = params.get("loss_function") in {"Logloss", "MultiClass"}
        model_class = CatBoostClassifier if is_classification else CatBoostRegressor
        model = model_class(**params)
        model.fit(
            x_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=(x_valid, y_valid),
            cat_features=categorical or None,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )
        return model

    raise ValueError(f"Bilinmeyen model tipi '{kind}'.")


def fit_without_validation(
    kind: ModelKind,
    params: dict[str, Any],
    features: pd.DataFrame,
    target: np.ndarray,
    categorical: list[str],
) -> Any:
    """Tum veriyle, DOGRULAMA KUMESI OLMADAN egitir.

    ``_fit_one_fold`` burada KULLANILAMAZ: o, erken durdurma icin bir ``eval_set``
    bekler. Ayrilmis dogrulama kumesi yokken egitim kumesini ``eval_set`` olarak
    vermek, modelin KENDI EGITIM KAYBINA gore durmasina yol acar -- yani asiri
    uyum noktasinda durur, tam tersi.

    Bu yuzden tur sayisi disaridan gelir. Kullanan yerler: ``refit.multi_seed_refit``
    (tur sayisi ``estimate_full_data_rounds`` ile CV'den devralinir) ve
    ``selection.null_importance_filter`` (sabit, kucuk bir tur sayisi yeter).
    """
    if kind == "lightgbm":
        import lightgbm as lgb

        is_classification = str(params.get("objective", "")).startswith(("binary", "multiclass"))
        model_class = lgb.LGBMClassifier if is_classification else lgb.LGBMRegressor
        model = model_class(**params)
        model.fit(features, target, categorical_feature=categorical or "auto")
        return model

    if kind == "xgboost":
        import xgboost as xgb

        is_classification = str(params.get("objective", "")).startswith(("binary", "multi"))
        model_class = xgb.XGBClassifier if is_classification else xgb.XGBRegressor
        model = model_class(**params)
        model.fit(features, target, verbose=False)
        return model

    if kind == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        is_classification = params.get("loss_function") in {"Logloss", "MultiClass"}
        model_class = CatBoostClassifier if is_classification else CatBoostRegressor
        model = model_class(**params)
        model.fit(features, target, cat_features=categorical or None, verbose=False)
        return model

    raise ValueError(f"Bilinmeyen model tipi '{kind}'.")


def _predict(model: Any, features: pd.DataFrame, *, needs_proba: bool) -> np.ndarray:
    """Metrigin ihtiyacina gore olasilik veya deger tahmini dondurur."""
    if needs_proba and hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba
    return np.asarray(model.predict(features)).ravel()


def _lightgbm_gain_importance(model: Any, feature_names: Sequence[str]) -> np.ndarray | None:
    """LightGBM modelinden 'gain' onemini cikarir; LightGBM degilse ``None``.

    NEDEN MODELIN KENDI ``importance_type``INE GUVENMIYORUZ: ``LGB_DEFAULTS``
    'gain' diyor ama ``feature_importances_`` yalnizca modelin KURULUS aninda
    verilen ``importance_type``i uygular. ``params`` disaridan gelen her yol
    (Optuna'nin best_params'i, ``selection.null_importance_filter``) o anahtari
    tasimaz ve model sessizce LightGBM varsayilani 'split'e duser.

    OLCULDU (3 gercek sinyal + 40 saf gurultu, 1000 agac, N=4000; params ile
    kurulmus LGBMRegressor):
      model.importance_type      = 'split'
      split gercek/gurultu orani =  1.31x
      gain  gercek/gurultu orani = 12.89x
    Yani ayni modelde iki farkli tablo -- ve 'split' gercek sinyali gurultuden
    neredeyse ayirt edemiyor.
    """
    booster = getattr(model, "booster_", None) or model
    getir = getattr(booster, "feature_importance", None)
    if not callable(getir):
        return None
    try:
        importance = np.asarray(getir(importance_type="gain"), dtype="float64").ravel()
    except (TypeError, ValueError):  # pragma: no cover - LightGBM disi bir 'feature_importance'
        return None
    return importance if importance.size == len(feature_names) else None


def _extract_importance(model: Any, feature_names: Sequence[str]) -> np.ndarray:
    """Model tipinden bagimsiz feature onem vektoru.

    LightGBM'de olcu HER ZAMAN 'gain'dir -- modelin ``importance_type``i ne
    olursa olsun (bkz. ``_lightgbm_gain_importance``). XGBoost ve CatBoost
    zaten gain tabanli bir olcu doner.

    DIKKAT -- OLCEK KUTUPHANELER ARASI KARSILASTIRILAMAZ. OLCULDU (10 feature,
    300 agac, N=4000) bu fonksiyonun donen vektorunun toplami:
      LightGBM (ham gain)          : 23228.6
      XGBoost  (normalize edilmis) :     1.0
    ``zoo`` icindeki farkli modellerin onem tablolarini yan yana koyup MUTLAK
    degerle karsilastirma; her tabloyu kendi icinde sirala.

    Cikaramadigi durumda sifir vektoru doner ama SESSIZ KALMAZ: sifirlarla dolu
    bir onem tablosu, "hicbir feature ise yaramiyor" diye yanlis okunur ve
    gereksiz feature elemesine yol acar. Sorun feature'larda degil, cikarim
    mekanizmasindadir -- bunu soylemek zorundayiz.
    """
    gain = _lightgbm_gain_importance(model, feature_names)
    if gain is not None:
        return gain

    for attribute in ("feature_importances_", "get_feature_importance"):
        value = getattr(model, attribute, None)
        if value is None:
            continue
        importance = value() if callable(value) else value
        importance = np.asarray(importance, dtype="float64").ravel()
        if importance.size == len(feature_names):
            return importance

    print(
        f"[cross_validate] UYARI: {type(model).__name__} icin feature onemi "
        "cikarilamadi (beklenen oznitelik yok veya boyut uyusmuyor). "
        "Onem tablosu SIFIRLARLA dolu -- bunu 'feature'lar ise yaramiyor' "
        "diye okuma."
    )
    return np.zeros(len(feature_names))


def _validate_sample_weight(
    sample_weight: np.ndarray | Sequence[float] | None, n_rows: int
) -> np.ndarray | None:
    """Ornek agirliklarini fit'ten ONCE dogrular.

    LightGBM boy uyusmazliginda anlasilmaz bir hata verir; negatif/NaN
    agirlik ise hicbir hata vermeden sacma bir modele yol acar. Ikisini de
    egitim baslamadan yakaliyoruz.
    """
    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight, dtype="float64").ravel()
    if len(weights) != n_rows:
        raise ValueError(f"sample_weight ({len(weights)}) ve train ({n_rows}) uzunluklari farkli.")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError(
            "sample_weight negatif/NaN/sonsuz deger iceriyor -- agirliklar sonlu ve >= 0 olmali."
        )
    return weights


def _prepare_cv_target(
    target: np.ndarray | pd.Series,
    n_rows: int,
    target_transform: TargetTransform,
    task_type: str,
    init_score: np.ndarray | Sequence[float] | None,
    test_init_score: np.ndarray | Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray], str | None]:
    """Ham hedef sozlesmesini ve fit uzayini CV dongusunden once dogrular."""
    raw_y = np.asarray(target).ravel()
    if len(raw_y) != n_rows:
        raise ValueError(f"train ({n_rows}) ve target ({len(raw_y)}) uzunluklari farkli.")
    assert_finite_target(raw_y)
    if target_transform is not None and (init_score is not None or test_init_score is not None):
        raise ValueError(
            "target_transform ile init_score/test_init_score birlikte kullanilamaz: "
            "offset'in fit mi ham hedef uzayinda oldugu belirsiz."
        )
    fit_y, inverse_target, transform_name = _resolve_target_transform(
        raw_y, target_transform, task_type
    )
    return raw_y, fit_y, inverse_target, transform_name


def cross_validate(
    train: pd.DataFrame,
    target: np.ndarray | pd.Series,
    folds: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    kind: ModelKind = "lightgbm",
    task_type: str = "regression",
    metric: str = "rmse",
    params: dict[str, Any] | None = None,
    test: pd.DataFrame | None = None,
    sample_weight: np.ndarray | Sequence[float] | None = None,
    init_score: np.ndarray | Sequence[float] | None = None,
    test_init_score: np.ndarray | Sequence[float] | None = None,
    target_transform: TargetTransform = None,
    early_stopping_metric: str | None = None,
    early_stopping_rounds: int = 200,
    verbose: bool = True,
) -> CVResult:
    """Capraz dogrulamali egitim. OOF ve test tahminlerini dondurur.

    Args:
        train: Feature'lar (hedef kolonu ICERMEZ).
        target: Hedef degerler.
        folds: ``(train_idx, valid_idx)`` ciftleri -- ``validation`` modulunden.
        kind: ``lightgbm`` | ``xgboost`` | ``catboost``.
        task_type: ``regression`` | ``binary`` | ``multiclass``.
        metric: ``metrics.METRIC_REGISTRY`` icindeki bir ad.
        params: Model parametreleri. ``None`` ise ``starter_params``.
        test: Verilirse fold ortalamasiyla test tahmini de uretilir.
        sample_weight: Satir basina egitim agirligi (``len(train)`` boyutlu;
            or. ``weighting.recency_activity_weights`` ciktisi). Her fold'da
            yalnizca TRAIN dilimi (``weight[train_idx]``) fit'e gecirilir;
            valid/eval tarafi agirliksiz kalir ki fold skorlari agirlikli ve
            agirliksiz kosular arasinda karsilastirilabilir olsun.
        init_score: Satir basina SABIT taban skor (exposure-offset, sigorta
            pratigi -- docs/10 bolum 3). Log-link'li sayim objective'lerinde
            (``poisson``/``tweedie``/``gamma``) ``np.log(maruziyet)`` dogal
            offset'tir: ilce nufusu gibi boyut farkini OGRENILECEK seyden
            VERILI oncule cevirir. LightGBM'de ``init_score``, XGBoost'ta
            ``base_margin`` olarak train VE valid dilimlerine TUTARLI
            gecirilir; tahminde offset geri eklenir (LightGBM: ham skor +
            offset, log-link'te exp; XGBoost bunu predict'te kendisi yapar).
            CatBoost icin ``NotImplementedError``: baseline yalnizca
            ``Pool(baseline=...)`` API'siyle ve Poisson'da
            ``prediction_type='Exponent'`` carpitmasiyla tasinabiliyor --
            sessizce yanlis baglamak yerine acikca reddediyoruz.
        test_init_score: ``test`` ve ``init_score`` birlikte verildiyse
            ZORUNLU (``len(test)`` boyutlu) -- test tahmini offset'siz
            uretilirse olcek sessizce kayar.
        target_transform: Ham hedefi fit icinde donusturur (``log1p`` veya
            ``sqrt`` ya da ``(forward, inverse)`` callable cifti). Model
            tahminleri OOF/test'e yazilmadan ve resmi metrik hesaplanmadan
            once inverse ile HAM hedef uzayina dondurulur.

            SART -- ``forward`` DURUMSUZ (stateless) ve ELEMAN-BAZLI olmalidir.
            Donusum fold dongusunun DISINDA, TUM train hedefine bir kez
            uygulanir. ``log1p``/``sqrt`` bu sarti saglar: her deger yalnizca
            kendisine bakilarak donusur. Ancak hedeften GLOBAL ISTATISTIK
            ogrenen bir cift -- ornegin
            ``lambda y: (y - y.mean()) / y.std()`` -- validation fold'unun
            hedef dagilimini train hedefine tasir ve CV'yi sessizce iyimser
            yapar. Boyle bir normalizasyon gerekiyorsa fold ICINDE, yalnizca
            train diliminden ogrenilerek yapilmalidir.
        early_stopping_metric: Genel metrik adi; backend'in ``eval_metric``
            degerine acikca cevrilir. ``params['eval_metric']`` farkli bir
            deger iceriyorsa sessiz oncelik yerine ``ValueError`` verir.
        early_stopping_rounds: Iyilesme olmadan kac tur beklenecegi.

    Returns:
        ``CVResult``. ``warnings`` alani skorun bilinen yanliliklarini tasir.

    Raises:
        ValueError: Fold listesi bossa veya boyutlar uyumsuzsa.

    ERKEN DURDURMA SKORU HAFIF IYIMSER YAPAR -- OLCULEN BUYUKLUK
    -----------------------------------------------------------
    Agac sayisi ``eval_set=[(x_valid, y_valid)]`` ile SECILIR, sonra ayni
    ``x_valid`` uzerinde OOF uretilip ayni fold'dan skor alinir. Yani her
    fold'un agac sayisi, skorlandigi veriye bakarak secilmis olur.

    OLCULDU (N=3000, 12 feature, KFold(4), lr=0.05, n_estimators=3000,
    early_stopping_rounds=200; referans = AYNI fold'larda sabit agac
    sayisiyla, eval_set olmadan egitilmis model):

        erken durdurmali CV  : 2.118068   (fold agaclari 86/88/70/94)
        sabit  87 agac       : 2.121460   -> fark 0.003392  (%0.16)
        sabit 100 agac       : 2.122541   -> fark 0.004472  (%0.21)
        sabit 200 agac       : 2.153995   -> fark 0.035927  (%1.67)

    Yanlilik %0.2 mertebesinde: tek basina fold sayisini veya semayi
    degistirmeyi HAK ETMEZ, bu yuzden mimari oldugu gibi birakildi.
    Ama ayni yonde calisan iki yanlilik daha var (Optuna'nin en iyi
    denemeyi secmesi ~%0.3, geri elemenin +0.0137'si) ve bunlar TOPLANIR.
    Bu yuzden sayi tek basina dolasmasin diye ``CVResult.warnings``a yazilir.
    ``early_stopping_rounds``u buyuterek "erken durdurmasiz referans" ALAMAZSIN:
    callback takiliyken model yine ``best_iteration``a kirpilir (olculdu:
    esr=20/200/10^6 ucunde de CV 2.091143, fold agaclari birebir ayni).
    Referans istiyorsan ``fit_without_validation`` ile sabit tur sayisi kullan.
    """
    fold_list = list(folds)
    # Hedefte NaN/inf SESSIZCE GECIYORDU ve bu en tehlikeli hata bicimidir:
    # LightGBM bu satirlari egitimde yok sayar, skor MAKUL GORUNEN bir sayi
    # cikar ve neyin yanlis oldugu anlasilmaz.
    #
    # OLCULDU: 400 satirin BIRINDE NaN olan bir hedefte skor 0.527 dondu --
    # tamamen inandirici. inf durumunda 4.1e+35 dondu; absurt ama yine
    # HATASIZ. Ikisi de sessizce yanlis CV skoru uretir.
    #
    # Siniflandirmada olasilik metrikleri icin de gecerli: NaN etiketle
    # egitilen model, hicbir zaman ogrenmedigi bir sinifi tahmin eder.
    raw_y, fit_y, inverse_target, target_transform_name = _prepare_cv_target(
        target,
        len(train),
        target_transform,
        task_type,
        init_score,
        test_init_score,
    )

    # Fold'lar bu frame icin mi uretildi? Kontrol etmezsek yanlis satirlar
    # sessizce train/valid olarak eslesir -- bkz. assert_folds_align.
    assert_folds_align(len(train), fold_list)

    weights = _validate_sample_weight(sample_weight, len(train))

    metric_fn, _, needs_proba = get_metric(metric)
    model_params = (
        merge_infrastructure_params(kind, params) if params else starter_params(kind, task_type)
    )
    model_params, backend_early_metric = _resolve_early_stopping_metric(
        kind,
        model_params,
        early_stopping_metric,
        target_transform=target_transform_name,
        early_stopping_rounds=early_stopping_rounds,
    )

    offsets, test_offsets = _resolve_offsets(
        kind, model_params, init_score, test_init_score, len(train), test
    )

    train_ready, test_ready, categorical = _prepare_categoricals(train, test, kind)
    feature_names = list(train_ready.columns)

    oof = np.zeros(len(train_ready), dtype="float64")
    oof_filled = np.zeros(len(train_ready), dtype=bool)
    test_predictions = np.zeros(len(test_ready)) if test_ready is not None else None
    importance_total = np.zeros(len(feature_names))
    fold_scores: list[float] = []
    models: list[Any] = []

    started = time.perf_counter()

    for fold_index, (train_idx, valid_idx) in enumerate(fold_list, start=1):
        x_train = train_ready.iloc[train_idx]
        x_valid = train_ready.iloc[valid_idx]

        model = _fit_one_fold(
            kind,
            model_params,
            x_train,
            fit_y[train_idx],
            x_valid,
            fit_y[valid_idx],
            categorical,
            early_stopping_rounds,
            sample_weight=(weights[train_idx] if weights is not None else None),
            init_train=(offsets[train_idx] if offsets is not None else None),
            init_valid=(offsets[valid_idx] if offsets is not None else None),
        )

        fold_prediction_model_space = (
            _predict(model, x_valid, needs_proba=needs_proba)
            if offsets is None
            else _predict_with_offset(kind, model, x_valid, offsets[valid_idx], model_params)
        )
        fold_prediction = _inverse_predictions(fold_prediction_model_space, inverse_target)
        oof[valid_idx] = fold_prediction
        oof_filled[valid_idx] = True

        score = float(metric_fn(raw_y[valid_idx], fold_prediction))
        fold_scores.append(score)
        models.append(model)
        importance_total += _extract_importance(model, feature_names)

        if test_ready is not None and test_predictions is not None:
            fold_test_model_space = (
                _predict(model, test_ready, needs_proba=needs_proba)
                if test_offsets is None
                else _predict_with_offset(kind, model, test_ready, test_offsets, model_params)
            )
            fold_test = _inverse_predictions(fold_test_model_space, inverse_target)
            test_predictions += fold_test / len(fold_list)

        if verbose:
            print(f"  fold {fold_index}/{len(fold_list)}  {metric}={score:.6f}")

    elapsed = time.perf_counter() - started

    # TimeSeriesSplit ilk fold'u hic valid yapmaz -- o satirlar OOF'ta bos kalir.
    # Genel skoru YALNIZCA doldurulmus satirlar uzerinden hesapla, aksi halde
    # sifirlar skoru sessizce bozar.
    coverage = float(oof_filled.mean())
    overall = (
        float(metric_fn(raw_y[oof_filled], oof[oof_filled])) if oof_filled.any() else float("nan")
    )

    if verbose and coverage < 0.999:
        print(
            f"  NOT: OOF kapsami %{coverage * 100:.1f} "
            "(TimeSeriesSplit ilk donemi hic dogrulamaz -- bu beklenen davranistir)."
        )

    importance = (
        pd.DataFrame({"feature": feature_names, "importance": importance_total / len(fold_list)})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    uyarilar = []
    if early_stopping_rounds > 0 and fold_scores:
        uyarilar.append(EARLY_STOPPING_BIAS_NOTE)
        if verbose:
            print(f"  NOT: {EARLY_STOPPING_BIAS_NOTE}")

    return CVResult(
        oof_predictions=oof,
        oof_covered=oof_filled,
        warnings=uyarilar,
        test_predictions=test_predictions,
        fold_scores=fold_scores,
        overall_score=overall,
        feature_importance=importance,
        models=models,
        elapsed_seconds=elapsed,
        metric_name=metric,
        model_kind=kind,
        score_space="raw",
        target_transform=target_transform_name,
        early_stopping_metric=backend_early_metric,
    )
