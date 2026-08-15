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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from .compat import MISSING_CATEGORY, categorical_columns, safe_str
from .metrics import get_metric
from .validation import assert_folds_align

__all__ = [
    "CVResult",
    "cross_validate",
    "fit_without_validation",
    "LGB_DEFAULTS",
    "XGB_DEFAULTS",
    "CAT_DEFAULTS",
    "starter_params",
]

ModelKind = Literal["lightgbm", "xgboost", "catboost"]

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

    2024 GDZ birincisi objective'i **Optuna arama uzayina koydu** -- yani
    hangisinin kazandigi veriye baglidir ve deneyle bulunur.
    """
    if kind == "lightgbm":
        params = dict(LGB_DEFAULTS)
        params["objective"] = objective or {
            "regression": "regression",
            "binary": "binary",
            "multiclass": "multiclass",
        }[task_type]
        return params

    if kind == "xgboost":
        params = dict(XGB_DEFAULTS)
        params["objective"] = objective or {
            "regression": "reg:squarederror",
            "binary": "binary:logistic",
            "multiclass": "multi:softprob",
        }[task_type]
        return params

    if kind == "catboost":
        params = dict(CAT_DEFAULTS)
        params["loss_function"] = objective or {
            "regression": "RMSE",
            "binary": "Logloss",
            "multiclass": "MultiClass",
        }[task_type]
        return params

    raise ValueError(f"Bilinmeyen model tipi '{kind}'.")


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
    "catboost": {"poisson": "Poisson", "tweedie": "Tweedie:variance_power=1.5",
                 "mae": "MAE", "l2": "RMSE"},
}


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
        top = self.feature_importance.head(15)
        lines.append("\nEn onemli 15 feature:")
        for _, row in top.iterrows():
            lines.append(f"  {row['feature']:<45} {row['importance']:>12.1f}")
        return "\n".join(lines)


def _prepare_categoricals(
    train: pd.DataFrame, test: pd.DataFrame | None, kind: ModelKind
) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
    """Kategorik kolonlari modele uygun tipe cevirir.

    Train ve test AYNI kategori kumesini paylasmak zorundadir; aksi halde
    LightGBM/XGBoost farkli kodlamalar uretir ve tahminler sessizce bozulur.
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


def _fit_one_fold(
    kind: ModelKind,
    params: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    categorical: list[str],
    early_stopping_rounds: int,
) -> Any:
    """Tek bir fold egitir ve modeli dondurur."""
    if kind == "lightgbm":
        import lightgbm as lgb

        is_classification = params.get("objective", "").startswith(("binary", "multiclass"))
        model_class = lgb.LGBMClassifier if is_classification else lgb.LGBMRegressor
        model = model_class(**params)
        model.fit(
            x_train, y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric=params.get("eval_metric"),
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
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
        return model

    if kind == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        is_classification = params.get("loss_function") in {"Logloss", "MultiClass"}
        model_class = CatBoostClassifier if is_classification else CatBoostRegressor
        model = model_class(**params)
        model.fit(
            x_train, y_train,
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

        is_classification = str(params.get("objective", "")).startswith(
            ("binary", "multiclass")
        )
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


def _extract_importance(model: Any, feature_names: Sequence[str]) -> np.ndarray:
    """Model tipinden bagimsiz feature onem vektoru.

    Cikaramadigi durumda sifir vektoru doner ama SESSIZ KALMAZ: sifirlarla dolu
    bir onem tablosu, "hicbir feature ise yaramiyor" diye yanlis okunur ve
    gereksiz feature elemesine yol acar. Sorun feature'larda degil, cikarim
    mekanizmasindadir -- bunu soylemek zorundayiz.
    """
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
        early_stopping_rounds: Iyilesme olmadan kac tur beklenecegi.

    Returns:
        ``CVResult``.

    Raises:
        ValueError: Fold listesi bossa veya boyutlar uyumsuzsa.
    """
    fold_list = list(folds)
    y = np.asarray(target).ravel()
    if len(y) != len(train):
        raise ValueError(f"train ({len(train)}) ve target ({len(y)}) uzunluklari farkli.")

    # Fold'lar bu frame icin mi uretildi? Kontrol etmezsek yanlis satirlar
    # sessizce train/valid olarak eslesir -- bkz. assert_folds_align.
    assert_folds_align(len(train), fold_list)

    metric_fn, _, needs_proba = get_metric(metric)
    model_params = dict(params) if params else starter_params(kind, task_type)

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
            kind, model_params, x_train, y[train_idx], x_valid, y[valid_idx],
            categorical, early_stopping_rounds,
        )

        fold_prediction = _predict(model, x_valid, needs_proba=needs_proba)
        oof[valid_idx] = fold_prediction
        oof_filled[valid_idx] = True

        score = float(metric_fn(y[valid_idx], fold_prediction))
        fold_scores.append(score)
        models.append(model)
        importance_total += _extract_importance(model, feature_names)

        if test_ready is not None and test_predictions is not None:
            fold_test = _predict(model, test_ready, needs_proba=needs_proba)
            test_predictions += fold_test / len(fold_list)

        if verbose:
            print(f"  fold {fold_index}/{len(fold_list)}  {metric}={score:.6f}")

    elapsed = time.perf_counter() - started

    # TimeSeriesSplit ilk fold'u hic valid yapmaz -- o satirlar OOF'ta bos kalir.
    # Genel skoru YALNIZCA doldurulmus satirlar uzerinden hesapla, aksi halde
    # sifirlar skoru sessizce bozar.
    coverage = float(oof_filled.mean())
    overall = float(metric_fn(y[oof_filled], oof[oof_filled])) if oof_filled.any() else float("nan")

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

    return CVResult(
        oof_predictions=oof,
        oof_covered=oof_filled,
        test_predictions=test_predictions,
        fold_scores=fold_scores,
        overall_score=overall,
        feature_importance=importance,
        models=models,
        elapsed_seconds=elapsed,
        metric_name=metric,
        model_kind=kind,
    )
