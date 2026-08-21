"""Optuna ile hiperparametre araması.

TASARIM KARARI: OBJECTIVE DE ARAMA UZAYINDA
-------------------------------------------
ATIF TARIHI (iki kez duzeltildi -- ders: 404 != yok)
2024 GDZ Datathon birincisi (Pikachow) objective'i GERCEKTEN Optuna arama
uzayina koydu -- final sunumu s.23: TPESampler; aranan parametre listesi
``objective`` ile basliyor. Bir onceki denetim, Kaggle sayfasi login'siz
404 verdigi icin "2024 yarismasi yok" diyip bu atifi silmisti; sunum PDF'i
(anilozturk.net) + coderspace.io etkinlik sayfasiyla yeniden dogrulandi.
2023 birincisinin notebook'unda ise Optuna HIC yok; parametreler elle
sabitlenmis (objective='MAE', eval_metric='MAPE', lr=0.03, depth=6).
Iki yil iki farkli yol -- aramak guvenli olandir.

Objective ilk bakista tuhaf bir arama boyutu gorunur -- "model ayari"
degil, problem tanimidir.

Ama sayim hedeflerinde degildir. ``poisson``, ``tweedie``, ``mae`` ve ``l2``
arasindaki secim VERIYE BAGLIDIR ve teoriden okunamaz:

  * Asiri yayilim (varyans/ortalama) 1'e yakinsa Poisson yeter
  * Sifir kutlesi buyukse Tweedie (variance_power 1.1-1.7) kazanir
  * Resmi metrik MAE ise ``mae`` objective'i dogrudan onu optimize eder --
    ama sifir-siskin veride bu her zaman en iyisi DEGILDIR

Tahmin etmek yerine ARAMAK, 12 gunluk yarismada bir saatlik yatirimdir.

BUTCE
-----
Her trial TAM bir CV kosusu demektir. 5 fold x 100 trial = 500 model egitimi.
``timeout`` ile sinirla, ``n_trials`` ile degil -- veri boyutunu bilmiyorsun.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .metrics import get_metric
from .models import COUNT_OBJECTIVES, ModelKind, cross_validate

__all__ = ["TuningResult", "suggest_params", "tune_with_optuna"]


@dataclass
class TuningResult:
    """Arama sonucu."""

    best_params: dict[str, Any]
    best_score: float
    n_trials: int
    study: Any = field(default=None, repr=False)
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Metrigin yonu. Varsayilan ``False`` cunku varsayilan metrik ``rmse``.
    #: ``tune_with_optuna`` bunu ``get_metric(metric)``ten DOLDURUR -- elle
    #: kurulan bir TuningResult'ta yanlis kalirsa tablolar ters siralanir.
    greater_is_better: bool = False

    @property
    def selection_optimism(self) -> float:
        """``best_score`` ile deneme skorlarinin ortalamasi arasindaki fark.

        ``best_score`` AYNI fold'lar uzerinde kosulan N korele denemenin en
        iyisidir; bagimsiz bir kumede beklenenden IYIMSERDIR. Bu sayi o
        iyimserligin kaba bir olcusudur. ``selection.SelectionResult`` ile
        BIREBIR ayni desen -- oradaki gerekce burada da gecerli.
        """
        if len(self.history) < 2 or "skor" not in self.history.columns:
            return 0.0
        ortalama = float(self.history["skor"].mean())
        return abs(self.best_score - ortalama)

    def summary(self) -> str:
        lines = [
            f"{self.n_trials} deneme, en iyi skor: {self.best_score:.6f}",
            "En iyi parametreler:",
        ]
        for key, value in sorted(self.best_params.items()):
            lines.append(f"  {key:<24} {value}")
        # SECIM YANLILIGI ACIKCA SOYLENIR -- selection.SelectionResult ile ayni desen.
        #
        # best_score, AYNI fold'lar uzerinde kosulan N denemenin en iyisidir ve
        # hicbir yerde yeniden tahmin edilmiyordu. OLCULDU (SIFIR sinyalli hedef,
        # y.std()=1.952159 teorik tavan, 12 deneme):
        #   raporlanan best 1.947733, deneme ortalamasi 1.951498 -> fark 0.003766
        # Ogrenilecek HICBIR SEY yokken raporlanan skor hedefin kendi std'sinin
        # ALTINA indi -- matematiksel olarak imkansiz bir iyilesme, tamamen secim
        # artefakti. Sayi yanlis degil, SUNUMU yaniltici; ve juriye giden slayt bu.
        if len(self.history) >= 2:
            lines.append(
                f"DIKKAT: bu skor {len(self.history)} korele denemenin EN IYISIDIR "
                f"(deneme ortalamasindan {self.selection_optimism:.6f} uzakta). "
                "Secim yanliligi tasir -- kazanci bagimsiz bir kumede dogrula."
            )
        return "\n".join(lines)

    def objective_comparison(self) -> pd.DataFrame:
        """Objective bazinda skor dagilimi.

        Aramanin en ogretici ciktisi: hangi kayip fonksiyonunun bu veride
        gercekten kazandigini gosterir ve juri sunumunda tek satirlik bir
        gerekce olur.

        YON: ``en_iyi`` sutunu ONCEDEN HER ZAMAN ``min`` ile hesaplaniyordu.
        r2/auc/f1/accuracy gibi buyuk-daha-iyi metriklerde bu, her ailenin EN
        KOTU denemesini "en iyi" diye gosterir ve tabloyu ters siralar.
        OLCULDU (A ailesi [0.60, 0.62, 0.91], B ailesi [0.55, 0.58, 0.59],
        metric='r2'):
            ONCE : basa B konuyor, A'nin en_iyi'si 0.60 yaziliyor (gercegi 0.91)
            SONRA: basa A konuyor, A'nin en_iyi'si 0.91
        Yani tablo, juri sunumunda KAYBEDEN aileyi oneriyordu.
        """
        if self.history.empty or "objective" not in self.history.columns:
            return pd.DataFrame()
        return (
            self.history.groupby("objective", observed=True)["skor"]
            .agg(deneme="size", en_iyi="max" if self.greater_is_better else "min", medyan="median")
            .sort_values("en_iyi", ascending=not self.greater_is_better)
            .reset_index()
        )


def suggest_params(
    trial: Any,
    kind: ModelKind,
    *,
    task_type: str = "regression",
    search_objective: bool = False,
) -> dict[str, Any]:
    """Bir Optuna trial'i icin parametre onerir.

    Araliklar 2025-2026 pratiginde ise yarayan degerlerdir; cok genis bir uzay
    12 gunluk bir yarismada bosa arama yapar.

    Args:
        search_objective: Sayim hedefi icin objective'i de arama uzayina koyar.
    """
    if kind == "lightgbm":
        params: dict[str, Any] = {
            "n_estimators": 4000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 0, 7),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "max_bin": trial.suggest_categorical("max_bin", [127, 255, 511]),
            "verbose": -1,
            "n_jobs": -1,
        }
    elif kind == "xgboost":
        params = {
            "n_estimators": 4000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 30, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "tree_method": "hist",
            "enable_categorical": True,
            "n_jobs": -1,
        }
    elif kind == "catboost":
        params = {
            "iterations": 4000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 30, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.1, 10, log=True),
            "bootstrap_type": "Bernoulli",
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "verbose": 0,
            "allow_writing_files": False,
        }
    else:
        raise ValueError(f"Bilinmeyen model tipi '{kind}'.")

    if search_objective:
        choice = trial.suggest_categorical("objective_family", ["l2", "mae", "poisson", "tweedie"])
        objective = COUNT_OBJECTIVES[kind][choice]
        key = "loss_function" if kind == "catboost" else "objective"
        params[key] = objective

        # Tweedie'nin varyans ussu de aranmali: 1.0 Poisson'a, 2.0 Gamma'ya
        # yakinsar. Sayim + sifir kutlesi icin arasi.
        #
        # ARALIK 1.9 -> 1.5'e DARALTILDI (2026-08-21, literatur taramasi).
        # Sayim hedeflerinde kanit dusuk ucu isaret ediyor: M5 birincisi 1.1
        # kullandi, ust-%7 bir cozum magaza basina 1.1/1.2 secti, XGBoost
        # perakende calismasi (arXiv:2208.12264) "1.1-1.3 vakalarin cogunda
        # en iyi" diyor ve us buyudukce yanliligin MONOTON negatiflestigini
        # olcuyor. 1.5-1.9 araligina harcanan her deneme, kanitin
        # gostermedigi bir bolgede harcanmis demektir.
        #
        # Bu bir ON KABULDUR, olcum degil -- daraltmak aramayi hizlandirir
        # ama yanlissa optimumu disarida birakir. Gercek veride 1.5 sinirina
        # YAPISAN bir optimum gorulurse ust sinir geri acilmalidir.
        if choice == "tweedie":
            power = trial.suggest_float("tweedie_variance_power", 1.1, 1.5)
            if kind == "catboost":
                params["loss_function"] = f"Tweedie:variance_power={power:.2f}"
            else:
                params["tweedie_variance_power"] = power

    return params


def tune_with_optuna(
    train: pd.DataFrame,
    target: np.ndarray | pd.Series,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    kind: ModelKind = "lightgbm",
    task_type: str = "regression",
    metric: str = "rmse",
    n_trials: int = 50,
    timeout: int | None = None,
    search_objective: bool = False,
    early_stopping_rounds: int = 100,
    seed: int = 42,
    storage: str | None = None,
    study_name: str | None = None,
    verbose: bool = True,
) -> TuningResult:
    """Optuna ile hiperparametre araması.

    Args:
        n_trials: Deneme sayisi. Her deneme TAM bir CV kosusu.
        timeout: Saniye cinsinden ust sinir. **Bunu kullan** -- veri boyutunu
            bilmeden ``n_trials`` secmek, aramanin ne kadar surecegini
            bilmemek demektir.
        search_objective: Sayim hedefinde objective'i de ara.
        storage: ``sqlite:///optuna.db`` gibi. Verilirse arama KESILDIGI YERDEN
            devam edebilir -- uzun aramada makine kapanirsa is kaybolmaz.

    Returns:
        ``TuningResult``. ``best_score`` N denemenin EN IYISIDIR ve secim
        yanliligi tasir -- ``selection_optimism`` bu yanliligin kaba olcusu,
        ``summary()`` de bunu acikca yazar. Zoo/harman skorlariyla veya LB ile
        karsilastirmadan once bagimsiz bir kumede dogrula.

    Raises:
        ImportError: Optuna kurulu degilse.
    """
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Arama icin optuna gerekli: pip install optuna") from exc

    # storage verilip study_name verilmezse Optuna HER SEFERINDE rastgele bir
    # ad uretir ("no-name-<uuid>") ve load_if_exists=True hicbir ise yaramaz:
    # arama kaldigi yerden DEVAM ETMEZ, sifirdan baslar. Ne hata ne uyari cikar.
    # OLCULDU: ayni sqlite dosyasina iki kosu -> iki ayri study, ikisi de 3 trial.
    #
    # Deterministik bir ad UYDURMUYORUZ cunku farkli feature setleriyle yapilan
    # iki arama ayni study'ye dolar ve birbirini kirletir. Karari kullaniciya
    # birakiyoruz -- ``embargo``da oldugu gibi.
    if storage is not None and study_name is None:
        raise ValueError(
            "storage verildi ama study_name verilmedi. Optuna bu durumda her "
            "kosuda rastgele bir ad uretir ve arama KALDIGI YERDEN DEVAM ETMEZ.\n"
            "Acik bir ad ver, orn: study_name='lgbm_takvim_lag_v2'.\n"
            "DIKKAT: ayni adi farkli feature setiyle tekrar kullanma -- eski "
            "trial'lar yeni arama uzayina karisir."
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    _, greater_is_better, _ = get_metric(metric)
    direction = "maximize" if greater_is_better else "minimize"
    y = np.asarray(target).ravel()

    records: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        params = suggest_params(trial, kind, task_type=task_type, search_objective=search_objective)
        result = cross_validate(
            train,
            y,
            folds,
            kind=kind,
            task_type=task_type,
            metric=metric,
            params=params,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )

        family = trial.params.get("objective_family", task_type)
        records.append(
            {
                "trial": trial.number,
                "skor": result.overall_score,
                "objective": family,
                "fold_std": result.fold_std,
                "kararli": result.is_stable,
            }
        )

        if verbose and (trial.number + 1) % 10 == 0:
            best = (
                min(r["skor"] for r in records)
                if not greater_is_better
                else max(r["skor"] for r in records)
            )
            print(f"  {trial.number + 1} deneme, en iyi {metric}={best:.6f}")

        return result.overall_score

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=storage,
        study_name=study_name,
        load_if_exists=storage is not None,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    # Kazanan trial'in TAM parametre sozlugunu yeniden uret: study.best_params
    # yalnizca aranan degerleri tutar, sabitleri (n_estimators, n_jobs...) tutmaz.
    best_params = suggest_params(
        optuna.trial.FixedTrial(study.best_params),
        kind,
        task_type=task_type,
        search_objective=search_objective,
    )

    result = TuningResult(
        best_params=best_params,
        best_score=float(study.best_value),
        n_trials=len(study.trials),
        study=study,
        history=pd.DataFrame(records),
        greater_is_better=greater_is_better,
    )

    if verbose:
        print("\n" + result.summary())
        comparison = result.objective_comparison()
        if not comparison.empty:
            print("\nObjective karsilastirmasi:")
            print(comparison.to_string(index=False))

    return result
