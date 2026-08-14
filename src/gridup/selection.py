"""Feature secimi: SHAP geri eleme ve null importance.

NEDEN BU MODUL VAR
------------------
Bu pipeline kolayca 400+ feature uretir (takvim x lag x rolling x grup x hava x
komsu). Hepsini modele vermek uc sorun yaratir:

  * **Varyans.** Gurultulu feature'lar fold'lar arasi sapmayi buyutur; kucuk
    iyilesmeler gorunmez hale gelir.
  * **Asiri uyum.** Ozellikle yuksek kardinaliteli kodlamalar.
  * **Aciklanabilirlik.** 400 feature'lik bir modeli juriye anlatamazsin.

2024 GDZ Datathon birincisi **490 -> 97 feature** indirdi ve skoru iyilestirdi.
Feature sayisi-skor egrisi ayrica sunumda guclu bir slayt olur.

IKI YONTEM, FARKLI SORULAR
--------------------------
``shap_backward_selection``
    "Modelin kararina en az katkida bulunan feature hangisi?" Katkiyi SHAP ile
    olcer, en dusukleri atar, CV'yi yeniden calistirir. Pahali ama en guvenilir.

``null_importance_filter``
    "Bu feature'in onemi SANS ESERI olabilir mi?" Hedefi karistirip modeli
    tekrar egitir; gercek onem, karistirilmis dagilimin ustunde degilse
    feature gurultudur. Ucuz, ilk eleme icin ideal.

Sira: once ``null_importance_filter`` ile bariz gurultuyu at (dakikalar),
sonra ``shap_backward_selection`` ile ince ayar yap (saatler).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .models import ModelKind, cross_validate, fit_without_validation, starter_params

__all__ = [
    "SelectionStep",
    "SelectionResult",
    "mean_absolute_shap",
    "null_importance_filter",
    "shap_backward_selection",
]


@dataclass(frozen=True)
class SelectionStep:
    """Geri elemenin tek adimi."""

    n_features: int
    score: float
    dropped: tuple[str, ...]
    elapsed_seconds: float


@dataclass
class SelectionResult:
    """Feature secimi sonucu."""

    best_features: list[str]
    best_score: float
    history: list[SelectionStep] = field(default_factory=list)
    greater_is_better: bool = False

    def curve(self) -> pd.DataFrame:
        """Feature sayisi - skor egrisi. Juri sunumunda dogrudan slayt olur."""
        return pd.DataFrame(
            [
                {"feature_sayisi": step.n_features, "skor": step.score}
                for step in self.history
            ]
        ).sort_values("feature_sayisi")

    def summary(self) -> str:
        if not self.history:
            return "Adim yok."
        first, last = self.history[0], self.history[-1]
        direction = "yukseldi" if self.best_score > first.score else "dusdu"
        return "\n".join(
            [
                f"Baslangic: {first.n_features} feature, skor {first.score:.6f}",
                f"Bitis:     {last.n_features} feature, skor {last.score:.6f}",
                f"En iyi:    {len(self.best_features)} feature, skor {self.best_score:.6f}"
                f"  (skor {direction})",
                f"Toplam {len(self.history)} adim.",
            ]
        )


def mean_absolute_shap(
    model: Any, features: pd.DataFrame, *, sample_size: int = 5000, seed: int = 42
) -> pd.Series:
    """Feature basina ortalama mutlak SHAP degeri.

    Args:
        sample_size: SHAP hesabi O(satir) maliyetlidir. 5.000 satir, siralama
            icin fazlasiyla yeterli -- 500.000 satirda hesaplamak 100 kat
            pahali ve siralamayi degistirmez.

    Returns:
        Feature adi -> ortalama |SHAP|, buyukten kucuge sirali.

    Raises:
        ImportError: ``shap`` kurulu degilse.
    """
    try:
        import shap
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Feature secimi icin 'shap' gerekli: pip install shap") from exc

    sample = features
    if len(features) > sample_size:
        sample = features.sample(n=sample_size, random_state=seed)

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)

    # Cok sinifli modeller liste veya 3B dizi dondurur; sinifar arasi ortalama al.
    if isinstance(values, list):
        values = np.mean([np.abs(block) for block in values], axis=0)
    else:
        values = np.abs(values)
        if values.ndim == 3:
            values = values.mean(axis=2)

    return (
        pd.Series(values.mean(axis=0), index=sample.columns)
        .sort_values(ascending=False)
    )


def null_importance_filter(
    train: pd.DataFrame,
    target: np.ndarray | pd.Series,
    *,
    kind: ModelKind = "lightgbm",
    task_type: str = "regression",
    params: dict[str, Any] | None = None,
    n_runs: int = 5,
    percentile: float = 75.0,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Onemi SANS ESERI aciklanabilecek feature'lari isaretler.

    YONTEM: Hedefi ``n_runs`` kez karistir ve her seferinde modeli egit. Bir
    feature karistirilmis hedefte de yuksek onem aliyorsa, o onem feature'in
    yapisindan gelir (yuksek kardinalite, cok benzersiz deger) -- gercek
    sinyalden degil.

    Karar kurali: gercek onem, null dagiliminin ``percentile``'inden BUYUK
    olmali. 75 muhafazakar bir esiktir; 50 daha agresif eler.

    Args:
        n_runs: Kac karistirma. 5 iyi bir denge; artirmak esigi stabilize eder.

    Returns:
        ``keep`` (tutulacak feature listesi), ``drop``, ``scores`` (DataFrame:
        gercek onem, null percentile, oran).
    """
    y = np.asarray(target).ravel()
    model_params = dict(params) if params else starter_params(kind, task_type)
    model_params.setdefault("n_estimators", 300)

    from .models import _extract_importance, _prepare_categoricals

    prepared, _, categorical = _prepare_categoricals(train, None, kind)
    columns = list(prepared.columns)

    def _importance(target_values: np.ndarray, run_seed: int) -> np.ndarray:
        run_params = dict(model_params)
        run_params["random_state"] = run_seed
        if kind == "catboost":
            run_params["random_seed"] = run_seed
            run_params.pop("random_state", None)
        model = fit_without_validation(kind, run_params, prepared, target_values, categorical)
        return _extract_importance(model, columns)

    if verbose:
        print(f"Gercek onem hesaplaniyor ({len(columns)} feature)...")
    actual = _importance(y, seed)

    rng = np.random.default_rng(seed)
    null_runs = []
    for run in range(n_runs):
        shuffled = rng.permutation(y)
        null_runs.append(_importance(shuffled, seed + run + 1))
        if verbose:
            print(f"  null kosusu {run + 1}/{n_runs}")

    null_matrix = np.vstack(null_runs)
    null_threshold = np.percentile(null_matrix, percentile, axis=0)

    # Sifira bolme: null esigi 0 ise gercek onem >0 olmasi yeterlidir.
    with np.errstate(divide="ignore", invalid="ignore"):
        no_null_signal = np.where(actual > 0, np.inf, 0.0)
        ratio = np.where(null_threshold > 0, actual / null_threshold, no_null_signal)

    scores = pd.DataFrame(
        {
            "feature": columns,
            "gercek_onem": actual,
            "null_esik": null_threshold,
            "oran": ratio,
        }
    ).sort_values("oran", ascending=False)

    keep = scores.loc[scores["oran"] > 1.0, "feature"].tolist()
    drop = scores.loc[scores["oran"] <= 1.0, "feature"].tolist()

    if verbose:
        print(f"\nTutulan: {len(keep)}   Atilan: {len(drop)}")
        if drop:
            print(f"  Atilanlardan ornekler: {drop[:8]}")

    return {"keep": keep, "drop": drop, "scores": scores}


def shap_backward_selection(
    train: pd.DataFrame,
    target: np.ndarray | pd.Series,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    kind: ModelKind = "lightgbm",
    task_type: str = "regression",
    metric: str = "rmse",
    params: dict[str, Any] | None = None,
    drop_per_step: int = 25,
    min_features: int = 20,
    max_steps: int = 20,
    patience: int = 3,
    shap_sample: int = 5000,
    progress: Callable[[str], None] | None = print,
) -> SelectionResult:
    """SHAP tabanli geri eleme. Her adimda en zayif feature'lari atar.

    2024 GDZ Datathon birincisinin birebir proseduru (490 -> 97 feature).

    Args:
        drop_per_step: Her adimda kac feature atilacak.
        min_features: Bu sayinin altina inme.
        max_steps: Ust sinir -- 12 gunluk yarismada zaman butcesi kontrolu.
        patience: Kac adim ust uste iyilesme olmazsa dur.
        shap_sample: SHAP hesabi icin ornek satir sayisi.

    Returns:
        ``SelectionResult``. ``best_features`` EN IYI CV skorunu veren kumedir --
        son adim degil. Egri genellikle once iyilesip sonra bozulur.

    UYARI -- SURE: Her adim TAM bir CV kosusu demektir. 20 adim x 5 fold =
    100 model egitimi. Once ``null_importance_filter`` ile kabaca ele, sonra
    bunu az sayida adimla calistir.
    """
    from .metrics import get_metric

    _, greater_is_better, _ = get_metric(metric)
    y = np.asarray(target).ravel()
    model_params = dict(params) if params else starter_params(kind, task_type)

    features = list(train.columns)
    history: list[SelectionStep] = []
    best_features, best_score = list(features), None
    stale = 0

    for step in range(max_steps):
        started = time.perf_counter()
        subset = train[features]

        result = cross_validate(
            subset, y, folds, kind=kind, task_type=task_type, metric=metric,
            params=model_params, early_stopping_rounds=100, verbose=False,
        )
        elapsed = time.perf_counter() - started

        improved = (
            best_score is None
            or (result.overall_score > best_score if greater_is_better
                else result.overall_score < best_score)
        )
        if improved:
            best_score = result.overall_score
            best_features = list(features)
            stale = 0
        else:
            stale += 1

        if progress:
            marker = " *" if improved else ""
            progress(
                f"adim {step + 1:>2}: {len(features):>4} feature  "
                f"{metric}={result.overall_score:.6f}  ({elapsed:.0f} sn){marker}"
            )

        if stale >= patience:
            if progress:
                progress(f"{patience} adimdir iyilesme yok -- duruluyor.")
            break

        if len(features) - drop_per_step < min_features:
            if progress:
                progress(f"Alt sinira ({min_features}) ulasildi -- duruluyor.")
            history.append(
                SelectionStep(len(features), result.overall_score, (), elapsed)
            )
            break

        # En zayif feature'lari SHAP ile bul. Fold modellerinin ilki temsili --
        # hepsinde hesaplamak maliyeti fold sayisi kadar artirir, siralamayi
        # anlamli olcude degistirmez.
        shap_scores = mean_absolute_shap(
            result.models[0], subset, sample_size=shap_sample
        )
        weakest = shap_scores.tail(drop_per_step).index.tolist()

        history.append(
            SelectionStep(len(features), result.overall_score, tuple(weakest), elapsed)
        )
        features = [column for column in features if column not in weakest]

    return SelectionResult(
        best_features=best_features,
        best_score=float(best_score) if best_score is not None else float("nan"),
        history=history,
        greater_is_better=greater_is_better,
    )
