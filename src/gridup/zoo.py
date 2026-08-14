"""Model zoo: uc kutuphaneyi AYNI fold'larda calistiran tek arayuz.

NEDEN AYNI FOLD'LAR SART
------------------------
Harmanlama ve stacking, modellerin OOF tahminlerini birlestirir. Bu ancak
tum modeller **ayni bolmeleri** kullanmissa gecerlidir. Farkli fold'larla
uretilmis OOF'lari harmanlamak sizinti yaratir: bir modelin valid satiri,
digerinin train satiri olur ve harman skoru yapay olarak yukselir.

Bu modul fold listesini bir kez alir ve her modele AYNISINI verir.

CESITLILIK NEDEN ONEMLI
-----------------------
Uc kutuphane ayni algoritmayi farkli uygular: LightGBM yaprak-bazli buyur,
XGBoost seviye-bazli, CatBoost sirali hedef kodlama yapar. Ayni veride
FARKLI hatalar uretirler -- harmanlamanin kazanci buradan gelir.

Korelasyonu 0,99 olan iki mukemmel model, korelasyonu 0,85 olan iki iyi
modelden daha kotu harmanlanir.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .metrics import get_metric
from .models import COUNT_OBJECTIVES, CVResult, ModelKind, cross_validate, starter_params

__all__ = ["ZooEntry", "ZooResult", "make_model_zoo", "sweep_count_objectives"]


@dataclass(frozen=True)
class ZooEntry:
    """Zoo'da tek bir model tanimi."""

    name: str
    kind: ModelKind
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ZooResult:
    """Zoo kosusunun ciktisi."""

    results: dict[str, CVResult]
    metric: str
    greater_is_better: bool
    elapsed_seconds: float = 0.0

    @property
    def oof_matrix(self) -> dict[str, np.ndarray]:
        """Harmanlamaya dogrudan verilebilecek OOF sozlugu."""
        return {name: result.oof_predictions for name, result in self.results.items()}

    @property
    def test_matrix(self) -> dict[str, np.ndarray]:
        return {
            name: result.test_predictions
            for name, result in self.results.items()
            if result.test_predictions is not None
        }

    def leaderboard(self) -> pd.DataFrame:
        """Modelleri skora gore siralar."""
        rows = [
            {
                "model": name,
                "skor": result.overall_score,
                "fold_std": result.fold_std,
                "kararli": result.is_stable,
                "sure_sn": round(result.elapsed_seconds, 1),
            }
            for name, result in self.results.items()
        ]
        return pd.DataFrame(rows).sort_values(
            "skor", ascending=not self.greater_is_better
        ).reset_index(drop=True)

    def correlation(self) -> pd.DataFrame:
        """Modeller arasi OOF korelasyonu.

        > 0,99 -> modeller aslinda ayni; harmanlamak kazanc getirmez
        0,90-0,98 -> saglikli cesitlilik
        """
        from .ensemble import correlation_matrix

        return correlation_matrix(self.oof_matrix)

    def summary(self) -> str:
        lines = [
            f"{len(self.results)} model, {self.elapsed_seconds / 60:.1f} dk",
            "",
            self.leaderboard().to_string(index=False),
        ]
        if len(self.results) > 1:
            correlations = self.correlation()
            values = correlations.to_numpy()
            off_diagonal = values[~np.eye(len(values), dtype=bool)]
            lines.append("")
            lines.append(
                f"Model korelasyonu: medyan {np.median(off_diagonal):.4f}, "
                f"max {off_diagonal.max():.4f}"
            )
            if off_diagonal.max() > 0.99:
                lines.append(
                    "  UYARI: iki model neredeyse ayni. Harmanlamak kazanc getirmez -- "
                    "farkli objective veya feature alt kumesi dene."
                )
        return "\n".join(lines)


def make_model_zoo(
    train: pd.DataFrame,
    target: np.ndarray | pd.Series,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    entries: Sequence[ZooEntry] | None = None,
    task_type: str = "regression",
    metric: str = "rmse",
    test: pd.DataFrame | None = None,
    early_stopping_rounds: int = 200,
    verbose: bool = True,
) -> ZooResult:
    """Birden fazla modeli AYNI fold'larda egitir.

    Args:
        entries: Egitilecek modeller. ``None`` ise uc kutuphanenin varsayilani.

    Returns:
        ``ZooResult`` -- ``oof_matrix`` dogrudan ``hill_climb_weights``a verilir.
    """
    if entries is None:
        entries = [
            ZooEntry("lightgbm", "lightgbm"),
            ZooEntry("xgboost", "xgboost"),
            ZooEntry("catboost", "catboost"),
        ]

    _, greater_is_better, _ = get_metric(metric)
    y = np.asarray(target).ravel()
    results: dict[str, CVResult] = {}
    started = time.perf_counter()

    for entry in entries:
        if verbose:
            print(f"\n--- {entry.name} ({entry.kind}) ---")
        params = entry.params or starter_params(entry.kind, task_type)
        results[entry.name] = cross_validate(
            train, y, folds,
            kind=entry.kind, task_type=task_type, metric=metric,
            params=params, test=test,
            early_stopping_rounds=early_stopping_rounds, verbose=verbose,
        )

    result = ZooResult(
        results=results,
        metric=metric,
        greater_is_better=greater_is_better,
        elapsed_seconds=time.perf_counter() - started,
    )
    if verbose:
        print("\n" + result.summary())
    return result


def sweep_count_objectives(
    train: pd.DataFrame,
    target: np.ndarray | pd.Series,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    kind: ModelKind = "lightgbm",
    metric: str = "mae",
    families: Sequence[str] = ("l2", "mae", "poisson", "tweedie"),
    test: pd.DataFrame | None = None,
    base_params: dict[str, Any] | None = None,
    early_stopping_rounds: int = 200,
    verbose: bool = True,
) -> ZooResult:
    """Sayim hedefi icin objective ailelerini AYNI fold'larda karsilastirir.

    Hangi kayip fonksiyonunun kazandigi VERIYE BAGLIDIR ve teoriden okunamaz.
    Bu supurme, tahmin etmek yerine olcmenin bir saatlik maliyetidir.

    Sonuc ayrica juri sunumunda tek satirlik bir gerekce olur:
    "Poisson, Tweedie ve L2'yi ayni bolmelerde karsilastirdik; Tweedie kazandi."
    """
    available = COUNT_OBJECTIVES[kind]
    entries = []
    for family in families:
        if family not in available:
            continue
        params = dict(base_params) if base_params else starter_params(kind, "regression")
        key = "loss_function" if kind == "catboost" else "objective"
        params[key] = available[family]
        entries.append(ZooEntry(f"{kind}_{family}", kind, params))

    if not entries:
        raise ValueError(f"'{kind}' icin gecerli objective ailesi yok: {list(families)}")

    return make_model_zoo(
        train, target, folds, entries=entries, task_type="regression",
        metric=metric, test=test, early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
    )
