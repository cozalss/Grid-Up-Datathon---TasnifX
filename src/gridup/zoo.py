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


def _kapsam_kesisimi(results: dict[str, CVResult]) -> np.ndarray:
    """Tum uyelerin AYNI ANDA gercek tahmin urettigi satirlarin maskesi.

    KESISIM aliriz cunku harman tek bir satirda TUM uyelerin degerini birlikte
    kullanir: uyelerden biri o satirda dolgu (0.0) ise harman da o satirda
    dolgudur ve skora sahte bir terim ekler.

    Raises:
        ValueError: Zoo bossa veya uyelerin OOF uzunluklari farkliysa (bu
            durumda uyeler ayni fold'larla uretilmemistir -- harmanlamak
            sizinti yaratir, sessizce kirpmak yerine hata firlatiriz).
    """
    if not results:
        raise ValueError("Zoo bos -- kapsam maskesi hesaplanamaz.")

    uzunluklar = {ad: len(sonuc.oof_predictions) for ad, sonuc in results.items()}
    n = next(iter(uzunluklar.values()))
    farkli = {ad: u for ad, u in uzunluklar.items() if u != n}
    if farkli:
        raise ValueError(
            f"Uyelerin OOF uzunluklari farkli ({farkli} vs {n}) -- ayni fold'larla "
            "uretilmemisler. Harmanlamak sizinti yaratir."
        )

    maske = np.ones(n, dtype=bool)
    for sonuc in results.values():
        # Elle kurulmus/eski CVResult'ta maske bostur; o uye kapsami daraltmaz.
        if sonuc.oof_covered.size == 0:
            continue
        maske &= sonuc.oof_covered
    return maske


@dataclass
class ZooResult:
    """Zoo kosusunun ciktisi."""

    results: dict[str, CVResult]
    metric: str
    greater_is_better: bool
    elapsed_seconds: float = 0.0

    @property
    def oof_matrix(self) -> dict[str, np.ndarray]:
        """TAM UZUNLUKTA OOF sozlugu -- kapsam DISI satirlar 0.0 DOLGUSUDUR.

        Yalnizca fold indeksleriyle calisan ``ensemble.stack_oof`` icin
        kullan; o zaman da ``base_covered=zoo.oof_covered`` ver.

        HARMANLAMAYA VE KORELASYONA DOGRUDAN VERME. purged_time_series_split
        ilk donemi hicbir fold'un valid tarafina koymaz; o satirlarda deger
        tahmin degil dolgudur ve skora sahte bir terim ekler.
        OLCULDU (3 uye, TimeSeriesSplit(4), N=3000, kapsam %80):
        maskesiz hill climbing rmse 2.754756, maskeli 2.213196 -- %24.5 sapma.
        Bunun yerine ``covered_oof_matrix()`` kullan.
        """
        return {name: result.oof_predictions for name, result in self.results.items()}

    @property
    def oof_covered(self) -> np.ndarray:
        """Tum uyelerin ortak OOF kapsam maskesi (``CVResult.oof_covered`` kesisimi).

        ``ensemble.stack_oof(..., base_covered=zoo.oof_covered)`` seklinde
        dogrudan verilebilir; stack_oof'un kendi tam-sifir sezgisinden daha
        guvenilirdir cunku tahmin degil OLCUM tasir.
        """
        return _kapsam_kesisimi(self.results)

    @property
    def coverage(self) -> float:
        """Ortak OOF kapsam orani (0..1). 1.0'dan kucukse harman maskesiz olamaz."""
        return float(self.oof_covered.mean())

    def covered_oof_matrix(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """``(kapsanan_satir_indeksleri, {model: o satirlarin OOF tahminleri})``.

        HARMANLAMA, KORELASYON VE HARMAN SKORU HER ZAMAN BUNUNLA YAPILIR.
        Hedefi de ayni indeksle kirp: ``y[indeks]``.

        ``CVResult.covered_predictions()`` ile ayni deseni izler; fark, burada
        maskenin tum uyelerin KESISIMI olmasidir.

        Raises:
            ValueError: Hicbir satir tum uyelerce kapsanmiyorsa -- bu durumda
                harman skoru tanimsizdir, sessizce bos dizi dondurmeyiz.
        """
        maske = self.oof_covered
        indeks = np.flatnonzero(maske)
        if indeks.size == 0:
            raise ValueError(
                "Hicbir satir tum uyelerce kapsanmiyor -- ortak OOF kapsami bos. "
                "Uyeler ayni fold'larla mi egitildi? Daha az fold dene."
            )
        return indeks, {ad: sonuc.oof_predictions[indeks] for ad, sonuc in self.results.items()}

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
        return (
            pd.DataFrame(rows)
            .sort_values("skor", ascending=not self.greater_is_better)
            .reset_index(drop=True)
        )

    def correlation(self) -> pd.DataFrame:
        """Modeller arasi OOF korelasyonu -- YALNIZCA ortak kapsamdaki satirlar.

        > 0,99 -> modeller aslinda ayni; harmanlamak kazanc getirmez
        0,90-0,98 -> saglikli cesitlilik

        Kapsam disi satirlar tum uyelerde ayni sabit (0.0) oldugu icin
        korelasyonu yapay olarak yukari ceker (olculdu: 0.885434 -> 0.885381,
        kucuk ama sistematik ve hep AYNI yonde).
        """
        from .ensemble import correlation_matrix

        _, kapsamli = self.covered_oof_matrix()
        return correlation_matrix(kapsamli)

    def summary(self) -> str:
        lines = [
            f"{len(self.results)} model, {self.elapsed_seconds / 60:.1f} dk",
            "",
            self.leaderboard().to_string(index=False),
        ]
        kapsam = self.coverage
        if kapsam < 0.999:
            lines.append("")
            lines.append(
                f"Ortak OOF kapsami: %{kapsam * 100:.1f} -- kalan satirlarda deger "
                "DOLGUDUR (0.0), tahmin degil."
            )
            lines.append(
                "  Harman ve korelasyon icin covered_oof_matrix() kullan; "
                "oof_matrix'i dogrudan harmanlamak skoru sisirir."
            )
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
        ``ZooResult``. Harman icin ``indeks, kapsamli = result.covered_oof_matrix()``
        kullan ve hedefi ``y[indeks]`` diye kirp -- ham ``oof_matrix`` kapsam
        disinda 0.0 DOLGUSU icerir ve harman skorunu sisirir.
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
            train,
            y,
            folds,
            kind=entry.kind,
            task_type=task_type,
            metric=metric,
            params=params,
            test=test,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
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
        train,
        target,
        folds,
        entries=entries,
        task_type="regression",
        metric=metric,
        test=test,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
    )
