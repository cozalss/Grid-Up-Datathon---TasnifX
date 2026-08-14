"""Cok tohumlu (multi-seed) tam veri yeniden egitimi.

NEDEN BU MODUL VAR
------------------
2024 GDZ Datathon birincisinin final mimarisi soyleydi::

    25 seed x full-data refit  ->  mean blend  ->  round  ->  clip

Ve ``refit_full=True`` tek basina skoru **3,02 -> 2,95** tasidi. Tek satirlik
bir degisiklik, yaklasik %2 iyilesme.

IKI AYRI FIKIR, KARISTIRMA
--------------------------
1. **Tam veri yeniden egitimi (full-data refit).** CV sirasinda her model verinin
   yalnizca ``(k-1)/k``'sini gorur. CV bittiginde en iyi konfigurasyon biliniyorsa,
   TUM veriyle bir kez daha egitmek modele %20 daha fazla ornek verir. Zaman
   serisinde bu ozellikle degerlidir cunku eklenen kisim EN GUNCEL donemdir --
   yani test'e en yakin olan.

   Bedeli: artik dogrulama kumesi yok, dolayisiyla **erken durdurma yapilamaz.**
   Agac sayisi CV'den devralinir (asagiya bak).

2. **Cok tohumlu ortalama (seed averaging).** GBDT'ler alt-orneklem ve feature
   orneklemesi nedeniyle tohuma duyarlidir. Ayni konfigurasyonu farkli tohumlarla
   egitip ortalamak varyansi dusurur. Bu bir "ensemble" degildir -- ayni modelin
   gurultusunu silmektir; bu yuzden aciklanabilirligi BOZMAZ ve juriye
   "25 farkli model" diye anlatilmasi gerekmez.

AGAC SAYISI KURALI
------------------
Erken durdurma olmadan kac tur egitilecegi kritik. Yaygin ve savunulabilir kural::

    n_estimators = ortalama_CV_best_iteration * (1 + 1/k)

Gerekce: CV'de her model ``(k-1)/k`` veriyle egitildi. Veri ``k/(k-1)`` kat
buyudugunde optimum tur sayisi yaklasik ayni oranda artar. 5 katli CV icin bu
%25 daha fazla agac demektir.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .models import ModelKind, _predict, _prepare_categoricals, fit_without_validation

__all__ = [
    "RefitResult",
    "estimate_full_data_rounds",
    "extract_best_iterations",
    "multi_seed_refit",
]


@dataclass
class RefitResult:
    """Cok tohumlu yeniden egitimin ciktisi."""

    predictions: np.ndarray
    per_seed_predictions: np.ndarray  # (n_seed, n_test)
    seeds: list[int]
    n_estimators: int
    elapsed_seconds: float = 0.0
    models: list[Any] = field(default_factory=list, repr=False)

    @property
    def seed_disagreement(self) -> float:
        """Tohumlar arasi ortalama standart sapma.

        YORUM: Tahmin olceginin %5'inden buyukse model tohuma asiri duyarlidir --
        daha fazla tohum ekle veya modeli duzenlilestir (regularize et).
        Kucukse zaten kararlidir ve 25 tohum israftir; 5 yeterlidir.
        """
        if self.per_seed_predictions.shape[0] < 2:
            return 0.0
        return float(self.per_seed_predictions.std(axis=0).mean())

    def summary(self) -> str:
        scale = float(np.abs(self.predictions).mean()) or 1.0
        ratio = self.seed_disagreement / scale
        verdict = "KARARLI" if ratio < 0.05 else "TOHUMA DUYARLI -- daha fazla tohum ekle"
        return "\n".join(
            [
                f"Tohum sayisi: {len(self.seeds)}   Agac sayisi: {self.n_estimators:,}",
                f"Tohumlar arasi sapma: {self.seed_disagreement:.6f} "
                f"(tahmin olceginin %{ratio * 100:.2f}'i)  [{verdict}]",
                f"Sure: {self.elapsed_seconds:.1f} sn",
            ]
        )


def estimate_full_data_rounds(
    fold_best_iterations: Sequence[int], *, n_folds: int, safety: float = 1.0
) -> int:
    """CV'deki en iyi tur sayilarindan tam veri icin tur sayisi tahmin eder.

    Args:
        fold_best_iterations: Her fold'un ``best_iteration_`` degeri.
        n_folds: CV kat sayisi.
        safety: Ek carpan. 1.0 = formul aynen; 0.9 = biraz muhafazakar.

    Returns:
        Tam veri egitimi icin onerilen ``n_estimators``.

    Raises:
        ValueError: Bos liste verilirse -- sessizce bir varsayilan uydurmak yerine.
    """
    valid = [int(value) for value in fold_best_iterations if value and value > 0]
    if not valid:
        raise ValueError(
            "Fold'lardan en iyi tur sayisi alinamadi. Erken durdurma calisti mi? "
            "Tam veri egitimi icin tur sayisini elle ver."
        )

    mean_rounds = float(np.mean(valid))
    scale = 1.0 + 1.0 / n_folds
    return max(1, int(round(mean_rounds * scale * safety)))


def extract_best_iterations(models: Sequence[Any]) -> list[int]:
    """``CVResult.models`` icinden erken durdurmanin buldugu tur sayilarini cikarir.

    Kullanim::

        result = cross_validate(...)
        rounds = estimate_full_data_rounds(
            extract_best_iterations(result.models), n_folds=len(result.fold_scores)
        )
    """
    rounds: list[int] = []
    for model in models:
        for attribute in ("best_iteration_", "best_iteration", "best_ntree_limit"):
            value = getattr(model, attribute, None)
            if callable(value):
                try:
                    value = value()
                except (TypeError, ValueError):
                    value = None
            if isinstance(value, (int, np.integer)) and value > 0:
                rounds.append(int(value))
                break
    return rounds


def multi_seed_refit(
    train: pd.DataFrame,
    target: np.ndarray | pd.Series,
    test: pd.DataFrame,
    *,
    kind: ModelKind = "lightgbm",
    params: dict[str, Any],
    n_estimators: int,
    seeds: Sequence[int] = tuple(range(5)),
    needs_proba: bool = False,
    verbose: bool = True,
) -> RefitResult:
    """Tum veriyle, birden fazla tohumla egitir ve tahminleri ortalar.

    CV BITTIKTEN SONRA calistirilir. Once ``cross_validate`` ile en iyi
    konfigurasyonu bul, sonra bunu cagir.

    Args:
        n_estimators: Agac sayisi. ``estimate_full_data_rounds`` ile hesapla --
            erken durdurma BURADA CALISMAZ (dogrulama kumesi yok).
        seeds: Tohum listesi. 5 iyi bir baslangictir; kazanan cozum 25 kullandi.
            ``seed_disagreement`` dusukse artirmanin faydasi azdir.
        needs_proba: Siniflandirmada olasilik isteniyorsa ``True``.

    Returns:
        ``RefitResult``.

    UYARI -- SURE: Bu, CV'nin ``len(seeds) / n_folds`` kati kadar surer.
    25 tohum, 5 katli CV'nin 5 katidir. 12 gunluk yarismada bunu SON GUNLER icin
    sakla; gelistirme sirasinda 3-5 tohum yeterlidir.
    """
    y = np.asarray(target).ravel()
    if len(y) != len(train):
        raise ValueError(f"train ({len(train)}) ve target ({len(y)}) uzunluklari farkli.")

    train_ready, test_ready, categorical = _prepare_categoricals(train, test, kind)
    if test_ready is None:
        raise ValueError("Tam veri egitimi icin test kumesi zorunludur.")

    started = time.perf_counter()
    per_seed: list[np.ndarray] = []
    models: list[Any] = []

    for index, seed in enumerate(seeds, start=1):
        seeded = dict(params)
        # Kutuphaneler tohum anahtarini farkli adlandirir; hepsini set ediyoruz ki
        # hangisi kullanilirsa kullanilsin tohum GERCEKTEN degissin. Sessizce ayni
        # tohumla 25 kez egitmek, cok tohumlu ortalamayi anlamsiz kilardi.
        seeded["random_state"] = seed
        if kind == "lightgbm":
            seeded["n_estimators"] = n_estimators
            seeded["bagging_seed"] = seed
            seeded["feature_fraction_seed"] = seed
        elif kind == "xgboost":
            seeded["n_estimators"] = n_estimators
        elif kind == "catboost":
            seeded["iterations"] = n_estimators
            seeded["random_seed"] = seed
            seeded.pop("random_state", None)

        model = fit_without_validation(kind, seeded, train_ready, y, categorical)
        prediction = _predict(model, test_ready, needs_proba=needs_proba)
        per_seed.append(prediction)
        models.append(model)

        if verbose:
            print(f"  tohum {index}/{len(seeds)} (seed={seed}) tamam")

    stacked = np.vstack(per_seed)
    result = RefitResult(
        predictions=stacked.mean(axis=0),
        per_seed_predictions=stacked,
        seeds=list(seeds),
        n_estimators=n_estimators,
        elapsed_seconds=time.perf_counter() - started,
        models=models,
    )

    if verbose:
        print(result.summary())

    return result
