"""Cok tohumlu (multi-seed) tam veri yeniden egitimi.

NEDEN BU MODUL VAR
------------------
2024 GDZ Datathon birincisinin (Pikachow, final sunumu s.24) mimarisi::

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

from .models import (
    ModelKind,
    _predict,
    _prepare_categoricals,
    assert_finite_target,
    fit_without_validation,
    merge_infrastructure_params,
)

__all__ = [
    "RefitResult",
    "estimate_full_data_rounds",
    "extract_best_iterations",
    "fold_train_fraction",
    "multi_seed_refit",
]

#: ``n_estimators``i sessizce ezen kutuphane takma adlari. LightGBM bunlarin
#: hepsini kabul eder ve sonuncusu kazanir; ``params`` icinde biri kalirsa
#: ``estimate_full_data_rounds``in hesabi ciope gider.
#: OLCULDU: n_estimators=500 istendi, params'ta num_iterations=5 vardi ->
#: gercek agac sayisi 5, RefitResult ise 500 raporladi.
_ROUND_ALIASES: dict[str, tuple[str, ...]] = {
    "lightgbm": (
        "num_iterations", "num_iteration", "n_iter", "num_trees", "num_round",
        "num_rounds", "nrounds", "num_boost_round", "max_iter",
    ),
    "xgboost": ("num_boost_round", "num_round", "num_rounds"),
    "catboost": ("num_boost_round", "num_trees"),
}


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


def fold_train_fraction(
    folds: Sequence[tuple[np.ndarray, np.ndarray]], n_rows: int
) -> float:
    """Fold'larin ORTALAMA train orani -- (k-1)/k varsayimi yerine olcum.

    Bunu ``estimate_full_data_rounds(mean_train_fraction=...)`` ile kullan.
    """
    if not folds or n_rows <= 0:
        raise ValueError("Fold listesi bos veya satir sayisi gecersiz.")
    return float(np.mean([len(train_idx) for train_idx, _ in folds]) / n_rows)


def estimate_full_data_rounds(
    fold_best_iterations: Sequence[int],
    *,
    n_folds: int,
    safety: float = 1.0,
    mean_train_fraction: float | None = None,
) -> int:
    """CV'deki en iyi tur sayilarindan tam veri icin tur sayisi tahmin eder.

    Args:
        fold_best_iterations: Her fold'un ``best_iteration_`` degeri.
        n_folds: CV kat sayisi.
        safety: Ek carpan. 1.0 = formul aynen; 0.9 = biraz muhafazakar.
        mean_train_fraction: Fold'larin gercek ortalama train orani.
            **Bu repoda genellikle VERILMELIDIR** -- asagiya bak.

    Returns:
        Tam veri egitimi icin onerilen ``n_estimators``.

    Raises:
        ValueError: Bos liste, gecersiz ``n_folds`` veya gecersiz oran.

    ``(k-1)/k`` VARSAYIMI BU REPODA YANLISTIR (olculdu)
    ---------------------------------------------------
    Klasik formul ``n_estimators = ortalama_best_iter * (1 + 1/k)`` her fold'un
    verinin ``(k-1)/k``'sini gordugunu varsayar. Bu, KFold icin dogrudur ama
    bu repoda varsayilan sema ``purged_time_series_split``tir ve o **genisleyen
    pencere** kullanir: ilk fold az, son fold cok veri gorur.

    200 gun x 20 ilce panelde, n_splits=5, test_span=20 gun ile OLCULDU::

        fold train oranlari : 0.35  0.45  0.55  0.65  0.75
        ortalama            : 0.550        <- (k-1)/k = 0.800 DEGIL
        formulun carpani    : 1.200
        GEREKEN carpan      : 1.818        <- 1 / 0.550

    Yani agac sayisi yaklasik **%50 eksik** tahmin ediliyordu. Son gun
    egitilen model, CV'nin dogruladigi modelden sistematik olarak daha az
    egitilmis oluyordu.

    ``mean_train_fraction`` verildiginde carpan ``1/oran`` olur::

        rounds = estimate_full_data_rounds(
            extract_best_iterations(result.models),
            n_folds=len(folds),
            mean_train_fraction=fold_train_fraction(folds, len(train)),
        )
    """
    valid = [int(value) for value in fold_best_iterations if value and value > 0]
    if not valid:
        raise ValueError(
            "Fold'lardan en iyi tur sayisi alinamadi. Erken durdurma calisti mi? "
            "Tam veri egitimi icin tur sayisini elle ver."
        )
    # n_folds dogrulanmiyordu: k=0 ham ZeroDivisionError veriyordu, k=-3 ise
    # SESSIZCE 67 agac (carpan 0.667) donduruyordu -- yani tam veriyle egitilen
    # model CV'dekinden AZ agacla kaliyordu. (olculdu)
    if n_folds < 2:
        raise ValueError(
            f"n_folds en az 2 olmali, {n_folds} verildi. Tek fold'dan tam veri "
            "tur sayisi cikarilamaz."
        )

    if mean_train_fraction is None:
        scale = 1.0 + 1.0 / n_folds
    else:
        if not 0.0 < mean_train_fraction <= 1.0:
            raise ValueError(
                f"mean_train_fraction (0, 1] araliginda olmali, "
                f"{mean_train_fraction} verildi."
            )
        scale = 1.0 / mean_train_fraction

    mean_rounds = float(np.mean(valid))
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


def _align_test_columns(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Test kolonlarini train'in ADI VE SIRASINA gore hizalar.

    NEDEN KRITIK (olculdu)
    ----------------------
    GBDT'ler feature'lari KONUMA gore okur, ada gore degil. Kolon SAYISI ayni
    ama SIRA farkliysa hicbir kutuphane hata vermez -- tahminler sessizce
    baska bir modelden gelmis gibi olur:

        duz sirali  test -> tahmin ortalamasi -0.1166
        karisik sirali   -> tahmin ortalamasi  1.4243
        200/200 satirda farkli tahmin, satir basina 2.6366 mutlak fark

    Bu, "dogrulanan model" ile "gonderilen model"in ayni olmamasidir; feature
    muhendisligi adimlari kolon sirasini degistirmeye cok yatkindir (merge,
    concat, groupby.agg hepsi degistirebilir).

    Kolon SAYISI farkliysa LightGBM zaten hata veriyor -- sessiz olan yalnizca
    bu durum, ve tam da yakalanmasi en zor olani.
    """
    eksik = [column for column in train.columns if column not in test.columns]
    if eksik:
        raise ValueError(
            f"Test'te {len(eksik)} train kolonu yok: {eksik[:10]}. "
            "Tahmin aninda erisilemeyen bir feature uretilmis olabilir."
        )
    fazla = [column for column in test.columns if column not in train.columns]
    if fazla:
        raise ValueError(
            f"Test'te train'de olmayan {len(fazla)} kolon var: {fazla[:10]}. "
            "Hangisinin dislanacagini TAHMIN ETMIYORUZ -- acikca sec: "
            "test[train.columns]"
        )
    if list(test.columns) == list(train.columns):
        return test
    return test[list(train.columns)]


def _drop_round_aliases(
    kind: str, params: dict[str, Any], n_estimators: int, *, verbose: bool
) -> dict[str, Any]:
    """``n_estimators``i ezecek takma adlari temizler.

    ``n_estimators`` acik bir argumandir ve ``estimate_full_data_rounds``in
    hesabini tasir; ``params`` icinde unutulmus bir takma ad onu sessizce
    ezerse hem model yanlis egitilir hem ``RefitResult`` yanlis rapor verir.
    """
    temiz = dict(params)
    kaldirilan = {
        alias: temiz.pop(alias)
        for alias in _ROUND_ALIASES.get(kind, ())
        if alias in temiz
    }
    if kaldirilan and verbose:
        print(
            f"  UYARI: params icindeki {list(kaldirilan)} anahtar(lari) "
            f"n_estimators={n_estimators:,} degerini ezerdi; kaldirildi "
            f"(degerleri: {kaldirilan})."
        )
    return temiz


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

    # cross_validate bu kontrolu YAPIYORDU, refit YAPMIYORDU. Yani CV'nin
    # reddettigi hedef, submission'i ureten yoldan gecebiliyordu.
    # OLCULDU: inf iceren hedefle refit tahmin ortalamasi 2.24e+35 dondu ve
    # postprocess_predictions onu TEMIZLEMEDI -- max 1.69e+36 submission'a gitti.
    assert_finite_target(y)

    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("En az bir tohum gerekli.")
    # Tekrarli tohum, cok tohumlu ortalamanin TERSINI yapar: hicbir varyans
    # dusumu saglamadan ``seed_disagreement=0`` uretir ve summary() "KARARLI"
    # der. OLCULDU: seeds=[7,7,7,7,7] -> sapma 0.000000 [KARARLI];
    # seeds=[1..5] -> 0.069315. Yani en tehlikeli girdi en guven verici raporu
    # uretiyordu.
    if len(set(seed_list)) != len(seed_list):
        tekrarlar = sorted({s for s in seed_list if seed_list.count(s) > 1})
        raise ValueError(
            f"Tohumlar benzersiz olmali; tekrar edenler: {tekrarlar}. "
            "Ayni tohumla birden fazla egitim varyansi DUSURMEZ ama "
            "seed_disagreement'i sifira cekip 'KARARLI' raporu uretir."
        )

    if test is None:
        raise ValueError("Tam veri egitimi icin test kumesi zorunludur.")

    test_aligned = _align_test_columns(train, test)
    train_ready, test_ready, categorical = _prepare_categoricals(
        train, test_aligned, kind
    )
    if test_ready is None:  # pragma: no cover -- test None degil, buraya gelinmez
        raise ValueError("Tam veri egitimi icin test kumesi zorunludur.")

    # CV'de calisan parametre sozlugu burada da calismali. cross_validate
    # merge_infrastructure_params cagiriyordu, refit CAGIRMIYORDU.
    # OLCULDU: Optuna tipi bir xgboost sozlugu CV'den GECTI ama refit'te
    # "DataFrame.dtypes for data must be int, float, bool or category" ile
    # coktu -- eksik olan tek sey enable_categorical=True idi.
    base_params = merge_infrastructure_params(kind, params)
    base_params = _drop_round_aliases(kind, base_params, n_estimators, verbose=verbose)

    started = time.perf_counter()
    per_seed: list[np.ndarray] = []
    models: list[Any] = []

    for index, seed in enumerate(seed_list, start=1):
        seeded = dict(base_params)
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
            print(f"  tohum {index}/{len(seed_list)} (seed={seed}) tamam")

    stacked = np.vstack(per_seed)
    result = RefitResult(
        predictions=stacked.mean(axis=0),
        per_seed_predictions=stacked,
        seeds=seed_list,
        n_estimators=n_estimators,
        elapsed_seconds=time.perf_counter() - started,
        models=models,
    )

    if verbose:
        print(result.summary())

    return result
