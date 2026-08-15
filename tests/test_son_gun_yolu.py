"""SIZINTI AVI TUR 6 -- SON GUN YOLU (``refit.py``).

NEDEN AYRI BIR SIZINTI SINIFI
-----------------------------
Onceki turlar CV'nin kendisine bakti. Bu tur farkli bir soru soruyor:
**CV dogru olsa bile, submission'i UREten yol ayni sey mi yapiyor?**

``cross_validate`` ile ``multi_seed_refit`` iki ayri kod yolu. Biri
dogruluyor, digeri gonderiyor. Aralarindaki her fark, "dogrulanan model"
ile "gonderilen model"in ayni olmamasi demektir -- ve bu, klasik sizintiyla
ayni sonucu verir: CV guzel, leaderboard berbat.

Her test ``t6_repro.py`` ile ONCE-SONRA olculmus bir bulguya karsilik gelir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.models import cross_validate, starter_params
from gridup.refit import (
    estimate_full_data_rounds,
    fold_train_fraction,
    multi_seed_refit,
)
from gridup.validation import purged_time_series_split

PARAMS = starter_params("lightgbm", "regression")


def _veri(n: int = 400, m: int = 200, tohum: int = 0):
    rng = np.random.default_rng(tohum)
    train = pd.DataFrame(
        {
            "a": rng.normal(size=n), "b": rng.normal(size=n),
            "c": rng.normal(size=n), "grup": rng.choice(["x", "y", "z"], n),
        }
    )
    test = pd.DataFrame(
        {
            "a": rng.normal(size=m), "b": rng.normal(size=m),
            "c": rng.normal(size=m), "grup": rng.choice(["x", "y", "z"], m),
        }
    )
    y = (train["a"] * 2 + train["b"] - train["c"] + rng.normal(0, 0.3, n)).to_numpy()
    return train, test, y


# --------------------------------------------------------------------------
# B0 -- hedefte NaN/inf
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bozuk", [np.nan, np.inf])
def test_refit_bozuk_hedefi_cross_validate_ile_ayni_sekilde_reddediyor(bozuk):
    """OLCULDU: cross_validate DURUYOR, multi_seed_refit GECIYORDU.

    inf iceren hedefle refit tahmin ortalamasi 2.24e+35 dondu ve
    ``postprocess_predictions`` onu TEMIZLEMEDI -- max 1.69e+36 submission'a
    gidiyordu. Son gun, harcanmis bir submission demektir.
    """
    train, test, y = _veri()
    y = y.copy()
    y[7] = bozuk
    folds = [(np.arange(0, 300), np.arange(300, 400))]

    with pytest.raises(ValueError, match="NaN/sonsuz"):
        cross_validate(train, y, folds, kind="lightgbm", params=PARAMS, verbose=False)
    with pytest.raises(ValueError, match="NaN/sonsuz"):
        multi_seed_refit(
            train, y, test, params=PARAMS, n_estimators=50, seeds=[0], verbose=False
        )


# --------------------------------------------------------------------------
# B1 -- kolon sirasi
# --------------------------------------------------------------------------


def test_test_kolon_sirasi_degisince_tahmin_degismiyor():
    """OLCULDU: ayni kolonlar farkli sirada verilince 200/200 satirda FARKLI
    tahmin cikiyordu (ortalama -0.1166 -> 1.4243, satir basina 2.6366 fark).

    GBDT'ler feature'lari KONUMA gore okur, ada gore degil; kolon SAYISI ayni
    oldugu icin hicbir kutuphane hata vermiyordu. Feature muhendisligindeki
    her merge/concat/groupby.agg kolon sirasini degistirebilir.
    """
    train, test, y = _veri()
    duz = multi_seed_refit(
        train, y, test, params=PARAMS, n_estimators=80, seeds=[0], verbose=False
    )
    karisik = multi_seed_refit(
        train, y, test[["grup", "c", "a", "b"]],
        params=PARAMS, n_estimators=80, seeds=[0], verbose=False,
    )
    np.testing.assert_allclose(duz.predictions, karisik.predictions)


def test_test_te_eksik_kolon_acik_hata_veriyor():
    train, test, y = _veri()
    with pytest.raises(ValueError, match="train kolonu yok"):
        multi_seed_refit(
            train, y, test.drop(columns=["c"]),
            params=PARAMS, n_estimators=50, seeds=[0], verbose=False,
        )


def test_test_te_fazla_kolon_tahmin_edilmeden_hata_veriyor():
    """Hangisinin dislanacagini TAHMIN ETMIYORUZ -- kullanici acikca secmeli."""
    train, test, y = _veri()
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="train'de olmayan"):
        multi_seed_refit(
            train, y, test.assign(ekstra=rng.normal(size=len(test))),
            params=PARAMS, n_estimators=50, seeds=[0], verbose=False,
        )


# --------------------------------------------------------------------------
# B2 -- altyapi parametreleri
# --------------------------------------------------------------------------


def test_cv_de_calisan_tuned_params_refit_te_de_calisiyor():
    """OLCULDU: Optuna tipi bir xgboost sozlugu CV'den GECTI ama refit'te
    "DataFrame.dtypes for data must be int, float, bool or category" ile
    coktu. Eksik olan tek sey ``enable_categorical=True`` idi -- yani son gun,
    tam da CV'nin sectigi konfigurasyonla model egitilemiyordu."""
    train, test, y = _veri()
    folds = [(np.arange(0, 300), np.arange(300, 400))]
    tuned = {
        "n_estimators": 200, "learning_rate": 0.05, "max_depth": 5,
        "subsample": 0.8, "colsample_bytree": 0.7,
    }
    cross_validate(train, y, folds, kind="xgboost", params=tuned, verbose=False)
    sonuc = multi_seed_refit(
        train, y, test, kind="xgboost", params=tuned,
        n_estimators=100, seeds=[0], verbose=False,
    )
    assert np.isfinite(sonuc.predictions).all()


# --------------------------------------------------------------------------
# B4 -- tur sayisi takma adlari
# --------------------------------------------------------------------------


def test_params_icindeki_num_iterations_n_estimatorsu_ezmiyor():
    """OLCULDU: n_estimators=500 istendi, params'ta num_iterations=5 vardi ->
    GERCEK agac sayisi 5, RefitResult ise 500 raporladi.

    Yani ``estimate_full_data_rounds``in tum hesabi ciope gidiyor ve rapor
    bunu gizliyordu.
    """
    train, test, y = _veri()
    sonuc = multi_seed_refit(
        train, y, test, params={**PARAMS, "num_iterations": 5},
        n_estimators=500, seeds=[0], verbose=False,
    )
    assert sonuc.models[0].booster_.num_trees() == 500
    assert sonuc.n_estimators == 500


# --------------------------------------------------------------------------
# B5 -- tekrarli tohum
# --------------------------------------------------------------------------


def test_tekrarli_tohum_reddediliyor():
    """OLCULDU: seeds=[7,7,7,7,7] -> sapma 0.000000 ve summary() 'KARARLI'.
    seeds=[1..5] -> 0.069315. Yani en tehlikeli girdi EN GUVEN VERICI raporu
    uretiyordu: hicbir varyans dusumu olmadan mukemmel kararlilik."""
    train, test, y = _veri()
    with pytest.raises(ValueError, match="benzersiz olmali"):
        multi_seed_refit(
            train, y, test, params=PARAMS, n_estimators=50,
            seeds=[7, 7, 7], verbose=False,
        )


def test_farkli_tohumlar_gercek_sapma_uretiyor():
    """Ters yon: farkli tohumlar sifir olmayan bir sapma vermeli."""
    train, test, y = _veri()
    sonuc = multi_seed_refit(
        train, y, test, params=PARAMS, n_estimators=50,
        seeds=[1, 2, 3], verbose=False,
    )
    assert sonuc.seed_disagreement > 0.0


# --------------------------------------------------------------------------
# B6 / B7 -- tur sayisi tahmini
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_folds", [1, 0, -3])
def test_gecersiz_n_folds_reddediliyor(n_folds: int):
    """OLCULDU: k=0 ham ZeroDivisionError, k=-3 ise SESSIZCE 67 agac
    (carpan 0.667) -- yani tam veriyle egitilen model CV'dekinden AZ agac."""
    with pytest.raises(ValueError, match="n_folds en az 2"):
        estimate_full_data_rounds([100, 110, 90], n_folds=n_folds)


def test_genisleyen_pencerede_k_eksi_bir_bolu_k_varsayimi_yanlis():
    """OLCULDU (200 gun x 20 ilce panel, n_splits=5, test_span=20 gun):

        fold train oranlari : 0.35  0.45  0.55  0.65  0.75
        ortalama            : 0.550        <- (k-1)/k = 0.800 DEGIL
        formulun carpani    : 1.200  -> 120 agac
        olcume dayali       : 1.818  -> 182 agac  (%52 daha fazla)

    Son gun egitilen model, CV'nin dogruladigi modelden sistematik olarak
    daha AZ egitilmis oluyordu.
    """
    zaman = pd.Series(
        np.repeat(pd.date_range("2024-01-01", periods=200, freq="D"), 20)
    )
    folds = purged_time_series_split(
        zaman, embargo=pd.Timedelta(days=30), n_splits=5,
        test_span=pd.Timedelta(days=20), verbose=False,
    )
    oran = fold_train_fraction(folds, len(zaman))
    assert oran == pytest.approx(0.550, abs=0.01)
    assert oran < 0.8, "genisleyen pencere (k-1)/k'dan az veri gorur"

    varsayimla = estimate_full_data_rounds([100, 110, 90], n_folds=5)
    olcumle = estimate_full_data_rounds(
        [100, 110, 90], n_folds=5, mean_train_fraction=oran
    )
    assert varsayimla == 120
    assert olcumle == 182
    assert olcumle > varsayimla * 1.4


def test_gecersiz_train_orani_reddediliyor():
    for gecersiz in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="mean_train_fraction"):
            estimate_full_data_rounds(
                [100], n_folds=5, mean_train_fraction=gecersiz
            )
