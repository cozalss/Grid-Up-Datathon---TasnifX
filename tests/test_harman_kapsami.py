"""Harman/zoo OOF KAPSAMI regresyon testleri.

TEMA: ``purged_time_series_split`` ve ``TimeSeriesSplit`` ilk donemi HIC
dogrulamaz. O satirlarda OOF degeri 0.0'dir -- tahmin degil DOLGU. Dolgu
satirlari harmana girdiginde metrige sabit bir terim ekler: agirliklarin
argmin'i degismez ama RAPORLANAN SAYI yanlis olur ve deney gunlugu, juri
slaydi ve LB karsilastirmasi bunun uzerine kurulur.

Bu dosya iki seyi birden korur:
  1. ``ZooResult`` kapsam maskesini OLCEREK tasiyor mu (yeni davranis),
  2. ``stack_oof``un kapsamsiz fold atlama duzeltmesi hala yerinde mi
     (onceki turda kapatildi, burada bir kez daha kilitleniyor).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from gridup.ensemble import (
    greedy_forward_selection,
    hill_climb_weights,
    prune_by_correlation,
    stack_oof,
)
from gridup.metrics import rmse
from gridup.models import CVResult

# =============================================================================
# Yardimcilar -- model egitmeden kapsam senaryosu kurar
# =============================================================================


def _cv_sonucu(oof: np.ndarray, kapsam: np.ndarray | None) -> CVResult:
    """Elle kurulmus CVResult -- yalnizca kapsam alanlari anlamli."""
    return CVResult(
        oof_predictions=oof,
        test_predictions=None,
        fold_scores=[1.0],
        overall_score=1.0,
        feature_importance=pd.DataFrame({"feature": [], "importance": []}),
        oof_covered=np.zeros(0, dtype=bool) if kapsam is None else kapsam,
    )


def _kapsamsiz_uyeler(n: int = 1200, dolgu_orani: float = 0.25, seed: int = 3):
    """``(uyeler, y, kapsam)`` -- ilk ``dolgu_orani`` blok TUM uyelerde sifir."""
    rng = np.random.default_rng(seed)
    y = rng.normal(10.0, 3.0, n)
    sinir = int(n * dolgu_orani)
    kapsam = np.zeros(n, dtype=bool)
    kapsam[sinir:] = True

    uyeler = {}
    for ad, gurultu in (("m1", 1.0), ("m2", 1.4), ("m3", 2.0)):
        dizi = np.zeros(n)
        dizi[kapsam] = y[kapsam] + rng.normal(0, gurultu, kapsam.sum())
        uyeler[ad] = dizi
    return uyeler, y, kapsam


# =============================================================================
# ZooResult kapsam maskesi -- model egitmeden
# =============================================================================


class TestZooKapsamMaskesi:
    def test_kapsam_maskesi_uyelerin_kesisimidir(self):
        """Harman bir satirda TUM uyeleri birlikte kullanir -> KESISIM dogru olan.

        Birlesim alsaydik, tek bir uyenin dolgu oldugu satir harmana sizardi.
        """
        from gridup.zoo import ZooResult

        a = np.array([False, True, True, True])
        b = np.array([False, False, True, True])
        zoo = ZooResult(
            results={
                "a": _cv_sonucu(np.arange(4.0), a),
                "b": _cv_sonucu(np.arange(4.0), b),
            },
            metric="rmse",
            greater_is_better=False,
        )
        assert zoo.oof_covered.dtype == bool
        assert np.array_equal(zoo.oof_covered, np.array([False, False, True, True]))
        assert zoo.coverage == pytest.approx(0.5)

    def test_covered_oof_matrix_indeks_ve_kirpilmis_dizi_dondurur(self):
        """``CVResult.covered_predictions()`` ile ayni desen: (indeks, tahminler)."""
        from gridup.zoo import ZooResult

        maske = np.array([False, False, True, True])
        zoo = ZooResult(
            results={"a": _cv_sonucu(np.array([0.0, 0.0, 7.0, 9.0]), maske)},
            metric="rmse",
            greater_is_better=False,
        )
        indeks, kapsamli = zoo.covered_oof_matrix()
        assert np.array_equal(indeks, np.array([2, 3]))
        assert np.array_equal(kapsamli["a"], np.array([7.0, 9.0]))

    def test_maskesiz_eski_cvresult_kapsami_daraltmaz(self):
        """YANLIS-POZITIF KORUMASI: elle kurulmus CVResult'ta maske bostur.

        O uyeyi 'hicbir satiri kapsamiyor' saymak, masum bir kosuyu bos
        kesisimle patlatirdi.
        """
        from gridup.zoo import ZooResult

        zoo = ZooResult(
            results={
                "eski": _cv_sonucu(np.arange(4.0), None),
                "yeni": _cv_sonucu(np.arange(4.0), np.array([False, True, True, True])),
            },
            metric="rmse",
            greater_is_better=False,
        )
        assert zoo.coverage == pytest.approx(0.75)

    def test_farkli_uzunlukta_uye_hata_firlatir(self):
        """Farkli uzunluk = ayni fold'larla uretilmemis. Kirpmak sizinti olurdu."""
        from gridup.zoo import ZooResult

        zoo = ZooResult(
            results={
                "a": _cv_sonucu(np.arange(4.0), np.ones(4, dtype=bool)),
                "b": _cv_sonucu(np.arange(3.0), np.ones(3, dtype=bool)),
            },
            metric="rmse",
            greater_is_better=False,
        )
        with pytest.raises(ValueError, match="OOF uzunluklari farkli"):
            _ = zoo.oof_covered

    def test_bos_kesisim_acik_hata_verir(self):
        """Ortak kapsam bossa harman skoru TANIMSIZDIR -- sessizce bos dondurme."""
        from gridup.zoo import ZooResult

        zoo = ZooResult(
            results={
                "a": _cv_sonucu(np.arange(4.0), np.array([True, True, False, False])),
                "b": _cv_sonucu(np.arange(4.0), np.array([False, False, True, True])),
            },
            metric="rmse",
            greater_is_better=False,
        )
        with pytest.raises(ValueError, match="Hicbir satir tum uyelerce kapsanmiyor"):
            zoo.covered_oof_matrix()

    def test_summary_kapsam_notunu_yalnizca_eksik_kapsamda_basar(self):
        """YANLIS-POZITIF KORUMASI: tam kapsamda gereksiz uyari cikmamali."""
        from gridup.zoo import ZooResult

        eksik = ZooResult(
            results={"a": _cv_sonucu(np.arange(4.0), np.array([False, True, True, True]))},
            metric="rmse",
            greater_is_better=False,
        )
        tam = ZooResult(
            results={"a": _cv_sonucu(np.arange(4.0), np.ones(4, dtype=bool))},
            metric="rmse",
            greater_is_better=False,
        )
        assert "Ortak OOF kapsami" in eksik.summary()
        assert "Ortak OOF kapsami" not in tam.summary()


# =============================================================================
# ensemble: dolgu satirlari artik SESSIZ gecmiyor
# =============================================================================


class TestDolguUyarisi:
    def test_hill_climb_dolgu_satirlarinda_uyarir(self):
        """Kapsam disi satirlar skoru sisirir -- kullaniciya SOYLENMELI.

        OLCULDU (gercek zoo, TimeSeriesSplit(4), N=3000, kapsam %80):
        maskesiz rmse 2.754756 / maskeli 2.213196 -- %24.5 sapma, tamamen sessiz.
        """
        uyeler, y, _ = _kapsamsiz_uyeler()
        with pytest.warns(UserWarning, match="TUM uyeler tam sifir"):
            hill_climb_weights(uyeler, y, metric="rmse", verbose=False)

    def test_prune_ve_greedy_de_uyarir(self):
        """Ayni girdi uc harman kapisindan da girebilir; ucu de konusmali."""
        uyeler, y, _ = _kapsamsiz_uyeler()
        with pytest.warns(UserWarning, match="TUM uyeler tam sifir"):
            prune_by_correlation(uyeler, y, metric="rmse")
        with pytest.warns(UserWarning, match="TUM uyeler tam sifir"):
            greedy_forward_selection(uyeler, y, metric="rmse", max_models=4, verbose=False)

    def test_tam_kapsamda_uyari_yok(self):
        """YANLIS-POZITIF KORUMASI: dolgu satiri yoksa hicbir uyari cikmamali."""
        uyeler, y, kapsam = _kapsamsiz_uyeler()
        kapsamli = {ad: dizi[kapsam] for ad, dizi in uyeler.items()}
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            hill_climb_weights(kapsamli, y[kapsam], metric="rmse", verbose=False)
            prune_by_correlation(kapsamli, y[kapsam], metric="rmse")

    def test_tek_uyede_sifir_olan_satir_dolgu_sayilmaz(self):
        """YANLIS-POZITIF KORUMASI: gercek bir tahmin tesadufen 0.0 olabilir.

        Dolgu isareti ancak TUM uyeler ayni satirda tam sifirsa gecerlidir.
        """
        n = 300
        rng = np.random.default_rng(11)
        y = rng.normal(5.0, 2.0, n)
        a = y + rng.normal(0, 0.5, n)
        b = y + rng.normal(0, 0.7, n)
        a[7] = 0.0  # yalnizca bir uyede sifir
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            hill_climb_weights({"a": a, "b": b}, y, metric="rmse", verbose=False)

    def test_maskeli_harman_skoru_maskesizden_dusuk_cikar(self):
        """Sifir dolgusu metrige sabit bir terim ekler; sapma tek yonludur.

        OLCULDU (elle kurulmus %25 dolgu): maskesiz rmse 5.3343 ->
        maskeli 0.7305 (7.30 kat). Gercek zoo kosusunda sisme daha kucuk ama
        ayni yonde: 2.754756 -> 2.213196 (%24.5).
        """
        uyeler, y, kapsam = _kapsamsiz_uyeler()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            agirlik_ham = hill_climb_weights(uyeler, y, metric="rmse", verbose=False)
        skor_ham = float(rmse(y, sum(agirlik_ham[a] * uyeler[a] for a in uyeler)))

        kapsamli = {ad: dizi[kapsam] for ad, dizi in uyeler.items()}
        agirlik = hill_climb_weights(kapsamli, y[kapsam], metric="rmse", verbose=False)
        skor = float(rmse(y[kapsam], sum(agirlik[a] * kapsamli[a] for a in kapsamli)))

        assert skor < skor_ham, "maskeleme skoru duzeltmedi"
        assert skor_ham / skor > 1.2, f"beklenen sisme gorunmuyor: {skor_ham} vs {skor}"


# =============================================================================
# stack_oof: onceki turda kapatilan P0 hala kapali mi
# =============================================================================


class TestStackOofKapsamiKapali:
    def test_kapsamsiz_fold_atlaniyor_ve_katsayilar_degenere_degil(self):
        """REGRESYON: fold-1'in meta-egitiminin %100'u sifir dolgusuydu.

        OLCULDU (maskesiz ESKI yol, gercek zoo + TimeSeriesSplit(4), N=3000):
        fold-1 Ridge katsayilari [0. 0. 0.], fold-1 test tahmin std 1.388e-17.
        Mevcut kod fold'u atlar: test RMSE 2.261405 -> 2.061546.
        """
        uyeler, y, _ = _kapsamsiz_uyeler(n=1200, dolgu_orani=0.25)
        folds = [
            (np.arange(0, 300), np.arange(300, 600)),
            (np.arange(0, 600), np.arange(600, 900)),
            (np.arange(0, 900), np.arange(900, 1200)),
        ]
        with warnings.catch_warnings(record=True) as yakalanan:
            warnings.simplefilter("always")
            sonuc = stack_oof(uyeler, y, folds, metric="rmse", verbose=False)

        assert any("ATLANDI" in str(u.message) for u in yakalanan), "atlama sessiz kaldi"
        katsayilar = np.array(list(sonuc["coefficients"].values()))
        assert np.any(np.abs(katsayilar) > 0.05), f"katsayilar dejenere: {katsayilar}"

    def test_zoo_maskesi_base_covered_olarak_verilebiliyor(self):
        """ZooResult.oof_covered tespit sezgisinin yerini alabilmeli.

        Maske OLCULMUS bilgidir; tam-sifir sezgisi yalnizca son savunmadir.
        """
        from gridup.zoo import ZooResult

        uyeler, y, kapsam = _kapsamsiz_uyeler(n=1200, dolgu_orani=0.25)
        folds = [
            (np.arange(0, 600), np.arange(600, 900)),
            (np.arange(0, 900), np.arange(900, 1200)),
        ]
        zoo = ZooResult(
            results={ad: _cv_sonucu(dizi, kapsam) for ad, dizi in uyeler.items()},
            metric="rmse",
            greater_is_better=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sonuc = stack_oof(
                uyeler, y, folds, base_covered=zoo.oof_covered,
                metric="rmse", verbose=False,
            )
        assert np.isfinite(sonuc["score"])
        assert sonuc["coverage"] == pytest.approx(0.5)


# =============================================================================
# Gercek zoo kosusu -- olculen sayilarin kaynagi
# =============================================================================


@pytest.fixture(scope="module")
def zoo_ve_veri():
    """TimeSeriesSplit(4), N=3000, 3 uye -> ortak kapsam %80.

    Modul kapsamli: uc modelin egitimi bir kez kosar (olculdu: ~0.6 sn).
    """
    from sklearn.model_selection import TimeSeriesSplit

    from gridup.models import starter_params
    from gridup.zoo import ZooEntry, make_model_zoo

    n = 3000
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n),
    })
    y = (3 * X["a"] - 2 * X["b"] + 0.5 * X["c"] + rng.normal(0, 2.0, n)).to_numpy()
    folds = list(TimeSeriesSplit(n_splits=4).split(X))

    lgb_a = starter_params("lightgbm", "regression")
    lgb_a["n_estimators"] = 80
    lgb_b = starter_params("lightgbm", "regression")
    lgb_b.update({"n_estimators": 80, "num_leaves": 15, "learning_rate": 0.08})
    xgb = starter_params("xgboost", "regression")
    xgb["n_estimators"] = 80

    zoo = make_model_zoo(
        X, y, folds,
        entries=[
            ZooEntry("lgbm_a", "lightgbm", lgb_a),
            ZooEntry("lgbm_b", "lightgbm", lgb_b),
            ZooEntry("xgb", "xgboost", xgb),
        ],
        metric="rmse", early_stopping_rounds=20, verbose=False,
    )
    return zoo, y


@pytest.mark.slow
class TestGercekZooKapsami:
    def test_kapsam_timeseries_ilk_blogunu_disliyor(self, zoo_ve_veri):
        """OLCULDU: TimeSeriesSplit(4) -> kapsam 0.8000 (3000 satirin 2400'u)."""
        zoo, _ = zoo_ve_veri
        assert zoo.coverage == pytest.approx(0.80)
        assert zoo.oof_covered.sum() == 2400
        assert not zoo.oof_covered[:600].any(), "ilk blok kapsanmis gorunuyor"

    def test_maskesiz_harman_skoru_sisiyor(self, zoo_ve_veri):
        """OLCULDU: sablon (maskesiz) rmse 2.754756 -> maskeli 2.213196, %24.5.

        Agirliklar BIREBIR ayni cikar (xgb 0.6452 / lgbm_b 0.3548) -- bozulan
        sadece raporlanan sayidir, secim degil.
        """
        zoo, y = zoo_ve_veri
        ham = zoo.oof_matrix
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            secilen_ham = prune_by_correlation(ham, y, metric="rmse", max_members=5)
            ag_ham = hill_climb_weights(
                {k: ham[k] for k in secilen_ham}, y, metric="rmse", verbose=False
            )
        skor_ham = float(rmse(y, sum(ag_ham[k] * ham[k] for k in secilen_ham)))

        indeks, kapsamli = zoo.covered_oof_matrix()
        y_k = y[indeks]
        secilen = prune_by_correlation(kapsamli, y_k, metric="rmse", max_members=5)
        ag = hill_climb_weights(
            {k: kapsamli[k] for k in secilen}, y_k, metric="rmse", verbose=False
        )
        skor = float(rmse(y_k, sum(ag[k] * kapsamli[k] for k in secilen)))

        assert skor == pytest.approx(2.213196, abs=0.05)
        assert skor_ham == pytest.approx(2.754756, abs=0.05)
        assert skor_ham > skor

    def test_leaderboard_skoru_zaten_maskeliydi(self, zoo_ve_veri):
        """YANLIS-POZITIF KORUMASI: CVResult.overall_score BOZULMAMIS olmali.

        OLCULDU: xgb 2.229975, lgbm_b 2.268656, lgbm_a 2.787499 -- hepsi
        kapsanan satirlar uzerinden; harman skoru (2.2132) ile ayni mertebede.
        """
        zoo, _ = zoo_ve_veri
        tablo = zoo.leaderboard()
        assert tablo["skor"].max() < 3.0, f"leaderboard sifirlarla bozulmus: {tablo}"
        assert set(tablo["model"]) == {"lgbm_a", "lgbm_b", "xgb"}

    def test_correlation_kapsanan_satirlari_kullaniyor(self, zoo_ve_veri):
        """OLCULDU: maskesiz 0.885434 -> maskeli 0.885381 (lgbm_a-xgb).

        Fark kucuk ama SISTEMATIK ve hep ayni yonde: dolgu satirlari tum
        uyelerde ayni sabit oldugu icin korelasyonu yukari ceker.
        """
        zoo, _ = zoo_ve_veri
        _, kapsamli = zoo.covered_oof_matrix()
        beklenen = pd.DataFrame(kapsamli).corr()
        alinan = zoo.correlation()
        assert alinan.loc["lgbm_a", "xgb"] == pytest.approx(
            beklenen.loc["lgbm_a", "xgb"], abs=1e-12
        )
        assert alinan.loc["lgbm_a", "xgb"] < pd.DataFrame(zoo.oof_matrix).corr().loc[
            "lgbm_a", "xgb"
        ]

    def test_summary_kapsam_uyarisini_basiyor(self, zoo_ve_veri):
        """Kapsam %80 iken kullanici bunu EKRANDA gormeli."""
        zoo, _ = zoo_ve_veri
        metin = zoo.summary()
        assert "Ortak OOF kapsami: %80.0" in metin
        assert "covered_oof_matrix()" in metin
