"""Harici katalog verisi testleri: deprem, yangin, turizm + EPDK sinifi.

Parquet testleri GERCEK indirilmis veri uzerinde calisir (data/external).
Veri yoksa o testler atlanir -- taze klonda kirilmasin, ama yerelde veri
varsa gercek dosya semasi dogrulanmis olsun (test_weather_spatial deseni).

EPDK bolge sinifi testleri veri indirmeye bagli DEGILDIR ve her zaman
calisir (yalnizca data/reference tablosunu kullanir, o repo icindedir).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridup.features.demografi import (
    KENTALTI_NUFUS_ESIGI,
    KENTSEL_NUFUS_ESIGI,
    epdk_bolge_sinifi,
)

_VERI = Path(__file__).resolve().parents[1] / "data"
DEPREM_YOLU = _VERI / "external" / "depremler.parquet"
YANGIN_YOLU = _VERI / "external" / "yanginlar.parquet"
TURIZM_YOLU = _VERI / "external" / "turizm_geceleme.parquet"
REFERANS_YOLU = _VERI / "reference" / "ilceler_gdz_adm.parquet"

#: Ege sinir kutusu -- fetch_deprem.py / fetch_yangin.py ile ayni degerler.
LAT_MIN, LAT_MAX = 36.0, 39.5
LON_MIN, LON_MAX = 26.0, 30.5


# --------------------------------------------------------------------------
# EPDK bolge sinifi (epdk_bolge_sinifi) -- her zaman calisir
# --------------------------------------------------------------------------


class TestEpdkBolgeSinifi:
    def test_esikler_tam_sinirda(self):
        """Resmi esikler KAPALI alt sinirdir: 50.000 kentsel, 2.000 kentalti."""
        assert epdk_bolge_sinifi(50_000) == "kentsel"
        assert epdk_bolge_sinifi(49_999) == "kentalti"
        assert epdk_bolge_sinifi(2_000) == "kentalti"
        assert epdk_bolge_sinifi(1_999) == "kirsal"

    def test_sabitler_yonetmelikle_ayni(self):
        assert KENTSEL_NUFUS_ESIGI == 50_000
        assert KENTALTI_NUFUS_ESIGI == 2_000

    def test_series_girdi_series_cikti(self):
        nufus = pd.Series([100, 5_000, 80_000], index=["a", "b", "c"])
        sonuc = epdk_bolge_sinifi(nufus)
        assert isinstance(sonuc, pd.Series)
        assert list(sonuc.index) == ["a", "b", "c"]
        assert sonuc.tolist() == ["kirsal", "kentalti", "kentsel"]

    def test_dizi_girdi_dizi_cikti(self):
        sonuc = epdk_bolge_sinifi(np.array([1_999, 2_000, 50_000]))
        assert isinstance(sonuc, np.ndarray)
        assert sonuc.tolist() == ["kirsal", "kentalti", "kentsel"]

    def test_nan_ve_negatif_reddedilir(self):
        """Eksik nufusu sessizce 'kirsal'a dusurmek yanlis sinif uretirdi."""
        with pytest.raises(ValueError, match="NaN"):
            epdk_bolge_sinifi(pd.Series([1000.0, np.nan]))
        with pytest.raises(ValueError, match="[Nn]egatif"):
            epdk_bolge_sinifi(-5)

    def test_referans_ilcelerde_siniflar(self):
        """96 ilcelik GDZ/ADM referansinda kentsel VE kentalti cikmali.

        'kirsal' ilce bazinda CIKMAZ: en kucuk ilce nufusu 5.266 (olculdu),
        yani her ilce kentalti esiginin (2.000) ustundedir. Yonetmelik esigi
        yerlesim-yeri nufusuna gore tanimlar; ilce toplami bizim panelde
        kullanilabilir tek granulerdir (bkz. demografi.py docstring).
        """
        referans = pd.read_parquet(REFERANS_YOLU)
        siniflar = epdk_bolge_sinifi(referans["nufus"])
        assert {"kentsel", "kentalti"} <= set(siniflar.unique())
        assert len(siniflar) == len(referans)
        # Sinif dagilimi mantikli olmali: iki sinif da onemsiz azinlik degil.
        sayimlar = siniflar.value_counts()
        assert sayimlar["kentsel"] >= 10
        assert sayimlar["kentalti"] >= 10


# --------------------------------------------------------------------------
# Deprem katalogu (data/external/depremler.parquet)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def depremler():
    if not DEPREM_YOLU.exists():
        pytest.skip("Deprem katalogu yok: scripts/fetch_deprem.py calistir")
    return pd.read_parquet(DEPREM_YOLU)


class TestDepremKatalogu:
    def test_sema(self, depremler):
        gerekli = {"tarih", "lat", "lon", "buyukluk", "derinlik_km"}
        assert gerekli <= set(depremler.columns)
        assert len(depremler) > 0

    def test_tarihler_aralikta(self, depremler):
        tarihler = pd.to_datetime(depremler["tarih"])
        assert tarihler.min() >= pd.Timestamp("2020-01-01")
        assert tarihler.max() <= pd.Timestamp.today()

    def test_koordinatlar_kutuda(self, depremler):
        assert depremler["lat"].between(LAT_MIN, LAT_MAX).all()
        assert depremler["lon"].between(LON_MIN, LON_MAX).all()

    def test_buyuklukler_makul(self, depremler):
        """M>=4 istendi; Ege'de 8 ustu tarihsel olarak gorulmedi."""
        assert depremler["buyukluk"].between(4.0, 8.0).all()

    def test_derinlik_pozitif(self, depremler):
        assert (depremler["derinlik_km"].dropna() >= 0).all()


# --------------------------------------------------------------------------
# Yangin sicak noktalari (data/external/yanginlar.parquet)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def yanginlar():
    if not YANGIN_YOLU.exists():
        pytest.skip("Yangin verisi yok: scripts/fetch_yangin.py calistir")
    return pd.read_parquet(YANGIN_YOLU)


class TestYanginVerisi:
    def test_sema(self, yanginlar):
        """FIRMS sicak nokta semasi -- alan_ha YOK, bu bilincli (bkz. betik)."""
        gerekli = {"tarih", "lat", "lon", "frp", "aygit"}
        assert gerekli <= set(yanginlar.columns)
        assert len(yanginlar) > 0

    def test_tarihler_aralikta(self, yanginlar):
        tarihler = pd.to_datetime(yanginlar["tarih"])
        assert tarihler.min() >= pd.Timestamp("2020-01-01")
        assert tarihler.max() <= pd.Timestamp.today()

    def test_koordinatlar_kutuda(self, yanginlar):
        assert yanginlar["lat"].between(LAT_MIN, LAT_MAX).all()
        assert yanginlar["lon"].between(LON_MIN, LON_MAX).all()

    def test_frp_negatif_degil(self, yanginlar):
        assert (yanginlar["frp"].dropna() >= 0).all()

    def test_marmaris_2021_gorunuyor(self, yanginlar):
        """2021 Marmaris/Mugla yangini (docs/10 bolum 5) veride iz birakmali."""
        tarihler = pd.to_datetime(yanginlar["tarih"])
        agustos_2021 = yanginlar[
            (tarihler >= "2021-07-28") & (tarihler <= "2021-08-15")
        ]
        assert len(agustos_2021) > 100


# --------------------------------------------------------------------------
# Turizm geceleme (data/external/turizm_geceleme.parquet)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def turizm():
    if not TURIZM_YOLU.exists():
        pytest.skip("Turizm verisi yok: scripts/fetch_turizm.py calistir")
    return pd.read_parquet(TURIZM_YOLU)


class TestTurizmGeceleme:
    def test_sema(self, turizm):
        gerekli = {"yil", "il", "ilce", "il_key", "ilce_key", "geceleme", "tesise_gelis"}
        assert gerekli <= set(turizm.columns)
        assert len(turizm) > 0

    def test_yillar_makul(self, turizm):
        assert turizm["yil"].between(2020, pd.Timestamp.today().year).all()
        # En az iki yil olmali ki yillar arasi interpolasyon mumkun olsun.
        assert turizm["yil"].nunique() >= 2

    def test_yalnizca_hedef_iller(self, turizm):
        assert set(turizm["il_key"].unique()) <= {
            "izmir", "manisa", "aydin", "denizli", "mugla"
        }

    def test_il_toplami_satirlari_yok(self, turizm):
        """'Toplam' ara satirlari ilce degildir; join'e sizmamali."""
        assert not turizm["ilce_key"].str.contains("toplam").any()

    def test_degerler_negatif_degil(self, turizm):
        assert (turizm["geceleme"] >= 0).all()
        assert (turizm["tesise_gelis"].dropna() >= 0).all()

    def test_mugla_turizm_agirligi(self, turizm):
        """docs/10: Mugla yaz nufusu 2-5 kat -- geceleme lideri Mugla olmali."""
        il_toplam = turizm.groupby("il_key")["geceleme"].sum()
        assert il_toplam.idxmax() == "mugla"

    def test_ilce_key_referansla_eslesiyor(self, turizm):
        """ilce_key referans tablonun anahtar bicimiyle buyuk oranda eslesmeli.

        KTB bazi turizm alt-bolgelerini ayri satir yazar (or. Alsancak);
        yuzde yuz eslesme beklenmez ama cogunluk eslesmeli.
        """
        referans = pd.read_parquet(REFERANS_YOLU)
        eslesen = turizm["ilce_key"].isin(set(referans["ilce_key"]))
        assert eslesen.mean() > 0.7
