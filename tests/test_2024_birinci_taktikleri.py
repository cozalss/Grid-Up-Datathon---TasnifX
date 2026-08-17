"""2024 GDZ BIRINCISININ (PIKACHOW) TAKTIKLERINDEN AKTARILAN OZELLIKLER.

NEDEN BU DOSYA
--------------
2024 GDZ Elektrik Datathon'unun gorevi 2026 Grid Up ile ayni ailedendir:
ilce bazli gunluk plansiz kesinti SAYISI, metrik MAE. Birincinin final
sunumu (Pikachow -- Anil Ozturk, Ahmet Tarik Karakas; anilozturk.net'te
halka acik, 29 slayt) satir satir incelendi ve pipeline'imizda OLMAYAN
teknikler buraya tasindi:

  * son tam ay istatistikleri (s.18 + s.26: last_month_same_day_* feature'lari
    importance listesinin EN USTUNDEYDI)
  * ileri bakisli tatil pencereleri (s.18: "ileriki 3-7-15 gun icinde bayram")
  * sqrt hedef donusumu (Rohlik Sales v2'nin 2. ve 3.'sunden, bagimsiz cifte
    kanit: sqrt+L2, ham MAE'yi VE yerli Tweedie'yi gecti)

Ayrica bir ATIF DUZELTMESININ bekcisi: onceki bir denetim "GDZ 2024 yarismasi
yok" diyerek DOGRU atiflari silmisti (404 != yok). Sunum PDF'i + coderspace.io
etkinlik sayfasi + kaggle.com/competitions/gdz-elektrik-datathon-2024 ile
yeniden dogrulandi; bu dosya geri donusu engeller.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.features.temporal import (
    MISSING_HOLIDAY_DISTANCE,
    add_previous_month_features,
    add_upcoming_holiday_features,
)
from gridup.metrics import inverse_sqrt_transform, sqrt_transform_target

# --------------------------------------------------------------------------
# Son tam ay istatistikleri
# --------------------------------------------------------------------------


def _iki_ayli_panel() -> pd.DataFrame:
    """Ocak (31 gun) + Subat (28 gun) 2023, tek ilce, deterministik degerler."""
    gunler = pd.date_range("2023-01-01", "2023-02-28", freq="D")
    # Ocak: gun numarasi ciftse 0, tekse gun numarasi. Subat: hep 5.
    degerler = [
        0.0 if gun.day % 2 == 0 else float(gun.day) if gun.month == 1 else 5.0 for gun in gunler
    ]
    return pd.DataFrame({"gun": gunler, "ilce": "bornova", "kesinti": degerler})


def test_subat_satirlari_ocak_istatistiklerini_aliyor():
    """horizon=1 = Pikachow'un 'gecen ay'i: Subat'in HER gunu ayni Ocak ozetini tasir."""
    panel = _iki_ayli_panel()
    sonuc = add_previous_month_features(
        panel, "kesinti", time_column="gun", horizon=1, group_columns=["ilce"]
    )

    subat = sonuc[sonuc["gun"].dt.month == 2]
    ocak_degerleri = panel.loc[panel["gun"].dt.month == 1, "kesinti"]

    # Ocak: tek gunler 1,3,...,31 -> toplam 256; cift gunler (15 adet) sifir.
    assert subat["kesinti_sontamay_toplam"].nunique() == 1
    assert subat["kesinti_sontamay_toplam"].iloc[0] == pytest.approx(float(ocak_degerleri.sum()))
    assert subat["kesinti_sontamay_max"].iloc[0] == pytest.approx(31.0)
    assert subat["kesinti_sontamay_olayli_gun"].iloc[0] == pytest.approx(16.0)
    assert subat["kesinti_sontamay_olaysiz_gun"].iloc[0] == pytest.approx(15.0)


def test_ayni_gun_feature_gecen_ayin_ayni_gununu_tasiyor():
    """s.26'daki last_month_same_day: 14 Subat satiri 14 Ocak'in degerini gorur."""
    panel = _iki_ayli_panel()
    sonuc = add_previous_month_features(
        panel, "kesinti", time_column="gun", horizon=1, group_columns=["ilce"]
    )
    on_dort_subat = sonuc[sonuc["gun"] == "2023-02-14"].iloc[0]
    # 14 Ocak cift gun -> deger 0.
    assert on_dort_subat["kesinti_sontamay_ayni_gun"] == pytest.approx(0.0)
    on_bes_subat = sonuc[sonuc["gun"] == "2023-02-15"].iloc[0]
    assert on_bes_subat["kesinti_sontamay_ayni_gun"] == pytest.approx(15.0)


def test_ilk_ay_gecmisi_olmadigi_icin_nan():
    panel = _iki_ayli_panel()
    sonuc = add_previous_month_features(
        panel, "kesinti", time_column="gun", horizon=1, group_columns=["ilce"]
    )
    ocak = sonuc[sonuc["gun"].dt.month == 1]
    assert ocak["kesinti_sontamay_toplam"].isna().all()


def test_olmayan_ay_gunu_nan():
    """31 Mart 'gecen ayin 31'ini' ister -- Subat 31 yoktur, NaN olmali."""
    gunler = pd.date_range("2023-02-01", "2023-03-31", freq="D")
    panel = pd.DataFrame({"gun": gunler, "kesinti": 1.0})
    sonuc = add_previous_month_features(panel, "kesinti", time_column="gun", horizon=1)
    otuz_bir_mart = sonuc[sonuc["gun"] == "2023-03-31"].iloc[0]
    assert np.isnan(otuz_bir_mart["kesinti_sontamay_ayni_gun"])
    # Ozet istatistikler yine de dolu: Subat mevcut.
    assert otuz_bir_mart["kesinti_sontamay_toplam"] == pytest.approx(28.0)


def test_ufuk_ay_sinirini_asarken_tamamlanmamis_ayi_kullanmiyor():
    """SIZINTI DISIPLINI: ufuk 31 gunken 1 Mart, Ocak'i bile goremez.

    1 Mart - 31 gun = 29 Ocak -> Ocak HENUZ TAMAMLANMAMIS sayilir (31 Ocak
    gorulmedi), son tam ay ARALIK'tir. Aralik veride yok -> NaN.
    31 Mart - 31 gun = 28 Subat -> Subat tamamlanmis, Subat kullanilir.
    'Gecen ay' kisayolu burada CV'de gorunmeyen dogrudan sizinti olurdu.
    """
    gunler = pd.date_range("2023-01-01", "2023-03-31", freq="D")
    panel = pd.DataFrame({"gun": gunler, "kesinti": 2.0})
    sonuc = add_previous_month_features(panel, "kesinti", time_column="gun", horizon=31)

    bir_mart = sonuc[sonuc["gun"] == "2023-03-01"].iloc[0]
    assert np.isnan(bir_mart["kesinti_ufuk31_sontamay_toplam"])

    otuz_bir_mart = sonuc[sonuc["gun"] == "2023-03-31"].iloc[0]
    assert otuz_bir_mart["kesinti_ufuk31_sontamay_toplam"] == pytest.approx(56.0)


def test_gruplar_birbirine_karismiyor():
    gunler = pd.date_range("2023-01-01", "2023-02-28", freq="D")
    parcalar = []
    for ilce, deger in (("bornova", 1.0), ("menemen", 10.0)):
        parcalar.append(pd.DataFrame({"gun": gunler, "ilce": ilce, "kesinti": deger}))
    panel = pd.concat(parcalar, ignore_index=True)

    sonuc = add_previous_month_features(
        panel, "kesinti", time_column="gun", horizon=1, group_columns=["ilce"]
    )
    subat = sonuc[sonuc["gun"].dt.month == 2]
    bornova = subat[subat["ilce"] == "bornova"]["kesinti_sontamay_toplam"]
    menemen = subat[subat["ilce"] == "menemen"]["kesinti_sontamay_toplam"]
    assert np.allclose(bornova, 31.0)
    assert np.allclose(menemen, 310.0)


def test_girdi_frame_degismiyor():
    panel = _iki_ayli_panel()
    kopya = panel.copy(deep=True)
    add_previous_month_features(
        panel, "kesinti", time_column="gun", horizon=1, group_columns=["ilce"]
    )
    pd.testing.assert_frame_equal(panel, kopya)


def test_gecersiz_ufuk_reddediliyor():
    panel = _iki_ayli_panel()
    with pytest.raises(ValueError, match="horizon"):
        add_previous_month_features(panel, "kesinti", time_column="gun", horizon=0)


# --------------------------------------------------------------------------
# Ileri bakisli tatil pencereleri
# --------------------------------------------------------------------------


def test_cumhuriyet_bayrami_oncesi_pencereler():
    """29 Ekim sabittir: 26 Ekim'den 3 gun, 20 Ekim'den 9 gun kalir."""
    panel = pd.DataFrame({"gun": pd.to_datetime(["2023-10-26", "2023-10-20"])})
    sonuc = add_upcoming_holiday_features(panel, "gun", windows=(3, 7, 15), include_half_days=False)
    yirmi_alti = sonuc.iloc[0]
    assert yirmi_alti["tatil_sonraki_mesafe"] == 3
    assert yirmi_alti["tatil_onumuzdeki_3g"] == 1

    yirmi = sonuc.iloc[1]
    assert yirmi["tatil_sonraki_mesafe"] == 9
    assert yirmi["tatil_onumuzdeki_7g"] == 0
    assert yirmi["tatil_onumuzdeki_15g"] == 1


def test_bayram_gunu_ileri_bakiyor_kendini_gormuyor():
    """29 Ekim'de 'sonraki tatil' 29 Ekim DEGILDIR -- bugunu tatil_mi tasir."""
    panel = pd.DataFrame({"gun": pd.to_datetime(["2023-10-29"])})
    sonuc = add_upcoming_holiday_features(panel, "gun", include_half_days=False)
    # 2023'te 29 Ekim'den sonraki resmi tatil 1 Ocak 2024: 64 gun.
    assert sonuc.iloc[0]["tatil_sonraki_mesafe"] == 64
    assert sonuc.iloc[0]["tatil_onumuzdeki_15g"] == 0


def test_yil_siniri_asiliyor():
    """Aralik sonu, SONRAKI yilin yilbasini gormeli -- takvim +2 yil uzatiliyor."""
    panel = pd.DataFrame({"gun": pd.to_datetime(["2023-12-30"])})
    sonuc = add_upcoming_holiday_features(panel, "gun", include_half_days=False)
    assert sonuc.iloc[0]["tatil_sonraki_mesafe"] == 2


def test_bozuk_tarih_sentinel_aliyor():
    panel = pd.DataFrame({"gun": [pd.NaT, pd.Timestamp("2023-10-26")]})
    sonuc = add_upcoming_holiday_features(panel, "gun", include_half_days=False)
    assert sonuc.iloc[0]["tatil_sonraki_mesafe"] == MISSING_HOLIDAY_DISTANCE
    assert sonuc.iloc[0]["tatil_onumuzdeki_3g"] == 0
    assert sonuc.iloc[1]["tatil_onumuzdeki_3g"] == 1


def test_bos_pencere_listesi_reddediliyor():
    panel = pd.DataFrame({"gun": pd.to_datetime(["2023-10-26"])})
    with pytest.raises(ValueError, match="pencere"):
        add_upcoming_holiday_features(panel, "gun", windows=())


# --------------------------------------------------------------------------
# sqrt hedef donusumu
# --------------------------------------------------------------------------


def test_sqrt_gidis_donus_kayipsiz():
    y = np.array([0.0, 1.0, 4.0, 9.0, 100.0])
    geri = inverse_sqrt_transform(sqrt_transform_target(y))
    np.testing.assert_allclose(geri, y)


def test_sqrt_negatif_hedefi_reddediyor():
    with pytest.raises(ValueError, match="negatif"):
        sqrt_transform_target(np.array([1.0, -0.1]))


def test_ters_donusum_once_kirpip_sonra_kare_aliyor():
    """ISARET HATASI KORUMASI: sqrt uzayinda -0.5 tahmini 0 olmali, +0.25 DEGIL.

    Dogrudan kare almak negatif tahmini sessizce pozitife cevirir.
    """
    assert inverse_sqrt_transform(np.array([-0.5]))[0] == pytest.approx(0.0)
    # Bilincli olarak kapatilirsa matematiksel kare doner (belgelenmis tehlike).
    assert inverse_sqrt_transform(np.array([-0.5]), clip_negative=False)[0] == pytest.approx(0.25)


# --------------------------------------------------------------------------
# Olasilik kalibrasyonu (arastirma taramasi #1 onerisi)
# --------------------------------------------------------------------------


def _bozuk_kalibrasyonlu_veri(n: int = 4000, seed: int = 7):
    """Gercek olasiligi p olan ama p**3 RAPORLAYAN bir siniflandirici taklidi."""
    rng = np.random.default_rng(seed)
    gercek_p = rng.uniform(0.05, 0.95, size=n)
    y = (rng.uniform(size=n) < gercek_p).astype("float64") * rng.gamma(2.0, 3.0, size=n)
    raporlanan = gercek_p**3  # sistematik olarak fazla-kotumser
    yarim = n // 2
    folds = [
        (np.arange(yarim), np.arange(yarim, n)),
        (np.arange(yarim, n), np.arange(yarim)),
    ]
    return raporlanan, y, folds


def test_kalibrasyon_bozuk_olasiligi_duzeltiyor():
    from gridup.two_stage import calibrate_positive_probability

    raporlanan, y, folds = _bozuk_kalibrasyonlu_veri()
    sonuc = calibrate_positive_probability(raporlanan, y, folds, verbose=False)
    assert sonuc.improved
    assert sonuc.brier_after < sonuc.brier_before
    assert sonuc.calibrated.min() >= 0.0
    assert sonuc.calibrated.max() <= 1.0
    # Test-zamani kalibratoru de duzeltmeli: p**3 raporlanan 0.512 -> ~0.8.
    assert float(sonuc.calibrator.predict([0.512])[0]) > 0.6


def test_kalibrasyon_tek_sinifta_atlaniyor():
    from gridup.two_stage import calibrate_positive_probability

    olasilik = np.array([0.2, 0.4, 0.6])
    hedef = np.zeros(3)
    sonuc = calibrate_positive_probability(olasilik, hedef, verbose=False)
    assert not sonuc.improved
    assert sonuc.calibrator is None
    np.testing.assert_array_equal(sonuc.calibrated, olasilik)
    assert any("tek sinif" in not_satiri for not_satiri in sonuc.notes)


def test_kalibrasyon_uzunluk_uyusmazligini_reddediyor():
    from gridup.two_stage import calibrate_positive_probability

    with pytest.raises(ValueError, match="uzunluklari"):
        calibrate_positive_probability(np.array([0.5]), np.array([1.0, 0.0]))


# --------------------------------------------------------------------------
# Atif bekcisi: "GDZ 2024 yoktur" yanlis-duzeltmesi geri donmesin
# --------------------------------------------------------------------------


def test_2024_atiflari_geri_gelmis_ve_yanlis_duzeltme_gitmis():
    import gridup.panel as panel_mod
    import gridup.selection as selection_mod
    import gridup.tuning as tuning_mod
    from gridup.models import starter_params

    for modul in (panel_mod, selection_mod, tuning_mod):
        assert "2024" in modul.__doc__, modul.__name__
        assert "yarismasi YOKTUR" not in modul.__doc__, modul.__name__
        assert "yarismasi yoktur" not in modul.__doc__, modul.__name__
    assert "Pikachow" in selection_mod.__doc__
    assert "2024 GDZ" in starter_params.__doc__
