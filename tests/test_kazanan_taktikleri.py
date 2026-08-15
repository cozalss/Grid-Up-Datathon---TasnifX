"""KAZANANLAR EL KITABI 2. DALGA TAKTIKLERI (docs/09 bolum 6 "Simdi").

Bes olculebilir aktarim, her biri kaynakli:

  * toplu-olay bayragi (M5 out-of-stock analogu) -- YALNIZCA ufuk-kaydirilmis
  * ornek agirligi: recency rampasi (Izmir Bombasi) x aktiflik (M5 14.)
  * kalibrasyon carpani (M5 2.si; M5 1.sinin "carpan yok" uyarisiyla)
  * yumusak IQR aykiri harmani (Izmir Bombasi, 0.38 ham + 0.62 kirpik)
  * hava sayaclari: ardisik esik-otesi gun + cok pencereli yagis anomalisi

Stil referansi: tests/test_2024_birinci_taktikleri.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.features.temporal import add_mass_event_features
from gridup.features.weather import add_consecutive_extreme_days, add_precip_anomaly
from gridup.metrics import soften_outliers, tune_final_multiplier
from gridup.models import cross_validate
from gridup.validation import purged_time_series_split
from gridup.weighting import recency_activity_weights

# --------------------------------------------------------------------------
# Toplu-olay bayragi (M5 out-of-stock analogu)
# --------------------------------------------------------------------------

_UFUK = 3


def _toplu_olay_paneli(n_gun: int = 20, n_grup: int = 4) -> tuple[pd.DataFrame, np.ndarray]:
    """Gun i'de ILK (i mod 5) grup olayli -> pay = (i mod 5) / 4, deterministik."""
    gunler = pd.date_range("2024-03-01", periods=n_gun, freq="D")
    satirlar = []
    for i, gun in enumerate(gunler):
        aktif = i % 5
        for g in range(n_grup):
            satirlar.append({"gun": gun, "ilce": f"g{g}", "kesinti": 1.0 if g < aktif else 0.0})
    paylar = np.array([(i % 5) / n_grup for i in range(n_gun)])
    return pd.DataFrame(satirlar), paylar


def test_toplu_olay_payi_ufuk_kadar_geriden_geliyor():
    """SIZINTI TESTI: d gunundeki feature d - horizon gununun payi, d'nin DEGIL.

    Paylar 5 gunluk dongude, ufuk 3: pay(d-3) != pay(d) HER gun icin dogru --
    yani esitlik yanlis gune kayarsa test kesin yakalar.
    """
    panel, paylar = _toplu_olay_paneli()
    sonuc = add_mass_event_features(
        panel, "kesinti", time_column="gun", horizon=_UFUK, group_columns=["ilce"]
    )
    kolon = f"kesinti_ufuk{_UFUK}_topluolay_pay_lag1"

    gunler = pd.date_range("2024-03-01", periods=20, freq="D")
    for i, gun in enumerate(gunler):
        deger = sonuc.loc[sonuc["gun"] == gun, kolon]
        assert deger.nunique(dropna=False) == 1  # gunun tum satirlari ayni payi gorur
        if i < _UFUK:
            assert deger.isna().all(), f"gun {i}: gecmis yokken pay NaN olmali"
        else:
            assert deger.iloc[0] == pytest.approx(paylar[i - _UFUK]), (
                f"gun {i}: pay {paylar[i - _UFUK]} (d-{_UFUK}) olmali"
            )
            # Ayni gunun payi ASLA yayinlanmamali -- dogrudan hedef sizintisi.
            assert deger.iloc[0] != pytest.approx(paylar[i])


def test_toplu_olay_kayan7_penceresi_ufukta_bitiyor():
    """Kayan ortalamanin penceresi de d - horizon'da BITMELI, d'de degil."""
    panel, paylar = _toplu_olay_paneli()
    sonuc = add_mass_event_features(
        panel, "kesinti", time_column="gun", horizon=_UFUK, group_columns=["ilce"]
    )
    kolon = f"kesinti_ufuk{_UFUK}_topluolay_pay_kayan7"

    gunler = pd.date_range("2024-03-01", periods=20, freq="D")
    for i in range(_UFUK, 20):
        beklenen = paylar[max(0, i - _UFUK - 6): i - _UFUK + 1].mean()
        deger = sonuc.loc[sonuc["gun"] == gunler[i], kolon].iloc[0]
        assert deger == pytest.approx(beklenen, abs=1e-6), f"gun {i}"


def test_toplu_olay_bayragi_esik_ve_nan():
    panel, paylar = _toplu_olay_paneli()
    sonuc = add_mass_event_features(
        panel, "kesinti", time_column="gun", horizon=_UFUK,
        group_columns=["ilce"], threshold=0.5,
    )
    kolon = f"kesinti_ufuk{_UFUK}_topluolay_bayrak_lag1"

    gunler = pd.date_range("2024-03-01", periods=20, freq="D")
    ilk = sonuc.loc[sonuc["gun"] == gunler[0], kolon]
    # Pay bilinmiyorsa bayrak 0 DEGIL NaN: "olay yok" ile "bilinmiyor" ayri.
    assert ilk.isna().all()
    for i in range(_UFUK, 20):
        deger = sonuc.loc[sonuc["gun"] == gunler[i], kolon].iloc[0]
        assert deger == pytest.approx(1.0 if paylar[i - _UFUK] >= 0.5 else 0.0)


def test_toplu_olay_ayni_gun_payi_reddediliyor():
    """horizon=0 ayni gunun payini yayinlardi -- yapisal olarak imkansiz olmali."""
    panel, _ = _toplu_olay_paneli()
    with pytest.raises(ValueError, match="horizon"):
        add_mass_event_features(
            panel, "kesinti", time_column="gun", horizon=0, group_columns=["ilce"]
        )


def test_toplu_olay_grup_kolonu_zorunlu():
    panel, _ = _toplu_olay_paneli()
    with pytest.raises(ValueError, match="group_columns"):
        add_mass_event_features(
            panel, "kesinti", time_column="gun", horizon=1, group_columns=[]
        )


# --------------------------------------------------------------------------
# Ornek agirligi: recency rampasi x aktiflik carpani
# --------------------------------------------------------------------------


def _agirlik_paneli(deger: float, n_gun: int = 5, ilce: str = "bornova") -> pd.DataFrame:
    gunler = pd.date_range("2024-01-01", periods=n_gun, freq="D")
    return pd.DataFrame({"gun": gunler, "ilce": ilce, "kesinti": deger})


def test_rampa_uclari_izmir_bombasi_degerleri():
    """En eski satir 0.05, en yeni 0.95, ara dogrusal (tam aktif grup: carpan=1)."""
    panel = _agirlik_paneli(deger=7.0)  # hep > 0 -> aktiflik 1 -> carpan 1
    agirlik = recency_activity_weights(
        panel, "kesinti", time_column="gun", group_columns=["ilce"]
    )
    np.testing.assert_allclose(agirlik, [0.05, 0.275, 0.5, 0.725, 0.95])


def test_aktiflik_carpani_olu_seriyi_tabana_indiriyor():
    """Hep-sifir grup rampa x floor alir; tam aktif grup rampa x 1 -- M5 14. taktigi."""
    olu = _agirlik_paneli(deger=0.0, ilce="olu")
    aktif = _agirlik_paneli(deger=3.0, ilce="aktif")
    panel = pd.concat([olu, aktif], ignore_index=True)

    agirlik = recency_activity_weights(
        panel, "kesinti", time_column="gun", group_columns=["ilce"], activity_floor=0.25
    )
    rampa = np.array([0.05, 0.275, 0.5, 0.725, 0.95])
    np.testing.assert_allclose(agirlik[:5], rampa * 0.25)   # olu -> taban
    np.testing.assert_allclose(agirlik[5:], rampa * 1.0)    # aktif -> tam


def test_tek_satirlik_grup_en_guncel_agirligi_alir():
    panel = _agirlik_paneli(deger=1.0, n_gun=1)
    agirlik = recency_activity_weights(
        panel, "kesinti", time_column="gun", group_columns=["ilce"]
    )
    assert agirlik[0] == pytest.approx(0.95)


def test_agirlik_girdi_sirasinda_donuyor():
    """Karisik satir sirasi: agirlik dogru SATIRA yazilmali, dogru konuma degil."""
    panel = _agirlik_paneli(deger=2.0)
    duz = recency_activity_weights(
        panel, "kesinti", time_column="gun", group_columns=["ilce"]
    )
    perm = np.array([3, 0, 4, 1, 2])
    karisik = panel.iloc[perm].reset_index(drop=True)
    karma = recency_activity_weights(
        karisik, "kesinti", time_column="gun", group_columns=["ilce"]
    )
    np.testing.assert_allclose(karma, duz[perm])


def test_agirlik_parametre_hatalari():
    panel = _agirlik_paneli(deger=1.0)
    with pytest.raises(ValueError, match="start"):
        recency_activity_weights(
            panel, "kesinti", time_column="gun", start=0.9, end=0.1
        )
    with pytest.raises(ValueError, match="activity_floor"):
        recency_activity_weights(
            panel, "kesinti", time_column="gun", activity_floor=1.5
        )
    with pytest.raises(KeyError, match="yok"):
        recency_activity_weights(panel, "olmayan", time_column="gun")


def _cv_verisi():
    rng = np.random.default_rng(0)
    n = 600
    tarih = pd.Series(np.tile(pd.date_range("2025-01-01", periods=200), 3))
    x = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    y = (x.a * 3 + rng.normal(0, 1, n)).to_numpy()
    folds = purged_time_series_split(
        tarih, embargo=pd.Timedelta(days=5), n_splits=2,
        test_span=pd.Timedelta(days=30), verbose=False,
    )
    return x, y, tarih, folds


@pytest.mark.slow
def test_cross_validate_agirlikla_kosuyor():
    """SMOKE: sample_weight yolu calisiyor; birim agirlik = agirliksizla AYNI skor."""
    x, y, tarih, folds = _cv_verisi()
    params = {"n_estimators": 50, "verbose": -1}

    duz = cross_validate(
        x, y, folds, kind="lightgbm", metric="mae", params=params, verbose=False
    )
    birim = cross_validate(
        x, y, folds, kind="lightgbm", metric="mae", params=params,
        sample_weight=np.ones(len(x)), verbose=False,
    )
    assert birim.overall_score == pytest.approx(duz.overall_score, rel=1e-9)

    panel = pd.DataFrame({"gun": tarih, "hedef": y})
    agirlik = recency_activity_weights(panel, "hedef", time_column="gun")
    gercek = cross_validate(
        x, y, folds, kind="lightgbm", metric="mae", params=params,
        sample_weight=agirlik, verbose=False,
    )
    assert np.isfinite(gercek.overall_score)
    assert np.array_equal(gercek.oof_covered, duz.oof_covered)


def test_cross_validate_agirlik_boyut_hatasi():
    x, y, _, folds = _cv_verisi()
    with pytest.raises(ValueError, match="sample_weight"):
        cross_validate(
            x, y, folds, kind="lightgbm", metric="mae",
            sample_weight=np.ones(3), verbose=False,
        )


def test_cross_validate_negatif_agirlik_reddediliyor():
    x, y, _, folds = _cv_verisi()
    kotu = np.ones(len(x))
    kotu[0] = -1.0
    with pytest.raises(ValueError, match="negatif"):
        cross_validate(
            x, y, folds, kind="lightgbm", metric="mae",
            sample_weight=kotu, verbose=False,
        )


# --------------------------------------------------------------------------
# Kalibrasyon carpani (M5 2.si -- 1.sinin uyarisiyla birlikte)
# --------------------------------------------------------------------------


def _yanli_tahmin(n: int = 400, carpan: float = 0.95, seed: int = 3):
    """Gercek deger y, tahmin y / carpan -> optimal duzeltme tam carpan'dir."""
    rng = np.random.default_rng(seed)
    y = rng.gamma(2.0, 50.0, size=n)
    return y, y / carpan


def test_carpan_sentetik_yanliligi_buluyor():
    y, tahmin = _yanli_tahmin(carpan=0.95)
    en_iyi, skor, tablo = tune_final_multiplier(y, tahmin)
    assert en_iyi == pytest.approx(0.95)
    # Carpansiz (1.0) skordan GERCEK bir kazanc olmali -- M5 karar kurali.
    carpansiz = float(tablo.loc[tablo["carpan"] == 1.0, "skor"].iloc[0])
    assert skor < carpansiz


def test_carpan_yansiz_tahminde_bir_donduruyor():
    y, _ = _yanli_tahmin()
    en_iyi, skor, _ = tune_final_multiplier(y, y)
    assert en_iyi == pytest.approx(1.0)
    assert skor == pytest.approx(0.0)


def test_carpan_izgarasi_biri_daima_iceriyor():
    """1.0 izgarada olmasa bile eklenir: 'carpan yok' secenegi hep denenir."""
    y, tahmin = _yanli_tahmin()
    _, _, tablo = tune_final_multiplier(y, tahmin, grid=np.array([0.93, 0.97]))
    assert 1.0 in set(np.round(tablo["carpan"], 6))
    assert len(tablo) == 3


def test_carpan_kapsam_maskesine_uyuyor():
    """Kapsanmayan (dolgu) satirlar taramaya girmemeli -- purged CV geregi."""
    y, tahmin = _yanli_tahmin(n=200)
    bozuk = tahmin.copy()
    bozuk[:50] = y[:50] * 5.0          # dolgu bolgesi: anlamsiz tahmin
    maske = np.ones(200, dtype=bool)
    maske[:50] = False
    en_iyi, _, _ = tune_final_multiplier(y, bozuk, covered=maske)
    assert en_iyi == pytest.approx(0.95)


def test_carpan_giris_hatalari():
    with pytest.raises(ValueError, match="uzunluklari"):
        tune_final_multiplier(np.ones(3), np.ones(4))
    with pytest.raises(ValueError, match="uzunluklari"):
        tune_final_multiplier(np.ones(3), np.ones(3), covered=np.ones(2, dtype=bool))
    with pytest.raises(ValueError, match="bos"):
        tune_final_multiplier(np.ones(3), np.ones(3), covered=np.zeros(3, dtype=bool))


# --------------------------------------------------------------------------
# Yumusak IQR aykiri harmani (Izmir Bombasi)
# --------------------------------------------------------------------------

#: 0..7 + 100: Q1=2, Q3=6, IQR=4 -> tavan = 6 + 1.5*4 = 12.
#: 100 -> 0.38*100 + 0.62*12 = 45.44; digerleri tavanin altinda, aynen kalir.
_AYKIRI = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0])


def test_yumusak_iqr_el_hesabi():
    sonuc = soften_outliers(_AYKIRI)
    assert sonuc[-1] == pytest.approx(45.44)
    np.testing.assert_allclose(sonuc[:-1], _AYKIRI[:-1])


def test_yumusak_iqr_grup_bazinda_ayri_tavan():
    """10x olcekli grup 10x tavan gormeli -- kuantiller grup ICINDEN."""
    degerler = np.concatenate([_AYKIRI, _AYKIRI * 10.0])
    gruplar = np.array(["a"] * 9 + ["b"] * 9)
    sonuc = soften_outliers(degerler, gruplar)
    assert sonuc[8] == pytest.approx(45.44)     # a: 100 -> 45.44
    assert sonuc[17] == pytest.approx(454.4)    # b: 1000 -> 0.38*1000 + 0.62*120


def test_yumusak_iqr_blend_uclari():
    assert soften_outliers(_AYKIRI, blend=0.0)[-1] == pytest.approx(100.0)  # ham
    assert soften_outliers(_AYKIRI, blend=1.0)[-1] == pytest.approx(12.0)   # sert kirpma


def test_yumusak_iqr_girdiyi_degistirmiyor():
    orijinal = _AYKIRI.copy()
    soften_outliers(_AYKIRI)
    np.testing.assert_array_equal(_AYKIRI, orijinal)


def test_yumusak_iqr_parametre_hatalari():
    with pytest.raises(ValueError, match="blend"):
        soften_outliers(_AYKIRI, blend=1.5)
    with pytest.raises(ValueError, match="iqr_factor"):
        soften_outliers(_AYKIRI, iqr_factor=-1.0)
    with pytest.raises(ValueError, match="uzunluklari"):
        soften_outliers(_AYKIRI, np.array(["a", "b"]))


# --------------------------------------------------------------------------
# Hava sayaclari: ardisik esik-otesi gun + yagis anomalisi
# --------------------------------------------------------------------------


def _hava_paneli() -> pd.DataFrame:
    gunler = pd.date_range("2024-07-01", periods=6, freq="D")
    return pd.DataFrame({
        "gun": gunler, "ilce": "bornova",
        "sicaklik_min": [10.0, 25.0, 26.0, 27.0, 5.0, 30.0],
    })


def test_ardisik_esik_ustu_sayaci():
    """[F,T,T,T,F,T] -> [0,1,2,3,0,1] -- sicak gece yorulma mekanizmasi."""
    sonuc = add_consecutive_extreme_days(
        _hava_paneli(), "sicaklik_min", time_column="gun",
        group_columns=["ilce"], threshold=22.0,
    )
    np.testing.assert_array_equal(
        sonuc["sicaklik_min_ardisik_ustu_gun"].to_numpy(), [0, 1, 2, 3, 0, 1]
    )


def test_ardisik_esik_alti_yonu():
    sonuc = add_consecutive_extreme_days(
        _hava_paneli(), "sicaklik_min", time_column="gun",
        group_columns=["ilce"], threshold=20.0, above=False,
    )
    np.testing.assert_array_equal(
        sonuc["sicaklik_min_ardisik_alti_gun"].to_numpy(), [1, 0, 0, 0, 1, 0]
    )


def test_ardisik_sayac_gruplar_ve_sira():
    """Karisik satir sirasi + iki grup: sayac grup ici kronolojiden, sira korunur."""
    gunler = pd.date_range("2024-07-01", periods=4, freq="D")
    panel = pd.concat([
        pd.DataFrame({"gun": gunler, "ilce": "b", "v": [30.0, 30.0, 10.0, 30.0]}),
        pd.DataFrame({"gun": gunler, "ilce": "a", "v": [10.0, 30.0, 30.0, 30.0]}),
    ], ignore_index=True).sample(frac=1, random_state=5).reset_index(drop=True)

    sonuc = add_consecutive_extreme_days(
        panel, "v", time_column="gun", group_columns=["ilce"], threshold=20.0
    )
    assert list(sonuc["gun"]) == list(panel["gun"])  # sira korundu
    beklenen = {("b", g): d for g, d in zip(gunler, [1, 2, 0, 1], strict=True)}
    beklenen |= {("a", g): d for g, d in zip(gunler, [0, 1, 2, 3], strict=True)}
    for _, satir in sonuc.iterrows():
        assert satir["v_ardisik_ustu_gun"] == beklenen[(satir["ilce"], satir["gun"])]


def test_yagis_anomalisi_el_hesabi():
    """pencere 2, yagis [0,0,10,0]: kayan toplam [0,0,10,10],
    genisleyen ort [0,0,10/3,5] -> anomali [0,0,20/3,5]."""
    panel = pd.DataFrame({
        "gun": pd.date_range("2024-01-01", periods=4, freq="D"),
        "ilce": "bornova", "yagis": [0.0, 0.0, 10.0, 0.0],
    })
    sonuc = add_precip_anomaly(
        panel, "yagis", time_column="gun", group_columns=["ilce"], windows=(2,)
    )
    np.testing.assert_allclose(
        sonuc["yagis_anomali2g"].to_numpy(), [0.0, 0.0, 20.0 / 3.0, 5.0], rtol=1e-6
    )


def test_yagis_anomalisi_grup_ici_referans():
    """Iklim referansi grubun KENDI gecmisi: ayni mm iki ilcede farkli anomali."""
    gunler = pd.date_range("2024-01-01", periods=2, freq="D")
    panel = pd.concat([
        pd.DataFrame({"gun": gunler, "ilce": "kurak", "yagis": [0.0, 10.0]}),
        pd.DataFrame({"gun": gunler, "ilce": "yagisli", "yagis": [100.0, 100.0]}),
    ], ignore_index=True)
    sonuc = add_precip_anomaly(
        panel, "yagis", time_column="gun", group_columns=["ilce"], windows=(2,)
    )
    kurak = sonuc[sonuc["ilce"] == "kurak"]["yagis_anomali2g"].to_numpy()
    yagisli = sonuc[sonuc["ilce"] == "yagisli"]["yagis_anomali2g"].to_numpy()
    np.testing.assert_allclose(kurak, [0.0, 5.0])       # [0,10] vs ort [0,5]
    np.testing.assert_allclose(yagisli, [0.0, 50.0])    # [100,200] vs ort [100,150]


def test_yagis_anomalisi_giris_hatalari():
    panel = pd.DataFrame({
        "gun": pd.date_range("2024-01-01", periods=2),
        "ilce": "x", "yagis": [1.0, 2.0],
    })
    with pytest.raises(ValueError, match="pencere"):
        add_precip_anomaly(
            panel, "yagis", time_column="gun", group_columns=["ilce"], windows=()
        )
    with pytest.raises(KeyError, match="yok"):
        add_precip_anomaly(
            panel, "olmayan", time_column="gun", group_columns=["ilce"]
        )
