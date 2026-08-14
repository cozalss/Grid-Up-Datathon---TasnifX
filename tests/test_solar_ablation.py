"""Gunes fizigi, Ramazan takvimi, sabit-uzunluklu fold ve ablasyon testleri.

Bu dosyadaki her test, 2023 GDZ Elektrik Datathon birincisinin cozumu
incelenerek bulunan bir eksigi kapatir. Beklenen degerlerin cogu **bagimsiz
kaynakla** dogrulanmistir (astronomi formulleri, frtgnn/turkish-calendar
veri seti); uydurma esik yoktur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.ablation import (
    FeatureGroup,
    _variant_columns,
    ablation_ensemble,
    leave_one_group_out,
)
from gridup.features.solar import (
    add_clearness_index,
    add_solar_features,
    clear_sky_daily,
    solar_geometry_daily,
)
from gridup.features.temporal import add_ramadan_features, ramadan_calendar
from gridup.validation import assert_folds_align, purged_time_series_split

IZMIR = (38.42, 27.14)


# --------------------------------------------------------------------------
# Gunes geometrisi -- kapali formul, pvlib gerektirmez
# --------------------------------------------------------------------------


def test_gun_uzunlugu_gundonumlerinde_izmir_gercegiyle_ortusuyor():
    """Izmir'de en kisa gun ~9.3, en uzun gun ~14.7 saattir."""
    # Arrange
    dates = pd.DatetimeIndex(["2025-12-21", "2025-06-21"])

    # Act
    geometry = solar_geometry_daily(IZMIR[0], dates)

    # Assert
    assert geometry.loc["2025-12-21", "gun_uzunlugu_saat"] == pytest.approx(9.3, abs=0.2)
    assert geometry.loc["2025-06-21", "gun_uzunlugu_saat"] == pytest.approx(14.7, abs=0.2)


def test_ekinoksta_gun_uzunlugu_12_saate_yakin():
    geometry = solar_geometry_daily(IZMIR[0], pd.DatetimeIndex(["2025-03-20"]))
    assert geometry.iloc[0]["gun_uzunlugu_saat"] == pytest.approx(12.0, abs=0.2)


def test_ogle_yuksekligi_teorik_degeri_veriyor():
    """Yaz gundonumunde ogle yuksekligi = 90 - (enlem - 23.45)."""
    geometry = solar_geometry_daily(IZMIR[0], pd.DatetimeIndex(["2025-06-21"]))
    beklenen = 90.0 - (IZMIR[0] - 23.45)
    assert geometry.iloc[0]["gunes_ogle_yuksekligi"] == pytest.approx(beklenen, abs=0.1)


def test_kuzey_enlem_yazin_daha_uzun_gun_gorur():
    """Kuzey yarimkurede yaz gundonumunde enlem arttikca gun uzar."""
    dates = pd.DatetimeIndex(["2025-06-21"])
    guney = solar_geometry_daily(36.6, dates).iloc[0]["gun_uzunlugu_saat"]
    kuzey = solar_geometry_daily(39.2, dates).iloc[0]["gun_uzunlugu_saat"]
    assert kuzey > guney


def test_kutupta_arccos_patlamiyor():
    """|enlem| > 66.5'te kutup gecesi/gunu olur; NaN degil 0 veya 24 bekleriz."""
    kis = solar_geometry_daily(80.0, pd.DatetimeIndex(["2025-12-21"]))
    yaz = solar_geometry_daily(80.0, pd.DatetimeIndex(["2025-06-21"]))
    assert not np.isnan(kis.iloc[0]["gun_uzunlugu_saat"])
    assert kis.iloc[0]["gun_uzunlugu_saat"] == pytest.approx(0.0, abs=0.01)
    assert yaz.iloc[0]["gun_uzunlugu_saat"] == pytest.approx(24.0, abs=0.01)


# --------------------------------------------------------------------------
# Acik-hava isinimi -- pvlib
# --------------------------------------------------------------------------


def test_acik_hava_isinimi_yazin_kistan_yuksek():
    pytest.importorskip("pvlib")
    daily = clear_sky_daily(*IZMIR, pd.date_range("2025-01-01", "2025-12-31", freq="D"))
    yaz = daily.loc["2025-06-21", "gunes_ghi_gunluk"]
    kis = daily.loc["2025-12-21", "gunes_ghi_gunluk"]
    assert yaz > kis
    # Izmir'de yaz/kis orani ~3 civaridir; 2-5 araligi genis ama anlamli bir bant.
    assert 2.0 < yaz / kis < 5.0


def test_acik_hava_yillik_ortalamasi_makul_bantta():
    """Izmir icin acik-hava GHI ortalamasi 4-6 kWh/m2/gun bandinda olmali."""
    pytest.importorskip("pvlib")
    daily = clear_sky_daily(*IZMIR, pd.date_range("2025-01-01", "2025-12-31", freq="D"))
    assert 4.0 < daily["gunes_ghi_gunluk"].mean() < 6.0
    assert daily.isna().sum().sum() == 0


def test_add_solar_features_frame_i_degistirmiyor():
    pytest.importorskip("pvlib")
    frame = pd.DataFrame({"tarih": pd.date_range("2025-01-01", periods=10, freq="D")})
    onceki = frame.copy()
    add_solar_features(frame, time_column="tarih", coordinates=IZMIR)
    pd.testing.assert_frame_equal(frame, onceki)


def test_add_solar_features_panelde_konum_basina_farkli_deger_uretiyor():
    pytest.importorskip("pvlib")
    gunler = pd.date_range("2025-06-01", periods=5, freq="D")
    koord = {"kuzey": (39.2, 27.1), "guney": (36.6, 28.4)}
    panel = pd.DataFrame(
        {
            "tarih": np.tile(gunler, 2),
            "yer": np.repeat(list(koord), len(gunler)),
        }
    )

    out = add_solar_features(panel, time_column="tarih", location_column="yer", coordinates=koord)

    kuzey = out.loc[out.yer == "kuzey", "gun_uzunlugu_saat"].to_numpy()
    guney = out.loc[out.yer == "guney", "gun_uzunlugu_saat"].to_numpy()
    assert np.all(kuzey > guney)
    assert out["gunes_ghi_gunluk"].isna().sum() == 0


def test_eksik_koordinat_sessiz_gecmiyor():
    panel = pd.DataFrame(
        {"tarih": pd.date_range("2025-01-01", periods=3, freq="D"), "yer": ["a", "b", "c"]}
    )
    with pytest.raises(KeyError, match="koordinati yok"):
        add_solar_features(
            panel, time_column="tarih", location_column="yer",
            coordinates={"a": IZMIR}, geometry_only=True,
        )


def test_berraklik_endeksi_sifir_boleni_nan_yapiyor():
    frame = pd.DataFrame(
        {"olculen": [3.0, 4.0, 5.0], "gunes_ghi_gunluk": [5.0, 0.0, 6.0]}
    )
    out = add_clearness_index(frame, observed_column="olculen")
    assert out["berraklik_endeksi"].iloc[0] == pytest.approx(0.6)
    assert np.isnan(out["berraklik_endeksi"].iloc[1])


# --------------------------------------------------------------------------
# Ramazan takvimi -- frtgnn/turkish-calendar ile dogrulandi
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("yil", "baslangic", "bitis"),
    [
        (2020, "2020-04-24", "2020-05-23"),
        (2021, "2021-04-13", "2021-05-12"),
        (2022, "2022-04-02", "2022-05-01"),
        (2023, "2023-03-23", "2023-04-20"),  # 29 gunluk Ramazan
        (2024, "2024-03-11", "2024-04-09"),
        (2026, "2026-02-18", "2026-03-19"),  # yarisma yili
    ],
)
def test_ramazan_takvimi_bagimsiz_kaynakla_ortusuyor(yil, baslangic, bitis):
    pytest.importorskip("hijridate")
    takvim = ramadan_calendar([yil])
    assert takvim[yil][0] == pd.Timestamp(baslangic)
    assert takvim[yil][1] == pd.Timestamp(bitis)


def test_ramazan_her_yil_geriye_kayiyor():
    """Hicri yil ~354 gun -- Ramazan miladi takvimde yilda ~11 gun GERI kayar.

    Bu testin varlik sebebi: ``ay``, ``dayofyear`` ve sinus/kosinus mevsimsellik
    kolonlari bu kaymayi ASLA ogrenemez. Ramazan icin ayri kolon sart.
    """
    pytest.importorskip("hijridate")
    takvim = ramadan_calendar([2024, 2025, 2026])
    baslangiclar = [takvim[y][0] for y in (2024, 2025, 2026)]

    # Kaydirmali ikili: uzunluklar KASITLI farkli, strict=True burada yanlis olur.
    for onceki, sonraki in zip(baslangiclar, baslangiclar[1:]):  # noqa: B905
        # Ardisik Ramazanlar arasi mesafe bir hicri yildir: 354-355 gun.
        assert 353 <= (sonraki - onceki).days <= 356
        # Ve takvim icindeki konumu her yil GERI gider.
        assert sonraki.dayofyear < onceki.dayofyear


def test_ramazan_feature_lari_sinirlarda_dogru():
    pytest.importorskip("hijridate")
    frame = pd.DataFrame({"tarih": pd.date_range("2026-02-16", "2026-03-22", freq="D")})

    out = add_ramadan_features(frame, "tarih")

    ilk = out.loc[out.tarih == "2026-02-18"].iloc[0]
    son = out.loc[out.tarih == "2026-03-19"].iloc[0]
    disari = out.loc[out.tarih == "2026-03-20"].iloc[0]

    assert ilk.ramazan_ayi == 1 and ilk.ramazan_gunu == 1
    assert ilk.ramazan_ilerleme == pytest.approx(0.0)
    assert son.ramazan_ayi == 1 and son.ramazan_gunu == 30
    assert son.ramazan_ilerleme == pytest.approx(1.0)
    assert son.ramazan_bayrama_kalan == 1
    assert disari.ramazan_ayi == 0 and disari.ramazan_gunu == 0


def test_ramazan_son_on_gun_tam_on_gun():
    pytest.importorskip("hijridate")
    frame = pd.DataFrame({"tarih": pd.date_range("2026-02-18", "2026-03-19", freq="D")})
    out = add_ramadan_features(frame, "tarih")
    assert int(out["ramazan_son_on_gun"].sum()) == 10


def test_ramazan_features_frame_i_degistirmiyor():
    pytest.importorskip("hijridate")
    frame = pd.DataFrame({"tarih": pd.date_range("2026-02-18", periods=5, freq="D")})
    onceki = frame.copy()
    add_ramadan_features(frame, "tarih")
    pd.testing.assert_frame_equal(frame, onceki)


def test_ramazan_gecersiz_tarihte_patlamiyor():
    pytest.importorskip("hijridate")
    frame = pd.DataFrame({"tarih": [None, None]})
    out = add_ramadan_features(frame, "tarih")
    assert (out["ramazan_ayi"] == 0).all()


# --------------------------------------------------------------------------
# Sabit zaman uzunlugunda fold -- 2023 birincisinin CV semasi
# --------------------------------------------------------------------------


def _panel_times(n_days: int = 400, n_units: int = 96) -> pd.Series:
    gunler = pd.date_range("2025-01-01", periods=n_days, freq="D")
    return pd.Series(np.repeat(gunler, n_units))


def test_test_span_tum_foldlari_ayni_zaman_uzunlugunda_yapiyor():
    times = _panel_times()
    folds = purged_time_series_split(
        times, embargo=pd.Timedelta(days=30), n_splits=3,
        test_span=pd.Timedelta(days=31), verbose=False,
    )
    assert len(folds) == 3
    values = times.to_numpy()
    uzunluklar = {
        (pd.Timestamp(values[v].max()) - pd.Timestamp(values[v].min())).days for _, v in folds
    }
    assert len(uzunluklar) == 1
    assert len({len(v) for _, v in folds}) == 1


def test_test_span_foldlari_kronolojik_sirada_donduruyor():
    times = _panel_times()
    folds = purged_time_series_split(
        times, embargo=pd.Timedelta(days=7), n_splits=3,
        test_span=pd.Timedelta(days=31), verbose=False,
    )
    values = times.to_numpy()
    baslangiclar = [pd.Timestamp(values[v].min()) for _, v in folds]
    assert baslangiclar == sorted(baslangiclar)


def test_son_fold_veri_sonuna_capali():
    times = _panel_times()
    folds = purged_time_series_split(
        times, embargo=pd.Timedelta(days=7), n_splits=3,
        test_span=pd.Timedelta(days=31), verbose=False,
    )
    values = times.to_numpy()
    assert pd.Timestamp(values[folds[-1][1]].max()) == pd.Timestamp(values.max())


def test_test_span_ambargoyu_koruyor():
    times = _panel_times()
    embargo = pd.Timedelta(days=30)
    folds = purged_time_series_split(
        times, embargo=embargo, n_splits=3, test_span=pd.Timedelta(days=31), verbose=False
    )
    values = times.to_numpy()
    for train_idx, valid_idx in folds:
        bosluk = pd.Timestamp(values[valid_idx].min()) - pd.Timestamp(values[train_idx].max())
        assert bosluk > embargo


def test_test_span_foldlari_cakismiyor():
    times = _panel_times()
    folds = purged_time_series_split(
        times, embargo=pd.Timedelta(days=7), n_splits=3,
        test_span=pd.Timedelta(days=31), verbose=False,
    )
    assert_folds_align(len(times), folds)
    hepsi = np.concatenate([v for _, v in folds])
    assert len(hepsi) == len(set(hepsi.tolist()))


def test_negatif_test_span_reddediliyor():
    with pytest.raises(ValueError, match="pozitif olmali"):
        purged_time_series_split(
            _panel_times(50, 2), embargo=pd.Timedelta(0),
            test_span=pd.Timedelta(days=-1), verbose=False,
        )


def test_test_span_verilmezse_eski_davranis_korunuyor():
    times = _panel_times(200, 4)
    eski = purged_time_series_split(times, embargo=pd.Timedelta(days=5), n_splits=3, verbose=False)
    assert len(eski) == 3
    # Satir sayisi esitlenmis pencereler: uzunluklar birbirine cok yakin olmali.
    boyutlar = [len(v) for _, v in eski]
    assert max(boyutlar) - min(boyutlar) <= 4


# --------------------------------------------------------------------------
# Ablasyon -- birincinin "2 asamali model" mimarisi
# --------------------------------------------------------------------------


def _ablasyon_verisi(seed: int = 7):
    rng = np.random.default_rng(seed)
    gunler = pd.date_range("2024-01-01", periods=400, freq="D")
    tarih = pd.Series(np.tile(gunler, 10))
    n = len(tarih)
    X = pd.DataFrame(
        {
            "takvim_ay": tarih.dt.month.to_numpy(),
            "lag_1": rng.normal(0, 1, n),
            "hava_sicaklik": rng.normal(18, 9, n),
            "deneysel_gurultu": rng.normal(0, 1, n),
        }
    )
    y = 120 + 3 * X.hava_sicaklik + 8 * X.lag_1 + rng.normal(0, 4, n)
    folds = purged_time_series_split(
        tarih, embargo=pd.Timedelta(days=14), n_splits=2,
        test_span=pd.Timedelta(days=60), verbose=False,
    )
    return X, y.to_numpy(), folds


def _gruplar():
    return [
        FeatureGroup("takvim", ("takvim_ay",), risk="cekirdek"),
        FeatureGroup("lag", ("lag_1",), risk="cekirdek"),
        FeatureGroup("hava", ("hava_sicaklik",), risk="harici"),
        FeatureGroup("deneysel", ("deneysel_gurultu",), risk="deneysel"),
    ]


def test_variant_columns_gercekten_daraliyor():
    """REGRESYON: onceki surumde 'gruplanmamis kolon' telafisi kasitli
    dislananlari geri getiriyordu ve her varyant AYNI kolonlari aliyordu --
    ablasyon sessizce hicbir sey yapmiyordu."""
    groups = _gruplar()
    hepsi_kolonlari = ["takvim_ay", "lag_1", "hava_sicaklik", "deneysel_gurultu"]

    cekirdek = _variant_columns(groups, "cekirdek", hepsi_kolonlari)
    saglam = _variant_columns(groups, "harici", hepsi_kolonlari)
    hepsi = _variant_columns(groups, "deneysel", hepsi_kolonlari)

    assert len(cekirdek) == 2
    assert len(saglam) == 3
    assert len(hepsi) == 4
    assert "hava_sicaklik" not in cekirdek
    assert "deneysel_gurultu" not in saglam


def test_gruplanmamis_kolon_cekirdek_sayiliyor():
    groups = [FeatureGroup("hava", ("hava_sicaklik",), risk="harici")]
    kolonlar = _variant_columns(groups, "cekirdek", ["hava_sicaklik", "unutulmus"])
    assert kolonlar == ["unutulmus"]


@pytest.mark.slow
def test_ablasyon_ic_ice_varyantlar_uretiyor():
    X, y, folds = _ablasyon_verisi()
    sonuc = ablation_ensemble(
        X, y, folds, groups=_gruplar(), metric="mape",
        params={"n_estimators": 120, "learning_rate": 0.1, "verbose": -1},
        verbose=False,
    )
    assert set(sonuc.variants) == {"hepsi", "saglam", "cekirdek"}
    boyutlar = [len(sonuc.variant_features[k]) for k in ("cekirdek", "saglam", "hepsi")]
    assert boyutlar == sorted(boyutlar)
    assert boyutlar[0] < boyutlar[-1]


@pytest.mark.slow
def test_ablasyon_harman_uyarisi_kotu_varyanti_yakaliyor():
    """Esit agirlik, varyantlar denk degilken harmanı bozar -- sessiz kalmamali."""
    X, y, folds = _ablasyon_verisi()
    sonuc = ablation_ensemble(
        X, y, folds, groups=_gruplar(), metric="mape",
        params={"n_estimators": 120, "learning_rate": 0.1, "verbose": -1},
        verbose=False,
    )
    rapor = sonuc.blend_check()
    skorlar = [r.overall_score for r in sonuc.variants.values()]
    if sonuc.blend_score > min(skorlar):
        assert "UYARI" in rapor and "hill_climb_weights" in rapor
    else:
        assert "Harman kazandi" in rapor


@pytest.mark.slow
def test_ablasyon_tek_risk_katmaninda_hata_veriyor():
    X, y, folds = _ablasyon_verisi()
    with pytest.raises(ValueError, match="en az IKI farkli risk"):
        ablation_ensemble(
            X, y, folds,
            groups=[FeatureGroup("hepsi", tuple(X.columns), risk="cekirdek")],
            metric="mape", verbose=False,
        )


def test_bilinmeyen_risk_reddediliyor():
    with pytest.raises(ValueError, match="Bilinmeyen risk"):
        FeatureGroup("x", ("a",), risk="belki")


def test_bos_grup_reddediliyor():
    with pytest.raises(ValueError, match="bos"):
        FeatureGroup("x", (), risk="cekirdek")


@pytest.mark.slow
def test_logo_gercek_sinyali_gurultuden_ayiriyor():
    """hava gercek sinyal, deneysel saf gurultu -- LOGO bunu ayirt etmeli."""
    X, y, folds = _ablasyon_verisi()
    tablo = leave_one_group_out(
        X, y, folds, groups=_gruplar(), metric="mape",
        params={"n_estimators": 120, "learning_rate": 0.1, "verbose": -1},
        verbose=False,
    )
    katki = tablo.set_index("grup")["katki"]
    assert katki["hava"] > katki["deneysel"]
    assert katki["lag"] > katki["deneysel"]
