"""Harici veri feature'larinin UFUK sizintisi testleri.

Bu dosyadaki testler tek bir soruyu sorar: **gelecekteki bir olay, gecmisteki
bir satirin feature'ina sizabiliyor mu?**

Yontem "olay enjeksiyonu": bilinen TEK bir tarihe tek bir olay konur ve
ciktida o olayin etkisinin hangi gunden itibaren gorundugu olculur. Etki
``olay_gunu + horizon``dan ONCE gorunuyorsa sizinti vardir.

Bu, "kolon adinda shift yaziyor" turu bir kontrolden cok daha gucludur:
kaydirmanin GERCEKTEN uygulandigini davranisla kanitlar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.features.national import (
    add_annual_district_attribute,
    add_national_series,
    daily_from_hourly,
)
from gridup.features.point_events import add_point_event_features, daily_point_intensity

HORIZON = 5
OLAY_GUNU = pd.Timestamp("2026-03-15")


def _panel(n_gun: int = 60) -> pd.DataFrame:
    gunler = pd.date_range("2026-02-01", periods=n_gun, freq="D")
    yerler = ["bornova", "aliaga"]
    return pd.DataFrame([(y, g) for y in yerler for g in gunler], columns=["ilce_key", "tarih"])


def _koordinatlar() -> pd.DataFrame:
    return pd.DataFrame(
        {"ilce_key": ["bornova", "aliaga"], "lat": [38.47, 38.80], "lon": [27.22, 26.97]}
    )


def _tek_olay() -> pd.DataFrame:
    """Bornova'nin uzerine dusen TEK bir olay."""
    return pd.DataFrame({"tarih": [OLAY_GUNU], "lat": [38.47], "lon": [27.22], "frp": [100.0]})


def test_nokta_olayi_ufuktan_once_gorunmez() -> None:
    """Olay gunu + horizon'dan ONCEKI hicbir satir olaydan etkilenmemeli.

    Bu testin kirilmasi, tahmin aninda var olmayan bir yanginin/depremin
    modele girdigi anlamina gelir -- CV'de harika gorunup leaderboard'da
    coken hatanin ta kendisi.
    """
    sonuc = add_point_event_features(
        _panel(),
        _tek_olay(),
        _koordinatlar(),
        key_column="ilce_key",
        time_column="tarih",
        horizon=HORIZON,
        weight_column="frp",
        radii_km=(25.0,),
        windows=(7,),
        prefix="yangin",
    )
    bornova = sonuc[sonuc["ilce_key"] == "bornova"].sort_values("tarih")
    uretilen = [k for k in sonuc.columns if k.startswith("yangin_")]
    assert uretilen, "Hicbir yangin kolonu uretilmedi -- test anlamsiz olurdu."

    erken = bornova[bornova["tarih"] < OLAY_GUNU + pd.Timedelta(days=HORIZON)]
    for kolon in uretilen:
        degerler = pd.to_numeric(erken[kolon], errors="coerce").fillna(0.0)
        assert (degerler == 0).all(), (
            f"SIZINTI: '{kolon}' olay gunu+{HORIZON}'dan once sifir degil. "
            f"Ilk sifir olmayan: {erken.loc[degerler.ne(0), 'tarih'].min()}"
        )


def test_nokta_olayi_ufuktan_sonra_gorunur() -> None:
    """Kaydirma dogru ama feature OLU olmamali -- etki sonradan gorunmeli.

    Bir onceki test tek basina 'her seyi sifirla' diyerek de gecilebilirdi;
    bu test onu imkansiz kilar.
    """
    sonuc = add_point_event_features(
        _panel(),
        _tek_olay(),
        _koordinatlar(),
        key_column="ilce_key",
        time_column="tarih",
        horizon=HORIZON,
        weight_column="frp",
        radii_km=(25.0,),
        windows=(7,),
        prefix="yangin",
    )
    bornova = sonuc[sonuc["ilce_key"] == "bornova"].sort_values("tarih")
    gec = bornova[bornova["tarih"] >= OLAY_GUNU + pd.Timedelta(days=HORIZON)]
    kayan = [k for k in sonuc.columns if k.startswith("yangin_sayi") and "kayan" in k]
    assert kayan, "Kayan pencere kolonu uretilmedi."
    toplam = sum(float(pd.to_numeric(gec[k], errors="coerce").fillna(0).sum()) for k in kayan)
    assert toplam > 0, "Olay ufuktan sonra da gorunmuyor -- feature olu."


def test_uzak_ilce_olaydan_etkilenmez() -> None:
    """Yaricap disindaki ilce, olayi HIC gormemeli.

    Mesafe filtresi calismazsa tum ilceler ayni degeri alir ve feature
    mekansal ayirt ediciligini kaybeder.
    """
    koordinat = _koordinatlar()
    sonuc = add_point_event_features(
        _panel(),
        _tek_olay(),
        koordinat,
        key_column="ilce_key",
        time_column="tarih",
        horizon=HORIZON,
        radii_km=(5.0,),  # Aliaga ~40 km uzakta -> disarida kalmali
        windows=(7,),
        prefix="yangin",
    )
    aliaga = sonuc[sonuc["ilce_key"] == "aliaga"]
    for kolon in [k for k in sonuc.columns if k.startswith("yangin_")]:
        degerler = pd.to_numeric(aliaga[kolon], errors="coerce").fillna(0.0)
        assert (degerler == 0).all(), f"Yaricap disindaki ilce '{kolon}' degeri aldi."


def test_ham_gunluk_kolon_ciktida_birakilmaz() -> None:
    """Kaydirilmamis gunluk yogunluk sizintidir; frame'de kalmamali."""
    sonuc = add_point_event_features(
        _panel(),
        _tek_olay(),
        _koordinatlar(),
        key_column="ilce_key",
        time_column="tarih",
        horizon=HORIZON,
        radii_km=(25.0,),
        windows=(7,),
        prefix="yangin",
    )
    ham = daily_point_intensity(
        _tek_olay(), _koordinatlar(), key_column="ilce_key", radii_km=(25.0,), prefix="yangin"
    )
    ham_kolonlar = [k for k in ham.columns if k not in ("ilce_key", "tarih")]
    sizan = [k for k in ham_kolonlar if k in sonuc.columns]
    assert not sizan, f"Ham (kaydirilmamis) kolonlar ciktida kalmis: {sizan}"


def test_ulusal_seri_ufuktan_once_gorunmez() -> None:
    """Ulusal seri gerceklesmis veridir; ufuk kadar kaydirilmis olmali."""
    gunler = pd.date_range("2026-02-01", periods=60, freq="D")
    # Tek gunde ani sicrama: kaydirma yanlissa o sicrama erken gorunur.
    deger = np.where(gunler == OLAY_GUNU, 1_000_000.0, 1.0)
    ulusal = pd.DataFrame({"tarih": gunler, "tuketim_mean": deger})

    sonuc = add_national_series(
        _panel(), ulusal, time_column="tarih", horizon=HORIZON, windows=(7,), prefix="tr"
    )
    uretilen = [k for k in sonuc.columns if k.startswith("tr_")]
    assert uretilen, "Ulusal kolon uretilmedi."

    erken = sonuc[sonuc["tarih"] < OLAY_GUNU + pd.Timedelta(days=HORIZON)]
    for kolon in uretilen:
        degerler = pd.to_numeric(erken[kolon], errors="coerce").fillna(0.0)
        assert (degerler < 1_000.0).all(), (
            f"SIZINTI: '{kolon}' ulusal sicramayi ufuktan once gosteriyor."
        )


def test_yillik_oznitelik_ayni_yili_kullanmaz() -> None:
    """``year_lag=1`` varsayilani gecerli olmali: 2026 paneli 2025'i gorur.

    Yillik istatistik yil bitmeden yayimlanmaz; ayni yili kullanmak tahmin
    aninda var olmayan bir sayiyi modele sokar.
    """
    yillik = pd.DataFrame(
        {
            "ilce_key": ["bornova", "aliaga", "bornova", "aliaga"],
            "yil": [2025, 2025, 2026, 2026],
            "geceleme": [100.0, 200.0, 999_999.0, 999_999.0],
        }
    )
    sonuc = add_annual_district_attribute(
        _panel(),
        yillik,
        key_column="ilce_key",
        time_column="tarih",
        value_columns=["geceleme"],
        prefix="turizm",
    )
    degerler = set(pd.to_numeric(sonuc["turizm_geceleme"], errors="coerce").dropna().unique())
    assert 999_999.0 not in degerler, "SIZINTI: ayni yilin (2026) degeri kullanilmis."
    assert degerler == {100.0, 200.0}, f"Beklenen 2025 degerleri degil: {degerler}"


def test_saatlikten_gunluge_yerel_gun_siniri() -> None:
    """Gun siniri Europe/Istanbul'a gore kesilmeli, UTC'ye gore degil.

    UTC birakilirsa her gun 3 saat kayar ve gunluk toplamlar yanlis gune
    yazilir -- panel ile hava/tatil hizalamasi sessizce bozulur.
    """
    # UTC 21:00 = Turkiye ertesi gun 00:00. Bu saat YENI gune ait olmali.
    zaman = pd.date_range("2026-03-14T21:00", periods=3, freq="h", tz="UTC")
    saatlik = pd.DataFrame({"zaman": zaman, "consumption": [1.0, 2.0, 3.0]})
    gunluk = daily_from_hourly(saatlik, time_column="zaman", value_columns=["consumption"])
    assert set(gunluk["tarih"]) == {pd.Timestamp("2026-03-15")}, (
        f"UTC 21:00 Turkiye'de ertesi gundur; gunler: {list(gunluk['tarih'])}"
    )


@pytest.mark.parametrize("horizon", [0, -1])
def test_gecersiz_ufuk_reddedilir(horizon: int) -> None:
    """Ufuk 1'den kucuk olamaz -- sessizce kabul edilirse koruma kalkar."""
    with pytest.raises(ValueError, match="horizon"):
        add_point_event_features(
            _panel(),
            _tek_olay(),
            _koordinatlar(),
            key_column="ilce_key",
            time_column="tarih",
            horizon=horizon,
        )
    with pytest.raises(ValueError, match="horizon"):
        add_national_series(
            _panel(),
            pd.DataFrame({"tarih": pd.date_range("2026-02-01", periods=5), "x": 1.0}),
            time_column="tarih",
            horizon=horizon,
        )
