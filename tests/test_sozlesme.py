"""TASARIM SOZLESMESI testleri -- vaadi belgeden koda tasir.

``gridup/__init__.py`` su sozu veriyor::

    * Feature fonksiyonlari girdi DataFrame'ini ASLA degistirmez, yeni frame dondurur

Bu dosyaya kadar bu soz **dokumantasyondu**: tek tek yazilmis dokuz test
bazi fonksiyonlari kapsiyordu, gerisi kapsamsizdi. Yeni bir feature
fonksiyonu eklendiginde sozlesmenin ihlal edilip edilmedigini kimse
kontrol etmiyordu.

Burada sozlesme **zorlanir**:

  1. ``gridup.features.*`` icindeki, ilk parametresi DataFrame olan her
     public fonksiyon otomatik kesfedilir.
  2. Her biri icin bir cagri senaryosu KAYITLI olmak zorundadir.
  3. Kayitli olmayan bir fonksiyon varsa test **KIRILIR** -- yani yeni bir
     feature fonksiyonu yazan kisi ya senaryoyu ekler ya da testi gorur.

Ucuncu madde bu dosyanin asil degeridir: kapsam kendiliginden buyur.
"""

from __future__ import annotations

import importlib
import inspect

import numpy as np
import pandas as pd
import pytest

#: Ilk parametresi "girdi cercevesi" anlamina gelen adlar.
_FRAME_PARAMS = frozenset({"frame", "hourly", "daily", "panel", "left", "table"})

FEATURE_MODULLERI = (
    "temporal",
    "weather",
    "spatial",
    "solar",
    "aggregate",
    "categorical",
    "outage_reason",
)


def _ornek_panel(n_gun: int = 40, n_yer: int = 4) -> pd.DataFrame:
    """Her feature fonksiyonunu besleyebilecek genis bir ornek panel."""
    rng = np.random.default_rng(0)
    gunler = pd.date_range("2026-02-10", periods=n_gun, freq="D")
    yerler = [f"yer_{i}" for i in range(n_yer)]
    tarih = pd.Series(np.tile(gunler, n_yer))
    n = len(tarih)
    return pd.DataFrame(
        {
            "tarih": tarih,
            "yer": np.repeat(yerler, n_gun),
            "il": np.repeat(["izmir", "izmir", "mugla", "mugla"][:n_yer], n_gun),
            "hedef": rng.normal(50, 10, n),
            "sicaklik": rng.normal(18, 7, n),
            "nem": rng.uniform(30, 90, n),
            "yagis_mm": rng.exponential(1.5, n),
            "ruzgar": rng.uniform(0, 40, n),
            "gunes_ghi_gunluk": rng.uniform(1, 8, n),
            "kategori": rng.choice(["a", "b", "c"], n),
            "kategori2": rng.choice(["x", "y"], n),
            "sebep": rng.choice(["Yildirim dusmesi", "Kablo arizasi", "Agac temasi"], n),
            "sayi": rng.integers(1, 100, n),
        }
    )


KOORDINATLAR = {
    "yer_0": (38.42, 27.14),
    "yer_1": (38.63, 27.42),
    "yer_2": (37.21, 28.36),
    "yer_3": (36.63, 29.12),
}


def _komsuluk() -> pd.DataFrame:
    """Her yerin en yakin iki komsusu -- spatial fonksiyonlarinin bekledigi sekil."""
    satirlar = []
    for yer in KOORDINATLAR:
        for komsu in KOORDINATLAR:
            if komsu != yer:
                satirlar.append({"yer": yer, "komsu": komsu, "mesafe_km": 50.0})
    return pd.DataFrame(satirlar)


def _saatlik() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    saatler = pd.date_range("2026-02-10", periods=48, freq="h")
    return pd.DataFrame(
        {
            "tarih": np.tile(saatler, 2),
            "yer": np.repeat(["yer_0", "yer_1"], len(saatler)),
            "sicaklik": rng.normal(18, 5, len(saatler) * 2),
            "ruzgar_yonu": rng.uniform(0, 360, len(saatler) * 2),
        }
    )


#: Fonksiyon adi -> (modul, cagri). Cagri bir panel alip fonksiyonu calistirir.
#: Yeni bir feature fonksiyonu eklersen BURAYA da bir satir ekle; aksi halde
#: ``test_her_feature_fonksiyonu_kayitli`` kirilir.
SENARYOLAR: dict[str, tuple[str, object]] = {
    # --- temporal ---
    "add_calendar_features": ("temporal", lambda m, f: m.add_calendar_features(f, "tarih")),
    "add_cyclical_features": (
        "temporal",
        lambda m, f: m.add_cyclical_features(f.assign(ay=f["tarih"].dt.month), {"ay": 12}),
    ),
    "add_ramadan_features": ("temporal", lambda m, f: m.add_ramadan_features(f, "tarih")),
    "add_turkish_holiday_features": (
        "temporal",
        lambda m, f: m.add_turkish_holiday_features(f, "tarih"),
    ),
    "add_lag_features": (
        "temporal",
        lambda m, f: m.add_lag_features(
            f, "hedef", [1, 7], time_column="tarih", group_columns=["yer"]
        ),
    ),
    "add_rolling_features": (
        "temporal",
        lambda m, f: m.add_rolling_features(
            f, "hedef", [7], time_column="tarih", group_columns=["yer"]
        ),
    ),
    "add_expanding_features": (
        "temporal",
        lambda m, f: m.add_expanding_features(
            f, "hedef", time_column="tarih", group_columns=["yer"]
        ),
    ),
    # --- weather ---
    "aggregate_hourly_to_daily": (
        "weather",
        lambda m, f: m.aggregate_hourly_to_daily(
            _saatlik(), time_column="tarih", group_columns=["yer"]
        ),
    ),
    "add_regional_aggregates": (
        "weather",
        lambda m, f: m.add_regional_aggregates(
            f, time_column="tarih", value_columns=["sicaklik"]
        ),
    ),
    "add_physical_derivatives": (
        "weather",
        lambda m, f: m.add_physical_derivatives(
            f.rename(columns={"sicaklik": "sicaklik_ort", "yagis_mm": "yagis_toplam",
                              "ruzgar": "ruzgar_max"}),
            group_columns=["yer"], time_column="tarih",
            temperature_max=None, temperature_min=None, gust_max=None,
        ),
    ),
    "add_weather_accumulators": (
        "weather",
        lambda m, f: m.add_weather_accumulators(
            f, group_columns=["yer"], time_column="tarih", value_columns=["yagis_mm"]
        ),
    ),
    # --- spatial ---
    "add_neighbour_target_lag": (
        "spatial",
        lambda m, f: m.add_neighbour_target_lag(
            f, _komsuluk(), key_column="yer", time_column="tarih",
            target_column="hedef", horizon=1,
        ),
    ),
    "add_neighbour_feature_mean": (
        "spatial",
        lambda m, f: m.add_neighbour_feature_mean(
            f, _komsuluk(), key_column="yer", time_column="tarih",
            value_columns=["sicaklik"],
        ),
    ),
    # --- solar ---
    "add_solar_features": (
        "solar",
        lambda m, f: m.add_solar_features(
            f, time_column="tarih", location_column="yer",
            coordinates=KOORDINATLAR, geometry_only=True,
        ),
    ),
    "add_clearness_index": (
        "solar",
        lambda m, f: m.add_clearness_index(f, observed_column="sicaklik"),
    ),
    # --- aggregate ---
    "add_group_statistics": (
        "aggregate",
        lambda m, f: m.add_group_statistics(f, group_columns=["yer"], value_columns=["sicaklik"]),
    ),
    "add_ratio_features": (
        "aggregate",
        lambda m, f: m.add_ratio_features(f, [("sicaklik", "nem")]),
    ),
    # --- categorical ---
    "add_frequency_encoding": (
        "categorical",
        lambda m, f: m.add_frequency_encoding(f, ["kategori"]),
    ),
    "add_count_encoding": ("categorical", lambda m, f: m.add_count_encoding(f, ["kategori"])),
    "add_combination_features": (
        "categorical",
        lambda m, f: m.add_combination_features(f, [("kategori", "kategori2")]),
    ),
    "reduce_rare_categories": (
        "categorical",
        lambda m, f: m.reduce_rare_categories(f, ["kategori"], min_count=1000),
    ),
    # --- outage_reason ---
    "add_reason_features": (
        "outage_reason",
        lambda m, f: m.add_reason_features(f, "sebep"),
    ),
}


def _frame_alan_fonksiyonlar() -> dict[str, str]:
    """``gridup.features.*`` icindeki, ilk parametresi cerceve olan public fonksiyonlar."""
    bulunan: dict[str, str] = {}
    for modul_adi in FEATURE_MODULLERI:
        modul = importlib.import_module(f"gridup.features.{modul_adi}")
        for ad in getattr(modul, "__all__", []):
            nesne = getattr(modul, ad, None)
            if not callable(nesne) or isinstance(nesne, type):
                continue
            try:
                parametreler = list(inspect.signature(nesne).parameters)
            except (TypeError, ValueError):  # pragma: no cover - C fonksiyonu
                continue
            if parametreler and parametreler[0] in _FRAME_PARAMS:
                bulunan[ad] = modul_adi
    return bulunan


def test_her_feature_fonksiyonu_kayitli():
    """Yeni bir feature fonksiyonu eklenirse bu test KIRILIR -- kasitli.

    Sozlesme testinin kapsami kendiliginden buyusun diye. Kirildiginda
    yapilacak sey: ``SENARYOLAR``a bir satir eklemek.
    """
    kesfedilen = set(_frame_alan_fonksiyonlar())
    kayitli = set(SENARYOLAR)

    eksik = sorted(kesfedilen - kayitli)
    fazla = sorted(kayitli - kesfedilen)

    assert not eksik, (
        f"Su feature fonksiyonlari sozlesme testinde KAPSANMIYOR: {eksik}. "
        "tests/test_sozlesme.py icindeki SENARYOLAR sozlugune ekle."
    )
    assert not fazla, (
        f"Su senaryolar artik var olmayan fonksiyonlara isaret ediyor: {fazla}."
    )


@pytest.mark.parametrize("fonksiyon_adi", sorted(SENARYOLAR))
def test_feature_fonksiyonu_girdiyi_degistirmiyor(fonksiyon_adi: str):
    """SOZLESME: feature fonksiyonu girdi frame'ini ASLA degistirmez."""
    modul_adi, cagri = SENARYOLAR[fonksiyon_adi]
    if modul_adi == "temporal" and fonksiyon_adi in {
        "add_ramadan_features", "add_turkish_holiday_features"
    }:
        pytest.importorskip("hijridate" if "ramadan" in fonksiyon_adi else "holidays")

    modul = importlib.import_module(f"gridup.features.{modul_adi}")
    panel = _ornek_panel()
    onceki = panel.copy(deep=True)

    cagri(modul, panel)

    pd.testing.assert_frame_equal(
        panel, onceki,
        obj=f"{fonksiyon_adi} girdi frame'ini DEGISTIRDI -- sozlesme ihlali",
    )


@pytest.mark.parametrize("fonksiyon_adi", sorted(SENARYOLAR))
def test_feature_fonksiyonu_yeni_frame_donduruyor(fonksiyon_adi: str):
    """SOZLESME: donen sey girdiyle AYNI NESNE olmamali."""
    modul_adi, cagri = SENARYOLAR[fonksiyon_adi]
    if modul_adi == "temporal" and fonksiyon_adi in {
        "add_ramadan_features", "add_turkish_holiday_features"
    }:
        pytest.importorskip("hijridate" if "ramadan" in fonksiyon_adi else "holidays")

    modul = importlib.import_module(f"gridup.features.{modul_adi}")
    panel = _ornek_panel()

    sonuc = cagri(modul, panel)

    assert isinstance(sonuc, pd.DataFrame)
    assert sonuc is not panel, f"{fonksiyon_adi} girdinin KENDISINI dondurdu"


@pytest.mark.parametrize("fonksiyon_adi", sorted(SENARYOLAR))
def test_feature_fonksiyonu_satir_sayisini_korumali(fonksiyon_adi: str):
    """Feature ekleme satir SAYISINI degistirmemeli.

    Istisna: ``aggregate_hourly_to_daily`` kasitli olarak indirger (saatlik
    -> gunluk). Onun sozlesmesi farklidir ve adinda yazar.
    """
    if fonksiyon_adi == "aggregate_hourly_to_daily":
        pytest.skip("kasitli indirgeme -- saatlikten gunluge")

    modul_adi, cagri = SENARYOLAR[fonksiyon_adi]
    if modul_adi == "temporal" and fonksiyon_adi in {
        "add_ramadan_features", "add_turkish_holiday_features"
    }:
        pytest.importorskip("hijridate" if "ramadan" in fonksiyon_adi else "holidays")

    modul = importlib.import_module(f"gridup.features.{modul_adi}")
    panel = _ornek_panel()

    sonuc = cagri(modul, panel)

    assert len(sonuc) == len(panel), (
        f"{fonksiyon_adi} satir sayisini {len(panel)} -> {len(sonuc)} degistirdi"
    )
