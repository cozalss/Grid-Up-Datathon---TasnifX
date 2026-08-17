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
    "school",
    # 2026-08-17: harici veriyi panele baglayan iki yeni modul. Listeye
    # EKLENMEZSE kendiliginden buyuyen tarama onlara ULASMAZ -- ayni bosluk
    # pipeline.py'de yasandi ve kapatildi; tekrarlamamak icin buradalar.
    "point_events",
    "national",
)


def _ornek_panel(n_gun: int = 40, n_yer: int = 4) -> pd.DataFrame:
    """Her feature fonksiyonunu besleyebilecek genis bir ornek panel."""
    rng = np.random.default_rng(0)
    gunler = pd.date_range("2026-02-10", periods=n_gun, freq="D")
    # Gorunum sirasi KASITLI olarak alfabetik DEGIL: pd.factorize ile
    # groupby(sort=True) arasindaki uyusmazlik ancak boyle yakalanir.
    yerler = ["zeytinburnu", "aliaga", "menemen", "bornova"][:n_yer]
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
    "zeytinburnu": (38.42, 27.14),
    "aliaga": (38.63, 27.42),
    "menemen": (37.21, 28.36),
    "bornova": (36.63, 29.12),
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
            "yer": np.repeat(["zeytinburnu", "aliaga"], len(saatler)),
            "sicaklik": rng.normal(18, 5, len(saatler) * 2),
            "ruzgar_yonu": rng.uniform(0, 360, len(saatler) * 2),
        }
    )


#: SATIR SAYISINI KORUMASI BEKLENMEYEN fonksiyonlar -- kasitli indirgeyiciler.
#: Bunlar "feature ekleyen" degil "cozunurluk dusuren" fonksiyonlardir
#: (saatlik -> gunluk). Sozlesme 1'in satir-sayisi ve satir-kimligi
#: kontrolleri onlara UYGULANMAZ; ama girdiyi degistirmeme kurali gecerlidir.
#: Liste ACIK tutulur: yeni bir indirgeyici eklenirse buraya yazilir, yoksa
#: test kirilir ve yazan kisi "bu gercekten indirgeyici mi" sorusunu cevaplar.
KASITLI_INDIRGEYENLER = frozenset({"aggregate_hourly_to_daily", "daily_from_hourly"})


#: Fonksiyon adi -> (modul, cagri). Cagri bir panel alip fonksiyonu calistirir.
#: Yeni bir feature fonksiyonu eklersen BURAYA da bir satir ekle; aksi halde
#: ``test_her_feature_fonksiyonu_kayitli`` kirilir.
SENARYOLAR: dict[str, tuple[str, object]] = {
    # --- point_events (harici nokta olaylari: yangin, deprem) ---
    "add_point_event_features": (
        "point_events",
        lambda m, f: m.add_point_event_features(
            f,
            pd.DataFrame(
                {
                    "tarih": pd.to_datetime(["2026-02-12", "2026-02-20", "2026-03-01"]),
                    "lat": [38.42, 38.63, 37.21],
                    "lon": [27.14, 27.42, 28.36],
                    "frp": [12.0, 45.0, 3.0],
                }
            ),
            pd.DataFrame(
                {
                    "yer": list(KOORDINATLAR),
                    "lat": [v[0] for v in KOORDINATLAR.values()],
                    "lon": [v[1] for v in KOORDINATLAR.values()],
                }
            ),
            key_column="yer",
            time_column="tarih",
            horizon=2,
            weight_column="frp",
            windows=(7,),
            prefix="yangin",
        ),
    ),
    # --- national (ulusal saatlik seri + yillik ilce ozniteligi) ---
    "daily_from_hourly": (
        "national",
        lambda m, f: m.daily_from_hourly(
            pd.DataFrame(
                {
                    "zaman": pd.date_range("2026-02-10", periods=48, freq="h", tz="UTC"),
                    "consumption": np.linspace(30000, 40000, 48),
                }
            ),
            time_column="zaman",
            value_columns=["consumption"],
        ),
    ),
    "add_national_series": (
        "national",
        lambda m, f: m.add_national_series(
            f,
            pd.DataFrame(
                {
                    "tarih": pd.date_range("2026-02-01", periods=60, freq="D"),
                    "consumption_mean": np.linspace(30000, 40000, 60),
                }
            ),
            time_column="tarih",
            horizon=2,
            windows=(7,),
            prefix="tr",
        ),
    ),
    "add_annual_district_attribute": (
        "national",
        lambda m, f: m.add_annual_district_attribute(
            f,
            pd.DataFrame(
                {
                    "yer": list(KOORDINATLAR) * 2,
                    "yil": [2025] * len(KOORDINATLAR) + [2026] * len(KOORDINATLAR),
                    "geceleme": list(range(1, 2 * len(KOORDINATLAR) + 1)),
                }
            ),
            key_column="yer",
            time_column="tarih",
            value_columns=["geceleme"],
            prefix="turizm",
        ),
    ),
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
            f, "hedef", shifts=[1, 7], time_column="tarih", horizon=1, group_columns=["yer"]
        ),
    ),
    "add_rolling_features": (
        "temporal",
        lambda m, f: m.add_rolling_features(
            f, "hedef", [7], time_column="tarih", horizon=1, group_columns=["yer"]
        ),
    ),
    "add_expanding_features": (
        "temporal",
        lambda m, f: m.add_expanding_features(
            f, "hedef", time_column="tarih", horizon=1, group_columns=["yer"]
        ),
    ),
    "add_previous_month_features": (
        "temporal",
        lambda m, f: m.add_previous_month_features(
            f, "hedef", time_column="tarih", horizon=1, group_columns=["yer"]
        ),
    ),
    "add_mass_event_features": (
        "temporal",
        lambda m, f: m.add_mass_event_features(
            f, "hedef", time_column="tarih", horizon=1, group_columns=["yer"]
        ),
    ),
    "add_event_decay_features": (
        "temporal",
        lambda m, f: m.add_event_decay_features(
            f, "hedef", time_column="tarih", horizon=1, group_columns=["yer"]
        ),
    ),
    "add_days_since_event_features": (
        "temporal",
        lambda m, f: m.add_days_since_event_features(
            f, "hedef", time_column="tarih", horizon=1, group_columns=["yer"]
        ),
    ),
    "add_upcoming_holiday_features": (
        "temporal",
        lambda m, f: m.add_upcoming_holiday_features(f, "tarih"),
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
        lambda m, f: m.add_regional_aggregates(f, time_column="tarih", value_columns=["sicaklik"]),
    ),
    "add_physical_derivatives": (
        "weather",
        lambda m, f: m.add_physical_derivatives(
            f.rename(
                columns={
                    "sicaklik": "sicaklik_ort",
                    "yagis_mm": "yagis_toplam",
                    "ruzgar": "ruzgar_max",
                }
            ),
            group_columns=["yer"],
            time_column="tarih",
            temperature_max=None,
            temperature_min=None,
            gust_max=None,
        ),
    ),
    "add_weather_accumulators": (
        "weather",
        lambda m, f: m.add_weather_accumulators(
            f, group_columns=["yer"], time_column="tarih", value_columns=["yagis_mm"]
        ),
    ),
    "add_consecutive_extreme_days": (
        "weather",
        lambda m, f: m.add_consecutive_extreme_days(
            f, "sicaklik", time_column="tarih", group_columns=["yer"], threshold=20.0
        ),
    ),
    "add_precip_anomaly": (
        "weather",
        lambda m, f: m.add_precip_anomaly(
            f, "yagis_mm", time_column="tarih", group_columns=["yer"], windows=(7,)
        ),
    ),
    # --- spatial ---
    "add_neighbour_target_lag": (
        "spatial",
        lambda m, f: m.add_neighbour_target_lag(
            f,
            _komsuluk(),
            key_column="yer",
            time_column="tarih",
            target_column="hedef",
            horizon=1,
        ),
    ),
    "add_neighbour_feature_mean": (
        "spatial",
        lambda m, f: m.add_neighbour_feature_mean(
            f,
            _komsuluk(),
            key_column="yer",
            time_column="tarih",
            value_columns=["sicaklik"],
            target_column=None,
        ),
    ),
    # --- solar ---
    "add_solar_features": (
        "solar",
        lambda m, f: m.add_solar_features(
            f,
            time_column="tarih",
            location_column="yer",
            coordinates=KOORDINATLAR,
            geometry_only=True,
        ),
    ),
    "add_clearness_index": (
        "solar",
        lambda m, f: m.add_clearness_index(f, observed_column="sicaklik"),
    ),
    # --- aggregate ---
    "add_group_statistics": (
        "aggregate",
        lambda m, f: m.add_group_statistics(
            f, group_columns=["yer"], value_columns=["sicaklik"], target_column=None
        ),
    ),
    "add_ratio_features": (
        "aggregate",
        lambda m, f: m.add_ratio_features(f, [("sicaklik", "nem")]),
    ),
    # --- school ---
    "add_school_calendar_features": (
        "school",
        lambda m, f: m.add_school_calendar_features(f, "tarih"),
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
    assert not fazla, f"Su senaryolar artik var olmayan fonksiyonlara isaret ediyor: {fazla}."


@pytest.mark.parametrize("fonksiyon_adi", sorted(SENARYOLAR))
def test_feature_fonksiyonu_girdiyi_degistirmiyor(fonksiyon_adi: str):
    """SOZLESME: feature fonksiyonu girdi frame'ini ASLA degistirmez."""
    modul_adi, cagri = SENARYOLAR[fonksiyon_adi]
    if modul_adi == "temporal" and fonksiyon_adi in {
        "add_ramadan_features",
        "add_turkish_holiday_features",
    }:
        pytest.importorskip("hijridate" if "ramadan" in fonksiyon_adi else "holidays")

    modul = importlib.import_module(f"gridup.features.{modul_adi}")
    panel = _ornek_panel()
    onceki = panel.copy(deep=True)

    cagri(modul, panel)

    pd.testing.assert_frame_equal(
        panel,
        onceki,
        obj=f"{fonksiyon_adi} girdi frame'ini DEGISTIRDI -- sozlesme ihlali",
    )


@pytest.mark.parametrize("fonksiyon_adi", sorted(SENARYOLAR))
def test_feature_fonksiyonu_yeni_frame_donduruyor(fonksiyon_adi: str):
    """SOZLESME: donen sey girdiyle AYNI NESNE olmamali."""
    modul_adi, cagri = SENARYOLAR[fonksiyon_adi]
    if modul_adi == "temporal" and fonksiyon_adi in {
        "add_ramadan_features",
        "add_turkish_holiday_features",
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
    if fonksiyon_adi in KASITLI_INDIRGEYENLER:
        pytest.skip("kasitli indirgeme -- cozunurluk dusuruyor")

    modul_adi, cagri = SENARYOLAR[fonksiyon_adi]
    if modul_adi == "temporal" and fonksiyon_adi in {
        "add_ramadan_features",
        "add_turkish_holiday_features",
    }:
        pytest.importorskip("hijridate" if "ramadan" in fonksiyon_adi else "holidays")

    modul = importlib.import_module(f"gridup.features.{modul_adi}")
    panel = _ornek_panel()

    sonuc = cagri(modul, panel)

    assert len(sonuc) == len(panel), (
        f"{fonksiyon_adi} satir sayisini {len(panel)} -> {len(sonuc)} degistirdi"
    )


# --------------------------------------------------------------------------
# SOZLESME 2: hedefe dokunan her fonksiyon ya fold-farkinda olmali
#             ya da NEDEN olmadigi kayitli olmali
# --------------------------------------------------------------------------

#: Hedef anlamina gelen parametre adlari.
_HEDEF_PARAMS = frozenset({"target", "target_column", "y", "y_true", "value_column"})

#: Fold farkindaligi anlamina gelen parametre adlari.
_FOLD_PARAMS = frozenset({"folds", "fold", "fold_list", "cv"})

HEDEF_MODULLERI = (
    "features.categorical",
    "features.aggregate",
    "features.spatial",
    "features.temporal",
    "selection",
    "two_stage",
    "ensemble",
    "models",
    "refit",
    "zoo",
    "tuning",
    "ablation",
    "neural",
    "metrics",
    "weighting",
    # 2026-08-17: ``pipeline`` bu listeye SONRADAN eklendi. Modul
    # ``gridup.features.*`` agacinin DISINDA dogdugu icin kendiliginden buyuyen
    # tarama ona ulasmiyordu: ``build_paired_history_features`` ve
    # ``build_paired_distribution_features`` hedef parametresi alip fold almadigi
    # halde hicbir gerekce yazilmadan geciyordu. Bu, kod duzeyinde degil KAPSAM
    # MEKANIZMASI duzeyinde bir bosluktu -- yarin pipeline.py'ye eklenecek ucuncu
    # bir fonksiyon hicbir testi kirmayacakti.
    "pipeline",
)

#: Hedefe dokunan ama fold ALMAYAN fonksiyonlar ve bunun NEDEN guvenli oldugu.
#: Bu sozluk bir muafiyet listesi degil, bir GEREKCE kaydidir: her satir
#: "bu fonksiyon neden sizinti yapmiyor" sorusunu cevaplar.
FOLDSUZ_GEREKCE: dict[str, str] = {
    # Yalnizca REDDEDER -- hedeften hicbir sey turetmez, hicbir sey dondurmez.
    "models.assert_finite_target": (
        "hedefte NaN/inf varsa hata firlatir; deger okumaz, istatistik "
        "cikarmaz, cikti uretmez -- fold kavrami uygulanamaz"
    ),
    # Saf skorlama -- model egitmez, hicbir sey ogrenmez.
    "metrics.rmse": "saf skorlama, fit yok",
    "metrics.rmsle": "saf skorlama, fit yok",
    "metrics.mape": "saf skorlama, fit yok",
    "metrics.mape_coverage": "saf tanisal olcum, fit yok",
    "metrics.smape": "saf skorlama, fit yok",
    "metrics.log_transform_target": "geri cevrilebilir donusum, istatistik ogrenmez",
    "metrics.sqrt_transform_target": "geri cevrilebilir donusum, istatistik ogrenmez",
    # OOF tahminler UZERINDE calisir -- girdisi zaten fold-disidir.
    "metrics.optimize_threshold": "OOF tahminler uzerinde calisir (docstring'de zorunlu kilinmis)",
    "metrics.tune_final_multiplier": (
        "OOF tahminler uzerinde carpan tarar (docstring'de zorunlu kilinmis; "
        "covered maskesi de bunun icin var); fit yok"
    ),
    "metrics.soften_outliers": (
        "hedef TANIMINI degistiren egitim-oncesi donusum; yalnizca TRAIN "
        "hedefine uygulanir, tahminlere asla (docstring); CV icinde train-fold "
        "dilimine uygulanmasi da docstring'de sart kosulmus -- fit/skor uretmez"
    ),
    "weighting.recency_activity_weights": (
        "agirlik uretir, feature degil: modele deger olarak girmez, yalnizca "
        "train satirlarinin kaybini olcekler; aktiflik orani geriye-donuk "
        "penceredir ve cross_validate agirligi train-fold dilimiyle kullanir"
    ),
    "ensemble.hill_climb_weights": "girdisi OOF tahmin matrisi -- fold zaten uygulanmis",
    "ensemble.greedy_forward_selection": "girdisi OOF tahmin matrisi",
    "ensemble.prune_by_correlation": "girdisi OOF tahmin matrisi",
    "two_stage.tune_threshold": "OOF olasiliklar uzerinde esik arar",
    "two_stage.zero_baseline_score": "sabit tahminin skoru, fit yok",
    # Giris noktalarinin paylastigi asamalar: fit YALNIZCA train uzerinde.
    "pipeline.build_paired_distribution_features": (
        "hedeften hicbir sey turetmez: grup istatistigi ve frekans kodlamasi "
        "yalnizca ``value_columns``/``frequency_columns`` uzerinde hesaplanir ve "
        "``reference=train`` ile YALNIZCA train'e fit edilip test'e uygulanir. "
        "``target_column`` burada bir VERI KAYNAGI degil, bir REDDETME kapisidir: "
        "``add_group_statistics`` icindeki ``_reject_target`` hedefin kazara "
        "``value_columns``a girmesini engeller. Hedef degeri hicbir ciktiya "
        "girmediginden fold kavrami uygulanmaz"
    ),
    "pipeline.build_paired_history_features": (
        "``value_column`` hedef OLABILIR, ama ucu birden kapali: (1) kaydirma "
        "``add_lag_features``/``add_rolling_features``a devredilir ve orada "
        "``shift >= horizon`` sozlesmesi ValueError ile zorlanir -- yani yalnizca "
        "gecmise bakar; (2) test hedefi tasiyorsa ACIK ValueError; (3) tasimiyorsa "
        "NaN yer tutucu konur, boylece hicbir test hedefi lag/rolling penceresine "
        "giremez. ``target_column`` ``_ZORUNLU`` nobetcisiyle acikca istenir, "
        "sessiz varsayilani yoktur (test_pipeline_core_contract ile kilitli)"
    ),
    # Nedensel olarak guvenli: yalnizca GECMISE bakar (shift/horizon).
    "features.temporal.add_lag_features": "shift(horizon) -- yalnizca gecmis, ileri bakmaz",
    "features.temporal.add_rolling_features": "closed='left' -- mevcut satiri DISLAR",
    "features.temporal.add_expanding_features": "shift(1) -- mevcut satiri DISLAR",
    "features.temporal.add_previous_month_features": (
        "sonu (satir - horizon)'u gecmeyen SON TAM takvim ayindan hesaplar -- "
        "yalnizca gecmis; ay siniri asan test blogu icin ufuk disiplini "
        "testle kanitli (test_2024_birinci_taktikleri)"
    ),
    "features.temporal.add_mass_event_features": (
        "horizon >= 1 ZORUNLU ve yalnizca takvim-kaydirilmis gunluk pay "
        "yayinlanir; ayni-gun payi hedef sizintisi olurdu ve URETILMEZ -- "
        "leak testi test_kazanan_taktikleri'nde"
    ),
    "features.temporal.add_event_decay_features": (
        "horizon >= 1 ZORUNLU; bozunum ham seri uzerinde hesaplanip horizon "
        "kadar kaydirilarak yayinlanir -- d gunundeki deger yalnizca "
        "<= d - horizon gozlemlerinden gelir; leak testi test_derin_kazi'da"
    ),
    "features.temporal.add_days_since_event_features": (
        "horizon >= 1 ZORUNLU; son olay ufuk-kaydirilmis seride aranir, ayni "
        "gunun olayi hicbir satira gorunmez -- leak testi test_derin_kazi'da"
    ),
    "ensemble.tune_power_mean": (
        "girdisi OOF tahmin matrisi -- fold zaten uygulanmis; covered maskesi "
        "dolgu satirlarini skordan dislamak icin var"
    ),
    "features.spatial.add_neighbour_target_lag": "horizon>=1 zorunlu -- komsunun gecmisi",
    "features.spatial.add_neighbour_feature_mean": (
        "target_column YALNIZCA reddetme icin; bu fonksiyon KAYDIRMA YAPMAZ, "
        "hedef value_columns'a girerse komsunun AYNI GUNKU hedefi feature olur "
        "(olculdu: corr 0.813099) -- bu yuzden target_column acikca istenir ve "
        "hedef verilirse ValueError firlatilir"
    ),
    # Hedefi feature'a CEVIRMEZ; hedefi yalnizca reddetmek icin tanir.
    "features.aggregate.add_group_statistics": (
        "target_column YALNIZCA reddetme icin; hedef value_columns'a girerse "
        "_reject_target ValueError firlatir"
    ),
    "features.aggregate.add_target_free_aggregates": "adinda: hedef kullanmaz",
    # Bilincli tam-veri egitimi -- dogrulama degil, final refit.
    "models.fit_without_validation": "bilincli tam-veri refit; skor URETMEZ",
    "refit.multi_seed_refit": "bilincli tam-veri refit; tur sayisi CV'den DISARIDAN gelir",
    # Kendi ic bolmesini kurar.
    "selection.null_importance_filter": "hedefi permute eder; ic bolmesini kendi kurar",
}


def _hedefe_dokunanlar() -> dict[str, bool]:
    """``tam_ad -> fold_farkinda_mi`` esleme."""
    sonuc: dict[str, bool] = {}
    for modul_adi in HEDEF_MODULLERI:
        modul = importlib.import_module(f"gridup.{modul_adi}")
        for ad in getattr(modul, "__all__", []):
            nesne = getattr(modul, ad, None)
            if not callable(nesne) or isinstance(nesne, type):
                continue
            try:
                parametreler = set(inspect.signature(nesne).parameters)
            except (TypeError, ValueError):  # pragma: no cover
                continue
            if parametreler & _HEDEF_PARAMS:
                sonuc[f"{modul_adi}.{ad}"] = bool(parametreler & _FOLD_PARAMS)
    return sonuc


def test_hedefe_dokunan_her_fonksiyon_ya_fold_alir_ya_gerekcelidir():
    """SOZLESME: 'Hedef kullanan her kodlama fold-disi calisir -- sizinti imkansiz'.

    Bu test o vaadi API yuzeyinde zorlar. Hedef parametresi alan bir
    fonksiyon ya ``folds`` da almalidir, ya da ``FOLDSUZ_GEREKCE`` icinde
    NEDEN guvenli oldugu yazili olmalidir.

    Yeni bir fonksiyon eklendiginde bu test kirilir ve yazan kisi
    "bu hedefe dokunuyor, sizinti yapar mi?" sorusunu cevaplamak zorunda kalir.
    """
    dokunanlar = _hedefe_dokunanlar()
    foldsuz = {ad for ad, fold_var in dokunanlar.items() if not fold_var}

    gerekcesiz = sorted(foldsuz - set(FOLDSUZ_GEREKCE))
    assert not gerekcesiz, (
        "Su fonksiyonlar hedefe dokunuyor, fold ALMIYOR ve gerekceleri KAYITLI DEGIL: "
        f"{gerekcesiz}. Ya folds parametresi ekle, ya FOLDSUZ_GEREKCE'ye neden "
        "guvenli oldugunu yaz. Bos birakma."
    )

    # Ters yon: artik var olmayan veya artik fold alan bir fonksiyon icin
    # gerekce tutmak, kaydin cürümesidir.
    gereksiz = sorted(set(FOLDSUZ_GEREKCE) - foldsuz)
    assert not gereksiz, (
        f"Su gerekceler artik gecersiz (fonksiyon yok ya da artik fold aliyor): {gereksiz}"
    )


def test_grup_istatistigi_hedefi_reddediyor():
    """add_group_statistics'in tek koruma noktasi GERCEKTEN atesleniyor mu?"""
    from gridup.features.aggregate import add_group_statistics

    panel = _ornek_panel()
    with pytest.raises(ValueError, match="fold-disi DEGILDIR"):
        add_group_statistics(
            panel, group_columns=["yer"], value_columns=["hedef"], target_column="hedef"
        )


def test_oof_target_encode_foldsuz_calismiyor():
    """Hedef kodlamanin fold'suz cagrilmasi YAPISAL olarak imkansiz olmali."""
    from gridup.features.categorical import oof_target_encode

    parametreler = inspect.signature(oof_target_encode).parameters
    assert "folds" in parametreler
    assert parametreler["folds"].default is inspect.Parameter.empty, (
        "folds'un varsayilani var -- fold'suz cagrilabilir hale gelmis"
    )


@pytest.mark.parametrize("fonksiyon_adi", sorted(SENARYOLAR))
def test_feature_fonksiyonu_satir_sirasini_da_korumali(fonksiyon_adi: str):
    """SOZLESME: feature ekleme satir SIRASINI da degistirmemeli.

    Satir sayisini korumak YETMEZ. Fold'lar KONUMSAL indekstir; bir feature
    fonksiyonu satirlari yeniden sirlarsa, daha once hesaplanmis fold'lar
    artik baska satirlara isaret eder ve **hata vermez**. Model yanlis
    hedeflerle egitilir, CV skoru anlamsizlasir.

    OLCULDU: ``add_weather_accumulators`` ve ``add_physical_derivatives``
    tam olarak bunu yapiyordu -- girdi [bornova x5, aliaga x5] iken cikti
    [aliaga x5, bornova x5] donuyordu.
    """
    if fonksiyon_adi in KASITLI_INDIRGEYENLER:
        pytest.skip("kasitli indirgeme -- satir kimligi degisir")

    modul_adi, cagri = SENARYOLAR[fonksiyon_adi]
    if modul_adi == "temporal" and fonksiyon_adi in {
        "add_ramadan_features",
        "add_turkish_holiday_features",
    }:
        pytest.importorskip("hijridate" if "ramadan" in fonksiyon_adi else "holidays")

    modul = importlib.import_module(f"gridup.features.{modul_adi}")
    panel = _ornek_panel()

    sonuc = cagri(modul, panel)

    for kimlik_kolonu in ("yer", "tarih"):
        if kimlik_kolonu in panel.columns and kimlik_kolonu in sonuc.columns:
            assert list(sonuc[kimlik_kolonu]) == list(panel[kimlik_kolonu]), (
                f"{fonksiyon_adi} satir SIRASINI degistirdi ('{kimlik_kolonu}' kolonunda). "
                "Fold'lar konumsaldir -- bu sessizce yanlis egitim demektir."
            )


# --------------------------------------------------------------------------
# REGRESYON: gorunum sirasi != alfabetik sira oldugunda grup kaymasi
# --------------------------------------------------------------------------


def _kayma_paneli() -> pd.DataFrame:
    """Gorunum sirasi alfabetik OLMAYAN, degerleri buyuk olcude ayrisan panel.

    Degerler kasitli olarak 1000 kat farkli: gruplar kayarsa fark aninda
    gorunur, kucuk sayisal fark degil.
    """
    adlar = ["zeytinburnu", "aliaga", "menemen"]
    kayit = []
    for i, il in enumerate(adlar):
        for gun, t in enumerate(pd.date_range("2025-01-01", periods=8)):
            kayit.append({"tarih": t, "ilce": il, "y": float((i + 1) * 1000 + gun)})
    return pd.DataFrame(kayit).sample(frac=1, random_state=7).reset_index(drop=True)


@pytest.mark.parametrize(
    ("fonksiyon", "kolon", "beklenen"),
    [
        ("add_rolling_features", "y_kayan3_mean", "rolling"),
        ("add_expanding_features", "y_genisleyen_mean", "expanding"),
    ],
)
def test_kayan_pencere_dogru_gruptan_okuyor(fonksiyon, kolon, beklenen):
    """REGRESYON P0: ic groupby sort=True iken degerler BASKA gruba yaziliyordu.

    ``_sorted_view`` satirlari ``pd.factorize`` ile GORUNUM sirasina dizer;
    ikinci groupby varsayilan ``sort=True`` ile ALFABETIK sirlar. Iki sira
    uyusmadiginda bornova'nin kayan ortalamasi aliaga'nin degerlerinden
    hesaplanir -- hatasiz, ayni satir sayisiyla, tamamen sessizce.

    Bu test ancak grup adlarinin gorunum sirasi alfabetik DEGILSE hatayi
    yakalar; onceki fixture'lar ('TR001','TR002') zaten alfabetikti.
    """
    from gridup.features import temporal

    panel = _kayma_paneli()
    if beklenen == "rolling":
        cikti = temporal.add_rolling_features(
            panel, "y", [3], time_column="tarih", horizon=1, group_columns=["ilce"]
        )
    else:
        cikti = temporal.add_expanding_features(
            panel, "y", time_column="tarih", horizon=1, group_columns=["ilce"]
        )

    for il in panel["ilce"].unique():
        grup = panel[panel.ilce == il].sort_values("tarih")
        kaydirilmis = grup["y"].shift(1)
        referans = (
            kaydirilmis.rolling(3, min_periods=1).mean()
            if beklenen == "rolling"
            else kaydirilmis.expanding(min_periods=1).mean()
        ).to_numpy()
        gercek = cikti[cikti.ilce == il].sort_values("tarih")[kolon].to_numpy()

        assert np.allclose(referans, gercek, equal_nan=True), (
            f"'{il}' icin {beklenen} degerleri BASKA gruptan geliyor.\n"
            f"  beklenen: {np.round(referans, 1)}\n"
            f"  gercek  : {np.round(gercek, 1)}"
        )


def test_gunes_tekrarli_indexte_ilceleri_karistirmiyor():
    """REGRESYON P0: pd.concat([train, test]) sonrasi index tekrarli olur ve
    etiket hizalamasi ilcelerin gunes degerlerini SESSIZCE takas ediyordu."""
    from gridup.features.solar import add_solar_features

    koordinat = {"kuzey": (60.0, 25.0), "guney": (20.0, 25.0)}
    gunler = pd.date_range("2024-01-10", periods=4, freq="D")
    panel = pd.concat(
        [
            pd.DataFrame({"tarih": gunler, "yer": "kuzey"}),
            pd.DataFrame({"tarih": gunler, "yer": "guney"}),
        ]
    )
    assert not panel.index.is_unique, "test kurulumu bozuk: index benzersiz olmamali"

    cikti = add_solar_features(
        panel,
        time_column="tarih",
        location_column="yer",
        coordinates=koordinat,
        geometry_only=True,
    )

    kuzey = cikti.loc[cikti.yer == "kuzey", "gun_uzunlugu_saat"].mean()
    guney = cikti.loc[cikti.yer == "guney", "gun_uzunlugu_saat"].mean()
    # Ocak ayinda 60N kutba yakin -> KISA gun; 20N ekvatora yakin -> uzun gun.
    assert kuzey < guney, (
        f"Ocak'ta 60N ({kuzey:.2f} sa) 20N'den ({guney:.2f} sa) KISA gun gormeli -- takas var"
    )


# --------------------------------------------------------------------------
# REGRESYON: OOF kapsam maskesi
# --------------------------------------------------------------------------


def _kucuk_cv():
    from gridup.models import cross_validate
    from gridup.validation import purged_time_series_split

    rng = np.random.default_rng(0)
    n = 600
    tarih = pd.Series(np.tile(pd.date_range("2025-01-01", periods=200), 3))
    X = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    y = (X.a * 3 + rng.normal(0, 1, n)).to_numpy()
    folds = purged_time_series_split(
        tarih,
        embargo=pd.Timedelta(days=5),
        n_splits=2,
        test_span=pd.Timedelta(days=30),
        verbose=False,
    )
    return (
        cross_validate(
            X,
            y,
            folds,
            kind="lightgbm",
            metric="rmse",
            params={"n_estimators": 50, "verbose": -1},
            verbose=False,
        ),
        y,
        folds,
    )


@pytest.mark.slow
def test_oof_kapsam_maskesi_foldlarla_ortusuyor():
    """REGRESYON P0: kapsanmayan satirlar OOF'ta SIFIR kalir -- gercek tahmin degil.

    Maske disari cikmadigi surece her cagiran onu yeniden hesaplamak zorunda
    kalir ve unutmak sessizce yanlis harman/korelasyon uretir.
    OLCULDU: gercek korelasyon 0.93 iken tum diziyle 0.47.
    """
    sonuc, _, folds = _kucuk_cv()

    beklenen = np.zeros(len(sonuc.oof_predictions), dtype=bool)
    for _, valid in folds:
        beklenen[valid] = True

    assert sonuc.oof_covered.dtype == bool
    assert np.array_equal(sonuc.oof_covered, beklenen)
    assert sonuc.coverage == pytest.approx(beklenen.mean())
    # Kapsanmayanlar gercekten sifir olmali (dolgu oldugunun kaniti).
    assert np.all(sonuc.oof_predictions[~beklenen] == 0.0)


@pytest.mark.slow
def test_covered_predictions_dogru_alt_kumeyi_veriyor():
    sonuc, y, _ = _kucuk_cv()
    indeks, tahminler = sonuc.covered_predictions()

    assert len(indeks) == len(tahminler) == int(sonuc.oof_covered.sum())
    assert np.array_equal(tahminler, sonuc.oof_predictions[indeks])
    # Dogru alt kumede korelasyon, tum diziden BELIRGIN sekilde yuksek olmali.
    dogru = abs(np.corrcoef(tahminler, y[indeks])[0, 1])
    hamdan = abs(np.corrcoef(sonuc.oof_predictions, y)[0, 1])
    assert dogru > hamdan


def test_maskesiz_cvresult_geriye_uyumlu():
    """Elle kurulmus veya eski CVResult'ta maske bos -- hepsi kapsanmis sayilmali."""
    from gridup.models import CVResult

    eski = CVResult(
        oof_predictions=np.arange(5.0),
        test_predictions=None,
        fold_scores=[1.0],
        overall_score=1.0,
        feature_importance=pd.DataFrame(),
    )
    indeks, tahminler = eski.covered_predictions()
    assert len(indeks) == 5
    assert eski.coverage == 1.0
    assert np.array_equal(tahminler, eski.oof_predictions)


@pytest.mark.parametrize("kind", ["lightgbm", "xgboost", "catboost"])
def test_genel_objective_anahtari_her_kutuphanede_cevriliyor(kind):
    """REGRESYON P1: objective='mae' XGBoost/CatBoost'ta cokuyordu.

    starter_params docstring'i 'COUNT_OBJECTIVES anahtarlarindan birini
    kullan' diyor ama kod degeri oldugu gibi geciriyordu:
      XGBoost  -> "Unknown objective function"
      CatBoost -> "mae loss is not supported"
    fit_two_stage bu yolu sabit kodladigi icin iki kutuphanede de tamamen
    kullanilamazdi.
    """
    from gridup.models import COUNT_OBJECTIVES, starter_params

    anahtar = "loss_function" if kind == "catboost" else "objective"
    for genel in ("mae", "poisson", "tweedie", "l2"):
        params = starter_params(kind, "regression", objective=genel)
        assert params[anahtar] == COUNT_OBJECTIVES[kind][genel]


def test_kutuphaneye_ozgu_objective_ezilmiyor():
    """Kullanici acikca kutuphane adi verdiyse cevirmeye kalkmayiz."""
    from gridup.models import starter_params

    assert (
        starter_params("xgboost", "regression", objective="reg:tweedie")["objective"]
        == "reg:tweedie"
    )
    assert (
        starter_params("catboost", "regression", objective="Huber:delta=1")["loss_function"]
        == "Huber:delta=1"
    )
