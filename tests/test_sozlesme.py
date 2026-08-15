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


# --------------------------------------------------------------------------
# SOZLESME 2: hedefe dokunan her fonksiyon ya fold-farkinda olmali
#             ya da NEDEN olmadigi kayitli olmali
# --------------------------------------------------------------------------

#: Hedef anlamina gelen parametre adlari.
_HEDEF_PARAMS = frozenset({"target", "target_column", "y", "y_true", "value_column"})

#: Fold farkindaligi anlamina gelen parametre adlari.
_FOLD_PARAMS = frozenset({"folds", "fold", "fold_list", "cv"})

HEDEF_MODULLERI = (
    "features.categorical", "features.aggregate", "features.spatial",
    "features.temporal", "selection", "two_stage", "ensemble", "models",
    "refit", "zoo", "tuning", "ablation", "neural", "metrics",
)

#: Hedefe dokunan ama fold ALMAYAN fonksiyonlar ve bunun NEDEN guvenli oldugu.
#: Bu sozluk bir muafiyet listesi degil, bir GEREKCE kaydidir: her satir
#: "bu fonksiyon neden sizinti yapmiyor" sorusunu cevaplar.
FOLDSUZ_GEREKCE: dict[str, str] = {
    # Saf skorlama -- model egitmez, hicbir sey ogrenmez.
    "metrics.rmse": "saf skorlama, fit yok",
    "metrics.rmsle": "saf skorlama, fit yok",
    "metrics.mape": "saf skorlama, fit yok",
    "metrics.mape_coverage": "saf tanisal olcum, fit yok",
    "metrics.smape": "saf skorlama, fit yok",
    "metrics.log_transform_target": "geri cevrilebilir donusum, istatistik ogrenmez",
    # OOF tahminler UZERINDE calisir -- girdisi zaten fold-disidir.
    "metrics.optimize_threshold": "OOF tahminler uzerinde calisir (docstring'de zorunlu kilinmis)",
    "ensemble.hill_climb_weights": "girdisi OOF tahmin matrisi -- fold zaten uygulanmis",
    "ensemble.greedy_forward_selection": "girdisi OOF tahmin matrisi",
    "ensemble.prune_by_correlation": "girdisi OOF tahmin matrisi",
    "two_stage.tune_threshold": "OOF olasiliklar uzerinde esik arar",
    "two_stage.zero_baseline_score": "sabit tahminin skoru, fit yok",
    # Nedensel olarak guvenli: yalnizca GECMISE bakar (shift/horizon).
    "features.temporal.add_lag_features": "shift(horizon) -- yalnizca gecmis, ileri bakmaz",
    "features.temporal.add_rolling_features": "closed='left' -- mevcut satiri DISLAR",
    "features.temporal.add_expanding_features": "shift(1) -- mevcut satiri DISLAR",
    "features.spatial.add_neighbour_target_lag": "horizon>=1 zorunlu -- komsunun gecmisi",
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
    if fonksiyon_adi == "aggregate_hourly_to_daily":
        pytest.skip("kasitli indirgeme -- satir kimligi degisir")

    modul_adi, cagri = SENARYOLAR[fonksiyon_adi]
    if modul_adi == "temporal" and fonksiyon_adi in {
        "add_ramadan_features", "add_turkish_holiday_features"
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
            panel, "y", [3], time_column="tarih", group_columns=["ilce"]
        )
    else:
        cikti = temporal.add_expanding_features(
            panel, "y", time_column="tarih", group_columns=["ilce"]
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
        panel, time_column="tarih", location_column="yer",
        coordinates=koordinat, geometry_only=True,
    )

    kuzey = cikti.loc[cikti.yer == "kuzey", "gun_uzunlugu_saat"].mean()
    guney = cikti.loc[cikti.yer == "guney", "gun_uzunlugu_saat"].mean()
    # Ocak ayinda 60N kutba yakin -> KISA gun; 20N ekvatora yakin -> uzun gun.
    assert kuzey < guney, (
        f"Ocak'ta 60N ({kuzey:.2f} sa) 20N'den ({guney:.2f} sa) KISA gun gormeli -- takas var"
    )
