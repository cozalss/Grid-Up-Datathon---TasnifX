"""Hava durumu feature muhendisligi: quantile, bolge-geneli agregat, fiziksel turevler.

NEDEN AYRI BIR MODUL
--------------------
Ham hava degerlerini modele vermek sinyalin cogunu kaybettirir. 2023 GDZ Datathon
birincisinin en yuksek onemli feature listesi ham degerlerle DEGIL, su turevlerle
doluydu::

    wind_dir_10md_date_allstates_q01
    effective_cloud_coverp_date_allstates_q08
    wind_speed_10mms_max
    t_apparentC_date_allstates_q01

Uc kalip goze carpiyor ve ucu de bu modulde uygulaniyor:

1. **MAX ve QUANTILE, ortalama degil.** Direk devirmeyen sey ortalama ruzgardir;
   hasari **tepe** deger yapar. Gunluk ortalama bu tepeyi silip supurur.

2. **Bolge-geneli ("allstates") agregat.** Tum ilcelerin ayni gunku degerinin
   quantile'i. Bir firtina bolgesel bir olaydir: komsu ilcede olculen ruzgar,
   o ilcenin kendi olcumundeki gurultuyu duzeltir.

3. **Fiziksel mekanizma turevleri.** "Derece-gun" iklimlendirme yukunu, "kuraklik
   sonrasi ilk yagmur" izolator kirlenmesini, "islak zemin + ruzgar" agac
   devrilmesini temsil eder. Modele fiziksel nedeni ogretmek, ham sayiyi
   vermekten daha az veriyle daha iyi ogrenmesini saglar.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "circular_mean",
    "aggregate_hourly_to_daily",
    "add_regional_aggregates",
    "add_physical_derivatives",
    "add_weather_accumulators",
    "DEFAULT_QUANTILES",
    "NEUTRAL_TEMPERATURE_C",
]

# Enerji sektorunun standart notr sicakligi: altinda isitma, ustunde sogutma
# yuku baslar. Derece-gun hesabinin dayanagi.
NEUTRAL_TEMPERATURE_C = 18.0

# Kazanan cozumde q01 ve q08 goruldu -- yani uc degerler (alt/ust kuyruk) ve
# ust-orta bolge ayri ayri tasiniyor. Ortalama tek basina bunlarin hicbirini vermez.
DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


def circular_mean(degrees: pd.Series | np.ndarray) -> float:
    """Dairesel degiskenin (ruzgar yonu) ortalamasi.

    NEDEN OZEL: 350 derece ile 10 derece komsudur ama aritmetik ortalamalari
    180'dir -- yani tam ters yon. Vektorel ortalama bunu duzeltir: acilari
    birim vektore cevir, topla, geri aciya cevir.

    >>> round(circular_mean(np.array([350.0, 10.0])), 4)
    0.0
    """
    values = np.asarray(degrees, dtype="float64")
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan")

    radians = np.deg2rad(values)
    angle = np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean()))

    # Yuvarlama ONCE, modulo SONRA. Aksi halde arctan2'nin urettigi -1e-14 gibi
    # bir deger `% 360` ile 359.99999...'a doner ve 360.0 olarak gorunur.
    # Aci olarak 0 ile 360 aynidir ama FEATURE DEGERI olarak sayisal uclardir --
    # yani bu fonksiyonun onlemek icin var oldugu sorunun ta kendisi.
    return float(np.round(angle, 9) % 360.0)


def aggregate_hourly_to_daily(
    hourly: pd.DataFrame,
    *,
    time_column: str,
    group_columns: Sequence[str],
    value_columns: Sequence[str] | None = None,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    direction_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Saatlik hava verisini gunluge indirir -- quantile'lari KORUYARAK.

    Args:
        hourly: Saatlik olcumler.
        time_column: Zaman damgasi kolonu.
        group_columns: Konum anahtari, or. ``["konum_key"]``.
        value_columns: Toplanacak sayisal kolonlar. ``None`` ise tum sayisallar.
        quantiles: Uretilecek quantile'lar. Varsayilan (0.1, 0.5, 0.9).
        direction_columns: Dairesel kolonlar (ruzgar yonu) -- bunlara aritmetik
            ortalama yerine ``circular_mean`` uygulanir ve ayrica sin/cos ciftine
            cevrilir.

    Returns:
        Yeni DataFrame: ``group_columns + [gun]`` bazinda tek satir.

    NOT -- GUN SINIRI: Turkiye 2016'dan beri kalici UTC+3 kullanir (yaz saati
    uygulamasi YOK). Bu, saatlik veriyi gune indirirken sinirin kaymadigi
    anlamina gelir; ama veriyi cekerken ``timezone=Europe/Istanbul`` vermeyi
    unutma, aksi halde gunler UTC'ye gore kesilir ve 3 saatlik kayma olusur.
    """
    frame = hourly.copy()
    frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce")
    frame["gun"] = frame[time_column].dt.normalize()

    group_list = list(group_columns) + ["gun"]

    if value_columns is None:
        value_columns = [
            column
            for column in frame.columns
            if column not in group_list + [time_column]
            and pd.api.types.is_numeric_dtype(frame[column])
            and column not in direction_columns
        ]

    aggregations: list[str] = ["mean", "min", "max", "std"]
    grouped = frame.groupby(group_list, observed=True)

    parts = [grouped[list(value_columns)].agg(aggregations)]
    parts[0].columns = [f"{column}_{stat}" for column, stat in parts[0].columns]

    for quantile in quantiles:
        block = grouped[list(value_columns)].quantile(quantile)
        label = f"q{int(round(quantile * 100)):02d}"
        block.columns = [f"{column}_{label}" for column in block.columns]
        parts.append(block)

    for column in direction_columns:
        if column not in frame.columns:
            continue
        angles = grouped[column].apply(circular_mean).rename(f"{column}_dairesel_ort")
        parts.append(angles.to_frame())

    daily = pd.concat(parts, axis=1).reset_index()

    # Dairesel kolonlari sin/cos'a cevir: agac modelleri 359 -> 0 sicramasini
    # bolme ile asabilir ama ucuz degildir; sin/cos komsulugu dogrudan verir.
    for column in direction_columns:
        source = f"{column}_dairesel_ort"
        if source not in daily.columns:
            continue
        radians = np.deg2rad(daily[source].to_numpy(dtype="float64"))
        daily[f"{column}_sin"] = np.sin(radians).astype("float32")
        daily[f"{column}_cos"] = np.cos(radians).astype("float32")

    return daily


def add_regional_aggregates(
    daily: pd.DataFrame,
    *,
    time_column: str,
    value_columns: Sequence[str],
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    prefix: str = "bolge",
) -> pd.DataFrame:
    """Ayni gunun BOLGE GENELI istatistiklerini her satira ekler. YENI frame dondurur.

    2024 birincisinin feature adlarindaki ``allstates`` tam olarak budur.

    NEDEN ISE YARIYOR: Firtina bolgesel bir olaydir. Bir ilcenin kendi olcumu
    gurultulu olabilir (istasyon konumu, yerel topografya), ama tum bolgenin
    o gunku ruzgar dagilimi olayin gercek siddetini gosterir. Model boylece
    "bu ilcede ruzgar yuksek" ile "bugun her yerde firtina var" arasindaki
    farki gorebilir.

    SIZINTI DEGILDIR: yalnizca feature'lar kullanilir, hedef kullanilmaz.
    Train ve test birlikte hesaplanabilir.
    """
    frame = daily.copy()
    frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce")

    available = [column for column in value_columns if column in frame.columns]
    if not available:
        raise KeyError(f"Bolge agregasyonu icin kolon bulunamadi: {list(value_columns)}")

    grouped = frame.groupby(time_column, observed=True)[available]

    blocks = []
    for stat in ("mean", "max", "std"):
        block = grouped.agg(stat)
        block.columns = [f"{prefix}_{column}_{stat}" for column in block.columns]
        blocks.append(block)

    for quantile in quantiles:
        block = grouped.quantile(quantile)
        label = f"q{int(round(quantile * 100)):02d}"
        block.columns = [f"{prefix}_{column}_{label}" for column in block.columns]
        blocks.append(block)

    regional = pd.concat(blocks, axis=1).reset_index()
    merged = frame.merge(regional, on=time_column, how="left", validate="many_to_one")

    # Satirin bolgeye gore SAPMASI: "burada her yerdekinden ne kadar siddetli?"
    for column in available:
        regional_mean = merged[f"{prefix}_{column}_mean"].to_numpy(dtype="float64")
        local = merged[column].to_numpy(dtype="float64")
        merged[f"{column}_{prefix}_fark"] = (local - regional_mean).astype("float32")

    return merged


def _consecutive_run(condition: np.ndarray) -> np.ndarray:
    """Her konum icin ardisik True sayisini dondurur (False'ta sifirlanir).

    ``[F, T, T, T, F, T]`` -> ``[0, 1, 2, 3, 0, 1]``
    """
    result = np.zeros(condition.shape, dtype="int32")
    running = 0
    for index, flag in enumerate(condition):
        running = running + 1 if flag else 0
        result[index] = running
    return result



def _orijinal_siraya_don(frame: pd.DataFrame, index) -> pd.DataFrame:
    """Gecici sira kolonuna gore geri sirala, kolonu at, girdi index'ini koy."""
    geri = frame.sort_values("_gridup_sira").drop(columns="_gridup_sira")
    geri.index = index
    return geri


def add_physical_derivatives(
    daily: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    time_column: str,
    temperature_mean: str = "sicaklik_ort",
    temperature_max: str | None = "sicaklik_max",
    temperature_min: str | None = "sicaklik_min",
    precipitation: str | None = "yagis_toplam",
    wind_max: str | None = "ruzgar_max",
    gust_max: str | None = "firtina_max",
    drought_days: int = 10,
) -> pd.DataFrame:
    """Fiziksel mekanizmalari temsil eden turev kolonlar. YENI frame dondurur.

    Her turev, kesintiye yol acan bilinen bir MEKANIZMAYI kodlar:

    ``isitma/sogutma_derece_gun``
        Iklimlendirme yuku. Sebeke asiri yuklendiginde trafo arizasi artar.

    ``ardisik_sicak_gece``
        Gece sogumasi olmayan gunler. Trafo gunduz isinir, gece sogur; sogumazsa
        yaglanma sicakligi birikir ve yalitim yaslanir. Tekil sicak gun degil,
        ARDISIK sicak GECE belirleyicidir.

    ``kuraklik_gunu`` / ``kuraklik_sonrasi_ilk_yagmur``
        Izolator kirlenmesi mekanizmasi. Uzun kuraklikta izolator uzerinde tuz/toz
        birikir; ilk hafif yagmur bunu iletken bir filme cevirir ve yuzey atlamasi
        (flashover) olur. En tehlikeli yagmur, siddetli olan degil, uzun kuraklik
        SONRASI ilk olandir.

    ``islak_ruzgar``
        Islak zemin agac koklerini gevsetir; ruzgar devirir. Ikisi ayri ayri zayif
        sinyal, birlikte gucludur.

    Args:
        drought_days: Kac gun yagissizlik "kuraklik" sayilir.
    """
    frame = daily.copy()
    frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce")
    # SIRA KORUMA: hesaplama icin siralamak ZORUNLU (kayan pencere kronolojik
    # olmali), ama girdi sirasini bozarak dondurmek TEHLIKELIDIR. Fold'lar
    # KONUMSAL indekstir: bu fonksiyon fold'lardan SONRA cagrilirsa, fold
    # indeksleri artik baska satirlara isaret eder ve hata VERMEZ.
    # validation.assert_folds_align docstring'i tam olarak bu kazayi uyariyor.
    # Bu yuzden orijinal konumu saklayip sonunda geri donuyoruz.
    _giris_sirasi = np.arange(len(frame))
    frame = frame.assign(_gridup_sira=_giris_sirasi)
    frame = frame.sort_values(list(group_columns) + [time_column])

    if temperature_mean not in frame.columns:
        raise KeyError(f"Sicaklik kolonu '{temperature_mean}' bulunamadi.")

    temperature = frame[temperature_mean].astype("float64")

    frame["isitma_derece_gun"] = (
        (NEUTRAL_TEMPERATURE_C - temperature).clip(lower=0).astype("float32")
    )
    frame["sogutma_derece_gun"] = (
        (temperature - NEUTRAL_TEMPERATURE_C).clip(lower=0).astype("float32")
    )

    has_range = (
        temperature_max and temperature_max in frame.columns
        and temperature_min and temperature_min in frame.columns
    )
    if has_range:
        # Gun ici genlik: dusuk genlik = gece sogumasi yok = termal birikme.
        frame["gunluk_genlik"] = (
            frame[temperature_max].astype("float64") - frame[temperature_min].astype("float64")
        ).astype("float32")

    grouped = frame.groupby(list(group_columns), observed=True, sort=False)

    def _run_length(series: pd.Series, *, above: float | None = None,
                    below: float | None = None) -> pd.Series:
        """Grup icinde ardisik kosul sayisi. Kosul bozulunca sifirlanir."""
        values = series.astype("float64").to_numpy()
        condition = values > above if above is not None else values < below
        return pd.Series(_consecutive_run(condition), index=series.index)

    # Tropik gece esigi: minimum sicaklik 22 C uzerindeyse trafo gece sogumaz.
    TROPICAL_NIGHT_C = 22.0
    # Yagissiz gun esigi: 1 mm altindaki yagis zemini islatmaz.
    DRY_DAY_MM = 1.0

    if temperature_min and temperature_min in frame.columns:
        frame["ardisik_sicak_gece"] = (
            grouped[temperature_min]
            .transform(lambda series: _run_length(series, above=TROPICAL_NIGHT_C))
            .astype("int16")
        )

    if precipitation and precipitation in frame.columns:
        frame["kuraklik_gunu"] = (
            grouped[precipitation]
            .transform(lambda series: _run_length(series, below=DRY_DAY_MM))
            .astype("int16")
        )

        # Kuraklik sonrasi ilk yagmur: bugun yagiyor VE dune kadar uzun kuraklik.
        # Izolator flashover'inin fiziksel tetikleyicisi budur -- siddetli yagmur
        # degil, uzun kurakligi bitiren ILK yagmur.
        is_wet = frame[precipitation].astype("float64") >= DRY_DAY_MM
        previous_drought = grouped["kuraklik_gunu"].shift(1).fillna(0)
        frame["kuraklik_sonrasi_ilk_yagmur"] = (
            is_wet & (previous_drought >= drought_days)
        ).astype("int8")

        # Kumulatif yagis: zemin doygunlugu proxy'si (7 gun).
        frame["yagis_7g_toplam"] = (
            grouped[precipitation]
            .transform(lambda series: series.rolling(7, min_periods=1).sum())
            .astype("float32")
        )

    wind_column = gust_max if gust_max and gust_max in frame.columns else wind_max
    if wind_column and wind_column in frame.columns:
        frame["ruzgar_3g_max"] = (
            grouped[wind_column]
            .transform(lambda series: series.rolling(3, min_periods=1).max())
            .astype("float32")
        )

        if "yagis_7g_toplam" in frame.columns:
            # Islak zemin + ruzgar: agac devrilmesinin fiziksel mekanizmasi.
            # Carpim, ikisinin de yuksek oldugu gunleri one cikarir.
            frame["islak_ruzgar"] = (
                frame["yagis_7g_toplam"].astype("float64")
                * frame[wind_column].astype("float64")
            ).astype("float32")

    return _orijinal_siraya_don(frame, daily.index)


def add_weather_accumulators(
    daily: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    time_column: str,
    value_columns: Sequence[str],
    windows: Sequence[int] = (3, 7, 14),
    horizon: int = 0,
    lead_windows: Sequence[int] = (),
) -> pd.DataFrame:
    """Hava degiskenleri icin geriye ve ILERIYE donuk pencereler. YENI frame dondurur.

    Args:
        horizon: Kac adim geriye kaydirilacak. **Hava icin genellikle 0 dogrudur**
            -- asagiya bak.
        lead_windows: ILERIYE bakan pencereler.

    HAVA VERISINDE UFUK NEDEN FARKLI
    ---------------------------------
    Hedef gecmisi icin ``horizon`` ZORUNLUDUR: tahmin gununde dunku kesinti
    sayisini bilmiyorsun.

    Hava icin durum farklidir. Yarisma test setinde hava verisini DE veriyorsa
    (2024'te veriyordu), tahmin gunundeki hava **bilinen-gelecek kovaryattir** --
    tipki gercek hayatta meteoroloji tahmininin bilinmesi gibi. O zaman:

      * ``horizon=0`` dogrudur (bugunun havasini kullanabilirsin)
      * ``lead_windows`` MESRUDUR: "onumuzdeki 3 gunun max ruzgari" gecerli bir
        feature'dir ve rakiplerin cogu bunu kullanmaz

    Ama test setinde hava YOKSA, hava da tahmin edilmesi gereken bir seydir ve
    ``horizon`` uygulanmalidir. **Veri geldiginde bunu ilk kontrol et.**
    """
    frame = daily.copy()
    frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce")
    # SIRA KORUMA: hesaplama icin siralamak ZORUNLU (kayan pencere kronolojik
    # olmali), ama girdi sirasini bozarak dondurmek TEHLIKELIDIR. Fold'lar
    # KONUMSAL indekstir: bu fonksiyon fold'lardan SONRA cagrilirsa, fold
    # indeksleri artik baska satirlara isaret eder ve hata VERMEZ.
    # validation.assert_folds_align docstring'i tam olarak bu kazayi uyariyor.
    # Bu yuzden orijinal konumu saklayip sonunda geri donuyoruz.
    _giris_sirasi = np.arange(len(frame))
    frame = frame.assign(_gridup_sira=_giris_sirasi)
    frame = frame.sort_values(list(group_columns) + [time_column])

    available = [column for column in value_columns if column in frame.columns]
    if not available:
        raise KeyError(f"Birikim icin kolon bulunamadi: {list(value_columns)}")

    grouped = frame.groupby(list(group_columns), observed=True, sort=False)
    new_columns: dict[str, np.ndarray] = {}

    for column in available:
        base = grouped[column].shift(horizon) if horizon else frame[column]
        shifted = base.groupby([frame[key] for key in group_columns], observed=True)

        for window in windows:
            roller = shifted.rolling(window, min_periods=1)
            new_columns[f"{column}_geri{window}_max"] = np.asarray(
                roller.max(), dtype="float32"
            )
            new_columns[f"{column}_geri{window}_ort"] = np.asarray(
                roller.mean(), dtype="float32"
            )

        for window in lead_windows:
            # Ileriye bakan pencere: ters cevir, kaydir, geri cevir.
            forward = (
                grouped[column]
                .transform(lambda s, w=window: s[::-1].rolling(w, min_periods=1).max()[::-1])
            )
            new_columns[f"{column}_ileri{window}_max"] = np.asarray(forward, dtype="float32")

    return _orijinal_siraya_don(frame.assign(**new_columns), daily.index)
