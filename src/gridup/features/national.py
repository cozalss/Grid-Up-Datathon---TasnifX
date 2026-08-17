"""ULUSAL saatlik seriler (EPIAS) ve YILLIK ilce oznitelikleri (turizm).

NEDEN BU MODUL VAR
------------------
Uc veri seti cekilmis ama HICBIR kod okumuyordu (olculdu: 2026-08-17):

  data/external/epias/tuketim_saatlik.parquet   58.044 saat, Turkiye toplami
  data/external/epias/uretim_saatlik.parquet    58.044 saat, 18 kaynak kirilimi
  data/external/turizm_geceleme.parquet         234 satir, YILLIK, ilce bazli

Ikisi ULUSAL (ilceye gore degismez), biri YILLIK (gune gore degismez).
Ikisi de panele ayni sekilde baglanamaz; bu yuzden tek modulde ama ayri
fonksiyonlarla ele alinirlar.

SIZINTI DISIPLINI
-----------------
**EPIAS gerceklesmis veridir.** Turkiye'nin dun ne tukettigini biliriz,
gelecek ay ne tuketecegini BILMEYIZ. Ileri bir tahmin blogunda bu seriler
elde YOKTUR. 2023 GDZ birincisi bunu acikca ele almis ve EPIAS'tan gelen
her degiskeni **en az 24 saat gecikmeli** kullandigini, hicbir forward leak
birakmadigini yazmisti. Biz daha katiyiz: gecikme TAHMIN UFKU kadardir.

``add_national_series`` ufuk kaydirmasini ZORUNLU tutar; kaydirmasiz ham
gunluk seri frame'de BIRAKILMAZ.

**Turizm yillik ve GECIKMELI yayimlanir.** 2026'nin konaklama bulteni
2026 icinde yayimlanmaz. Bu yuzden ``add_annual_district_attribute``
varsayilan olarak BIR ONCEKI yilin degerini kullanir (``year_lag=1``);
"ayni yil" secenegi acikca istenmeli ve neden guvenli oldugu bilinmelidir.

ULUSAL SERININ SINIRI -- DURUST BEKLENTI
---------------------------------------
Turkiye toplami, ilce x gun kesinti tahmininde ZAYIF bir sinyaldir: tum
ilceler icin ayni degeri alir, yani yalnizca ZAMAN eksenini aciklar
(ekonomik aktivite, bayram etkisi, genel hava dalgasi). Takvim ve tatil
feature'lariyla buyuk olcude ortusur. 2023 yarismasinda hedef ENERJI
TUKETIMI oldugu icin cok degerliydi; hedef KESINTI ise degeri dusuktur.
Ablasyonla olculmeden "faydali" varsayilmamalidir.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .temporal import add_lag_features, add_rolling_features

__all__ = [
    "daily_from_hourly",
    "add_national_series",
    "add_annual_district_attribute",
    "RENEWABLE_COLUMNS",
]

#: EPIAS uretim kirilimindaki yenilenebilir kaynaklar. Dagitik uretimin
#: sebekeye bindirdigi yuk (ADM/GDZ'nin DERMS gundemi) icin toplam uretime
#: orani, ham MWh'dan daha anlamlidir.
RENEWABLE_COLUMNS: tuple[str, ...] = ("wind", "sun", "geothermal", "biomass", "river")


def daily_from_hourly(
    hourly: pd.DataFrame,
    *,
    time_column: str,
    value_columns: Sequence[str],
    aggregations: Sequence[str] = ("mean", "max", "min"),
) -> pd.DataFrame:
    """Saatlik ulusal seriyi gunluk ozetlere indirir. YENI frame dondurur.

    Gun siniri YEREL zamana gore kesilir (Turkiye kalici UTC+3). Saat dilimi
    bilgisi tasiyan damgalar once yerel saate cevrilir, sonra normalize
    edilir -- aksi halde her gun 3 saat kayar.

    Raises:
        KeyError: Zaman veya deger kolonu yoksa.
    """
    if time_column not in hourly.columns:
        raise KeyError(f"hourly icinde '{time_column}' kolonu yok.")
    eksik = [k for k in value_columns if k not in hourly.columns]
    if eksik:
        raise KeyError(f"hourly icinde su kolonlar yok: {eksik}")

    frame = hourly.copy()
    zaman = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
    zaman = zaman.dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
    frame["_gun"] = zaman.dt.normalize()

    gun = frame.groupby("_gun")[list(value_columns)].agg(list(aggregations))
    gun.columns = [f"{kolon}_{islev}" for kolon, islev in gun.columns]
    return gun.reset_index().rename(columns={"_gun": "tarih"})


def add_national_series(
    frame: pd.DataFrame,
    daily_national: pd.DataFrame,
    *,
    time_column: str,
    horizon: int,
    value_columns: Sequence[str] | None = None,
    national_time_column: str = "tarih",
    windows: Sequence[int] = (7, 28),
    prefix: str = "ulusal",
) -> pd.DataFrame:
    """Ulusal gunluk seriyi UFUK-GUVENLI olarak panele ekler. YENI frame dondurur.

    Seri tum ilceler icin AYNIDIR; bu yuzden gruplama yapilmaz ve kaydirma
    dogrudan tarih ekseninde uygulanir. Ham (kaydirilmamis) kolon cikti
    frame'inde BIRAKILMAZ -- birakilsaydi tahmin gununun gerceklesmis ulusal
    tuketimi modele girerdi ve bu, tanimi geregi gelecekten bilgidir.

    Args:
        daily_national: ``daily_from_hourly`` ciktisi; tarih + deger kolonlari.
        horizon: Tahmin ufku (gun). Kaydirma bu kadar olur.
        value_columns: Kullanilacak kolonlar. None ise tarih disindaki hepsi.

    Raises:
        ValueError: ``horizon`` 1'den kucukse.
    """
    if horizon < 1:
        raise ValueError(f"horizon en az 1 olmali, {horizon} verildi.")
    if time_column not in frame.columns:
        raise KeyError(f"frame icinde '{time_column}' kolonu yok.")

    ulusal = daily_national.copy()
    if national_time_column not in ulusal.columns:
        raise KeyError(f"daily_national icinde '{national_time_column}' kolonu yok.")
    kolonlar = (
        list(value_columns)
        if value_columns
        else [k for k in ulusal.columns if k != national_time_column]
    )
    if not kolonlar:
        return frame.copy()

    ulusal = ulusal[[national_time_column, *kolonlar]].copy()
    ulusal[national_time_column] = pd.to_datetime(ulusal[national_time_column]).dt.normalize()
    yeniden = {k: f"{prefix}_{k}" for k in kolonlar}
    ulusal = ulusal.rename(columns=yeniden)
    ham = list(yeniden.values())

    # KAYDIRMA PANELDE DEGIL, ULUSAL TABLODA YAPILIR.
    #
    # ``add_lag_features`` kaydirmayi SATIR cinsinden uygular. Panelde her
    # tarih ilce sayisi kadar tekrar eder; 96 ilcelik bir panelde
    # ``shift(31)`` yalnizca ~8 SAAT geriye gider, 31 gun degil. Bu, testle
    # OLCULDU: 2 ilcelik panelde ulusal sicrama ufuktan once sizdi
    # (test_harici_sizinti::test_ulusal_seri_ufuktan_once_GORUNMEZ).
    #
    # Ulusal seri gun basina TEK satir oldugundan, kaydirma burada
    # matematiksel olarak gun cinsindendir. Kaydirilmis kolonlar panele
    # sonra baglanir; ayni tarihteki tum ilceler ayni degeri alir -- ulusal
    # serinin tanimi zaten budur.
    ulusal = ulusal.drop_duplicates(subset=[national_time_column])
    ulusal = ulusal.sort_values(national_time_column).reset_index(drop=True)
    for kolon in ham:
        ulusal = add_lag_features(
            ulusal, kolon, shifts=[horizon], time_column=national_time_column, horizon=horizon
        )
        ulusal = add_rolling_features(
            ulusal,
            kolon,
            windows,
            time_column=national_time_column,
            horizon=horizon,
            aggregations=("mean", "max"),
        )
    # Ham (kaydirilmamis) kolonlar panele HIC girmez.
    ulusal = ulusal.drop(columns=ham).rename(columns={national_time_column: "_ulusal_gun"})

    cikti = frame.copy()
    cikti["_ulusal_gun"] = pd.to_datetime(cikti[time_column]).dt.normalize()
    cikti = cikti.merge(ulusal, on="_ulusal_gun", how="left")
    return cikti.drop(columns=["_ulusal_gun"])


def add_annual_district_attribute(
    frame: pd.DataFrame,
    annual: pd.DataFrame,
    *,
    key_column: str,
    time_column: str,
    value_columns: Sequence[str],
    annual_key_column: str | None = None,
    year_column: str = "yil",
    year_lag: int = 1,
    population: pd.DataFrame | None = None,
    population_key_column: str | None = None,
    population_column: str = "nufus",
    prefix: str = "yillik",
) -> pd.DataFrame:
    """Yillik ilce ozniteligini panele ekler. YENI frame dondurur.

    ``year_lag=1`` (varsayilan) BILINCLIDIR: yillik istatistikler yil
    bitmeden yayimlanmaz. 2026 panelinde 2026 gecelemesini kullanmak,
    tahmin aninda var olmayan bir sayiyi modele sokmaktir. Bir onceki yilin
    degeri hem elde vardir hem de "bu ilce ne kadar turistik" sorusunun
    kararli cevabidir.

    ``population`` verilirse kisi basina normalize edilmis bir yogunluk
    kolonu da uretilir: mutlak geceleme buyuk ilcelerde her zaman yuksektir,
    oysa SEBEKEYI ZORLAYAN sey nufusa GORE fazlaliktir (Bodrum'un yazin
    nufusu katlanir; Konak'in katlanmaz).

    Raises:
        ValueError: ``year_lag`` negatifse.
    """
    if year_lag < 0:
        raise ValueError(f"year_lag negatif olamaz, {year_lag} verildi.")
    for kolon in (key_column, time_column):
        if kolon not in frame.columns:
            raise KeyError(f"frame icinde '{kolon}' kolonu yok.")
    yillik_anahtar = annual_key_column or key_column
    for kolon in (yillik_anahtar, year_column, *value_columns):
        if kolon not in annual.columns:
            raise KeyError(f"annual icinde '{kolon}' kolonu yok.")

    tablo = annual[[yillik_anahtar, year_column, *value_columns]].copy()
    tablo[yillik_anahtar] = tablo[yillik_anahtar].astype(str)
    # Gecikme, KAYNAK yilina eklenir: 2023 verisi 2024 panelinde gorunur.
    tablo["_eslesme_yil"] = tablo[year_column].astype(int) + int(year_lag)
    yeniden = {k: f"{prefix}_{k}" for k in value_columns}
    tablo = tablo.rename(columns={yillik_anahtar: "_eslesme_anahtar", **yeniden})
    tablo = tablo.drop(columns=[year_column])

    cikti = frame.copy()
    cikti["_eslesme_anahtar"] = cikti[key_column].astype(str)
    cikti["_eslesme_yil"] = pd.to_datetime(cikti[time_column]).dt.year
    cikti = cikti.merge(tablo, on=["_eslesme_anahtar", "_eslesme_yil"], how="left")

    if population is not None:
        nufus_anahtar = population_key_column or key_column
        if nufus_anahtar not in population.columns or population_column not in population.columns:
            raise KeyError(f"population icinde '{nufus_anahtar}' veya '{population_column}' yok.")
        nufus = population[[nufus_anahtar, population_column]].copy()
        nufus[nufus_anahtar] = nufus[nufus_anahtar].astype(str)
        nufus = nufus.rename(columns={nufus_anahtar: "_eslesme_anahtar"})
        cikti = cikti.merge(nufus, on="_eslesme_anahtar", how="left")
        for kaynak, hedef in yeniden.items():
            pay = pd.to_numeric(cikti[hedef], errors="coerce")
            payda = pd.to_numeric(cikti[population_column], errors="coerce")
            # Nufus 0 veya eksikse oran tanimsizdir; 0'a bolup sonsuz
            # uretmek yerine NaN birakiyoruz -- submission dogrulamasi
            # sonsuz degeri zaten reddeder, ama once buraya girmemeli.
            cikti[f"{hedef}_kisi_basi"] = pay.where(payda.gt(0)) / payda.where(payda.gt(0))
            del kaynak
        if population_column not in frame.columns:
            cikti = cikti.drop(columns=[population_column])

    return cikti.drop(columns=["_eslesme_anahtar", "_eslesme_yil"])
