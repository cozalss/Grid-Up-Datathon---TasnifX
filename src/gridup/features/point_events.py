"""NOKTA OLAYLARINI panele baglar: yangin tespitleri ve depremler.

NEDEN BU MODUL VAR
------------------
``data/external/yanginlar.parquet`` (NASA FIRMS, 30.575 sicak nokta) ve
``data/external/depremler.parquet`` (AFAD, M>=4) cekilmisti ama HICBIR kod
onlari okumuyordu -- yani indirilmis, Kaggle paketine konmus ve
kullanilmamis veriydi (olculdu: 2026-08-17 kod taramasi).

Sebebi mesru: ikisi de NOKTA KOORDINATI tasir, panel ise ilce x gun. Aradaki
birlestirme mesafe, yaricap ve zaman penceresi kararlari gerektirir ve bu
kararlar panel anahtarlari bilinmeden yazilamaz. Bu modul o birlestirmeyi
PARAMETRELI hale getirir: veri gunu hangi anahtar gelirse gelsin calisir.

NEDENSELLIK ZINCIRLERI
----------------------
* **Yangin** -> hat uzerinden gecen duman ve alev iyonize hava birakir;
  iletken ile toprak arasinda ATLAMA (flashover) olusur ve koruma acar.
  Orman yangini ayrica direk yakar ve saha ekibinin erisimini keser.
  FRP (Fire Radiative Power) yanginin siddetini olcer -- sayidan daha
  bilgilendiricidir, bu yuzden agirlik olarak kullanilabilir.
* **Deprem** -> direk temeli kayar, izolator catlar, yeralti kablosu
  gerilir. Etki YARICAPI buyukluge gore genisler; M4 yereldir, M6 il
  capinda hissedilir.

SIZINTI DISIPLINI -- BU MODULUN EN ONEMLI KISMI
-----------------------------------------------
Bir yangin veya deprem GELECEKTE ne zaman olacagi bilinmez. Tahmin ufku
ileri bir blok ise (or. 31 gun), o blogun icindeki olaylar tahmin aninda
ELDE YOKTUR. Bu yuzden bu modul ham gunluk sayimi frame'de BIRAKMAZ:
yalnizca ``add_rolling_features``/``add_lag_features`` ile ufuk kadar
kaydirilmis turevleri dondurur. Ham kolon uretilir, kullanilir ve DUSURULUR.

Kaydirma matematigi bu modulde YENIDEN YAZILMAZ -- denetlenmis
``features.temporal`` fonksiyonlarina devredilir, boylece ufuk semantigi
paneldeki diger tum lag/rolling ile birebir ayni kalir.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .spatial import haversine_cross
from .temporal import add_lag_features, add_rolling_features

__all__ = [
    "daily_point_intensity",
    "add_point_event_features",
    "DEFAULT_RADII_KM",
]

#: Varsayilan yaricaplar (km). 25 km "ayni ilce ve bitisigi", 50 km "hava
#: olayinin tipik olcegi", 100 km "bolgesel bask". Uc olcek birlikte, olayin
#: yerel mi bolgesel mi oldugunu modele ayirt ettirir.
DEFAULT_RADII_KM: tuple[float, ...] = (25.0, 50.0, 100.0)


def daily_point_intensity(
    events: pd.DataFrame,
    coordinates: pd.DataFrame,
    *,
    key_column: str,
    event_time_column: str = "tarih",
    latitude_column: str = "lat",
    longitude_column: str = "lon",
    radii_km: Sequence[float] = DEFAULT_RADII_KM,
    weight_column: str | None = None,
    prefix: str = "olay",
) -> pd.DataFrame:
    """Nokta olaylarini ``(anahtar, tarih)`` gunluk yogunluk tablosuna cevirir.

    Her ilce merkezi icin, verilen yaricaplarin ICINDE kalan olaylar gune
    gore toplanir. ``weight_column`` verilirse (or. yangin icin ``frp``,
    deprem icin ``buyukluk``) sayimin yaninda agirlik toplami ve maksimumu
    da uretilir.

    Bu fonksiyon HAM gunluk yogunlugu dondurur -- sizinti korumasi YOKTUR.
    Dogrudan modele verilmemeli; ``add_point_event_features`` uzerinden
    kullanilmalidir.

    Args:
        events: Nokta olaylari; lat/lon ve tarih kolonlari zorunlu.
        coordinates: Ilce merkezleri; ``key_column``, lat, lon.
        radii_km: Artan sirada olmasi gerekmez; cikti kolon adlarinda
            yuvarlanmis tam sayi olarak gecer (``..._r50``).

    Raises:
        KeyError: Zorunlu kolonlardan biri yoksa.
        ValueError: Yaricap pozitif degilse veya koordinat tablosunda
            tekrarlanan anahtar varsa.
    """
    for kolon in (latitude_column, longitude_column, event_time_column):
        if kolon not in events.columns:
            raise KeyError(f"events icinde '{kolon}' kolonu yok.")
    for kolon in (key_column, latitude_column, longitude_column):
        if kolon not in coordinates.columns:
            raise KeyError(f"coordinates icinde '{kolon}' kolonu yok.")
    if weight_column is not None and weight_column not in events.columns:
        raise KeyError(f"events icinde agirlik kolonu '{weight_column}' yok.")
    if any(float(r) <= 0 for r in radii_km):
        raise ValueError(f"Yaricaplar pozitif olmali: {list(radii_km)}")
    if coordinates[key_column].duplicated().any():
        raise ValueError(
            f"coordinates icinde tekrarlanan '{key_column}' var; mesafe matrisi belirsizlesir."
        )

    olaylar = events.dropna(subset=[latitude_column, longitude_column, event_time_column]).copy()
    if olaylar.empty:
        return pd.DataFrame({key_column: [], "tarih": []})

    olaylar["_gun"] = pd.to_datetime(olaylar[event_time_column]).dt.normalize()

    mesafe = haversine_cross(
        coordinates[latitude_column].to_numpy(),
        coordinates[longitude_column].to_numpy(),
        olaylar[latitude_column].to_numpy(),
        olaylar[longitude_column].to_numpy(),
    )

    anahtarlar = coordinates[key_column].astype(str).to_numpy()
    gunler = olaylar["_gun"].to_numpy()
    agirlik = (
        pd.to_numeric(olaylar[weight_column], errors="coerce").fillna(0.0).to_numpy()
        if weight_column is not None
        else None
    )

    parcalar: list[pd.DataFrame] = []
    for yaricap in radii_km:
        etiket = f"r{int(round(float(yaricap)))}"
        icinde = mesafe <= float(yaricap)
        satir_idx, olay_idx = np.nonzero(icinde)
        if satir_idx.size == 0:
            continue
        kayit = {
            key_column: anahtarlar[satir_idx],
            "tarih": gunler[olay_idx],
            f"{prefix}_sayi_{etiket}": np.ones(satir_idx.size, dtype="float64"),
        }
        if agirlik is not None:
            kayit[f"{prefix}_agirlik_{etiket}"] = agirlik[olay_idx]
        uzun = pd.DataFrame(kayit)
        toplama: dict[str, str] = {f"{prefix}_sayi_{etiket}": "sum"}
        if agirlik is not None:
            toplama[f"{prefix}_agirlik_{etiket}"] = "sum"
        gunluk = uzun.groupby([key_column, "tarih"], as_index=False).agg(toplama)
        if agirlik is not None:
            enb = (
                uzun.groupby([key_column, "tarih"], as_index=False)[f"{prefix}_agirlik_{etiket}"]
                .max()
                .rename(columns={f"{prefix}_agirlik_{etiket}": f"{prefix}_agirlik_max_{etiket}"})
            )
            gunluk = gunluk.merge(enb, on=[key_column, "tarih"], how="left")
        parcalar.append(gunluk)

    if not parcalar:
        return pd.DataFrame({key_column: [], "tarih": []})

    sonuc = parcalar[0]
    for parca in parcalar[1:]:
        sonuc = sonuc.merge(parca, on=[key_column, "tarih"], how="outer")
    return sonuc.sort_values([key_column, "tarih"]).reset_index(drop=True)


def add_point_event_features(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    coordinates: pd.DataFrame,
    *,
    key_column: str,
    time_column: str,
    horizon: int,
    event_time_column: str = "tarih",
    latitude_column: str = "lat",
    longitude_column: str = "lon",
    radii_km: Sequence[float] = DEFAULT_RADII_KM,
    windows: Sequence[int] = (7, 30, 90),
    weight_column: str | None = None,
    prefix: str = "olay",
) -> pd.DataFrame:
    """Nokta olay gecmisini UFUK-GUVENLI feature'lara cevirir. YENI frame dondurur.

    Uretilen her kolon ``shift(horizon)`` sonrasidir: tahmin aninda (yani
    ``t - horizon`` gunu) bilinen olaylardan hesaplanir. Ham gunluk yogunluk
    kolonlari ARA URUNDUR ve cikti frame'inde BIRAKILMAZ -- birakilsalardi
    tahmin gununun kendi yangini/depremi modele girerdi.

    Args:
        frame: Panel. ``key_column`` ve ``time_column`` tasimali.
        events: Nokta olaylari (yangin veya deprem).
        coordinates: Ilce merkezleri.
        horizon: Tahmin ufku (gun). ``add_rolling_features`` ile ayni anlam.
        windows: Geriye bakis pencereleri (gun).
        weight_column: Olay siddeti (``frp``, ``buyukluk``). None ise yalnizca
            sayim uretilir.

    Raises:
        ValueError: ``horizon`` 1'den kucukse.
    """
    if horizon < 1:
        raise ValueError(f"horizon en az 1 olmali, {horizon} verildi.")
    for kolon in (key_column, time_column):
        if kolon not in frame.columns:
            raise KeyError(f"frame icinde '{kolon}' kolonu yok.")

    gunluk = daily_point_intensity(
        events,
        coordinates,
        key_column=key_column,
        event_time_column=event_time_column,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
        radii_km=radii_km,
        weight_column=weight_column,
        prefix=prefix,
    )

    cikti = frame.copy()
    ham_kolonlar = [k for k in gunluk.columns if k not in (key_column, "tarih")]
    if not ham_kolonlar:
        return cikti

    zaman = pd.to_datetime(cikti[time_column]).dt.normalize()
    cikti["_birlesim_gun"] = zaman
    cikti["_birlesim_anahtar"] = cikti[key_column].astype(str)

    gunluk = gunluk.rename(columns={key_column: "_birlesim_anahtar", "tarih": "_birlesim_gun"})
    cikti = cikti.merge(gunluk, on=["_birlesim_anahtar", "_birlesim_gun"], how="left")

    # Olay OLMAYAN gun/ilce = 0 olaydir, eksik veri degil. Doldurulmazsa
    # rolling toplami NaN yayar ve "olay yok" bilgisi kaybolur.
    for kolon in ham_kolonlar:
        cikti[kolon] = cikti[kolon].fillna(0.0)

    # Ufuk disiplini DEVREDILIYOR: kaydirma matematigi burada yeniden
    # yazilmaz, denetlenmis temporal fonksiyonlari kullanilir.
    for kolon in ham_kolonlar:
        cikti = add_lag_features(
            cikti,
            kolon,
            shifts=[horizon],
            time_column=time_column,
            horizon=horizon,
            group_columns=[key_column],
        )
        cikti = add_rolling_features(
            cikti,
            kolon,
            windows,
            time_column=time_column,
            horizon=horizon,
            group_columns=[key_column],
            aggregations=("sum", "max"),
        )

    # Ham kolonlar SIZINTIDIR: tahmin gununun kendi olayini tasirlar.
    return cikti.drop(columns=[*ham_kolonlar, "_birlesim_gun", "_birlesim_anahtar"])
