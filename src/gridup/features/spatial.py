"""Mekansal feature'lar: komsu ilce sinyali (spatial lag).

NEDEN BU MODUL VAR
------------------
Firtina, sicak hava dalgasi ve yildirim **ilce siniri tanimaz**. Komsu ilcede
dun yasanan kesinti, bugun bu ilcede olacak kesinti icin guclu bir sinyaldir --
cunku ayni hava olayi bolgeyi tarar.

2023 GDZ Datathon birincisinin cozumunde komsuluk/bolgesel sinyal fikri
vardi ve
arastirma bunu "ucuz ve yuksek getirili" diye isaretledi.

MEKANSAL SIZINTI UYARISI
------------------------
Komsu ilcenin **hedefini** kullanmak, sizinti riski tasir: eger CV bolmesi
zamana degil de rastgele yapiliyorsa, komsunun AYNI GUNDEKI hedefi valid
kumesinden gelir. Bu modul bunu iki sekilde onler:

  1. ``add_neighbour_target_lag`` ``horizon`` kadar kaydirir -- yani komsunun
     GECMISI kullanilir, bugunu degil. Kaydirma SATIR bazlidir; bu yuzden
     fonksiyon girdinin ``(varlik, gun)`` bazinda TEK satir olmasini SART
     kosar ve tekrarli panelde ``ValueError`` firlatir. Bu sart olmadan
     "horizon satir once" cogu zaman AYNI GUN demektir (olculdu: 40 satir/gun
     + horizon=30 -> satirlarin %25.0'i ayni gunden gelir).
  2. ``add_neighbour_feature_mean`` KAYDIRMAZ; bu yuzden ``target_column``
     ACIKCA verilmelidir ve hedef ``value_columns``a girerse reddedilir.
  3. Zaman bazli CV kullaniliyorsa (``purged_time_series_split``) komsu satirlar
     da ayni zaman diliminde oldugu icin dogal olarak ayrilir.

Komsunun **hava durumunu** kullanmak sizinti DEGILDIR ve kaydirma gerektirmez.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .aggregate import _ZORUNLU

__all__ = [
    "haversine_matrix",
    "nearest_neighbours",
    "add_neighbour_target_lag",
    "add_neighbour_feature_mean",
    "EARTH_RADIUS_KM",
]

EARTH_RADIUS_KM = 6371.0088


def haversine_matrix(
    latitudes: np.ndarray, longitudes: np.ndarray
) -> np.ndarray:
    """Tum nokta ciftleri arasi buyuk daire mesafesi (km). NxN matris dondurur.

    Duz Oklid mesafesi Turkiye enleminde ciddi hata verir: 1 derece boylam
    ekvatorda ~111 km iken 39. enlemde ~86 km'dir. Ilceler arasi siralamayi
    bozacak kadar buyuk bir fark.

    >>> import numpy as np
    >>> d = haversine_matrix(np.array([38.42, 38.62]), np.array([27.14, 27.43]))
    >>> bool(30 < d[0, 1] < 35)
    True
    """
    latitude_rad = np.deg2rad(np.asarray(latitudes, dtype="float64"))
    longitude_rad = np.deg2rad(np.asarray(longitudes, dtype="float64"))

    delta_lat = latitude_rad[:, None] - latitude_rad[None, :]
    delta_lon = longitude_rad[:, None] - longitude_rad[None, :]

    inner = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(latitude_rad)[:, None] * np.cos(latitude_rad)[None, :]
        * np.sin(delta_lon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(inner, 0, 1)))


def nearest_neighbours(
    coordinates: pd.DataFrame,
    *,
    key_column: str,
    latitude_column: str = "lat",
    longitude_column: str = "lon",
    k: int = 3,
    max_distance_km: float | None = 120.0,
) -> pd.DataFrame:
    """Her varlik icin en yakin ``k`` komsuyu bulur.

    Gercek sinir poligonu (GeoJSON) yoksa mesafe tabanli komsuluk iyi bir
    yaklasimdir: bir hava olayinin etki yaricapi zaten idari sinirlari degil,
    kilometreyi takip eder.

    Args:
        coordinates: ``key_column``, ``latitude_column``, ``longitude_column``
            iceren tablo. Her varlik icin TEK satir.
        k: Kac komsu.
        max_distance_km: Bundan uzak komsulari eleme. ``None`` = sinirsiz.
            Ege bolgesinde 120 km makul bir firtina olcegidir.

    Returns:
        Uzun format: ``[key_column, "komsu", "mesafe_km", "komsu_sirasi"]``.

    Raises:
        ValueError: Anahtar tekrarliysa (her varlik tek satir olmali).
    """
    for column in (key_column, latitude_column, longitude_column):
        if column not in coordinates.columns:
            raise KeyError(f"Kolon '{column}' koordinat tablosunda yok.")

    if coordinates[key_column].duplicated().any():
        duplicated = coordinates.loc[coordinates[key_column].duplicated(), key_column]
        raise ValueError(
            f"Koordinat tablosunda tekrarlayan anahtar var: {duplicated.unique()[:5]}. "
            "Her varlik icin tek satir olmali."
        )

    keys = coordinates[key_column].to_numpy()
    distances = haversine_matrix(
        coordinates[latitude_column].to_numpy(),
        coordinates[longitude_column].to_numpy(),
    )
    np.fill_diagonal(distances, np.inf)  # kendisi komsusu degil

    rows = []
    for index, key in enumerate(keys):
        order = np.argsort(distances[index])[:k]
        for rank, neighbour_index in enumerate(order, start=1):
            distance = float(distances[index, neighbour_index])
            if max_distance_km is not None and distance > max_distance_km:
                continue
            rows.append(
                {
                    key_column: key,
                    "komsu": keys[neighbour_index],
                    "mesafe_km": round(distance, 2),
                    "komsu_sirasi": rank,
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError(
            f"Hicbir komsu bulunamadi. max_distance_km={max_distance_km} cok dar olabilir."
        )
    return result


#: pandas'in string kisayolu YANLIS bos-deger semantigi tasiyan istatistikler.
#:
#: ``sum`` varsayilani ``min_count=0``dir ve tum-NaN grup icin 0.0 uretir --
#: boylece "komsu hakkinda BILGI YOK" ile "komsuda kesinti YOK" ayni degere
#: duser. LightGBM NaN'i ayri bir dal olarak isler, 0.0'i gercek sifir olarak;
#: hata her varligin ilk ``horizon`` gununde sistematik olur (olculdu: serinin
#: ilk gunu mean NaN orani 1.00 iken sum NaN orani 0.00).
#:
#: Duzeltme ``.sum(min_count=1)`` ile YAPILIR -- ``.agg(python_fonksiyonu)``
#: ile DEGIL; ikincisi ayni sonucu 69 kat yavas uretir (bkz. _komsu_agrege).
_ISTATISTIK_DUZELTMESI = frozenset({"sum"})


def _hedefi_reddet(value_columns: Sequence[str], target_column: Any) -> None:
    """Hedef kolonu ``value_columns`` icindeyse ACIKCA reddeder.

    ``add_neighbour_feature_mean`` KAYDIRMAZ -- "komsunun bugunku havasi
    tahmin aninda zaten biliniyor" varsayimina dayanir. Hedef buraya girerse
    uretilen kolon komsunun AYNI GUNKU hedefi olur; olculdu: corr(komsu hedef
    ortalamasi, satirin kendi ayni gunku hedefi) = 0.813099.

    Ikinci savunma hattina guvenilemez: ``leakage_report`` korelasyon esigi
    0.95'tir ve bu deger onun ALTINDA kalir -- rapor "0 kritik" der.
    """
    if target_column is _ZORUNLU:
        raise TypeError(
            "target_column ACIKCA verilmelidir.\n"
            "Bu fonksiyon kaydirma YAPMAZ; hedef value_columns'a girerse komsunun "
            "AYNI GUNKU hedefi feature olur (olculdu: corr 0.813099) ve leakage_report "
            "0.95 esiginin altinda kaldigi icin bunu yakalamaz.\n"
            "  hedef varsa   : target_column='HEDEF'\n"
            "  hedef yoksa   : target_column=None  (bilincli karar)"
        )
    if target_column and target_column in value_columns:
        raise ValueError(
            f"Hedef kolon '{target_column}' value_columns icinde. Komsu ortalamasi "
            "KAYDIRMAZ; komsunun ayni gunku hedefi dogrudan sizintidir. Komsunun "
            "GECMIS hedefi icin features.spatial.add_neighbour_target_lag kullan."
        )


def _neighbour_aggregate(
    frame: pd.DataFrame,
    neighbours: pd.DataFrame,
    *,
    key_column: str,
    time_column: str,
    value_column: str,
    statistics: Sequence[str],
    output_prefix: str,
) -> pd.DataFrame:
    """Komsu degerlerini varlik-zaman bazinda toplayip geri baglar."""
    source = frame[[key_column, time_column, value_column]].rename(
        columns={key_column: "komsu", value_column: "_komsu_deger"}
    )

    expanded = neighbours.merge(source, on="komsu", how="inner")

    # ``sum``i AYRI hesapliyoruz. ``.agg([python_fonksiyonu])`` pandas'in
    # Cython yolundan cikip her grup icin Python cagrisina duser; olculdu
    # (96-200 ilce x 600-2409 gun, varsayilan istatistikler):
    #     grup= 57.600  0.04s -> 0.77s   17.8x
    #     grup=231.264  0.16s -> 2.96s   18.0x
    #     grup=481.800  0.29s -> 6.20s   21.6x   (grup sayisiyla DOGRUSAL)
    # Izole agregasyon adiminda fark 69 kat. ``.sum(min_count=1)`` AYNI
    # semantigi (tum-NaN grup -> NaN) Cython yolunda verir.
    grouped = expanded.groupby([key_column, time_column], observed=True)["_komsu_deger"]
    duz = [stat for stat in statistics if stat not in _ISTATISTIK_DUZELTMESI]
    parcalar: list[pd.DataFrame | pd.Series] = []
    if duz:
        parcalar.append(grouped.agg(duz))
    if "sum" in statistics:
        parcalar.append(grouped.sum(min_count=1).rename("sum"))
    aggregated = pd.concat(parcalar, axis=1)[list(statistics)].reset_index()
    aggregated.columns = [key_column, time_column] + [
        f"{output_prefix}_{stat}" for stat in statistics
    ]
    return aggregated


def _tek_satir_dogrula(
    frame: pd.DataFrame, *, key_column: str, time_column: str
) -> None:
    """Girdi ``(varlik, gun)`` bazinda tek satir degilse hata firlatir.

    ``shift(horizon)`` SATIR kaydirir, GUN degil. Ilce basina gunde birden cok
    satir varsa "horizon satir once" cogu zaman AYNI GUN olur ve komsunun
    bugunku hedefi feature'a sizar. Olculdu (20 ilce x 60 gun):
      40 satir/gun, horizon=30 -> satirlarin %25.0'i ayni gunden, corr=+0.1682
      60 satir/gun, horizon= 7 -> satirlarin %88.3'u ayni gunden, corr=+0.8098
    Once gunluk topla (dogru kullanim ayni veride corr=-0.1771).

    Tahmin etmek yerine hata firlatiyoruz: burada "muhtemelen gunluk toplamak
    istiyordu" varsayimi sessiz bir veri donusumu olurdu ve toplama kurali
    (sum mu mean mi) yalnizca cagiran tarafta bellidir.
    """
    tekrarli = int(frame.duplicated([key_column, time_column]).sum())
    if tekrarli:
        ornek = frame.loc[frame.duplicated([key_column, time_column]), key_column]
        raise ValueError(
            f"('{key_column}', '{time_column}') ikilisi {tekrarli} satirda tekrarliyor "
            f"(ornek anahtarlar: {list(ornek.unique()[:3])}). add_neighbour_target_lag "
            "shift(horizon) ile SATIR kaydirir, GUN degil: tekrarli panelde bu, "
            "komsunun AYNI GUNKU hedefini feature yapar (olculdu: 40 satir/gun + "
            "horizon=30 -> %25.0 ayni gun, corr +0.1682).\n"
            f"  Once varlik-gun bazinda topla: frame.groupby(['{key_column}', "
            f"'{time_column}'])[hedef].sum().reset_index()"
        )


def add_neighbour_target_lag(
    frame: pd.DataFrame,
    neighbours: pd.DataFrame,
    *,
    key_column: str,
    time_column: str,
    target_column: str,
    horizon: int,
    statistics: Sequence[str] = ("mean", "max", "sum"),
) -> pd.DataFrame:
    """Komsu ilcelerin GECMIS hedef degerlerini feature olarak ekler.

    Args:
        neighbours: ``nearest_neighbours`` ciktisi.
        horizon: **ZORUNLU.** Komsunun hedefi kac adim geriden alinacak.
            ``add_lag_features`` ile AYNI ufku kullan -- komsunun bugunku
            kesintisini de bilmiyorsun.

    Returns:
        Yeni DataFrame.

    Raises:
        ValueError: ``horizon < 1`` ise. Sifir ufuk, komsunun AYNI GUNDEKI
            hedefini kullanmak demektir -- bu dogrudan sizintidir.
        ValueError: ``(key_column, time_column)`` ikilisi tekrarliysa. Kaydirma
            SATIR bazlidir; tekrarli panelde "horizon satir once" ayni gune
            denk gelir ve komsunun bugunku hedefi sizar.
    """
    if horizon < 1:
        raise ValueError(
            f"horizon >= 1 olmali (verilen: {horizon}). Sifir ufuk, komsunun ayni "
            "gunku hedefini kullanmak demektir ve dogrudan sizintidir."
        )

    working = frame.copy()
    working[time_column] = pd.to_datetime(working[time_column], errors="coerce")
    _tek_satir_dogrula(working, key_column=key_column, time_column=time_column)

    # Once her varligin KENDI hedefini kaydir, sonra komsulara dagit.
    working = working.sort_values([key_column, time_column])
    working["_kaydirilmis"] = working.groupby(key_column, observed=True)[
        target_column
    ].shift(horizon)

    shifted = working[[key_column, time_column, "_kaydirilmis"]]
    aggregated = _neighbour_aggregate(
        shifted,
        neighbours,
        key_column=key_column,
        time_column=time_column,
        value_column="_kaydirilmis",
        statistics=statistics,
        output_prefix=f"komsu_{target_column}_ufuk{horizon}",
    )

    original = frame.copy()
    original[time_column] = pd.to_datetime(original[time_column], errors="coerce")
    return original.merge(
        aggregated, on=[key_column, time_column], how="left", validate="many_to_one"
    )


def add_neighbour_feature_mean(
    frame: pd.DataFrame,
    neighbours: pd.DataFrame,
    *,
    key_column: str,
    time_column: str,
    value_columns: Sequence[str],
    target_column: str | None = _ZORUNLU,
    statistics: Sequence[str] = ("mean", "max"),
) -> pd.DataFrame:
    """Komsularin HAVA/FEATURE degerlerini ekler -- kaydirma GEREKMEZ.

    Hedef kullanilMADIGI icin sizinti yoktur: komsu ilcedeki bugunku ruzgar,
    tahmin aninda bilinen bir bilgidir (hava tahmini gibi). Bu vaadi tutan sey
    ``target_column`` nobetcisidir; onsuz vaat sadece bir temenniydi.

    Kullanim: bir ilcenin kendi olcumu gurultuluyse (istasyon konumu, yerel
    topografya), komsularin ortalamasi gercek bolgesel kosulu daha iyi verir.

    Args:
        target_column: **ZORUNLU.** Hedef kolonun adi -- yalnizca REDDETMEK
            icin kullanilir, hicbir yerde okunmaz. Hedef yoksa acikca ``None``
            ver (bilincli karar). Varsayilani ``None`` yapsaydik "vermedim" ile
            "hedef yok" ayirt edilemezdi ve koruma pratikte hic calismazdi.

    Raises:
        TypeError: ``target_column`` verilmediyse.
        ValueError: Hedef ``value_columns`` icindeyse -- uretilen kolon
            komsunun AYNI GUNKU hedefi olurdu (olculdu: corr 0.813099).
        KeyError: ``value_columns`` frame'de yoksa.
    """
    _hedefi_reddet(value_columns, target_column)

    working = frame.copy()
    working[time_column] = pd.to_datetime(working[time_column], errors="coerce")

    missing = [column for column in value_columns if column not in working.columns]
    if missing:
        raise KeyError(f"Komsu agregasyonu icin eksik kolonlar: {missing}")

    result = working
    for column in value_columns:
        aggregated = _neighbour_aggregate(
            working,
            neighbours,
            key_column=key_column,
            time_column=time_column,
            value_column=column,
            statistics=statistics,
            output_prefix=f"komsu_{column}",
        )
        result = result.merge(
            aggregated, on=[key_column, time_column], how="left", validate="many_to_one"
        )

    return result
