"""Gunes geometrisi ve acik-hava (clear-sky) isinim feature'lari.

NEDEN BU MODUL VAR
------------------
2023 GDZ Elektrik Datathon **birincisi** ``pvlib`` ile uretilmis gunes
isinimi feature'lari kullandi (GHI/DNI/DHI + gunes konumu + Linke bulaniklik).
Prophet baseline 4.27 MAPE alirken birincinin 1.48 almasinin bir bileseni buydu.

Elektrik dagitiminda gunesin iki ayri etkisi vardir ve ikisi de guclu:

1. **Cati ustu GES uretimi sebekeden cekilen yuku DUSURUR.** Ogle saatlerinde
   yuksek isinim, dagitilan enerjiyi mekanik olarak azaltir. Bu, sicaklikla
   ACIKLANAMAYAN bir varyans kaynagidir.
2. **Isinim -> sicaklik -> klima yuku -> trafo stresi.** Ariza/kesinti
   problemlerinde yaz tepe yuku, ekipman arizasinin en buyuk tetikleyicisidir.

SIZINTI ACISINDAN NEDEN BEDAVA
------------------------------
Acik-hava isinimi ve gunes konumu **saf astronomidir**: sadece enlem, boylam,
rakim ve zamana baglidir. Havanin ne olacagini bilmeye gerek YOKTUR.

Bu, onlari yarismadaki en degerli feature sinifi yapar: test doneminin
degerleri **kusursuz dogrulukla ve sizinti olmadan** hesaplanabilir. Hava
tahmini icin bunu soyleyemezsin -- orada tahmin hatasi tasirsin.

Bulutluluk etkisi zaten hava verisinden gelir; bu modul "bulut olmasaydi ne
kadar isinim olurdu" tabanini verir. Ikisinin ORANI (gerceklesen/acik-hava)
bulutluluk endeksidir ve tek basina cok guclu bir feature'dir.

SAAT DILIMI
-----------
Turkiye 2016'dan beri kalici UTC+3 kullanir (yaz saati uygulamasi kaldirildi).
Veri araligimiz 2020+ oldugu icin DST belirsizligi YOKTUR. Yine de zaman
damgalarini acikca ``Europe/Istanbul`` ile yerellestiriyoruz -- 2016 oncesi
veri gelirse sessizce yanlis hesaplamak yerine dogru davransin.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "SOLAR_COLUMNS",
    "add_clearness_index",
    "add_solar_features",
    "clear_sky_daily",
    "solar_geometry_daily",
]

TIMEZONE = "Europe/Istanbul"

#: Gunes sabiti (W/m2). Dunya atmosferi disina dusen ortalama isinim.
SOLAR_CONSTANT = 1361.0

#: Bir gunu ornekleyecek saat sayisi. 24 yeterlidir: acik-hava egrisi puruzsuzdur
#: ve gunluk toplamda 10 dakikalik ornekleme ile fark binde birin altinda kalir.
HOURS_PER_DAY = 24

SOLAR_COLUMNS = (
    "gunes_ghi_gunluk",
    "gunes_dni_gunluk",
    "gunes_dhi_gunluk",
    "gunes_ghi_tepe",
    "gun_uzunlugu_saat",
    "gunes_ogle_yuksekligi",
    "gunes_deklinasyon",
)


def _require_pvlib():
    """pvlib'i tembel import eder ve yoksa ne yapilacagini soyler."""
    try:
        import pvlib
    except ImportError as exc:  # pragma: no cover - kurulum yolu
        raise ImportError(
            "Gunes feature'lari icin pvlib gerekli: pip install pvlib\n"
            "Kaggle'da internet kapaliysa clear_sky_daily ciktisini parquet olarak "
            "onceden hesaplayip Dataset olarak yukle -- pvlib'e orada ihtiyacin olmaz."
        ) from exc
    return pvlib


def solar_geometry_daily(
    latitude: float,
    dates: pd.DatetimeIndex | Sequence[pd.Timestamp],
) -> pd.DataFrame:
    """Gun uzunlugu, deklinasyon ve ogle yuksekligi -- pvlib'siz, kapali formul.

    Bu uc buyukluk basit astronomi formulleriyle hesaplanir ve pvlib
    gerektirmez. ``clear_sky_daily`` calistirilamadiginda bile bu feature'lar
    kullanilabilir; gun uzunlugu tek basina mevsimselligin ``dayofyear``dan
    daha duzgun bir temsilidir (sinus/kosinus ciftine gerek kalmaz).

    Args:
        latitude: Enlem (derece, kuzey pozitif).
        dates: Gunluk tarih dizisi.

    Returns:
        ``gun_uzunlugu_saat``, ``gunes_ogle_yuksekligi``, ``gunes_deklinasyon``
        kolonlarini iceren, ``dates`` ile ayni uzunlukta frame.

    Kutup gecesi/gunu (|enlem| > 66.5) durumunda gun uzunlugu 0 veya 24'e
    sabitlenir; Turkiye icin bu dal hic calismaz ama fonksiyon genel kalsin.
    """
    index = pd.DatetimeIndex(dates)
    day_of_year = index.dayofyear.to_numpy(dtype=float)

    # Cooper (1969) deklinasyon yaklasimi -- gunluk cozunurlukte hatasi <0.5 derece.
    declination = 23.45 * np.sin(np.radians(360.0 * (284.0 + day_of_year) / 365.0))

    lat_rad = np.radians(latitude)
    dec_rad = np.radians(declination)

    # Gun batimi saat acisi: cos(w) = -tan(enlem) * tan(deklinasyon)
    cos_hour_angle = -np.tan(lat_rad) * np.tan(dec_rad)
    # Kutup gecesi/gunu: arccos tanimsiz olur, sinirlara kirp.
    cos_hour_angle = np.clip(cos_hour_angle, -1.0, 1.0)
    sunset_hour_angle = np.arccos(cos_hour_angle)
    day_length = 2.0 * np.degrees(sunset_hour_angle) / 15.0

    # Yerel gunes ogleninde gunesin ufuktan yuksekligi (derece).
    noon_elevation = 90.0 - abs(np.degrees(lat_rad) - declination)

    return pd.DataFrame(
        {
            "gun_uzunlugu_saat": day_length,
            "gunes_ogle_yuksekligi": noon_elevation,
            "gunes_deklinasyon": declination,
        },
        index=index,
    )


def clear_sky_daily(
    latitude: float,
    longitude: float,
    dates: pd.DatetimeIndex | Sequence[pd.Timestamp],
    *,
    altitude: float = 0.0,
    model: str = "ineichen",
) -> pd.DataFrame:
    """Bir konum icin gunluk acik-hava isinim toplamlari.

    Saatlik acik-hava egrisi hesaplanip gune toplanir. Cikti birimi
    **kWh/m2/gun** -- W/m2 saatlik degerlerin toplami 1000'e bolunerek.

    Args:
        latitude: Enlem (derece).
        longitude: Boylam (derece).
        dates: Gunluk tarihler (saat bileseni yok sayilir).
        altitude: Rakim (metre). Ineichen modeli rakimla artan isinimi hesaba katar.
        model: pvlib acik-hava modeli (``ineichen``, ``haurwitz``, ``simplified_solis``).

    Returns:
        ``gunes_ghi_gunluk`` (global yatay, kWh/m2), ``gunes_dni_gunluk``
        (dogrudan normal), ``gunes_dhi_gunluk`` (yayili yatay),
        ``gunes_ghi_tepe`` (gun icindeki en yuksek anlik GHI, W/m2)
        ve ``solar_geometry_daily`` kolonlari.

    Raises:
        ImportError: pvlib kurulu degilse.
    """
    pvlib = _require_pvlib()

    unique_dates = pd.DatetimeIndex(pd.DatetimeIndex(dates).normalize().unique()).sort_values()
    if len(unique_dates) == 0:
        return pd.DataFrame(columns=list(SOLAR_COLUMNS))

    # Saatlik grid: her gunun 24 saati. tz-aware olmak zorunda -- pvlib gunes
    # konumunu UTC uzerinden hesaplar ve naive zaman damgasini reddeder.
    hourly = pd.date_range(
        start=unique_dates[0],
        end=unique_dates[-1] + pd.Timedelta(hours=HOURS_PER_DAY - 1),
        freq="h",
        tz=TIMEZONE,
    )

    location = pvlib.location.Location(
        latitude=latitude, longitude=longitude, altitude=altitude, tz=TIMEZONE
    )
    clear_sky = location.get_clearsky(hourly, model=model)

    # Yerel takvim gunune gore grupla. tz-aware index'in .date'i yerel gundur.
    local_day = pd.DatetimeIndex(clear_sky.index.tz_convert(TIMEZONE).date)
    grouped = clear_sky.groupby(local_day)

    # Her model uc bileseni de dondurmez: ``haurwitz`` YALNIZCA ``ghi``
    # uretir (pvlib 0.15.2'de olculdu), ``ineichen`` ve ``simplified_solis``
    # ucunu de. Kosulsuz ``grouped["dni"]`` istemek haurwitz'de
    # ``KeyError: Column not found: dni`` ile cokuyordu -- ustelik docstring
    # haurwitz'i desteklenen model olarak listeliyordu.
    sutunlar: dict[str, pd.Series] = {
        # W/m2 x 1 saat = Wh/m2; /1000 -> kWh/m2
        "gunes_ghi_gunluk": grouped["ghi"].sum() / 1000.0,
        "gunes_ghi_tepe": grouped["ghi"].max(),
    }
    for bilesen, ad in (("dni", "gunes_dni_gunluk"), ("dhi", "gunes_dhi_gunluk")):
        if bilesen in clear_sky.columns:
            sutunlar[ad] = grouped[bilesen].sum() / 1000.0

    daily = pd.DataFrame(sutunlar)
    daily.index = pd.DatetimeIndex(daily.index)

    geometry = solar_geometry_daily(latitude, daily.index)
    combined = daily.join(geometry)

    # Istenen gunlerle hizala: bazi gunler eksikse NaN degil, hesaplanmis deger
    # dondurmek istiyoruz -- unique_dates zaten tam araligi kapsiyor.
    return combined.reindex(unique_dates)


def add_solar_features(
    frame: pd.DataFrame,
    *,
    time_column: str,
    location_column: str | None = None,
    coordinates: Mapping[str, tuple[float, float]] | tuple[float, float],
    altitudes: Mapping[str, float] | float = 0.0,
    geometry_only: bool = False,
) -> pd.DataFrame:
    """Panele gunes feature'larini ekler.

    Girdi frame'i **degistirmez**, yeni frame dondurur.

    Args:
        frame: Panel (satir = konum x gun).
        time_column: Zaman kolonu adi.
        location_column: Konum anahtari kolonu. ``None`` ise tum panel tek
            konum kabul edilir ve ``coordinates`` bir ``(lat, lon)`` cifti olmalidir.
        coordinates: Konum anahtari -> ``(enlem, boylam)`` esleme, veya tek
            konum icin dogrudan ``(enlem, boylam)``.
        altitudes: Konum anahtari -> rakim (metre), veya tek deger.
        geometry_only: ``True`` ise pvlib gerektirmeyen sadece geometri
            kolonlarini uretir. pvlib yoksa veya hiz onemliyse kullan.

    Returns:
        Gunes kolonlari eklenmis yeni frame.

    Raises:
        KeyError: ``coordinates`` icinde olmayan bir konum anahtari varsa.
    """
    if time_column not in frame.columns:
        raise KeyError(f"Zaman kolonu '{time_column}' frame'de yok.")

    result = frame.copy()
    times = pd.to_datetime(result[time_column]).dt.normalize()

    if location_column is None:
        if not isinstance(coordinates, tuple):
            raise TypeError(
                "location_column verilmediginde coordinates bir (enlem, boylam) "
                f"cifti olmali, {type(coordinates).__name__} verildi."
            )
        latitude, longitude = coordinates
        altitude = float(altitudes) if not isinstance(altitudes, Mapping) else 0.0
        table = _solar_table(latitude, longitude, times, altitude, geometry_only)
        aligned = table.reindex(times.to_numpy())
        for column in table.columns:
            result[column] = aligned[column].to_numpy()
        return result

    if location_column not in frame.columns:
        raise KeyError(f"Konum kolonu '{location_column}' frame'de yok.")
    if not isinstance(coordinates, Mapping):
        raise TypeError("location_column verildiginde coordinates bir esleme (dict) olmali.")

    keys = result[location_column].astype(str)
    missing = sorted(set(keys.unique()) - set(coordinates))
    if missing:
        raise KeyError(
            f"{len(missing)} konumun koordinati yok: {missing[:5]}"
            f"{' ...' if len(missing) > 5 else ''}\n"
            "data/reference/ilceler_gdz_adm.parquet dosyasindaki 'il_key|ilce_key' "
            "anahtar formatiyla eslestirdiginden emin ol."
        )

    # KONUMSAL yazim, etiket hizalamasi DEGIL.
    #
    # Onceki surum ``pd.concat(pieces).reindex(result.index)`` kullaniyordu.
    # Frame'in index'i benzersizse bu dogru calisir; ama ``pd.concat([train,
    # test])`` sonrasi index TEKRARLIDIR ve bu repoda o kalip belgelenmis bir
    # kullanimdir (features/categorical.py docstring'i onu oneriyor).
    # Tekrarli index'te ``reindex`` hizalama YAPMAZ -- diziyi oldugu gibi
    # gecirir. Sonuc: her ilcenin gunes degerleri BASKA bir ilceye yazilir.
    #
    # OLCULDU: Ocak ayinda 60N enlemi 6.1 saatlik gun gormesi gerekirken
    # 10.9 saat aliyordu (20N'nin degeri). Hata firlamiyor, satir sayisi ve
    # kolonlar ayni -- tamamen sessiz. Fizik kontrolu olmasa fark edilmezdi.
    #
    # Konumsal yazim index'ten tamamen bagimsizdir, dolayisiyla bu tuzak
    # yapisal olarak ortadan kalkar.
    konum = np.arange(len(result))
    ciktilar: dict[str, np.ndarray] = {}

    for key, group_times in times.groupby(keys):
        latitude, longitude = coordinates[key]
        altitude = (
            float(altitudes.get(key, 0.0)) if isinstance(altitudes, Mapping) else float(altitudes)
        )
        table = _solar_table(latitude, longitude, group_times, altitude, geometry_only)
        aligned = table.reindex(group_times.to_numpy())
        # group_times.index -> result icindeki KONUMLAR
        satirlar = konum[keys.to_numpy() == key]
        for column in aligned.columns:
            if column not in ciktilar:
                ciktilar[column] = np.full(len(result), np.nan, dtype="float64")
            ciktilar[column][satirlar] = aligned[column].to_numpy(dtype="float64")

    for column, degerler in ciktilar.items():
        result[column] = degerler
    return result


def _solar_table(
    latitude: float,
    longitude: float,
    times: pd.Series,
    altitude: float,
    geometry_only: bool,
) -> pd.DataFrame:
    """Tek konum icin gunluk gunes tablosu (tarih indeksli)."""
    unique = pd.DatetimeIndex(times.unique()).sort_values()
    if geometry_only:
        return solar_geometry_daily(latitude, unique)
    return clear_sky_daily(latitude, longitude, unique, altitude=altitude)


def add_clearness_index(
    frame: pd.DataFrame,
    *,
    observed_column: str,
    clear_sky_column: str = "gunes_ghi_gunluk",
    output_column: str = "berraklik_endeksi",
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    """Bulutluluk endeksi: gerceklesen isinim / acik-hava isinimi.

    Bu oran, hava durumu verisi ile astronomi arasindaki koprudir ve tek basina
    ``ghi``den daha bilgilendiricidir: 0.75 yazin da kisin da "acik gun" demektir,
    ham ``ghi`` ise mevsimle karisir.

    Args:
        frame: Hem gerceklesen hem acik-hava isinim kolonunu iceren frame.
        observed_column: Olculen/tahmin edilen isinim kolonu (ayni birimde).
        clear_sky_column: ``clear_sky_daily`` ciktisindaki acik-hava kolonu.
        output_column: Uretilecek kolon adi.
        epsilon: Sifira bolmeyi engelleyen esik.

    Returns:
        Endeks kolonu eklenmis yeni frame. Acik-hava degeri ~0 olan gunlerde
        (kutup gecesi gibi) sonuc NaN'dir -- 0/0 icin uydurma deger uretmeyiz.
    """
    for column in (observed_column, clear_sky_column):
        if column not in frame.columns:
            raise KeyError(f"'{column}' kolonu frame'de yok.")

    result = frame.copy()
    denominator = result[clear_sky_column].to_numpy(dtype=float)
    numerator = result[observed_column].to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        index = np.where(denominator > epsilon, numerator / denominator, np.nan)

    # Fiziksel ust sinir ~1.2'dir (bulut kenari yansimasi kisa sureli asim yapar).
    # Bunun ustu olcum hatasi veya birim uyusmazligidir; kirpmak yerine BIRAKIYORUZ
    # ki profiling asamasinda gorulsun.
    result[output_column] = index
    return result
