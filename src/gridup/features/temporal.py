"""Zaman tabanli feature'lar: takvim, dongusel kodlama, gecikme (lag), kayan pencere.

Elektrik yuku ve ariza problemlerinde sinyalin buyuk kismi ZAMANDADIR:
gunun saati, haftanin gunu, mevsim, tatil, sicaklik gecmisi. Bu modul o
sinyali cikarir.

SIZINTI UYARISI
---------------
``add_lag_features`` ve ``add_rolling_features`` GECMISE bakar ve bu dogrudur.
Ama iki kural ihlal edilirse sizinti olur:

  1. Veri ZAMANA GORE SIRALI olmali. Sirali degilse ``shift(1)`` rastgele bir
     satiri alir ve gelecegi sizdirir. Bu fonksiyonlar sirayi KENDILERI garanti
     eder ve girdiyi degistirmez.

  2. Kayan pencere ``closed="left"`` olmali -- yani icinde bulunulan satirin
     kendi degeri pencereye GIRMEMELI. pandas varsayilani mevcut satiri dahil
     eder; bu, hedef turevli bir kolonda dogrudan hedef sizintisidir.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "add_calendar_features",
    "add_cyclical_features",
    "add_turkish_holiday_features",
    "add_lag_features",
    "add_rolling_features",
    "add_expanding_features",
    "TURKISH_SEASONS",
]

# Ege bolgesi icin mevsimsellik: yaz turizmi ve tarimsal sulama yuku belirleyicidir.
TURKISH_SEASONS = {
    12: "kis", 1: "kis", 2: "kis",
    3: "ilkbahar", 4: "ilkbahar", 5: "ilkbahar",
    6: "yaz", 7: "yaz", 8: "yaz",
    9: "sonbahar", 10: "sonbahar", 11: "sonbahar",
}


def add_calendar_features(
    frame: pd.DataFrame,
    time_column: str,
    *,
    prefix: str | None = None,
    include_year: bool = True,
) -> pd.DataFrame:
    """Zaman damgasindan takvim feature'lari cikarir. YENI frame dondurur.

    Args:
        frame: Girdi (degistirilmez).
        time_column: Datetime kolonu adi.
        prefix: Uretilen kolon oneki. Varsayilan: ``time_column``.
        include_year: Yil kolonu eklensin mi? DIKKAT: test donemi train'den
            sonraysa yil bir sizinti/ekstrapolasyon riskidir -- agac modelleri
            gorulmemis yil degerini ele alamaz. Zaman ayrimi varsa ``False`` yap.

    Returns:
        Yeni DataFrame.
    """
    prefix = prefix or time_column
    times = pd.to_datetime(frame[time_column], errors="coerce")

    new_columns = {
        f"{prefix}_ay": times.dt.month.astype("int8"),
        f"{prefix}_gun": times.dt.day.astype("int8"),
        f"{prefix}_haftanin_gunu": times.dt.dayofweek.astype("int8"),
        f"{prefix}_yilin_gunu": times.dt.dayofyear.astype("int16"),
        f"{prefix}_hafta": times.dt.isocalendar().week.astype("int8"),
        f"{prefix}_ceyrek": times.dt.quarter.astype("int8"),
        f"{prefix}_ayin_ilk_gunu": times.dt.is_month_start.astype("int8"),
        f"{prefix}_ayin_son_gunu": times.dt.is_month_end.astype("int8"),
        f"{prefix}_hafta_sonu": (times.dt.dayofweek >= 5).astype("int8"),
        f"{prefix}_mevsim": times.dt.month.map(TURKISH_SEASONS).astype("category"),
    }

    # Saat bilgisi yalnizca gercekten varsa eklenir; hepsi 00:00 ise gurultudur.
    if times.dt.hour.nunique(dropna=True) > 1:
        new_columns[f"{prefix}_saat"] = times.dt.hour.astype("int8")
        new_columns[f"{prefix}_mesai_saati"] = (
            times.dt.hour.between(8, 18) & (times.dt.dayofweek < 5)
        ).astype("int8")
        # Elektrik yukunde puant (pik) saatler: aksam 17-22 arasi tuketim zirvesi.
        new_columns[f"{prefix}_puant_saat"] = times.dt.hour.between(17, 22).astype("int8")

    if include_year:
        new_columns[f"{prefix}_yil"] = times.dt.year.astype("int16")

    # Mutlak zaman: agac modellerinin trendi yakalamasi icin tek monoton kolon.
    origin = times.min()
    if pd.notna(origin):
        new_columns[f"{prefix}_gun_sayaci"] = (times - origin).dt.days.astype("int32")

    return frame.assign(**new_columns)


def add_cyclical_features(
    frame: pd.DataFrame,
    columns: dict[str, int],
    *,
    drop_original: bool = False,
) -> pd.DataFrame:
    """Dongusel degiskenleri sin/cos ciftine cevirir. YENI frame dondurur.

    NEDEN: Saat 23 ile saat 0 zamansal olarak KOMSUDUR ama sayisal olarak 23
    birim uzaktir. Agac modelleri bunu bolme yaparak asabilir, ama dogrusal
    modeller ve sinir aglari asamaz. sin/cos kodlamasi komsulugu korur.

    Args:
        columns: ``{kolon_adi: periyot}`` -- or. ``{"saat": 24, "ay": 12}``.

    >>> import pandas as pd
    >>> df = pd.DataFrame({"saat": [0, 6, 12, 18, 23]})
    >>> out = add_cyclical_features(df, {"saat": 24})
    >>> sorted(c for c in out.columns if c != "saat")
    ['saat_cos', 'saat_sin']
    """
    new_columns = {}
    for column, period in columns.items():
        if column not in frame.columns:
            raise KeyError(f"Kolon '{column}' frame icinde yok.")
        values = frame[column].astype(float)
        angle = 2.0 * np.pi * values / period
        new_columns[f"{column}_sin"] = np.sin(angle).astype("float32")
        new_columns[f"{column}_cos"] = np.cos(angle).astype("float32")

    result = frame.assign(**new_columns)
    if drop_original:
        result = result.drop(columns=list(columns))
    return result


def add_turkish_holiday_features(
    frame: pd.DataFrame,
    time_column: str,
    *,
    prefix: str = "tatil",
    window_days: int = 3,
) -> pd.DataFrame:
    """TR resmi tatil feature'lari ekler. YENI frame dondurur.

    NEDEN ONEMLI: Elektrik tuketimi tatillerde ciddi degisir -- sanayi durur,
    konut tuketimi artar, tatil bolgelerinde (Mugla, Aydin, Izmir sahili) nufus
    patlar. Dini bayramlar HICRI takvime gore her yil ~11 gun KAYAR, bu yuzden
    ``ay + gun`` kolonlari onlari YAKALAYAMAZ. Ayri bir kolon sart.

    ``window_days``: bayram oncesi/sonrasi kopru gunlerini yakalamak icin
    tatile olan gun mesafesi de uretilir.
    """
    try:
        import holidays as holidays_lib
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "TR tatil feature'lari icin 'holidays' paketi gerekli: pip install holidays"
        ) from exc

    times = pd.to_datetime(frame[time_column], errors="coerce")
    valid_years = times.dt.year.dropna()
    if valid_years.empty:
        return frame.assign(**{f"{prefix}_mi": np.zeros(len(frame), dtype="int8")})

    years = range(int(valid_years.min()) - 1, int(valid_years.max()) + 2)
    calendar = holidays_lib.country_holidays("TR", years=list(years))

    dates = times.dt.date
    is_holiday = dates.map(lambda day: day in calendar if pd.notna(day) else False)

    holiday_dates = np.array(sorted(calendar.keys()), dtype="datetime64[D]")
    day_values = times.dt.normalize().to_numpy(dtype="datetime64[D]")

    def _distance_to_nearest(values: np.ndarray) -> np.ndarray:
        if holiday_dates.size == 0:
            return np.full(values.shape, 999, dtype="int16")
        # Her gun icin en yakin tatile mesafe (gun cinsinden, isaretsiz).
        differences = np.abs(
            (values[:, None] - holiday_dates[None, :]).astype("timedelta64[D]").astype(int)
        )
        return differences.min(axis=1).astype("int16")

    # Buyuk veri setlerinde NxM matris bellek yer; benzersiz gunler uzerinden hesapla.
    unique_days, inverse = np.unique(day_values, return_inverse=True)
    distances = _distance_to_nearest(unique_days)[inverse]

    new_columns = {
        f"{prefix}_mi": is_holiday.astype("int8"),
        f"{prefix}_mesafe": distances,
        f"{prefix}_yakininda": (distances <= window_days).astype("int8"),
        f"{prefix}_veya_haftasonu": (
            is_holiday.to_numpy() | (times.dt.dayofweek >= 5).to_numpy()
        ).astype("int8"),
    }
    return frame.assign(**new_columns)


def _sorted_view(frame: pd.DataFrame, sort_by: Sequence[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Sirali bir kopya ve orijinal siraya donmek icin indeks dondurur."""
    order = np.lexsort([frame[column].to_numpy() for column in reversed(list(sort_by))])
    return frame.iloc[order], order


def add_lag_features(
    frame: pd.DataFrame,
    value_column: str,
    lags: Sequence[int],
    *,
    time_column: str,
    group_columns: Sequence[str] | None = None,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Gecikmeli (lag) feature'lar ekler. YENI frame dondurur, sirayi korur.

    Zaman serisi problemlerinde en guclu feature ailesi genellikle budur:
    "bir onceki gunun tuketimi", "gecen haftanin ayni gunu".

    Args:
        value_column: Gecikmesi alinacak kolon (genellikle hedef veya bir olcum).
        lags: Gecikme adimlari, or. ``[1, 7, 14, 28]``.
        time_column: Siralama icin zaman kolonu -- ZORUNLU. Sirasiz veride
            ``shift`` rastgele satir alir ve gelecegi sizdirir.
        group_columns: Varsa her varlik icin ayri gecikme (or. trafo bazinda).

    Returns:
        Yeni DataFrame (girdi sirasinda).
    """
    if value_column not in frame.columns:
        raise KeyError(f"Kolon '{value_column}' frame icinde yok.")

    prefix = prefix or value_column
    sort_keys = list(group_columns or []) + [time_column]
    ordered, order = _sorted_view(frame, sort_keys)

    source = (
        ordered.groupby(list(group_columns), observed=True)[value_column]
        if group_columns
        else ordered[value_column]
    )

    lagged = {}
    for lag in lags:
        shifted = source.shift(lag)
        lagged[f"{prefix}_lag{lag}"] = np.asarray(shifted)

    # Orijinal siraya geri dondur.
    restore = np.empty_like(order)
    restore[order] = np.arange(len(order))
    return frame.assign(**{name: values[restore] for name, values in lagged.items()})


def add_rolling_features(
    frame: pd.DataFrame,
    value_column: str,
    windows: Sequence[int],
    *,
    time_column: str,
    group_columns: Sequence[str] | None = None,
    aggregations: Sequence[str] = ("mean", "std", "min", "max"),
    prefix: str | None = None,
) -> pd.DataFrame:
    """Kayan pencere istatistikleri ekler. YENI frame dondurur.

    KRITIK: Pencere ``shift(1)`` sonrasi hesaplanir, yani MEVCUT SATIR DAHIL
    DEGILDIR. Bu, hedef turevli bir kolonda dogrudan hedef sizintisini onler.
    pandas'in varsayilan davranisi mevcut satiri DAHIL EDER -- bu fonksiyon
    onu bilerek degistirir.

    Args:
        windows: Pencere boylari, or. ``[7, 14, 30]``.
        aggregations: ``mean``, ``std``, ``min``, ``max``, ``median``, ``sum``.
    """
    if value_column not in frame.columns:
        raise KeyError(f"Kolon '{value_column}' frame icinde yok.")

    prefix = prefix or value_column
    sort_keys = list(group_columns or []) + [time_column]
    ordered, order = _sorted_view(frame, sort_keys)

    if group_columns:
        grouped = ordered.groupby(list(group_columns), observed=True)[value_column]
        shifted = grouped.shift(1)
        rolling_source = shifted.groupby(
            [ordered[column].to_numpy() for column in group_columns]
        )
    else:
        shifted = ordered[value_column].shift(1)
        rolling_source = shifted

    computed = {}
    for window in windows:
        roller = rolling_source.rolling(window=window, min_periods=1)
        for aggregation in aggregations:
            values = getattr(roller, aggregation)()
            computed[f"{prefix}_kayan{window}_{aggregation}"] = np.asarray(values, dtype="float32")

    restore = np.empty_like(order)
    restore[order] = np.arange(len(order))
    return frame.assign(**{name: values[restore] for name, values in computed.items()})


def add_expanding_features(
    frame: pd.DataFrame,
    value_column: str,
    *,
    time_column: str,
    group_columns: Sequence[str] | None = None,
    aggregations: Sequence[str] = ("mean", "std"),
    prefix: str | None = None,
) -> pd.DataFrame:
    """Genisleyen pencere (tum gecmis) istatistikleri. YENI frame dondurur.

    Kayan pencerenin aksine tum gecmisi kullanir; bir varligin "genel seviyesini"
    yakalar. Yine ``shift(1)`` ile mevcut satir haric tutulur.
    """
    prefix = prefix or value_column
    sort_keys = list(group_columns or []) + [time_column]
    ordered, order = _sorted_view(frame, sort_keys)

    if group_columns:
        shifted = ordered.groupby(list(group_columns), observed=True)[value_column].shift(1)
        source = shifted.groupby([ordered[column].to_numpy() for column in group_columns])
    else:
        source = ordered[value_column].shift(1)

    expander = source.expanding(min_periods=1)
    computed = {
        f"{prefix}_genisleyen_{aggregation}": np.asarray(
            getattr(expander, aggregation)(), dtype="float32"
        )
        for aggregation in aggregations
    }

    restore = np.empty_like(order)
    restore[order] = np.arange(len(order))
    return frame.assign(**{name: values[restore] for name, values in computed.items()})
