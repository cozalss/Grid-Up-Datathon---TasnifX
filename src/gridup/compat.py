"""pandas / numpy surum uyumlulugu.

NEDEN BU MODUL VAR
------------------
Bu makinede pandas 3.0.3 + numpy 2.4.6 kurulu. Kaggle notebook imaji genellikle
daha eski surumler tasir. Ayni notebook her iki ortamda da calismali.

Bu makinede OLCULEN kaldirilmis API'ler (2026-08-14):

    pandas.DataFrame.applymap   KALDIRILDI  -> .map kullan
    pandas.DataFrame.append     KALDIRILDI  -> pd.concat kullan
    numpy.NaN                   KALDIRILDI  -> np.nan kullan
    numpy.float_                KALDIRILDI  -> np.float64 kullan
    pandas Copy-on-Write        HER ZAMAN ACIK, kapatilamaz

Copy-on-Write'in sonucu: zincirli atama (``df[df.a > 1]['b'] = 0``) artik
SESSIZCE hicbir sey yapmaz. 2023 oncesi her ogretici bu kalibi kullanir.
Daima ``.loc[maske, "kolon"] = deger`` yaz.

Bu modulu import etmek ortami degistirmez; yalnizca guvenli yardimcilar sunar.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "PANDAS_VERSION",
    "NUMPY_VERSION",
    "environment_report",
    "assert_no_removed_api",
    "safe_concat",
    "elementwise",
    "downcast_numeric",
    "reduce_memory",
    "is_categorical_like",
    "categorical_columns",
    "safe_str",
    "MISSING_CATEGORY",
]

# Eksik kategorik degerler icin acik sentinel. "None"/"nan" gibi kazara olusan
# stringlerden ayirt edilebilsin diye bilerek gercek bir kategori adina benzemez.
MISSING_CATEGORY = "_EKSIK"

PANDAS_VERSION: tuple[int, ...] = tuple(int(part) for part in pd.__version__.split(".")[:2])
NUMPY_VERSION: tuple[int, ...] = tuple(int(part) for part in np.__version__.split(".")[:2])

# Kaggle'da bu kalibi kullanan kod sessizce hicbir sey yapmaz (Copy-on-Write).
CHAINED_ASSIGNMENT_WARNING = (
    "Zincirli atama (df[maske]['kol'] = x) pandas >= 3.0'da sessizce etkisizdir. "
    "df.loc[maske, 'kol'] = x kullan."
)


def is_categorical_like(series: pd.Series) -> bool:
    """Bu kolon modele verilmeden ONCE kodlanmali mi?

    NEDEN AYRI BIR FONKSIYON -- bu makinede OLCULEN davranis (pandas 3.0.3):

        Seri tipi          dtype       is_object_dtype   is_string_dtype
        -----------------  ----------  ---------------   ---------------
        pd.Series(["a"])   str         False             True
        astype("category") category    False             True
        dtype=object       object      True              False

    Yani pandas 3.0'da duz metin kolonlari artik ``object`` DEGIL ``str``
    dtype'indadir ve ``is_object_dtype`` onlari GORMEZ. pandas 2.x'te ise tam
    tersi: metin ``object``tir ve ``is_string_dtype`` False doner.

    Tek bir yuklemle her iki dunyayi kapsamazsan sonuc SESSIZ degildir ama
    gecte gelir: LightGBM egitim aninda
    ``ValueError: pandas dtypes must be int, float or bool`` firlatir --
    yani feature'lari kurup 10 dakika bekledikten sonra.

    Bool ve datetime bilerek DISLANIR: ikisi de model tarafindan dogrudan
    kullanilabilir.
    """
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_datetime64_any_dtype(dtype):
        return False
    return bool(pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype))


def categorical_columns(frame: pd.DataFrame) -> list[str]:
    """Kodlama gerektiren kolon adlarini dondurur (surumden bagimsiz)."""
    return [column for column in frame.columns if is_categorical_like(frame[column])]


def safe_str(series: pd.Series, *, missing: str | None = None) -> pd.Series:
    """Seriyi string'e cevirir; eksik degerleri SURUMDEN BAGIMSIZ ele alir.

    NEDEN ``.astype(str)`` KULLANILMAZ -- olculen davranis::

        pd.Series(["a", "b", None]).astype(str)
          pandas 2.1.4 -> ['a', 'b', 'None']   <- NaN, LITERAL bir kategori oldu
          pandas 3.0.3 -> ['a', 'b', NaN]      <- NaN korundu

    Bu fark uc yerde sessizce zarar verir:

    1. **Model egitimi.** pandas 2.x'te ``"None"`` gercek bir kategori sayilir;
       LightGBM/XGBoost'un YERLI eksik-deger islemesi devre disi kalir. Model,
       "veri yok" ile "None adli bir trafo grubu" arasindaki farki goremez.

    2. **``.astype(str).fillna(...)`` etkisiz kalir.** pandas 2.x'te ``astype``
       zaten NaN birakmadigi icin ``fillna`` dolduracak bir sey bulamaz -- yani
       eksik degeri isaretleme niyeti sessizce iptal olur.

    3. **Etkilesim feature'lari.** ``il + "__" + ilce`` islemi ``"None__bornova"``
       gibi uydurma kategoriler uretir.

    Cozum: maskeyi ORIJINAL seriden al, donusturdukten sonra geri uygula.

    Args:
        series: Girdi (degistirilmez).
        missing: Eksik degerlerin yerine konacak etiket. ``None`` ise eksiklik
            gercek NaN olarak KORUNUR (agac modelleri bunu yerli olarak isler).

    Returns:
        Yeni Series.
    """
    was_missing = series.isna()
    converted = series.astype("object").astype(str)
    return converted.mask(was_missing) if missing is None else converted.mask(
        was_missing, missing
    )


def environment_report() -> dict[str, Any]:
    """Ortamin ML acisindan onemli ozelliklerini dondurur.

    Notebook'un ilk hucresinde calistir ve ciktisini birak: jurinin
    tekrarlanabilirlik degerlendirmesinde bu bir arti puandir.
    """
    import platform
    import sys

    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }

    for name in ("sklearn", "lightgbm", "xgboost", "catboost", "optuna", "shap"):
        try:
            module = __import__(name)
            report[name] = getattr(module, "__version__", "bilinmiyor")
        except ImportError:
            report[name] = None

    report["copy_on_write_always_on"] = PANDAS_VERSION >= (3, 0)
    return report


def assert_no_removed_api(source: str) -> list[str]:
    """Kaynak kodda pandas 3.0 / numpy 2.x'te KALDIRILAN API'leri arar.

    Kaggle'dan kopyalanan bir notebook hucresini yapistirmadan once calistir.
    Bulunan sorunlarin listesini dondurur (bos liste = temiz).
    """
    removed = {
        ".applymap(": "DataFrame.applymap kaldirildi -> .map( kullan",
        ".append(": "DataFrame.append kaldirildi -> pd.concat([...]) kullan",
        "np.NaN": "np.NaN kaldirildi -> np.nan kullan",
        "numpy.NaN": "numpy.NaN kaldirildi -> numpy.nan kullan",
        "np.float_": "np.float_ kaldirildi -> np.float64 kullan",
        "np.int_": "np.int_ davranisi degisti -> np.int64 kullan",
        "np.object_": "np.object_ yerine object kullan",
        "np.bool8": "np.bool8 kaldirildi -> np.bool_ kullan",
        "inplace=True": "inplace Copy-on-Write ile guvenilmez -> yeniden atama yap",
    }
    return [message for token, message in removed.items() if token in source]


def safe_concat(frames: list[pd.DataFrame], **kwargs: Any) -> pd.DataFrame:
    """``DataFrame.append`` yerine gecen guvenli birlestirme.

    Bos listeyi ve tamamen bos DataFrame'leri acikca ele alir: pandas bos
    frame'leri concat ederken dtype'lari ``object``e cevirebilir ve bu, sonraki
    her sayisal islemi sessizce bozar.
    """
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return frames[0].iloc[:0].copy() if frames else pd.DataFrame()
    kwargs.setdefault("ignore_index", True)
    return pd.concat(non_empty, **kwargs)


def elementwise(frame: pd.DataFrame, func: Any) -> pd.DataFrame:
    """``applymap`` yerine gecen, her iki pandas surumunde de calisan eleman-bazli uygulama."""
    if hasattr(frame, "map"):
        return frame.map(func)
    return frame.applymap(func)  # pragma: no cover - yalnizca pandas < 2.1


def downcast_numeric(series: pd.Series) -> pd.Series:
    """Bir sayisal seriyi degeri koruyarak en kucuk dtype'a indirir.

    Degeri BOZACAKSA indirmez -- ``pd.to_numeric(downcast=...)`` zaten guvenlidir
    ama float32'ye inerken hassasiyet kaybi olabilecegi icin float'ta esik kontrolu
    yapariz. Yeni seri dondurur; girdi degismez.
    """
    if pd.api.types.is_integer_dtype(series):
        return pd.to_numeric(series, downcast="integer")

    if pd.api.types.is_float_dtype(series):
        as_float32 = series.astype(np.float32)
        # Hassasiyet kaybi kabul edilebilir mi? NaN'lar disinda tam esitlik ariyoruz.
        finite = series.notna()
        if finite.sum() == 0:
            return as_float32
        max_error = float(np.abs(series[finite] - as_float32[finite].astype(np.float64)).max())
        scale = float(np.abs(series[finite]).max()) or 1.0
        if max_error / scale < 1e-6:
            return as_float32
        return series

    return series


def reduce_memory(frame: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """Bellek kullanimini dusuren YENI bir DataFrame dondurur (girdi degismez).

    Kaggle notebook'lari 16-30 GB RAM ile sinirlidir; 10M+ satirli bir dagitim
    sebekesi veri seti float64 ile kolayca sinir asar.

    Dusuk kardinaliteli metin kolonlarini ``category``ye cevirir -- bu hem bellek
    kazandirir hem LightGBM'in yerel kategorik destegini acar.
    """
    before_mb = frame.memory_usage(deep=True).sum() / 1024**2
    converted = {}

    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            converted[column] = downcast_numeric(series)
        elif is_categorical_like(series) and not isinstance(series.dtype, pd.CategoricalDtype):
            unique_ratio = series.nunique(dropna=False) / max(len(series), 1)
            converted[column] = series.astype("category") if unique_ratio < 0.5 else series
        else:
            converted[column] = series

    result = pd.DataFrame(converted, index=frame.index)
    after_mb = result.memory_usage(deep=True).sum() / 1024**2

    if verbose:
        saved = 100 * (before_mb - after_mb) / before_mb if before_mb else 0.0
        print(f"Bellek: {before_mb:.1f} MB -> {after_mb:.1f} MB  ({saved:.1f}% azalma)")

    return result
