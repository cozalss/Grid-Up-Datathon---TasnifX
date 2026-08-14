"""Kategorik kodlama: frekans, sayim ve fold-disi (out-of-fold) hedef kodlama.

HEDEF KODLAMA TABULAR YARISMALARIN EN GUCLU VE EN TEHLIKELI TEKNIGIDIR.

Fikir: her kategoriyi, o kategorideki hedefin ortalamasiyla degistir. Yuksek
kardinaliteli kolonlarda (trafo_id, ilce, abone_grubu) one-hot'un yapamadigi
seyi yapar.

TEHLIKE: Naif uygulama -- tum train uzerinde ortalama alip ayni train'e
uygulamak -- her satirin KENDI hedefini feature'ina karistirir. Sonuc: CV
skorun ucar, leaderboard skorun cakilir. Bu, Kaggle'da en sik yapilan hatadir.

COZUM: Kodlamayi FOLD ICINDE hesapla. Bir satirin kodlamasi, o satirin
bulunmadigi fold'lardan gelmelidir. Bu modul bunu zorunlu kilar --
``oof_target_encode`` fold indeksi olmadan CALISMAZ.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "add_frequency_encoding",
    "add_count_encoding",
    "oof_target_encode",
    "add_combination_features",
    "reduce_rare_categories",
]


def add_frequency_encoding(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    reference: pd.DataFrame | None = None,
    suffix: str = "_frekans",
) -> pd.DataFrame:
    """Her kategoriyi goreli frekansiyla kodlar. YENI frame dondurur.

    Hedefi KULLANMAZ, dolayisiyla sizinti riski yoktur ve fold disina cikmasi
    gerekmez. Cogu zaman hedef kodlamaya yakin performans verir -- once bunu dene.

    Args:
        reference: Frekanslarin hesaplanacagi referans (genellikle
            ``pd.concat([train, test])``). ``None`` ise ``frame`` kullanilir.
            Train+test uzerinden hesaplamak MESRUDUR: test'in hedefini degil,
            yalnizca dagilimini kullanir.
    """
    source = reference if reference is not None else frame
    new_columns = {}

    for column in columns:
        if column not in frame.columns:
            raise KeyError(f"Kolon '{column}' frame icinde yok.")
        frequencies = source[column].value_counts(normalize=True, dropna=False)
        new_columns[f"{column}{suffix}"] = (
            frame[column].map(frequencies).astype("float32").fillna(0.0)
        )

    return frame.assign(**new_columns)


def add_count_encoding(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    reference: pd.DataFrame | None = None,
    suffix: str = "_sayim",
) -> pd.DataFrame:
    """Her kategoriyi mutlak gorulme sayisiyla kodlar. YENI frame dondurur.

    Frekans kodlamasindan farki: mutlak buyukluk bilgisini korur. Bir trafonun
    veri setinde 10.000 kez gorunmesi ile 3 kez gorunmesi farkli seylerdir --
    ikincisi muhtemelen yeni kurulmus veya arizali bir kayittir.
    """
    source = reference if reference is not None else frame
    new_columns = {}

    for column in columns:
        if column not in frame.columns:
            raise KeyError(f"Kolon '{column}' frame icinde yok.")
        counts = source[column].value_counts(dropna=False)
        new_columns[f"{column}{suffix}"] = (
            frame[column].map(counts).astype("float32").fillna(0.0)
        )

    return frame.assign(**new_columns)


def _smoothed_means(
    categories: pd.Series,
    target: pd.Series,
    *,
    prior: float,
    smoothing: float,
) -> pd.Series:
    """Bayesci yumusatma ile kategori ortalamalari.

    Formul::

        kodlama = (n * kategori_ortalamasi + m * genel_ortalama) / (n + m)

    ``n`` az oldugunda kodlama genel ortalamaya yaklasir -- yani 3 ornekli bir
    kategori, 3.000 ornekli bir kategori kadar guvenilir sayilmaz. Yumusatma
    olmadan nadir kategoriler saf gurultu tasir ve model onlari ezberler.
    """
    stats = target.groupby(categories, observed=True).agg(["sum", "count"])
    return (stats["sum"] + prior * smoothing) / (stats["count"] + smoothing)


def oof_target_encode(
    train: pd.DataFrame,
    target: pd.Series,
    columns: Sequence[str],
    folds: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    test: pd.DataFrame | None = None,
    smoothing: float = 20.0,
    noise_level: float = 0.0,
    seed: int = 42,
    suffix: str = "_hedef_kod",
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Sizintisiz (fold-disi) hedef kodlama. YENI frame'ler dondurur.

    Her train satirinin kodlamasi, o satirin AIT OLMADIGI fold'lardan hesaplanir.
    Test kodlamasi tum train'den hesaplanir (test hedefi zaten bilinmiyor,
    sizinti yok).

    Args:
        train: Egitim feature'lari.
        target: Hedef degerler (``train`` ile ayni uzunlukta, ayni sirada).
        columns: Kodlanacak kategorik kolonlar.
        folds: ``(train_idx, valid_idx)`` ciftleri -- CV bolucusunun ciktisi.
            ZORUNLU: sizintisiz kodlamanin tek yolu budur.
        smoothing: Bayesci yumusatma agirligi. Yuksek deger = daha muhafazakar.
            Kategori basina ortalama ornek sayisiyla ayni mertebede sec.
        noise_level: Kodlamaya eklenecek gaussian gurultunun standart sapmasi
            (hedefin std'sine oranla). Asiri uyumu kirar; 0.01-0.05 tipiktir.
        seed: Gurultu icin tohum.

    Returns:
        ``(kodlanmis_train, kodlanmis_test)``. ``test`` verilmemisse ikincisi ``None``.

    Raises:
        ValueError: ``folds`` bossa -- sessizce sizintili kodlama uretmek yerine.
    """
    fold_list = list(folds)
    if not fold_list:
        raise ValueError(
            "oof_target_encode fold gerektirir. Fold'suz hedef kodlama HER SATIRIN "
            "KENDI HEDEFINI sizdirir ve CV'yi anlamsiz kilar."
        )

    if len(train) != len(target):
        raise ValueError(f"train ({len(train)}) ve target ({len(target)}) uzunluklari farkli.")

    target_values = pd.Series(np.asarray(target, dtype=float), index=range(len(target)))
    prior = float(target_values.mean())
    rng = np.random.default_rng(seed)

    train_encoded: dict[str, np.ndarray] = {}
    test_encoded: dict[str, np.ndarray] = {}

    for column in columns:
        if column not in train.columns:
            raise KeyError(f"Kolon '{column}' train icinde yok.")

        categories = train[column].reset_index(drop=True)
        oof = np.full(len(train), np.nan, dtype="float64")

        for fit_idx, apply_idx in fold_list:
            means = _smoothed_means(
                categories.iloc[fit_idx],
                target_values.iloc[fit_idx],
                prior=prior,
                smoothing=smoothing,
            )
            oof[apply_idx] = categories.iloc[apply_idx].map(means).to_numpy(dtype="float64")

        # Hicbir fold'un gormedigi kategori -> genel ortalama.
        oof = np.where(np.isnan(oof), prior, oof)

        if noise_level > 0:
            scale = float(target_values.std()) * noise_level
            oof = oof + rng.normal(0.0, scale, size=len(oof))

        train_encoded[f"{column}{suffix}"] = oof.astype("float32")

        if test is not None:
            if column not in test.columns:
                raise KeyError(f"Kolon '{column}' test icinde yok.")
            full_means = _smoothed_means(
                categories, target_values, prior=prior, smoothing=smoothing
            )
            mapped = test[column].map(full_means).to_numpy(dtype="float64")
            test_encoded[f"{column}{suffix}"] = np.where(
                np.isnan(mapped), prior, mapped
            ).astype("float32")

    encoded_train = train.assign(**train_encoded)
    encoded_test = test.assign(**test_encoded) if test is not None else None
    return encoded_train, encoded_test


def add_combination_features(
    frame: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    *,
    separator: str = "__",
) -> pd.DataFrame:
    """Iki kategorik kolonu birlestirip yeni kategori uretir. YENI frame dondurur.

    NEDEN: ``ilce`` ve ``ariza_tipi`` ayri ayri zayif olabilir ama
    ``ilce__ariza_tipi`` bir etkilesim yakalayabilir -- or. kirsal bir ilcede
    agac temasi arizasi sik, sanayi ilcesinde asiri yuk arizasi sik.

    Kardinalite patlamasina dikkat: 50 ilce x 20 ariza tipi = 1000 kategori.
    ``reduce_rare_categories`` ile birlikte kullan.
    """
    new_columns = {}
    for left, right in pairs:
        for column in (left, right):
            if column not in frame.columns:
                raise KeyError(f"Kolon '{column}' frame icinde yok.")
        name = f"{left}{separator}{right}"
        new_columns[name] = (
            frame[left].astype(str) + separator + frame[right].astype(str)
        ).astype("category")
    return frame.assign(**new_columns)


def reduce_rare_categories(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    min_count: int = 10,
    other_label: str = "_DIGER",
    reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Nadir kategorileri tek bir kova altinda toplar. YENI frame dondurur.

    NEDEN: Bir kategori 3 kez gorunuyorsa, model onun icin ogrendigi sey
    gurultudur ve test'te muhtemelen hic gorunmeyecektir. Bunlari birlestirmek
    hem varyansi hem bellek kullanimini dusurur.

    ``reference`` verilirse esik ORADA hesaplanir -- train ve test'in ayni
    kategorileri birlestirmesi icin bu gereklidir.
    """
    source = reference if reference is not None else frame
    new_columns = {}

    for column in columns:
        if column not in frame.columns:
            raise KeyError(f"Kolon '{column}' frame icinde yok.")
        counts = source[column].value_counts(dropna=False)
        keep = set(counts[counts >= min_count].index)
        as_object = frame[column].astype(object)
        new_columns[column] = (
            as_object.where(as_object.isin(keep), other_label).astype("category")
        )

    return frame.assign(**new_columns)
