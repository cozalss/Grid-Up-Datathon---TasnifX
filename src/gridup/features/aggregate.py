"""Grup istatistikleri ve oran feature'lari.

Tabular yarismalarin en verimli feature ailesi: bir satiri KENDI GRUBUNUN
istatistigine gore konumlandirmak. "Bu trafonun tuketimi, ayni ilcedeki
trafolarin ortalamasinin kac kati?" sorusu, ham tuketim degerinden cok daha
ayirt edicidir.

SIZINTI KURALI
--------------
``add_group_statistics`` HEDEFI KULLANMAZ -- yalnizca feature kolonlari uzerinden
istatistik alir. Bu yuzden train+test birlikte hesaplanabilir ve fold'a ihtiyac
duymaz. Hedef bazli grup istatistigi istiyorsan ``categorical.oof_target_encode``
kullan; o fold-disi calisir.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["add_group_statistics", "add_ratio_features", "add_target_free_aggregates"]

_DEFAULT_AGGREGATIONS = ("mean", "std", "min", "max", "median")

#: ``target_column``in ACIKCA verilmesini zorunlu kilan nobetci. Varsayilan
#: ``None`` olsaydi "vermedim" ile "hedef yok" ayirt edilemezdi ve koruma
#: pratikte hic calismazdi -- nitekim calismiyordu (olculdu: hedefle 0.96
#: korelasyonlu kolonlar sessizce uretiliyordu).
_ZORUNLU: Any = object()


def _reject_target(value_columns: Sequence[str], target_column: Any) -> None:
    """Hedef kolonu ``value_columns`` icindeyse ACIKCA reddeder.

    Bu fonksiyon ailesi hedefi kullanmadigi ICIN fold'a ihtiyac duymaz ve
    train+test uzerinde guvenle calisir. Ama hedef kazara ``value_columns``a
    girerse, satirin KENDI hedefi kendi grup ortalamasina karisir -- yani
    ``oof_target_encode``in onlemek icin var oldugu sizinti, baska bir kapidan
    geri gelir. CV skoru siser, leaderboard coker.
    """
    if target_column is _ZORUNLU:
        raise TypeError(
            "target_column ACIKCA verilmelidir.\n"
            "Bu koruma eskiden opt-in'di ve varsayilan None oldugu icin pratikte "
            "HIC calismiyordu: hedef value_columns'a girince sessizce hedefle "
            "0.96 korelasyonlu kolonlar uretiliyordu (olculdu).\n"
            "  hedef varsa   : target_column='HEDEF'\n"
            "  hedef yoksa   : target_column=None  (bilincli karar)"
        )
    if target_column and target_column in value_columns:
        raise ValueError(
            f"Hedef kolon '{target_column}' value_columns icinde. Grup istatistikleri "
            "fold-disi DEGILDIR; hedefle kullanilirsa dogrudan sizinti olur. "
            "Hedef bazli kodlama icin features.oof_target_encode kullan."
        )


def add_group_statistics(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    value_columns: Sequence[str],
    *,
    aggregations: Sequence[str] = _DEFAULT_AGGREGATIONS,
    reference: pd.DataFrame | None = None,
    add_deviation: bool = True,
    target_column: str | None = _ZORUNLU,
) -> pd.DataFrame:
    """Grup bazli istatistikler ve satirin gruptan sapmasini ekler.

    Args:
        group_columns: Gruplama anahtari, or. ``["ilce"]`` veya ``["ilce", "ay"]``.
        value_columns: Istatistigi alinacak sayisal kolonlar.
        aggregations: ``mean``, ``std``, ``min``, ``max``, ``median``, ``sum``, ``count``.
        reference: Istatistiklerin hesaplanacagi frame (genellikle train+test).
            ``None`` ise ``frame``. Hedef kullanilmadigi icin train+test guvenlidir.
        add_deviation: Satirin grup ortalamasindan farkini ve oranini da ekler --
            genellikle ham istatistikten daha guclu sinyaldir.

    Returns:
        Yeni DataFrame.
    """
    _reject_target(value_columns, target_column)

    source = reference if reference is not None else frame
    group_list = list(group_columns)

    for column in group_list + list(value_columns):
        if column not in frame.columns:
            raise KeyError(f"Kolon '{column}' frame icinde yok.")

    group_label = "_".join(group_list)
    new_columns: dict[str, pd.Series] = {}

    for value_column in value_columns:
        stats = source.groupby(group_list, observed=True)[value_column].agg(list(aggregations))
        stats.columns = [f"{value_column}_bazinda_{group_label}_{name}" for name in aggregations]

        merged = frame[group_list].merge(
            stats.reset_index(), on=group_list, how="left", validate="many_to_one"
        )

        for column in stats.columns:
            new_columns[column] = merged[column].astype("float32").to_numpy()

        if add_deviation and "mean" in aggregations:
            mean_column = f"{value_column}_bazinda_{group_label}_mean"
            group_mean = merged[mean_column].to_numpy(dtype="float64")
            row_value = frame[value_column].to_numpy(dtype="float64")

            new_columns[f"{value_column}_{group_label}_fark"] = (
                (row_value - group_mean).astype("float32")
            )
            # Sifira bolme: 0 yerine NaN uret -- 0 yanlis bir "oran yok" sinyali verir.
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(group_mean != 0, row_value / group_mean, np.nan)
            new_columns[f"{value_column}_{group_label}_oran"] = ratio.astype("float32")

            if "std" in aggregations:
                std_column = f"{value_column}_bazinda_{group_label}_std"
                group_std = merged[std_column].to_numpy(dtype="float64")
                with np.errstate(divide="ignore", invalid="ignore"):
                    zscore = np.where(group_std > 0, (row_value - group_mean) / group_std, np.nan)
                new_columns[f"{value_column}_{group_label}_zskor"] = zscore.astype("float32")

    return frame.assign(**new_columns)


def add_ratio_features(
    frame: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    *,
    epsilon: float = 1e-9,
) -> pd.DataFrame:
    """Kolon ciftleri arasinda oran, fark ve carpim uretir. YENI frame dondurur.

    Agac modelleri toplama/cikarma yapamaz -- yalnizca esik bolmesi yapar. Bu
    yuzden ``a / b`` gibi turetilmis bir kolon, model tek basina kesfedemedigi
    bir iliskiyi acikca verir.

    Elektrik ornegi: ``kesinti_suresi / abone_sayisi`` -> abone basina etki
    (SAIDI'nin bilesenleri).
    """
    new_columns = {}
    for numerator, denominator in pairs:
        for column in (numerator, denominator):
            if column not in frame.columns:
                raise KeyError(f"Kolon '{column}' frame icinde yok.")

        top = frame[numerator].to_numpy(dtype="float64")
        bottom = frame[denominator].to_numpy(dtype="float64")

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = top / np.where(np.abs(bottom) < epsilon, np.nan, bottom)

        new_columns[f"{numerator}_bolu_{denominator}"] = ratio.astype("float32")
        new_columns[f"{numerator}_eksi_{denominator}"] = (top - bottom).astype("float32")
        new_columns[f"{numerator}_carpi_{denominator}"] = (top * bottom).astype("float32")

    return frame.assign(**new_columns)


def add_target_free_aggregates(
    train: pd.DataFrame,
    test: pd.DataFrame,
    group_columns: Sequence[str],
    value_columns: Sequence[str],
    *,
    aggregations: Sequence[str] = _DEFAULT_AGGREGATIONS,
    target_column: str | None = _ZORUNLU,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train ve test'e AYNI grup istatistiklerini uygular. Iki yeni frame dondurur.

    Neden ayri fonksiyon: istatistikleri yalnizca train'den hesaplamak, test'te
    train'de gorulmemis gruplar icin NaN birakir. Train+test birlestirip
    hesaplamak hedefi kullanmadigi icin MESRUDUR ve bu bosluğu kapatir.

    Bu, Kaggle'da "test-time feature engineering" olarak bilinen ve kural disi
    OLMAYAN standart bir tekniktir -- test'in HEDEFINI degil, yalnizca
    feature dagilimini kullanir.
    """
    _reject_target(value_columns, target_column)

    shared = list(group_columns) + list(value_columns)
    missing_train = [column for column in shared if column not in train.columns]
    missing_test = [column for column in shared if column not in test.columns]
    if missing_train or missing_test:
        raise KeyError(
            f"Eksik kolonlar -- train: {missing_train}, test: {missing_test}"
        )

    reference = pd.concat([train[shared], test[shared]], ignore_index=True)

    return (
        add_group_statistics(
            train, group_columns, value_columns,
            aggregations=aggregations, reference=reference, target_column=target_column,
        ),
        add_group_statistics(
            test, group_columns, value_columns,
            aggregations=aggregations, reference=reference, target_column=target_column,
        ),
    )
