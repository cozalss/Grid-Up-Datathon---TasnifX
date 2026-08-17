"""Ornek agirliklari: yenilik (recency) rampasi x aktiflik carpani.

NEREDEN GELDI (docs/09 bolum 1 + 2.2)
-------------------------------------
Iki bagimsiz yarisma kaniti tek fonksiyonda birlesiyor:

* **Zamana gore dogrusal rampa** -- Izmir Bombasi (GDZ 2024 3.su,
  github.com/sercanyesiloz/Gdz-Elektrik-Datathon-2024): ilce basina en eski
  satir 0.05, en yeni satir 0.95 agirlik alir. Yakin gecmis, rejimi en iyi
  temsil eden donemdir; model ona daha cok bassin.
* **Aktiflik carpani** -- M5 14.su: hep-sifir (olu) seriler egitimi domine
  etmesin diye serinin son-N-gun aktiflik orani agirliga carpilir. Bizim
  analogumuz: son ``activity_window`` gunde hedefi > 0 olan gun payi.

SIZINTI NOTU -- AGIRLIK FEATURE DEGILDIR
----------------------------------------
Agirliklar yalnizca EGITIM satirlarinin kayip fonksiyonunu olcekler; modele
deger olarak girmez ve tahmin aninda var olmazlar. Aktiflik orani GERIYE
donuk pencereyle hesaplanir (satirin kendi gunu dahil) ve
``models.cross_validate`` agirligi yalnizca train-fold dilimiyle kullanir --
valid satirlarinin hedefi hicbir egitime agirlik uzerinden sizamaz.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .validation import parse_time_series

__all__ = ["recency_activity_weights"]


def recency_activity_weights(
    frame: pd.DataFrame,
    value_column: str,
    *,
    time_column: str,
    group_columns: Sequence[str] | None = None,
    start: float = 0.05,
    end: float = 0.95,
    activity_window: int = 28,
    activity_floor: float = 0.25,
) -> np.ndarray:
    """Satir basina egitim agirligi: dogrusal zaman rampasi x aktiflik carpani.

    Formul (grup ici, zamana gore sirali)::

        rampa(i)    = start + (end - start) * i / (n - 1)      # i = 0..n-1
        aktiflik(d) = son activity_window gunde deger > 0 olan gun payi
        carpan(d)   = activity_floor + (1 - activity_floor) * aktiflik(d)
        agirlik     = rampa * carpan

    Carpan afin haritadir (sert ``max`` degil): tam-olu seri
    ``activity_floor``a iner ama hic sifirlanmaz -- sifir agirlik, o ilcenin
    p(0) sinyalini de silerdi. Tam aktif seri 1.0 alir.

    Args:
        frame: Girdi (DEGISTIRILMEZ).
        value_column: Hedef benzeri kolon; ``> 0`` "aktif gun" sayilir
            (NaN aktif SAYILMAZ).
        time_column: Grup ici kronoloji icin zaman kolonu.
        group_columns: Rampa ve aktiflik bu gruplar icinde ayri hesaplanir
            (or. ilce). ``None`` = tek grup.
        start: En eski satirin rampa degeri.
        end: En yeni satirin rampa degeri. Tek satirlik grup ``end`` alir
            (en guncel gozlem odur).
        activity_window: Aktiflik payinin geriye donuk pencere boyu (satir).
        activity_floor: Carpanin tabani (0..1).

    Returns:
        ``len(frame)`` boyutlu float64 dizi, GIRDI SIRASINDA.

    Raises:
        KeyError: Kolonlar frame'de yoksa.
        ValueError: Parametreler tanim araligi disindaysa.
    """
    for column in (value_column, time_column):
        if column not in frame.columns:
            raise KeyError(f"Kolon '{column}' frame icinde yok.")
    if start < 0 or end < 0 or start > end:
        raise ValueError(f"0 <= start <= end olmali, verilen: start={start}, end={end}")
    if not 0.0 <= activity_floor <= 1.0:
        raise ValueError(f"activity_floor [0, 1] araliginda olmali: {activity_floor}")
    if activity_window < 1:
        raise ValueError(f"activity_window >= 1 olmali: {activity_window}")
    if len(frame) == 0:
        return np.zeros(0, dtype="float64")

    groups = list(group_columns or [])
    degerler = pd.to_numeric(frame[value_column], errors="coerce")

    calisma = pd.DataFrame(
        {
            "_zaman": parse_time_series(frame[time_column], strict=False),
            "_olayli": (degerler > 0).astype("float64"),
        }
    )
    for column in groups:
        calisma[column] = frame[column].to_numpy()
    calisma["_sira"] = np.arange(len(calisma))

    # kind="stable": ayni zamanli satirlarin girdi sirasi korunur -- rampa
    # deterministik kalir.
    sirali = calisma.sort_values([*groups, "_zaman"], kind="stable")

    if groups:
        grup = sirali.groupby(groups, observed=True, sort=False)
        boyut = grup["_olayli"].transform("size").to_numpy(dtype="float64")
        konum = grup.cumcount().to_numpy(dtype="float64")
        aktiflik = (
            grup["_olayli"]
            .transform(lambda s: s.rolling(activity_window, min_periods=1).mean())
            .to_numpy(dtype="float64")
        )
    else:
        boyut = np.full(len(sirali), float(len(sirali)))
        konum = np.arange(len(sirali), dtype="float64")
        aktiflik = (
            sirali["_olayli"]
            .rolling(activity_window, min_periods=1)
            .mean()
            .to_numpy(dtype="float64")
        )

    rampa = np.where(boyut > 1, start + (end - start) * konum / np.maximum(boyut - 1.0, 1.0), end)
    carpan = activity_floor + (1.0 - activity_floor) * aktiflik
    agirlik = rampa * carpan

    cikti = np.empty(len(frame), dtype="float64")
    cikti[sirali["_sira"].to_numpy()] = agirlik
    return cikti
