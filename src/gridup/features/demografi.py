"""Demografik ilce oznitelikleri: EPDK bolge sinifi.

NEDEN BU MODUL VAR
------------------
EPDK'nin Elektrik Dagitimi ve Perakende Satisa Iliskin Hizmet Kalitesi
Yonetmeligi, dagitim bolgelerini NUFUSA gore uc sinifa ayirir ve tazminat/
kalite esikleri bu siniflara gore degisir. Yani DSO'nun KENDI operasyonu
(bakim onceligi, ekip konuslanmasi, kesinti toleransi) bu siniflamaya
gore sekillenir -- hazir ve resmi bir ilce oznitelifi (docs/10 bolum 1:
"DSO'larin tazminat rejiminin kendi siniflamasi; hazir ilce oznitelifi").

RESMI ESIKLER (yonetmelik tanimlari; alomaliye.com yonetmelik metni +
docs/10 bolum 1):
  * Kentsel  : nufus >= 50.000
  * Kentalti : 2.000 <= nufus < 50.000
  * Kirsal   : nufus < 2.000

NOT: Yonetmelik esikleri YERLESIM YERI (imar alani) nufusuna gore tanimlar;
biz ilce toplam nufusuna uyguluyoruz -- GDZ/ADM'nin 96 ilcelik panelinde
kullanilabilir tek granularite bu. Ilce bazinda en kucuk nufus 5.266
oldugundan referans tabloda "kirsal" sinifi cikmaz; sinif ancak yerlesim
bazli nufusla ayrisir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "epdk_bolge_sinifi",
    "KENTSEL_NUFUS_ESIGI",
    "KENTALTI_NUFUS_ESIGI",
]

#: EPDK Hizmet Kalitesi Yonetmeligi esikleri (docs/10 bolum 1).
KENTSEL_NUFUS_ESIGI = 50_000
KENTALTI_NUFUS_ESIGI = 2_000


def epdk_bolge_sinifi(
    nufus: pd.Series | np.ndarray | int | float,
) -> pd.Series | np.ndarray | str:
    """Nufusu EPDK bolge sinifina cevirir: kentsel / kentalti / kirsal.

    Vektorludur: Series girdiye ayni indeksli Series, dizi girdiye dizi,
    skalere str doner.

    >>> epdk_bolge_sinifi(50_000)
    'kentsel'
    >>> epdk_bolge_sinifi(49_999)
    'kentalti'
    >>> epdk_bolge_sinifi(1_999)
    'kirsal'

    Raises:
        ValueError: Negatif veya NaN nufus varsa. Nufus verisi eksikse bunu
            sessizce "kirsal"a dusurmek yanlis sinif uretirdi -- once veriyi
            duzelt.
    """
    degerler = np.asarray(nufus, dtype="float64")
    if np.isnan(degerler).any() or (degerler < 0).any():
        raise ValueError(
            "Nufus negatif veya NaN olamaz. Eksik nufusu sessizce siniflamak "
            "yerine once kaynak veriyi duzelt (data/reference tablosuna bak)."
        )

    siniflar = np.select(
        [degerler >= KENTSEL_NUFUS_ESIGI, degerler >= KENTALTI_NUFUS_ESIGI],
        ["kentsel", "kentalti"],
        default="kirsal",
    )
    if isinstance(nufus, pd.Series):
        return pd.Series(siniflar, index=nufus.index, name="epdk_bolge_sinifi")
    if np.isscalar(nufus):
        return str(siniflar.item())
    return siniflar
