"""ONARIMIN SINAVI -- kirli katlar kesilince blok-cifti korelasyonlari duzeliyor mu?

Tasarim: her blok tarihe gore ikiye bolunur (erken / gec). Ayni yontemle
(buzulmus grup ofseti, n0=200) her yarim icin ofset haritasi cikarilir.
Uc karsilastirma:

  (a) TEMIZ-TEMIZ  kis26_erken  vs kis26_gec    (iki ay arayla, TEMIZ model)
  (b) KIRLI-TEMIZ  guz25_gec    vs kis26_erken  (BITISIK gunler, kirli->temiz)
  (c) KIRLI-KIRLI  yaz25_gec    vs guz25_erken  (bitisik gunler, kirli->kirli)
  (d) KIRLI-ICI    yaz25_erken  vs yaz25_gec    (ayni kirli model)

(b) zamanda (a)'dan DAHA YAKIN. Eger negatif korelasyonun sebebi mevsim/zaman
uzakligi olsaydi (b) >= (a) olurdu. Kirlilikse (b) << (a) olur.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BURA = Path(__file__).resolve().parent
sys.path.insert(0, str(BURA))
sys.path.insert(0, str(BURA.parent / "sicak_kaldirac"))
from olcut import delta_coz, grup_ofseti, hazirla, zincir  # noqa: E402
from ortak import BLOKLAR, bloklari_kur  # noqa: E402

ANAHTARLAR = ["hg", "kova", "ilce", "seviye_d", "seviye_d10", "gecmis_k", "sifir_k", "tanim"]
ORTA = {"yaz25": "2025-06-01", "guz25": "2025-10-01", "kis26": "2026-02-01"}


def main() -> int:
    bl = bloklari_kur()
    yarim = {}
    for k in BLOKLAR:
        b = bl[k]
        hazirla(b)
        r = zincir(b)
        tar = pd.to_datetime(b.cerceve["tarih"]).to_numpy()
        kes = np.datetime64(ORTA[k])
        for ad, m in (("erken", tar < kes), ("gec", tar >= kes)):
            rr = r + delta_coz(b, r, m)
            yarim[f"{k}_{ad}"] = (b, rr, m)
            print(
                f"  {k}_{ad:6} {int(m.sum()):>8,} satir  "
                f"{pd.Timestamp(tar[m].min()):%Y-%m-%d}..{pd.Timestamp(tar[m].max()):%Y-%m-%d}"
            )

    ciftler = [
        ("(a) TEMIZ-TEMIZ", "kis26_erken", "kis26_gec"),
        ("(b) KIRLI-TEMIZ", "guz25_gec", "kis26_erken"),
        ("(c) KIRLI-KIRLI", "yaz25_gec", "guz25_erken"),
        ("(d) KIRLI-ICI  ", "yaz25_erken", "yaz25_gec"),
        ("(e) TEMIZ-KIRLI", "kis26_gec", "yaz25_erken"),
    ]

    print()
    print("=" * 104)
    print("GRUP OFSETI KORELASYONU -- yarim-blok ciftleri (buzulmus, n0=200)")
    print("=" * 104)
    bas = f"{'cift':18}{'A':14}{'B':14}"
    for a in ANAHTARLAR:
        bas += f"{a[:9]:>11}"
    print(bas)
    print("-" * 104)
    for etiket, ka, kb in ciftler:
        sat = f"{etiket:18}{ka:14}{kb:14}"
        for a in ANAHTARLAR:
            ba, ra, ma = yarim[ka]
            bb, rb, mb = yarim[kb]
            ha = pd.Series(grup_ofseti(ba, ra, ma, a))
            hb = pd.Series(grup_ofseti(bb, rb, mb, a))
            x = pd.concat([ha, hb], axis=1, join="inner").dropna()
            if len(x) < 3:
                sat += f"{'n<3':>11}"
            else:
                sat += f"{x.iloc[:, 0].corr(x.iloc[:, 1]):>+11.3f}"
        print(sat)

    print()
    print("n (ortak grup sayisi):")
    for etiket, ka, kb in ciftler:
        sat = f"{etiket:18}"
        for a in ANAHTARLAR:
            ba, ra, ma = yarim[ka]
            bb, rb, mb = yarim[kb]
            ha = pd.Series(grup_ofseti(ba, ra, ma, a))
            hb = pd.Series(grup_ofseti(bb, rb, mb, a))
            sat += f"{len(pd.concat([ha, hb], axis=1, join='inner').dropna()):>11,}"
        print(sat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
