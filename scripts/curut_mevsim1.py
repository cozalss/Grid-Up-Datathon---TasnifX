# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""CURUTME: mevsimsel yakinlik agirligi -- yapisal tani (model egitimi YOK)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))
import deney as d
import tuketim_model as tm  # noqa: E402


def dairesel(tarih, hedef_doy):
    doy = pd.to_datetime(tarih).dt.dayofyear.to_numpy()
    h = np.unique(hedef_doy)
    f = np.abs(doy[:, None] - h[None, :])
    return np.minimum(f, 365 - f).min(axis=1).astype("float64")


egitim, test = d.cerceveleri_kur()
ek = d._ek_kokenler_kur(False)
ek = ek[ek["_blok"].isin([a for a, _, _ in tm.EK_KOKENLER])]
ortak = [k for k in egitim.columns if k in ek.columns]
genis = pd.concat([egitim[ortak], ek[ortak]], ignore_index=True)
print(f"ana egitim {len(egitim):,}   ek koken {len(ek):,}   genis {len(genis):,}")
print("\nEK KOKEN bloklarinin satir sayisi ve ozet penceresi:")
for a in genis["_blok"].unique():
    s = genis[genis["_blok"] == a]
    op = s["ozet_pencere_gun"].unique() if "ozet_pencere_gun" in s.columns else ["?"]
    print(
        f"  {a:8} {len(s):>9,}  ozet_pencere_gun={sorted(op)}  "
        f"aylar={sorted(pd.to_datetime(s['tarih']).dt.month.unique())}"
    )

print("\n" + "=" * 78)
print("1) YAPISAL OLGU: her blogun EGITIM parcasinda HEDEF TAKVIM AYLARI")
print("=" * 78)
for b in tm.BLOKLAR:
    dog = egitim[egitim["_blok"] == b.ad]
    parca = tm.kokenleri_ayikla(genis, b.ad)
    hedef_ay = sorted(pd.to_datetime(dog["tarih"]).dt.month.unique())
    pay = pd.to_datetime(parca["tarih"]).dt.month
    print(f"\n{b.ad}: dogrulama {len(dog):,}  hedef aylar {hedef_ay}  egitim {len(parca):,}")
    tot = 0
    for m in hedef_ay:
        alt = parca[pay.to_numpy() == m]
        kk = alt["_blok"].value_counts().to_dict() if len(alt) else {}
        tot += len(alt)
        print(f"   ay {m:>2}: {len(alt):>8,}   koken dagilimi {kk}")
    print(f"   HEDEF AY TOPLAM {tot:,}  = %{100 * tot / len(parca):.1f}")

print("\nTEST tarafi:")
te_ay = sorted(pd.to_datetime(test["tarih"]).dt.month.unique())
uretim = genis  # uretimde hicbir koken atilmaz
pay = pd.to_datetime(uretim["tarih"]).dt.month
print(f"  test hedef aylar {te_ay}  uretim egitimi {len(uretim):,}")
tot = 0
for m in te_ay:
    alt = uretim[pay.to_numpy() == m]
    tot += len(alt)
    print(f"   ay {m:>2}: {len(alt):>8,}   koken {alt['_blok'].value_counts().to_dict()}")
print(f"   HEDEF AY TOPLAM {tot:,}  = %{100 * tot / len(uretim):.1f}")
