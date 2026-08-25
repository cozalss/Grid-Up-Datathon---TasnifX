# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""CURUTME 6: mevsim agirligi URETIMDE hangi kolonlarin dolulugunu eziyor?

kis26 bu maliyeti GOREMEZ: docs/41 §6.1'e gore t_gy_* ve t_ay_sapma kis26
EGITIMINDE %0,0 dolu. Yani CV'de agirligin bu kolonlara verdigi zarar
olculemez; uretimde ise kolonlar %21,2 dolu ve TEST %52,6 istiyor.
"""

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
    h = np.unique(np.asarray(hedef_doy))
    f = np.abs(doy[:, None] - h[None, :])
    return np.minimum(f, 365 - f).min(axis=1).astype("float64")


egitim, test = d.cerceveleri_kur()
ek = d._ek_kokenler_kur(False)
ek = ek[ek["_blok"].isin([a for a, _, _ in tm.EK_KOKENLER])]
ortak = [k for k in egitim.columns if k in ek.columns]
genis = pd.concat([egitim[ortak], ek[ortak]], ignore_index=True)
TAU = 45.0
hd = pd.to_datetime(test["tarih"])
u = dairesel(genis["tarih"], hd.dt.dayofyear.unique())
w = np.exp(-u / TAU)
w = w / w.mean()

KOLONLAR = [
    k
    for k in (
        "t_ay_sapma",
        "t_gy_log_ort",
        "t_gy_sifir_orani",
        "t_gy_gun",
        "t_egim_cdd22",
        "t_son_kayit_yasi",
        "yas",
        "t_gun_sayisi",
        "ulusal_yil_once",
    )
    if k in genis.columns
]
print(f"{'kolon':>20}{'URETIM duz':>12}{'MEVSIM agr.':>13}{'kayip':>9}{'TEST':>9}")
for k in KOLONLAR:
    v = genis[k].to_numpy()
    dolu = np.isfinite(pd.to_numeric(pd.Series(v), errors="coerce").to_numpy())
    p_d = 100 * dolu.mean()
    p_a = 100 * (w * dolu).sum() / w.sum()
    tv = (
        pd.to_numeric(test[k], errors="coerce").to_numpy()
        if k in test.columns
        else np.array([np.nan])
    )
    p_t = 100 * np.isfinite(tv).mean()
    print(f"{k:>20}{p_d:>11.1f}%{p_a:>12.1f}%{p_a - p_d:>+8.1f}{p_t:>8.1f}%")

# 2026 satirlarinin agirlik payi
y26 = (pd.to_datetime(genis["tarih"]).dt.year == 2026).to_numpy()
print(
    f"\n2026 satirlari: duz pay %{100 * y26.mean():.1f}  ->  MEVSIM agirlikli %{100 * (w * y26).sum() / w.sum():.1f}"
    f"   (kat {((w * y26).sum() / w.sum()) / y26.mean():.2f})"
)
for ay in (1, 2, 3):
    m = y26 & (pd.to_datetime(genis["tarih"]).dt.month == ay).to_numpy()
    print(
        f"  2026-{ay:02d}: duz %{100 * m.mean():.2f}  agirlikli %{100 * (w * m).sum() / w.sum():.2f}"
        f"  kat {((w * m).sum() / w.sum()) / m.mean():.2f}"
    )
