# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""CURUTME 4: olcut ESS + mevsim agirliginin bloklar arasi YAPISAL yordayicilari."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))
import deney as d
import deney_ileri as di
import olcut as ol
import tuketim_model as tm  # noqa: E402


def dairesel(tarih, hedef_doy):
    doy = pd.to_datetime(tarih).dt.dayofyear.to_numpy()
    h = np.unique(np.asarray(hedef_doy))
    f = np.abs(doy[:, None] - h[None, :])
    return np.minimum(f, 365 - f).min(axis=1).astype("float64")


egitim, test = d.cerceveleri_kur()
tm.kategorik_kodla(egitim, test)
gk = ol.guc_kenarlari(test)
te_s = test[test["soguk_mu"] != 1]
ek = d._ek_kokenler_kur(False)
ek = ek[ek["_blok"].isin([a for a, _, _ in tm.EK_KOKENLER])]
ortak = [k for k in egitim.columns if k in ek.columns]
genis = pd.concat([egitim[ortak], ek[ortak]], ignore_index=True)
TAU = 45.0

print(
    f"{'blok':>10} {'egitim':>10} {'u_ort':>7} {'u_med':>7} {'<=45g pay':>10} "
    f"{'ESS_agirlik':>12} {'olcut_ESS':>10} {'mevsim kazanc':>14}"
)
KAZANC = {"yaz25": +0.00264, "guz25": +0.00407, "kis26": +0.04774}
for b in tm.BLOKLAR:
    _, dogrulama, gercek, soguk = di.blok_parcalari(egitim, b.ad)
    sicak = ~soguk
    dg = dogrulama[sicak]
    parca = tm.kokenleri_ayikla(genis, b.ad)
    hd = pd.to_datetime(dg["tarih"])
    u = dairesel(parca["tarih"], hd.dt.dayofyear.unique())
    w = np.exp(-u / TAU)
    w = w / w.mean()
    ess = w.sum() ** 2 / (w**2).sum() / len(w)
    wt, tani = ol.test_agirliklari(dg, te_s, gk)
    oess = tani.get("ess_pay", tani.get("ess", np.nan))
    print(
        f"{b.ad:>10} {len(parca):>10,} {u.mean():>7.1f} {np.median(u):>7.0f} "
        f"{100 * (u <= 45).mean():>9.1f}% {100 * ess:>11.1f}% {oess:>10} {KAZANC[b.ad]:>+14.5f}"
    )

hd = pd.to_datetime(test["tarih"])
u = dairesel(genis["tarih"], hd.dt.dayofyear.unique())
w = np.exp(-u / TAU)
w = w / w.mean()
ess = w.sum() ** 2 / (w**2).sum() / len(w)
print(
    f"{'URETIM':>10} {len(genis):>10,} {u.mean():>7.1f} {np.median(u):>7.0f} "
    f"{100 * (u <= 45).mean():>9.1f}% {100 * ess:>11.1f}% {'-':>10} {'?':>14}"
)
print("\nolcut tani anahtarlari ornegi:", tani)
