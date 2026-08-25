# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""CURUTME 7: kis26 kazanci HANGI DOGRULAMA AYINDA? (kaydedilmis tahminlerden)

Iddianin mekanizmasi dogruysa kazanc, egitimde AYNI TAKVIM AYI satiri BULUNAN
aylarda (Subat, Mart -- sub25) yogunlasmali. Aralik ve Ocak'ta egitimde o
takvim ayindan TEK SATIR yok; kapsama hipotezi oralarda kazanc ONGORMEZ.
Ayrica ESKI (duz) olcutle de skorlanir -- 24 Agustos'ta ayni mekanizmayi
0/3 REDDEDEN olcut buydu.
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
import deney_ileri as di
import olcut as ol
import tuketim_model as tm  # noqa: E402

ONB = KOK / "data" / "interim" / "aile_onbellek"
CIK = KOK / "data" / "interim" / "curut_mevsim_tahmin"
TOHUMLAR = (1000, 1001, 1002)
BLOK = sys.argv[1] if len(sys.argv) > 1 else "kis26"

egitim, test = d.cerceveleri_kur()
tm.kategorik_kodla(egitim, test)
gk = ol.guc_kenarlari(test)
te_s = test[test["soguk_mu"] != 1]
_, dogrulama, gercek, soguk = di.blok_parcalari(egitim, BLOK)
sicak = ~soguk
dg = dogrulama[sicak]
y = gercek[sicak]
wt, _ = ol.test_agirliklari(dg, te_s, gk)
g = np.log1p(np.clip(y, 0, None))
ay = pd.to_datetime(dg["tarih"]).dt.month.to_numpy()

r0 = {t: np.load(ONB / f"{BLOK}_{t}_cat_uretim.npy").astype("float64") for t in TOHUMLAR}
r1 = {t: np.load(CIK / f"{BLOK}_{t}_mevsim45.npy").astype("float64") for t in TOHUMLAR}


def wrmse(e, w):
    return float(np.sqrt((w * e * e).sum() / w.sum()))


print(f"{BLOK}: DOGRULAMA AYI BAZINDA (agirlikli RMSLE, eslenik 3 tohum)")
print(
    f"  {'ay':>4}{'satir':>9}{'agr.pay':>9}{'egitimde ayni ay':>18}{'fark':>10}{'SH':>9}{'t':>7}{'tohum':>7}"
)
EGIT_VAR = {
    12: 0,
    1: 0,
    2: 58267,
    3: 65206,
    4: 64048,
    5: 67262,
    6: 67976,
    7: 75643,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
}
for m in sorted(set(ay)):
    sel = ay == m
    f = np.array(
        [
            wrmse(g[sel] - r0[t][sel], wt[sel]) - wrmse(g[sel] - r1[t][sel], wt[sel])
            for t in TOHUMLAR
        ]
    )
    sh = f.std(ddof=1) / np.sqrt(len(f))
    print(
        f"  {m:>4}{int(sel.sum()):>9,}{100 * wt[sel].sum() / wt.sum():>8.1f}%"
        f"{EGIT_VAR.get(m, 0):>18,}{f.mean():>+10.5f}{sh:>9.5f}{f.mean() / sh:>+7.2f}"
        f"{int((f > 0).sum()):>5}/3"
    )

print("\nOLCUT KARSILASTIRMASI (tum blok):")
for ad, w in (
    ("TESTE AGIRLIKLANDIRILMIS (docs/40)", wt),
    ("DUZ (24 Agustos olcutu)", np.ones_like(wt)),
):
    f = np.array([wrmse(g - r0[t], w) - wrmse(g - r1[t], w) for t in TOHUMLAR])
    sh = f.std(ddof=1) / np.sqrt(len(f))
    print(
        f"  {ad:38}{f.mean():>+10.5f}  SH {sh:.5f}  t={f.mean() / sh:+.2f}  {int((f > 0).sum())}/3"
    )
