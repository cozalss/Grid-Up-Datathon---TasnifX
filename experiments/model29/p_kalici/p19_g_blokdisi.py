"""p19-G: p19-F'deki dort 3/1/1 yapisi icin DURUST BLOK-DISI SECIM.

p19-F'de C3'un 3/3 pozitif cikmasi, dort yapi arasindan TAM VERIYLE secildi.
Burada her hedef blok icin secim YALNIZ diger iki bloktan yapilir, sonra
hedefteki sonuc RAPORLANIR (ne cikarsa).
"""

import json, os, sys
import numpy as np, pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
from p11_agirlik import PG_HAKIM, agirlik, kova, onyukleme_w, rmsle, test_dagilimi, wrmsle
from p11_ortak import BLOKLAR, DN, egitim, kirp, sicak_rmsle, toplam

W = (0.6, 0.2, 0.2)
d8 = lambda b, t: os.path.join(BURA, f"p19_{b}_{t}_hp_derin8.npy")
hb = lambda b, t: os.path.join(BURA, f"p11b_{b}_{t}_huber.npy")

T = pd.read_parquet(os.path.join(DN, "test.parquet"))
q = test_dagilimi(T[T.soguk_mu == 1])
del T
E = egitim()
MW, KZ, TS = {}, {}, {}
for b in BLOKLAR:
    blk = E[E._blok == b]
    sog = blk[blk.soguk_mu == 1].reset_index(drop=True)
    y = np.log1p(sog.tuketim.to_numpy(dtype="float64").clip(0))
    w = agirlik(sog, q)
    m = kova(sog)[0] == PG_HAKIM
    sic = sicak_rmsle(b)
    z = np.load(os.path.join(DN, f"soguk_tahmin_{b}.npz"))
    ts = [
        t
        for t in sorted({int(k.split("_")[0]) for k in z.files})
        if os.path.exists(d8(b, t)) and os.path.exists(hb(b, t))
    ]
    TS[b] = ts
    o = lambda f: np.mean([f(t) for t in ts], axis=0)
    cat7 = o(lambda t: z[f"{t}_cat"].astype(np.float64))
    xgb = o(lambda t: z[f"{t}_xgb"].astype(np.float64))
    lgb = o(lambda t: z[f"{t}_lgbm"].astype(np.float64))
    cat8 = o(lambda t: np.load(d8(b, t)).astype(np.float64))
    hub = o(lambda t: np.load(hb(b, t)).astype(np.float64))
    Y = {
        "B_std": W[0] * cat7 + W[1] * xgb + W[2] * lgb,
        "C1_lgbm_huber": W[0] * cat7 + W[1] * xgb + W[2] * hub,
        "C2_cat_derin8": W[0] * cat8 + W[1] * xgb + W[2] * lgb,
        "C3_ikisi": W[0] * cat8 + W[1] * xgb + W[2] * hub,
    }
    MW[b] = {}
    KZ[b] = {}
    rB = y - kirp(Y["B_std"])
    for k, v in Y.items():
        r = y - kirp(v)
        MW[b][k] = float(np.sum(w * r * r) / w.sum())
        KZ[b][k] = dict(
            agr=round(wrmsle(rB, w) - wrmsle(r, w), 5),
            pg=round(rmsle(rB[m]) - rmsle(r[m]), 5),
            bil=round(toplam(wrmsle(rB, w), sic) - toplam(wrmsle(r, w), sic), 5),
            oy=onyukleme_w(sog.tanim.values, rB, r, w, 500)["pozitif_oran"],
        )

R = {"ortak_tohum": TS, "secim": {}}
for h in BLOKLAR:
    dis = [b for b in BLOKLAR if b != h]
    puan = {k: round(sum(MW[b][k] / MW[b]["B_std"] for b in dis), 5) for k in MW[h]}
    s = min(puan, key=puan.get)
    R["secim"][h] = dict(secilen=s, puan=puan, hedefte=(None if s == "B_std" else KZ[h][s]))
    print(f"hedef={h:6} secilen={s:14} puan={puan}")
    print(f"   -> hedefte: {KZ[h][s] if s != 'B_std' else 'TABAN secildi'}")
b = [R["secim"][h]["hedefte"]["bil"] for h in BLOKLAR if R["secim"][h]["hedefte"]]
R["durust_blokdisi_bilesim"] = dict(
    bloklar=b, ort=round(float(np.mean(b)), 5), pozitif=int(sum(1 for x in b if x > 0)), n=len(b)
)
print("\nDURUST BLOK-DISI:", R["durust_blokdisi_bilesim"])
json.dump(
    R,
    open(os.path.join(BURA, "p19_g_blokdisi.json"), "w", encoding="utf-8"),
    indent=1,
    ensure_ascii=False,
)
