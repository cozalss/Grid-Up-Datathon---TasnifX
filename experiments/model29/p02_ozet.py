"""p02 NIHAI OZET: yaz25 blogunda mevcut boru hatti vs sifirdan temiz taban."""

import json
import os

import numpy as np
import pandas as pd

K = "c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = f"{K}/data/interim/deney"
AO = f"{K}/data/interim/aile_onbellek"
SC = (
    "C:/Users/Cem/AppData/Local/Temp/claude/"
    "c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/"
    "e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad"
)

e = pd.read_parquet(
    f"{DN}/egitim.parquet", columns=["tanim", "tarih", "tuketim", "soguk_mu", "_blok"]
)
blk = e[e._blok == "yaz25"]
sic, sog = blk[blk.soguk_mu == 0], blk[blk.soguk_mu == 1]
P = [
    np.load(f"{AO}/yaz25_{t}_{a}_uretim.npy").astype(np.float64)
    for t in (1000, 1001, 1002)
    for a in ("cat", "xgb", "lgbm")
    if os.path.exists(f"{AO}/yaz25_{t}_{a}_uretim.npy")
]
z = np.load(f"{DN}/soguk_tahmin_yaz25.npz")
idx = np.concatenate([sic.index.values, sog.index.values])
bf = e.loc[idx].copy()
bf["tanim"] = bf.tanim.astype(str)
bf["mevcut"] = np.concatenate([np.mean(P, axis=0), np.mean([z[q] for q in z.files], axis=0)])

m = bf
for et, dosya, kol in (
    ("p02_v1_karisim", "p02_yaz25.parquet", "p_kar"),
    ("p02_v1_ofsetli", "p02_yaz25.parquet", "p_ofsetli"),
    ("p02_v2", "p02_yaz25_v2.parquet", "p2"),
    ("p02_v3", "p02_yaz25_v3.parquet", "p3"),
    ("p02_v4", "p02_yaz25_v4.parquet", "p4"),
):
    d = pd.read_parquet(f"{SC}/{dosya}", columns=["tanim", "tarih", "y", kol])
    d["tanim"] = d.tanim.astype(str)
    d = d.rename(columns={kol: et}).drop_duplicates(["tanim", "tarih"])
    if "y" in m.columns:
        d = d.drop(columns=["y"])
    n0 = len(m)
    m = m.merge(d, on=["tanim", "tarih"], how="left")
    assert len(m) == n0 and m[et].notna().all(), et

assert np.allclose(m.y.to_numpy(), np.log1p(m.tuketim.to_numpy()))
y = m.y.to_numpy()
sg = m.soguk_mu.to_numpy().astype(float)
HED = 0.2216
R = {}
print(f"{'model':20s} {'duz':>9s} {'sicak':>9s} {'soguk':>9s} {'test-agirlikli':>14s}")
for c in ["mevcut", "p02_v1_karisim", "p02_v1_ofsetli", "p02_v2", "p02_v3", "p02_v4"]:
    r = m[c].to_numpy() - y
    a = float(np.sqrt((r * r).mean()))
    b = float(np.sqrt((r[sg == 0] ** 2).mean()))
    d2 = float(np.sqrt((r[sg == 1] ** 2).mean()))
    w = float(np.sqrt((1 - HED) * b**2 + HED * d2**2))
    R[c] = dict(duz=a, sicak=b, soguk=d2, test_agirlikli=w)
    print(f"{c:20s} {a:9.5f} {b:9.5f} {d2:9.5f} {w:14.5f}")
# en iyi olasi harman (kopya cekerek) -- ust sinir
for c in ["p02_v1_karisim", "p02_v4"]:
    best = min(
        ((np.sqrt(((w2 * m[c] + (1 - w2) * m.mevcut - y) ** 2).mean())), w2)
        for w2 in np.arange(0, 1.01, 0.05)
    )
    print(f"en iyi harman mevcut+{c}: RMSLE={best[0]:.5f} agirlik={best[1]:.2f}")
    R[f"harman_{c}"] = dict(rmsle=float(best[0]), agirlik=float(best[1]))
R["_not"] = (
    "yaz25=2025-04-01..07-31, 274929 satir, 2891 trafo, %7.5 soguk. "
    "test-agirlikli: soguk orani test blogundaki %22.16'ya tasinmis hali."
)
json.dump(R, open(f"{K}/experiments/model29/p02_temiz_taban.json", "w"), indent=1)
print("\nyazildi: experiments/model29/p02_temiz_taban.json")
