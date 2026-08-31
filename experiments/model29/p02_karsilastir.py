"""p02: yaz25'te MEVCUT boru hatti vs SIFIRDAN taban -- ayni satirlarda."""
import json
import os

import numpy as np
import pandas as pd

K = "c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = f"{K}/data/interim/deney"
AO = f"{K}/data/interim/aile_onbellek"
SC = ("C:/Users/Cem/AppData/Local/Temp/claude/"
      "c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/"
      "e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad")

e = pd.read_parquet(f"{DN}/egitim.parquet",
                    columns=["tanim", "tarih", "tuketim", "soguk_mu", "_blok"])
blk = e[e._blok == "yaz25"]
sic, sog = blk[blk.soguk_mu == 0], blk[blk.soguk_mu == 1]
P = [np.load(f"{AO}/yaz25_{t}_{a}_uretim.npy").astype(np.float64)
     for t in (1000, 1001, 1002) for a in ("cat", "xgb", "lgbm")
     if os.path.exists(f"{AO}/yaz25_{t}_{a}_uretim.npy")]
print("sicak uye sayisi", len(P), "uzunluk", [len(p) for p in P][:3], "sicak satir", len(sic))
z = np.load(f"{DN}/soguk_tahmin_yaz25.npz")
print("soguk uye", z.files, "uzunluk", len(z[z.files[0]]), "soguk satir", len(sog))
idx = np.concatenate([sic.index.values, sog.index.values])
pb = np.concatenate([np.mean(P, axis=0), np.mean([z[q] for q in z.files], axis=0)])

bf = e.loc[idx].copy()
bf["p_mevcut"] = pb
bf["tanim"] = bf.tanim.astype(str)

# benim tahminlerim
mine = pd.read_parquet(f"{SC}/p02_yaz25.parquet",
                       columns=["tanim", "tarih", "y", "soguk", "p_ofsetli", "p_ham", "p_kar"])
mine["tanim"] = mine.tanim.astype(str)
m = bf.merge(mine, on=["tanim", "tarih"], how="inner", validate="1:1")
assert len(m) == len(bf) == len(mine), (len(m), len(bf), len(mine))
assert np.allclose(m.y.to_numpy(), np.log1p(m.tuketim.to_numpy()))
assert (m.soguk.to_numpy() == m.soguk_mu.to_numpy()).mean() > 0.99
print("ESLESME TAM:", len(m), "satir")

y = m.y.to_numpy()
sg = m.soguk_mu.to_numpy().astype(float)
HED = 0.2216  # test blogundaki soguk orani


def rap(ad, p):
    r = p - y
    duz = float(np.sqrt((r * r).mean()))
    sc = float(np.sqrt((r[sg == 0] ** 2).mean()))
    so = float(np.sqrt((r[sg == 1] ** 2).mean()))
    agr = float(np.sqrt(HED * so ** 2 + (1 - HED) * sc ** 2))
    print(f"{ad:22s} duz={duz:.6f}  sicak={sc:.6f}  soguk={so:.6f}  test-agirlikli={agr:.6f}")
    return dict(duz=duz, sicak=sc, soguk=so, test_agirlikli=agr)


R = {}
R["mevcut_boru_hatti"] = rap("MEVCUT boru hatti", m.p_mevcut.to_numpy())
for c in ("p_ofsetli", "p_ham", "p_kar"):
    R[c] = rap("p02 " + c, m[c].to_numpy())
# mevcut + benim ortalamasi
for w in (0.25, 0.5, 0.75):
    R[f"harman_{w}"] = rap(f"harman {w:.2f}*p02+{1-w:.2f}*mevcut",
                           w * m.p_kar.to_numpy() + (1 - w) * m.p_mevcut.to_numpy())
json.dump(R, open(f"{K}/experiments/model29/p02_karsilastirma.json", "w"), indent=1)
