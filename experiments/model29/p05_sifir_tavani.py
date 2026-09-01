"""p05: URETIM hattinda SIFIR ele alisinin TAVANI.

Sorular:
  1) Kahin (oracle): gercek sifir satirlarinin tahminini 0 yapsak kazanc ne?
     -> hicbir sifir-siniflandiricisinin asamayacagi TAVAN.
  2) Elimizdeki siniflandirici (p03'un P0'i, guz25+kis26'da egitilmis) bu
     tavana ne kadar yaklasiyor? AUC, esik taramasi (KAHIN esik secimi --
     yani gercekte ulasilamaz, yalnizca TAVAN).
  3) Yumusak buzme p = (1-P0)^g * pb: g'nin KAHIN degeriyle en iyi ne olur?
Hepsi TAVAN olcumu; yaz25 hedefi kullanildigi icin URETIME TASINAMAZ.
Amac, kalan basligin buyuklugunu bilmek.
"""

import json
import os

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = os.path.join(KOK, "data/interim/deney")
BURA = os.path.dirname(os.path.abspath(__file__))
ARA = os.environ.get(
    "ARA",
    r"C:/Users/Cem/AppData/Local/Temp/claude/"
    r"c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/"
    r"e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad",
)
HEDEF_SOGUK = 0.222

e = pd.read_parquet(os.path.join(DN, "egitim.parquet"), columns=["tuketim", "soguk_mu", "_blok"])
blk = e[e._blok == "yaz25"]
sic, sog = blk[blk.soguk_mu == 0], blk[blk.soguk_mu == 1]
idx = np.concatenate([sic.index.values, sog.index.values])
bf = e.loc[idx]
y = bf.tuketim.to_numpy(dtype=np.float64)
yv = np.log1p(y)
sgm = bf.soguk_mu.to_numpy(dtype=np.float64)
pb = np.load(os.path.join(ARA, "p03_pb_yaz25.npy"))
P0 = np.load(os.path.join(ARA, "p03_P0_yaz25.npy"))
ww = np.where(sgm == 1, HEDEF_SOGUK / sgm.mean(), (1 - HEDEF_SOGUK) / (1 - sgm.mean()))
ww = ww / ww.mean()
s = sgm == 1
z = y == 0


def olc(p):
    r = np.asarray(p) - yv
    return {
        "duz": float(np.sqrt(np.mean(r * r))),
        "test_agirlikli": float(np.sqrt(np.mean(ww * r * r))),
        "soguk": float(np.sqrt(np.mean(r[s] ** 2))),
        "sicak": float(np.sqrt(np.mean(r[~s] ** 2))),
    }


T = olc(pb)
R = {
    "aciklama": __doc__.strip(),
    "taban": T,
    "sifir_orani": float(z.mean()),
    "sifir_orani_soguk": float(z[s].mean()),
    "sifir_orani_sicak": float(z[~s].mean()),
}

# 1) kahin
pk = pb.copy()
pk[z] = 0.0
R["kahin_sifirlari_sifirla"] = olc(pk)
R["kahin_kazanc"] = {k: T[k] - v for k, v in R["kahin_sifirlari_sifirla"].items()}

# 2) siniflandirici kalitesi
o = np.argsort(P0)
rank = np.empty(len(P0))
rank[o] = np.arange(len(P0))
n1, n0 = z.sum(), (~z).sum()
auc = float((rank[z].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))
R["P0_auc"] = auc
R["P0_ort"] = float(P0.mean())
R["esik_taramasi_KAHIN"] = []
for th in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
    m = P0 > th
    if m.sum() == 0:
        continue
    q = np.where(m, 0.0, pb)
    R["esik_taramasi_KAHIN"].append(
        {
            "esik": th,
            "n": int(m.sum()),
            "kesinlik": float(z[m].mean()),
            "anma": float(m[z].mean()),
            **olc(q),
            "kazanc_agirlikli": T["test_agirlikli"] - olc(q)["test_agirlikli"],
        }
    )

# 3) yumusak buzme (1-P0)^g
R["gama_taramasi_KAHIN"] = []
for g in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
    q = ((1 - P0) ** g) * pb
    R["gama_taramasi_KAHIN"].append(
        {"gama": g, **olc(q), "kazanc_agirlikli": T["test_agirlikli"] - olc(q)["test_agirlikli"]}
    )

# 4) sifir satirlarin toplam hata payi
r = pb - yv
R["sifir_satir_hata_payi"] = float((r[z] ** 2).sum() / (r * r).sum())
R["sifir_satir_ort_tahmin"] = float(pb[z].mean())

json.dump(
    R,
    open(os.path.join(BURA, "p05_sifir_tavani.json"), "w", encoding="utf-8"),
    indent=1,
    ensure_ascii=False,
)
print(json.dumps(R, indent=1, ensure_ascii=False))
