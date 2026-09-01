# -*- coding: utf-8 -*-
"""ELENME SINIRI: hangi cep TURLERI kahin tavaniyla dogrudan elenir?
   Esik: MSE payi < %1.5103  <=>  oran_max < 0.12290."""
import os, sys, json
import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
sys.path.insert(0, os.path.join(KOK, "experiments/model29/p_kalici"))
import p27_ortak as P
CIK = r"C:/Users/Cem/AppData/Local/Temp/claude/c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/d8509f77-6f9b-4e1d-b980-62e299ed4fc5/scratchpad"
ESIK = 0.12298 / np.sqrt(1.0013719)

out = {}
for bad in ("yaz25",):
    d = P.blok(bad, soguk_harman="cat", son_islem=True).reset_index(drop=True)
    d["w"] = P.agirlik(d)
    w = d["w"].values; r = d["r"].values
    TOP = float(np.sum(w * r * r))
    v = w * r * r
    print("BLOK", bad, " esik MSE payi >= %.4f%%" % (100 * ESIK ** 2))
    for ad, key in [("ilce", d.ilce_key.astype(str).values),
                    ("gun", d.tarih.astype(str).values),
                    ("ay", d.ay.values),
                    ("trafo", d.tanim.astype(str).values),
                    ("bolge", d.bolge.astype(str).values),
                    ("haftanin gunu", d.hg.values)]:
        g = pd.DataFrame({"k": key, "v": v}).groupby("k", observed=True)["v"].sum() / TOP
        g = g.sort_values(ascending=False)
        gecen = int((np.sqrt(g.values) >= ESIK).sum())
        print("  %-14s hucre=%5d  en buyuk pay=%7.4f%% (oran %.4f)  medyan pay=%.5f%%  "
              "tek-hucre ESIGI GECEN: %d/%d" % (
                  ad, len(g), 100 * g.values[0], np.sqrt(g.values[0]),
                  100 * float(np.median(g.values)), gecen, len(g)))
        out[ad] = dict(hucre=int(len(g)), en_buyuk_pay=float(g.values[0]),
                       en_buyuk_oran=float(np.sqrt(g.values[0])),
                       gecen=gecen, toplam=int(len(g)),
                       ilk5=[[str(i), float(x)] for i, x in zip(g.index[:5], g.values[:5])])
    # kucuk y-araliklari
    tuk = d.tuketim.values.astype(float)
    for ad, m in [("tuketim 0<y<1", (tuk > 0) & (tuk < 1)),
                  ("tuketim 1<=y<5", (tuk >= 1) & (tuk < 5)),
                  ("tuketim 5<=y<10", (tuk >= 5) & (tuk < 10)),
                  ("tuketim >=10000", tuk >= 10000),
                  ("hafta ici tek gun (ilk)", d.tarih.astype(str).values == d.tarih.astype(str).values[0])]:
        p = float(np.sum(w[m] * r[m] ** 2) / TOP)
        print("  %-24s pay=%7.4f%%  oran=%.4f  %s" % (
            ad, 100 * p, np.sqrt(p), "GECER" if np.sqrt(p) >= ESIK else "ELENDI"))
        out[ad] = dict(pay=p, oran=float(np.sqrt(p)), gecer=bool(np.sqrt(p) >= ESIK))
with open(os.path.join(CIK, "k06_elenen.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
