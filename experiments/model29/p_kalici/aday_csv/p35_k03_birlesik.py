# -*- coding: utf-8 -*-
"""BIRLESIK TAVANLAR: cep birlesimlerinin kahin tavani + cok-bayrakli izdusum."""
import os, sys, json, itertools
import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
sys.path.insert(0, os.path.join(KOK, "experiments/model29/p_kalici"))
import p27_ortak as P
CIK = r"C:/Users/Cem/AppData/Local/Temp/claude/c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/d8509f77-6f9b-4e1d-b980-62e299ed4fc5/scratchpad"


def kur(bad):
    d = P.blok(bad, soguk_harman="cat", son_islem=True).reset_index(drop=True)
    d["w"] = P.agirlik(d)
    return d


def maskeler(d):
    tuk = d["tuketim"].values.astype(float)
    r = d["r"].values
    sog = d["soguk_mu"].values.astype(int)
    w = d["w"].values
    tp = pd.DataFrame({"t": d["tanim"].values, "v": tuk}).groupby("t", observed=True)["v"].max()
    olu = pd.Series(d["tanim"].values).isin(set(tp.index[tp.values == 0])).values
    tm = pd.DataFrame({"t": d["tanim"].values, "v": w * r * r}).groupby("t", observed=True)["v"].sum()
    tm = tm.sort_values(ascending=False)
    ust1 = pd.Series(d["tanim"].values).isin(set(tm.index[: max(1, int(0.01 * len(tm)))])).values
    return {
        "sifir": tuk == 0,
        "olu_trafo": olu,
        "r<-2": r < -2,
        "r>+2": r > 2,
        "trafo_ust1": ust1,
        "soguk": sog == 1,
        "soguk_sifir": (sog == 1) & (tuk == 0),
        "sicak_sifir": (sog == 0) & (tuk == 0),
        "y>=100": tuk >= 100,
    }


def kahin_oran(d, m):
    w = d["w"].values; r = d["r"].values
    W = w.sum(); TOP = float(np.sum(w * r * r))
    return float(np.sqrt(np.sum(w[m] * r[m] ** 2) / TOP))


def cok_bayrak_izdusum(d, listem):
    """span{1_S1..1_Sk} uzerine agirlikli izdusum -> oran."""
    w = d["w"].values; r = d["r"].values
    W = w.sum(); TOP = float(np.sum(w * r * r))
    A = np.column_stack([m.astype(np.float64) for m in listem])
    G = A.T @ (w[:, None] * A)
    b = A.T @ (w * r)
    c = np.linalg.lstsq(G, b, rcond=None)[0]
    val = float(c @ b) / W
    return float(np.sqrt(max(val, 0.0) / (TOP / W)))


if __name__ == "__main__":
    out = {}
    for bad in ("yaz25", "guz25", "kis26"):
        d = kur(bad)
        M = maskeler(d)
        S = {"tekil": {k: kahin_oran(d, v) for k, v in M.items()}}
        birlesimler = {
            "sifir + r<-2": ["sifir", "r<-2"],
            "sifir + r<-2 + r>+2": ["sifir", "r<-2", "r>+2"],
            "sifir + trafo_ust1": ["sifir", "trafo_ust1"],
            "sifir + r<-2 + trafo_ust1": ["sifir", "r<-2", "trafo_ust1"],
            "sifir + abs(r)>2 + trafo_ust1": ["sifir", "r<-2", "r>+2", "trafo_ust1"],
            "olu + sifir": ["olu_trafo", "sifir"],
            "soguk + sifir": ["soguk", "sifir"],
            "EN IYI 3 (sifir,r<-2,trafo_ust1)": ["sifir", "r<-2", "trafo_ust1"],
        }
        S["birlesim_kahin"] = {}
        S["birlesim_cok_bayrak"] = {}
        for ad, ks in birlesimler.items():
            u = np.zeros(len(d), bool)
            for k in ks:
                u |= M[k]
            S["birlesim_kahin"][ad] = kahin_oran(d, u)
            S["birlesim_cok_bayrak"][ad] = cok_bayrak_izdusum(d, [M[k] for k in ks])
        # tekil cok-bayrak (tek bayrak izdusumu = k01 oran_bayrak)
        S["tekil_bayrak"] = {k: cok_bayrak_izdusum(d, [v]) for k, v in M.items()}
        # ORTOGONALLIK: 9 bayragin HEPSI birlikte
        S["dokuz_bayrak_hepsi"] = cok_bayrak_izdusum(d, list(M.values()))
        out[bad] = S
        print("=== " + bad + " tamam"); sys.stdout.flush()
    with open(os.path.join(CIK, "k03_birlesik.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    for bad in out:
        print("\n### " + bad)
        print(" TEKIL   kahin / bayrak")
        for k in out[bad]["tekil"]:
            print("   %-16s %7.4f  %+7.4f" % (k, out[bad]["tekil"][k], out[bad]["tekil_bayrak"][k]))
        print(" BIRLESIM  kahin / cok-bayrak-izdusum")
        for k in out[bad]["birlesim_kahin"]:
            print("   %-34s %7.4f  %7.4f" % (k, out[bad]["birlesim_kahin"][k],
                                             out[bad]["birlesim_cok_bayrak"][k]))
        print("   9 bayragin hepsi (izdusum): %.4f" % out[bad]["dokuz_bayrak_hepsi"])
