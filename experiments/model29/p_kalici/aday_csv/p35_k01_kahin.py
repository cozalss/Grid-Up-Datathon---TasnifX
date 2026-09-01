# -*- coding: utf-8 -*-
"""KAHIN TAVAN HARITASI -- cep bazli rho_max (oracle) + sert-bayrak gercek rho.
Olcek-serbest cerceve:
   rho_max(S)/||r|| = sqrt( MSE_payi(S) )
   LB'de gereken oran = 0.12298/1.000686 = 0.12290  ->  MSE payi >= %1.511
"""
import os, sys, json
import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
sys.path.insert(0, os.path.join(KOK, "experiments/model29/p_kalici"))
import p27_ortak as P

CIK = r"C:/Users/Cem/AppData/Local/Temp/claude/c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/d8509f77-6f9b-4e1d-b980-62e299ed4fc5/scratchpad"

MSE_LB = 1.0013719
NORM_LB = np.sqrt(MSE_LB)          # 1.000686
RHO_GEREK = 0.12298                # T = 0.99310
RHO_KABUL = 0.11436                # T = 0.99413
ORAN_GEREK = RHO_GEREK / NORM_LB   # 0.122896
ORAN_KABUL = RHO_KABUL / NORM_LB


def kur(bad):
    d = P.blok(bad, soguk_harman="cat", son_islem=True)
    d = d.reset_index(drop=True)
    d["w"] = P.agirlik(d)
    return d


def cep_olc(d, m, ad):
    w = d["w"].values; r = d["r"].values
    W = w.sum(); TOP = float(np.sum(w * r * r))
    m = np.asarray(m, dtype=bool)
    n = int(m.sum())
    if n == 0:
        return None
    ws = w[m]; rs = r[m]
    A = float(ws.sum())
    S_mse = float(np.sum(ws * rs * rs))
    mse_payi = S_mse / TOP
    rho_max = np.sqrt(S_mse / W)
    oran_max = np.sqrt(mse_payi)
    rho_bayrak = float(np.sum(ws * rs)) / np.sqrt(A * W)
    oran_bayrak = rho_bayrak / np.sqrt(TOP / W)
    rbar = float(np.sum(ws * rs) / A)
    tg = d["tanim"].values[m]
    dfk = pd.DataFrame({"g": tg, "v": ws * (rs - rbar)})
    gg = dfk.groupby("g", observed=True)["v"].sum().values
    var = float(np.sum(gg * gg)) / (A * W)
    se = np.sqrt(max(var, 0.0))
    return dict(ad=ad, n=n, satir_payi=n / len(d), agirlik_payi=A / W,
                mse_payi=mse_payi, rho_max=rho_max, oran_max=oran_max,
                rho_bayrak=rho_bayrak, oran_bayrak=oran_bayrak, se_bayrak=se)


def bolum_olc(d, key, ad):
    w = d["w"].values; r = d["r"].values
    W = w.sum(); TOP = float(np.sum(w * r * r))
    k = pd.Series(np.asarray(key)).astype("string").fillna("NA").values
    df = pd.DataFrame({"k": k, "w": w, "wr": w * r})
    g = df.groupby("k", observed=True).agg(sw=("w", "sum"), swr=("wr", "sum"))
    val = float(np.sum(g.swr.values ** 2 / g.sw.values) / W)
    rho = np.sqrt(max(val, 0.0))
    return dict(ad=ad, hucre=int(len(g)), rho_grup=rho, oran_grup=rho / np.sqrt(TOP / W))


def cepleri_kur(d):
    r = d["r"].values
    tuk = d["tuketim"].values.astype(float)
    sog = d["soguk_mu"].values.astype(int)
    w = d["w"].values
    ceps = []
    ceps.append(("TUM SATIRLAR", np.ones(len(d), bool)))
    ceps.append(("gercek sifir (y==0)", tuk == 0))
    tp = pd.DataFrame({"t": d["tanim"].values, "tuk": tuk}).groupby("t", observed=True)["tuk"].max()
    olu = set(tp.index[tp.values == 0])
    m_olu = pd.Series(d["tanim"].values).isin(olu).values
    ceps.append(("olu trafo (4 ay sifir) satirlari", m_olu))
    ceps.append(("sifir AMA olu-olmayan trafo", (tuk == 0) & (~m_olu)))
    ceps.append(("soguk kohort", sog == 1))
    ceps.append(("sicak kohort", sog == 0))
    ceps.append(("soguk & sifir", (sog == 1) & (tuk == 0)))
    ceps.append(("soguk & pozitif", (sog == 1) & (tuk > 0)))
    ceps.append(("sicak & sifir", (sog == 0) & (tuk == 0)))
    ceps.append(("sicak & pozitif", (sog == 0) & (tuk > 0)))
    ceps.append(("y >= 100", tuk >= 100))
    ceps.append(("10 <= y < 100", (tuk >= 10) & (tuk < 100)))
    ceps.append(("0 < y < 10", (tuk > 0) & (tuk < 10)))
    ceps.append(("buyuk kacirma r > +2", r > 2))
    ceps.append(("buyuk kacirma r < -2", r < -2))
    ceps.append(("abs(r) > 2 (ikisi)", np.abs(r) > 2))
    ceps.append(("abs(r) <= 2 (govde)", np.abs(r) <= 2))
    tm = pd.DataFrame({"t": d["tanim"].values, "v": w * r * r}).groupby("t", observed=True)["v"].sum()
    tm = tm.sort_values(ascending=False)
    nt = len(tm)
    for q in (0.01, 0.05, 0.10):
        st = set(tm.index[: max(1, int(round(q * nt)))])
        ceps.append(("trafo ust yuzde " + str(int(q * 100)) + " (MSE)",
                     pd.Series(d["tanim"].values).isin(st).values))
    gq = pd.qcut(d["guc"].values.astype(float), 5, labels=False, duplicates="drop")
    for i in range(int(np.nanmax(gq)) + 1):
        ceps.append(("guc bandi " + str(i + 1) + "/5", gq == i))
    uf = d["ufuk_gun"].values.astype(float)
    for a, b in [(1, 15), (16, 45), (46, 75), (76, 105), (106, 122)]:
        ceps.append(("ufuk " + str(a) + "-" + str(b), (uf >= a) & (uf <= b)))
    for a in sorted(pd.unique(d["ay"].values)):
        ceps.append(("ay=" + str(a), d["ay"].values == a))
    ceps.append(("hafta sonu", d["hg"].values >= 5))
    return ceps


def blok_rapor(bad):
    d = kur(bad)
    w = d["w"].values; r = d["r"].values
    W = w.sum(); TOP = float(np.sum(w * r * r))
    out = {"blok": bad, "n": int(len(d)),
           "mse_w": TOP / W, "rmsle_w": float(np.sqrt(TOP / W)),
           "ort_artik_w": float(np.sum(w * r) / W)}
    sat = []
    for ad, m in cepleri_kur(d):
        s = cep_olc(d, m, ad)
        if s:
            sat.append(s)
    out["cepler"] = sat
    bol = []
    parcalar = [
        (d["ilce_key"].values, "BOLUNTU ilce"),
        (d["bolge"].values, "BOLUNTU bolge"),
        (d["ay"].values, "BOLUNTU ay"),
        (d["hg"].values, "BOLUNTU haftanin gunu"),
        (pd.qcut(d["guc"].values.astype(float), 10, labels=False, duplicates="drop"), "BOLUNTU guc desil"),
        (np.minimum(d["ufuk_gun"].values.astype(int) // 15, 8), "BOLUNTU ufuk 9 kova"),
        (d["tarih"].astype(str).values, "BOLUNTU gun (122)"),
        (d["soguk_mu"].values, "BOLUNTU rejim"),
        (d["tanim"].values, "BOLUNTU trafo kimlik"),
        (d["ilce_key"].astype(str).values + "|" + d["ay"].astype(str).values, "BOLUNTU ilce x ay"),
        (d["tanim"].astype(str).values + "|" + d["ay"].astype(str).values, "BOLUNTU trafo x ay"),
    ]
    for key, ad in parcalar:
        bol.append(bolum_olc(d, key, ad))
    out["boluntular"] = bol
    return out, d


if __name__ == "__main__":
    hepsi = {}
    for bad in ("yaz25", "guz25", "kis26"):
        rap, d = blok_rapor(bad)
        hepsi[bad] = rap
        print("=== %s: n=%d rmsle_w=%.4f ort_artik=%+.4f" %
              (bad, rap["n"], rap["rmsle_w"], rap["ort_artik_w"]))
        sys.stdout.flush()
    hepsi["_esik"] = dict(oran_gerek=ORAN_GEREK, oran_kabul=ORAN_KABUL,
                          mse_payi_gerek=ORAN_GEREK ** 2, mse_payi_kabul=ORAN_KABUL ** 2)
    with open(os.path.join(CIK, "k01_kahin.json"), "w", encoding="utf-8") as f:
        json.dump(hepsi, f, indent=1)
    print("YAZILDI k01_kahin.json")
    print("esik: oran>=%.5f  => MSE payi >= %.4f yuzde" % (ORAN_GEREK, 100 * ORAN_GEREK ** 2))
