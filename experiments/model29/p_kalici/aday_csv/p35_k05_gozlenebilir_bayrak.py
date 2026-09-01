# -*- coding: utf-8 -*-
"""GOZLENEBILIR BAYRAK rho'lari: uyelik YALNIZCA test'te de bulunan oznitelklerden.
   Model egitimi YOK, esik secimi hedefe bakmadan (sabit izgara) -> sizinti yok;
   uc blokta ayni izgara, isaret kararliligi okunur."""
import os, sys, json, gc
import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
sys.path.insert(0, os.path.join(KOK, "experiments/model29/p_kalici"))
import p27_ortak as P
CIK = r"C:/Users/Cem/AppData/Local/Temp/claude/c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/d8509f77-6f9b-4e1d-b980-62e299ed4fc5/scratchpad"


def rho_olc(d, u_ham, merkezle=False):
    w = d["w"].values; r = d["r"].values
    W = w.sum()
    u = np.asarray(u_ham, dtype=np.float64).copy()
    if merkezle:
        u -= np.sum(w * u) / W
    nrm = np.sqrt(np.sum(w * u * u) / W)
    if nrm <= 0:
        return None
    u = u / nrm
    rho = float(np.sum(w * r * u) / W)
    v = w * (r * u - rho)
    gg = pd.DataFrame({"g": d["tanim"].values, "v": v}).groupby("g", observed=True)["v"].sum().values
    se = float(np.sqrt(np.sum(gg * gg)) / W)
    rms = float(np.sqrt(np.sum(w * r * r) / W))
    return dict(rho=rho, se=se, oran=rho / rms, t=rho / se if se > 0 else 0.0)


def yonler(d):
    Y = []
    sf = d.t_sifir_orani.fillna(0).values.astype(float)
    ol = d.t_olu_mu.fillna(0).values.astype(float) if "t_olu_mu" in d.columns else None
    kq = d.t_kuyruk_sifir.fillna(0).values.astype(float)
    sg = d.soguk_mu.values.astype(float)
    p = d["p"].values
    Y.append(("SABIT", np.ones(len(d))))
    if ol is not None:
        Y.append(("t_olu_mu==1", ol))
    for e in (0.5, 0.8, 0.9, 0.99):
        Y.append(("t_sifir_orani>%.2f" % e, (sf > e).astype(float)))
    Y.append(("t_sifir_orani (surekli)", sf))
    for e in (7, 30, 60):
        Y.append(("t_kuyruk_sifir>%d" % e, (kq > e).astype(float)))
    Y.append(("sifir-adayi(sf>0.9) & SICAK", (sf > 0.9) * (1 - sg)))
    Y.append(("sifir-adayi(sf>0.9) & SOGUK", (sf > 0.9) * sg))
    Y.append(("SOGUK gostergesi", sg))
    Y.append(("p<1 (model dusuk)", (p < 1).astype(float)))
    Y.append(("p<0.5", (p < 0.5).astype(float)))
    Y.append(("p>6 (model yuksek)", (p > 6).astype(float)))
    Y.append(("p (surekli)", p))
    if "t_log_son30" in d.columns:
        s30 = d.t_log_son30.fillna(-1).values.astype(float)
        Y.append(("t_log_son30<0.1", (s30 < 0.1).astype(float)))
    if "t_son_kayit_yasi" in d.columns:
        ya = d.t_son_kayit_yasi.fillna(-1).values.astype(float)
        Y.append(("t_son_kayit_yasi>60", (ya > 60).astype(float)))
    Y.append(("ufuk>90", (d.ufuk_gun.values.astype(float) > 90).astype(float)))
    Y.append(("hafta sonu", (d.hg.values >= 5).astype(float)))
    return Y


if __name__ == "__main__":
    out = {}
    for bad in ("yaz25", "guz25", "kis26"):
        d = P.blok(bad, soguk_harman="cat", son_islem=True).reset_index(drop=True)
        d["w"] = P.agirlik(d)
        S = {}
        for ad, u in yonler(d):
            s = rho_olc(d, u)
            if s:
                # bu bayragin KAHIN tavani (bayrakli kumedeki artik normu)
                m = np.asarray(u) > 0
                w = d["w"].values; r = d["r"].values
                W = w.sum(); TOP = float(np.sum(w * r * r))
                s["kahin_oran"] = float(np.sqrt(np.sum(w[m] * r[m] ** 2) / TOP))
                s["pay"] = float(m.mean())
                S[ad] = s
        out[bad] = S
        print("=== " + bad + " tamam"); sys.stdout.flush()
        del d; gc.collect()
    with open(os.path.join(CIK, "k05_gozlenebilir.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    adlar = list(out["yaz25"].keys())
    print("")
    h = "%-30s %5s %7s | %9s %7s %6s | %9s %9s  %s" % (
        "GOZLENEBILIR BAYRAK", "pay%", "kahin", "yaz25 oran", "SE", "t", "guz25", "kis26", "isaret")
    print(h); print("-" * len(h))
    for a in adlar:
        v = [out[b][a] for b in ("yaz25", "guz25", "kis26")]
        ss = "AYNI" if (v[0]["oran"] > 0) == (v[1]["oran"] > 0) == (v[2]["oran"] > 0) else "ZIT"
        print("%-30s %5.1f %7.4f | %+9.4f %7.4f %6.1f | %+9.4f %+9.4f  %s" % (
            a[:30], 100 * v[0]["pay"], v[0]["kahin_oran"], v[0]["oran"],
            v[0]["se"] / np.sqrt(np.maximum(1e-12, 1.0)), v[0]["t"],
            v[1]["oran"], v[2]["oran"], ss))
