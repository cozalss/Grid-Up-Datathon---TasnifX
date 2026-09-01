"""p36-a: SIFIR CEBI ANATOMISI -- docs/82 bolum 4 vs bolum 6 celiskisini COZ.

Karar sayisi: bayrak/siniflandirici SATIR duyarliligi vs MSE-KUTLE duyarliligi.
Sifir cebi MSE'nin cogu, bayragin YAKALADIGI satirlarda MI yoksa KACIRDIGI
satirlarda mi?  MSE_sifir = E_w[z * p^2] (cunku z satirinda r = -p).
"""
import json, os, sys
import numpy as np, pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
sys.path.insert(0, os.path.join(KOK, "experiments/model29/p_kalici"))
from p27_ortak import agirlik, blok, rmsle  # noqa

CIK = r"C:/Users/Cem/AppData/Local/Temp/claude/c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/d8509f77-6f9b-4e1d-b980-62e299ed4fc5/scratchpad"
BLOKLAR = ("yaz25", "guz25", "kis26")
R = {}

D = {b: blok(b, soguk_harman="cat", son_islem=True) for b in BLOKLAR}

# ---- 1) bolum 6 teyidi + bolum 4 teyidi ----
t1 = []
for b in BLOKLAR:
    d = D[b]; w = agirlik(d)
    p, y, r = d.p.values, d.y.values, d.r.values
    z = d.tuketim.values <= 0
    sw = w.sum()
    MSE = float(np.sum(w * r * r) / sw)
    MSE_z = float(np.sum(w[z] * r[z] ** 2) / sw)
    # sifir satirlarinda r == -p mi (kontrol)
    assert np.allclose(r[z], -p[z], atol=1e-9), b
    # olu trafo
    g = pd.DataFrame(dict(t=d.tanim.values, z=z))
    olu = (g.groupby("t").z.mean() == 1.0)
    om = olu.reindex(d.tanim.values).to_numpy()
    t1.append(dict(blok=b, MSE=round(MSE, 5), RMSE=round(np.sqrt(MSE), 5),
        sifir_pay_agir=round(float(np.sum(w * z) / sw), 5),
        MSE_sifir_payi=round(MSE_z / MSE, 4),
        sifir_ort_p=round(float(np.average(p[z], weights=w[z])), 4),
        sifir_rms_p=round(float(np.sqrt(np.average(p[z] ** 2, weights=w[z]))), 4),
        sifir_medyan_p=round(float(np.median(p[z])), 4),
        olu_trafo_ort_p=round(float(np.average(p[om], weights=w[om])), 4),
        olu_trafo_rms_p=round(float(np.sqrt(np.average(p[om] ** 2, weights=w[om]))), 4),
        canli_poz_ort_p=round(float(np.average(p[~z], weights=w[~z])), 4)))
    print(t1[-1])
R["01_bolum6_bolum4"] = t1

# ---- 2) SIFIR MSE'sinin p-desilleri uzerindeki dagilimi ----
print("\n--- sifir satirlarinin MSE'si p'ye gore nasil dagiliyor ---")
t2 = []
for b in BLOKLAR:
    d = D[b]; w = agirlik(d)
    p = d.p.values; z = d.tuketim.values <= 0
    pz, wz = p[z], w[z]
    kes = np.quantile(pz, np.linspace(0, 1, 11))
    kes[-1] += 1e-9
    idx = np.clip(np.digitize(pz, kes[1:-1]), 0, 9)
    tot = float(np.sum(wz * pz ** 2))
    for k in range(10):
        m = idx == k
        t2.append(dict(blok=b, desil=k + 1,
            p_ort=round(float(np.average(pz[m], weights=wz[m])), 3),
            satir_pay=round(float(wz[m].sum() / wz.sum()), 4),
            MSE_pay_sifir_icinde=round(float(np.sum(wz[m] * pz[m] ** 2) / tot), 4)))
    print(b, [x["MSE_pay_sifir_icinde"] for x in t2[-10:]])
R["02_desil"] = t2
with open(os.path.join(CIK, "p36_a.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1, default=str)
print("\nasama1 yazildi")
