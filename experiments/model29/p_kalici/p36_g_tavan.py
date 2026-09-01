"""p36-g: TAVAN CEBIRI -- MUKEMMEL kesinlikte bile 0.1230'a ulasilir mi?

delta = -p*f  icin:
  rho = ( E_w[f z p^2] - E_w[f(1-z) p (y-p)] ) / sqrt(E_w[f p^2])
Mukemmel kesinlik (f subset z) => rho = sqrt(E_w[f z p^2]) : yani sifir-kutlesinin
karekoku.  Gereken rho=0.1230 icin gereken yakalanan kutle = 0.01513.
Gercek siniflandiricinin DP/YP ayrisimini da veriyoruz.
"""
import json, os, sys, pickle
import numpy as np, pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
sys.path.insert(0, os.path.join(KOK, "experiments/model29/p_kalici"))
from p27_ortak import agirlik, blok  # noqa
CIK = r"C:/Users/Cem/AppData/Local/Temp/claude/c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/d8509f77-6f9b-4e1d-b980-62e299ed4fc5/scratchpad"
BLOKLAR = ("yaz25", "guz25", "kis26")
D = {b: blok(b, soguk_harman="cat", son_islem=True) for b in BLOKLAR}
W = {b: agirlik(D[b]) for b in BLOKLAR}
Q = pickle.load(open(os.path.join(CIK, "p36_q.pkl"), "rb"))
GEREK_T = 0.99310; GEREK_RHO = 0.12298; KABUL_RHO = 0.11436
R = {}

print("=== TAVAN: mukemmel kesinlik (sadece gercek sifirlari sifirla) ===")
t = []
for b in BLOKLAR:
    d, w = D[b], W[b]; p, y, r = d.p.values, d.y.values, d.r.values
    q = Q[b]["q"]; z = d.tuketim.values <= 0; sw = w.sum()
    z_kutle = float(np.sum(w[z] * p[z] ** 2) / sw)
    print(f"\n{b}: TUM sifir kutlesi E_w[z p^2] = {z_kutle:.5f} -> "
          f"mukemmel-kesinlik rho tavani = {np.sqrt(z_kutle):.5f}")
    for e in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        f = q > e
        if f.sum() < 50: continue
        kap2 = float(np.sum(w * f * p * p) / sw)
        dp_k = float(np.sum(w[f & z] * p[f & z] ** 2) / sw)      # yakalanan kutle
        yp_pay = float(np.sum(w[f & ~z] * (-p[f & ~z]) * r[f & ~z]) / sw)
        rho = (dp_k + yp_pay) / np.sqrt(kap2)
        t.append(dict(blok=b, esik=e,
            kesinlik=round(float(np.sum(w[f]*z[f])/np.sum(w[f])), 4),
            yakalanan_kutle=round(dp_k, 5),
            kutle_orani=round(dp_k / z_kutle, 4),
            kappa=round(float(np.sqrt(kap2)), 5),
            rho_DP_bileseni=round(dp_k / np.sqrt(kap2), 5),
            rho_YP_bileseni=round(yp_pay / np.sqrt(kap2), 5),
            rho_net=round(rho, 5),
            # ayni f'yi MUKEMMEL kesinlikle uygulasak (yalniz f&z sifirlansa)
            rho_mukemmel_kesinlik=round(float(np.sqrt(dp_k)), 5)))
        print("  ", t[-1])
R["01_tavan"] = t

print("\n=== 0.1230 icin gereken yakalanan sifir-kutlesi = %.5f ===" % GEREK_RHO**2)
for b in BLOKLAR:
    d, w = D[b], W[b]; p = d.p.values; z = d.tuketim.values <= 0
    zk = float(np.sum(w[z]*p[z]**2)/w.sum())
    print(f"{b}: sifir kutlesi {zk:.5f}; gereken pay = {GEREK_RHO**2/zk*100:.2f}% "
          f"(MUKEMMEL kesinlikle)")

# --- DURUST blok-disi secim, dik(sabit) rho uzerinden, tam aile ---
F = json.load(open(os.path.join(CIK, "p36_f.json"), encoding="utf-8"))
T = pd.DataFrame(F["02_rho_dik"])
P = T.pivot(index="yon", columns="blok", values="rho_dik1")[list(BLOKLAR)]
S = T.pivot(index="yon", columns="blok", values="se_dik1")[list(BLOKLAR)]
C = json.load(open(os.path.join(CIK, "p36_c.json"), encoding="utf-8"))
P2 = pd.DataFrame(C["01_rho"])[list(BLOKLAR)]
S2 = pd.DataFrame(C["02_se"])[list(BLOKLAR)]
P2 = P2[~P2.index.str.startswith("KAHIN")]; S2 = S2.loc[P2.index]
PT = pd.concat([P, P2]); ST = pd.concat([S, S2])
PT = PT[~PT.index.duplicated()].dropna(); ST = ST.loc[PT.index]
print(f"\n=== DURUST SECIM, aday havuzu n={len(PT)} ===")
dur = []
for hed in BLOKLAR:
    dis = [b for b in BLOKLAR if b != hed]
    for olcut, fn in (("ortalama", np.mean), ("minimaks", np.min)):
        sk = PT[dis].apply(fn, axis=1)
        en = sk.idxmax()
        dur.append(dict(hedef=hed, olcut=olcut, secilen=en,
                        disarda=round(float(sk[en]), 5),
                        hedefte_rho=round(float(PT.loc[en, hed]), 5),
                        hedefte_se=round(float(ST.loc[en, hed]), 5)))
        print(dur[-1])
R["02_durust"] = dur
oy = [x["hedefte_rho"] for x in dur if x["hedef"] == "yaz25"]
R["03_yaz25_durust_rho"] = oy
print(f"\nyaz25 DURUST rho = {oy}")
print(f"3 blok DURUST ortalama rho = {np.mean([x['hedefte_rho'] for x in dur]):.5f}")

# --- LB olcegine cevir ---
print("\n=== LB olcegi ===")
MSE_LB = 1.0013719
for ad, rho in [("yaz25 DURUST(en iyi)", max(oy)),
                ("yaz25 SIZINTILI tavan (kendinde secilmis)", float(PT.yaz25.max())),
                ("3-blok DURUST ort", float(np.mean([x["hedefte_rho"] for x in dur]))),
                ("mukemmel-kesinlik q>0.5 (yaz25)",
                 float([x["rho_mukemmel_kesinlik"] for x in t
                        if x["blok"]=="yaz25" and x["esik"]==0.5][0]))]:
    for tas in (1.0, 0.185):
        rl = rho * tas
        sk = float(np.sqrt(max(MSE_LB - rl**2, 1e-9)))
        print(f"{ad:44s} rho={rho:+.5f} tasima={tas:<5} -> LB {sk:.5f} "
              f"(kazanc {1.00069-sk:+.5f})")
R["04_gerek"] = dict(gereken_rho=GEREK_RHO, kabul_rho=KABUL_RHO)
with open(os.path.join(CIK, "p36_g.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1, default=str)
print("\nyazildi p36_g.json")
