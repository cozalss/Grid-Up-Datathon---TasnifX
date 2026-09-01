"""p36-f: KUTLENIN OLDUGU YERI HEDEFLE.
(1) 'uzun gecmis + uzun sifir kuyrugu ama model CANLI diyor' cebi: gercekten olu mu?
(2) o cebe ozel yon aileleri ve rho (blok-disi q ile), SE'li.
(3) span-benzeri dikleme: sabit ve p-seviyesi bilesenleri cikarilinca rho ne olur.
"""
import json, os, sys, pickle
import numpy as np, pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
sys.path.insert(0, os.path.join(KOK, "experiments/model29/p_kalici"))
from p27_ortak import agirlik, blok  # noqa
CIK = r"C:/Users/Cem/AppData/Local/Temp/claude/c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/d8509f77-6f9b-4e1d-b980-62e299ed4fc5/scratchpad"
BLOKLAR = ("yaz25", "guz25", "kis26")
rng = np.random.default_rng(11)
D = {b: blok(b, soguk_harman="cat", son_islem=True) for b in BLOKLAR}
W = {b: agirlik(D[b]) for b in BLOKLAR}
Q = pickle.load(open(os.path.join(CIK, "p36_q.pkl"), "rb"))
R = {}

# ---------- 1) 'model canli diyor ama kuyruk sifir' cebi ----------
print("=== cep: t_kuyruk_sifir>=K ve p>4 (model CANLI diyor) ===")
t = []
for b in BLOKLAR:
    d, w = D[b], W[b]; p = d.p.values; z = d.tuketim.values <= 0
    ks = np.nan_to_num(d.t_kuyruk_sifir.values.astype(np.float64), nan=-1.0)
    tot = float(np.sum(w * d.r.values ** 2))
    pen = float(np.nanmedian(d.ozet_pencere_gun.values))
    for K in (1, 30, 60, 90):
        m = (ks >= K) & (p > 4)
        if m.sum() < 30: continue
        t.append(dict(blok=b, pencere=pen, K=K, n=int(m.sum()),
            pay=round(float(np.sum(w * m) / w.sum()), 5),
            gercekten_sifir_orani=round(float(np.average(z[m], weights=w[m])), 4),
            ort_p=round(float(np.average(p[m], weights=w[m])), 3),
            ort_y=round(float(np.average(d.y.values[m], weights=w[m])), 3),
            ort_r=round(float(np.average(d.r.values[m], weights=w[m])), 3),
            MSE_payi=round(float(np.sum(w[m] * d.r.values[m] ** 2) / tot), 4),
            sifirlamanin_kahin_MSE=round(float(np.sum(w[m] * z[m] * p[m] ** 2) / tot), 4)))
        print(t[-1])
R["01_cep"] = t

# ---------- 2) rho, dikleme ile ----------
def rho_se(r, dl, w, tid, dik=None, B=200):
    dl = dl.astype(np.float64).copy()
    if dik is not None:                      # w-agirlikli dikleme
        X = np.column_stack(dik).astype(np.float64)
        A = X.T @ (w[:, None] * X); bvec = X.T @ (w * dl)
        dl = dl - X @ np.linalg.lstsq(A, bvec, rcond=None)[0]
    k2 = float(np.sum(w * dl * dl) / np.sum(w))
    if k2 <= 1e-14: return np.nan, np.nan, np.nan
    kap = np.sqrt(k2); u = dl / kap
    rho = float(np.sum(w * r * u) / np.sum(w))
    kod, inv = np.unique(tid, return_inverse=True); nk = len(kod)
    num = np.bincount(inv, weights=w * r * u, minlength=nk)
    den = np.bincount(inv, weights=w, minlength=nk)
    idx = rng.integers(0, nk, size=(B, nk))
    return rho, float(np.std(num[idx].sum(1) / den[idx].sum(1))), kap

def yonler(b):
    d = D[b]; p = d.p.values
    q = np.clip(Q[b]["q"], 0, 1 - 1e-6)
    ks = np.nan_to_num(d.t_kuyruk_sifir.values.astype(np.float64), nan=-1.0)
    o = {}
    for K in (1, 30, 60):
        for pe in (0.0, 3.0, 4.0):
            m = (ks >= K) & (p > pe)
            o[f"cep_K{K}_p>{pe}_tam_sifirla"] = -p * m
    for qe in (0.3, 0.5, 0.7):
        for pe in (0.0, 4.0):
            o[f"q>{qe}_p>{pe}_tam_sifirla"] = -p * ((q > qe) & (p > pe))
    o["q_p*q"] = -p * q
    o["q_p*q^2"] = -p * q ** 2
    return o

print("\n=== rho: ham vs dikleme(sabit) vs dikleme(sabit,p,soguk) ===")
sat = []
for b in BLOKLAR:
    d, w = D[b], W[b]; r = d.r.values
    tid = pd.factorize(d.tanim.values)[0]
    bir = np.ones(len(d)); pv = d.p.values; sg = (d.soguk_mu.values == 1).astype(float)
    for ad, dl in yonler(b).items():
        a = rho_se(r, dl, w, tid)
        c = rho_se(r, dl, w, tid, dik=[bir])
        e = rho_se(r, dl, w, tid, dik=[bir, pv, sg])
        sat.append(dict(blok=b, yon=ad, rho=a[0], se=a[1], kappa=a[2],
                        rho_dik1=c[0], se_dik1=c[1], rho_dik3=e[0], se_dik3=e[1]))
T = pd.DataFrame(sat)
for b in BLOKLAR:
    print(f"\n--- {b} ---")
    print(T[T.blok == b].drop(columns="blok").to_string(
        index=False, float_format=lambda x: f"{x:+.5f}"))
R["02_rho_dik"] = json.loads(T.to_json(orient="records"))

Pv = T.pivot(index="yon", columns="blok", values="rho_dik1")[list(BLOKLAR)]
Sv = T.pivot(index="yon", columns="blok", values="se_dik1")[list(BLOKLAR)]
print("\n=== dik(sabit) rho ozet, yaz25 sirali ===")
o = pd.concat([Pv, Sv.add_prefix("se_")], axis=1).sort_values("yaz25", ascending=False)
print(o.to_string(float_format=lambda x: f"{x:+.5f}"))
R["03_ozet_dik1"] = json.loads(Pv.to_json())

# ---------- 3) pencere uzunlugu etkisi ----------
print("\n=== pencere uzunlugu vs bayrakli satirlarda ort_p (TEST=455) ===")
for b in BLOKLAR:
    d = D[b]; ks = np.nan_to_num(d.t_kuyruk_sifir.values.astype(float), nan=-1.0)
    f = ks >= 1
    print(f"{b}: pencere={np.nanmedian(d.ozet_pencere_gun.values):.0f} "
          f"ort_p_bayrak={d.p.values[f].mean():.3f} "
          f"p90={np.quantile(d.p.values[f],0.9):.3f} "
          f"bayrakta_gercek_sifir_orani={(d.tuketim.values[f]<=0).mean():.4f}")
with open(os.path.join(CIK, "p36_f.json"), "w", encoding="utf-8") as fh:
    json.dump(R, fh, ensure_ascii=False, indent=1, default=str)
print("\nyazildi p36_f.json")
