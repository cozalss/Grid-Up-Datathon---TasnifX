"""p36-c: ESAS OLCUM -- birim yon basina rho, 4 aile, 3 blok, trafo-kumeli SE.

  u_ham = delta ; norm = sqrt(E_w[u_ham^2]) ; u = u_ham/norm
  rho   = E_w[r*u]           (kazanc = rho^2, optimum kappa = rho)
SE: trafo duzeyinde kume onyuklemesi (artiklar trafo icinde bagimli).
"""
import json, os, sys, pickle
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
sys.path.insert(0, os.path.join(KOK, "experiments/model29/p_kalici"))
from p27_ortak import agirlik, blok  # noqa

CIK = r"C:/Users/Cem/AppData/Local/Temp/claude/c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/d8509f77-6f9b-4e1d-b980-62e299ed4fc5/scratchpad"
BLOKLAR = ("yaz25", "guz25", "kis26")
rng = np.random.default_rng(7)

D = {b: blok(b, soguk_harman="cat", son_islem=True) for b in BLOKLAR}
W = {b: agirlik(D[b]) for b in BLOKLAR}
Q = pickle.load(open(os.path.join(CIK, "p36_q.pkl"), "rb"))
R = {}

def aile(b):
    d = D[b]; p = d.p.values
    q = np.clip(Q[b]["q"], 0, 1 - 1e-6)
    qw = np.clip(Q[b]["qw"], 0, 1 - 1e-6)
    z = d.tuketim.values <= 0
    ks = np.nan_to_num(d.t_kuyruk_sifir.values.astype(np.float64), nan=-1.0)
    o = {}
    o["KAHIN_mukemmel_sifir"] = -p * z
    o["KAHIN_tam_artik"] = d.r.values.copy()
    for e in (1, 7, 30):
        o[f"KURAL_kuyruk>={e}_sabit"] = -1.0 * (ks >= e)
        o[f"KURAL_kuyruk>={e}_p"] = -p * (ks >= e)
    for ad, qq in (("q", q), ("qw", qw)):
        for e in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95):          # (a) sert bayrak sabit -1
            o[f"a_{ad}_sabit1[q>{e}]"] = -1.0 * (qq > e)
        for a in (0.5, 1.0, 2.0, 3.0):                           # (b) olasilikla orantili
            o[f"b_{ad}_q^{a}"] = -(qq ** a)
        for a in (0.5, 1.0, 2.0, 3.0):                           # (c) p ile olcekli
            o[f"c_{ad}_p*q^{a}"] = -p * (qq ** a)
        for e in (0.2, 0.3, 0.5, 0.7, 0.9):
            o[f"c_{ad}_p*1[q>{e}]"] = -p * (qq > e)
        for g in (0.25, 0.5, 1.0, 2.0):                          # (d) yumusak carpansal
            o[f"d_{ad}_gamma{g}_log(1-q)"] = g * np.log(1.0 - qq)
            o[f"d_{ad}_gamma{g}_log(1-q)_kapili0.3"] = g * np.log(1.0 - qq) * (qq > 0.3)
    return o

def rho_se(r, dl, w, tid, B=200):
    k2 = float(np.sum(w * dl * dl) / np.sum(w))
    if k2 <= 1e-14: return np.nan, np.nan, np.nan
    kap = np.sqrt(k2); u = dl / kap
    rho = float(np.sum(w * r * u) / np.sum(w))
    # trafo-kumeli onyukleme
    kod, inv = np.unique(tid, return_inverse=True)
    nk = len(kod)
    num = np.bincount(inv, weights=w * r * u, minlength=nk)
    den = np.bincount(inv, weights=w, minlength=nk)
    n2 = np.bincount(inv, weights=w * dl * dl, minlength=nk)
    idx = rng.integers(0, nk, size=(B, nk))
    bs = (num[idx].sum(1) / np.maximum(n2[idx].sum(1) / den[idx].sum(1), 1e-14) ** 0.5) / den[idx].sum(1)
    return rho, float(np.std(bs)), kap

sat = []
for b in BLOKLAR:
    d, w = D[b], W[b]; r = d.r.values
    tid = pd.factorize(d.tanim.values)[0]
    for ad, dl in aile(b).items():
        rho, se, kap = rho_se(r, dl, w, tid)
        sat.append(dict(blok=b, yon=ad, rho=rho, se=se, kappa_tam=kap))
T = pd.DataFrame(sat)
P = T.pivot(index="yon", columns="blok", values="rho")[list(BLOKLAR)]
S = T.pivot(index="yon", columns="blok", values="se")[list(BLOKLAR)]
K = T.pivot(index="yon", columns="blok", values="kappa_tam")[list(BLOKLAR)]
P["ORT"] = P.mean(1); P["MIN"] = P.min(1)
P["yaz25_t"] = P.yaz25 / S.yaz25
Pg = P.sort_values("yaz25", ascending=False)
print("=== rho (birim yon), SE trafo-kumeli; yaz25'e gore sirali ===")
out = pd.concat([Pg[["yaz25"]], S[["yaz25"]].rename(columns={"yaz25": "se_yaz"}),
                 Pg[["guz25", "kis26", "ORT", "MIN"]]], axis=1).loc[Pg.index]
print(out.to_string(float_format=lambda x: f"{x:+.5f}"))
R["01_rho"] = json.loads(P.to_json()); R["02_se"] = json.loads(S.to_json())
R["03_kappa"] = json.loads(K.to_json())

# ---- DURUST blok-disi secim: parametre DIGER iki bloktan, olcum hedefte ----
ad_list = [x for x in P.index if not x.startswith("KAHIN")]
dur = []
for hed in BLOKLAR:
    dis = [b for b in BLOKLAR if b != hed]
    sk = {a: float(np.mean([P.loc[a, b] for b in dis])) for a in ad_list}
    skmin = {a: float(min(P.loc[a, b] for b in dis)) for a in ad_list}
    e1 = max(sk, key=sk.get); e2 = max(skmin, key=skmin.get)
    dur.append(dict(hedef=hed, olcut="ortalama", secilen=e1,
                    disarda=round(sk[e1], 5), hedefte=round(float(P.loc[e1, hed]), 5),
                    hedefte_se=round(float(S.loc[e1, hed]), 5)))
    dur.append(dict(hedef=hed, olcut="minimaks", secilen=e2,
                    disarda=round(skmin[e2], 5), hedefte=round(float(P.loc[e2, hed]), 5),
                    hedefte_se=round(float(S.loc[e2, hed]), 5)))
print("\n=== DURUST blok-disi secim ===")
print(pd.DataFrame(dur).to_string(index=False))
R["04_durust"] = dur

# ---- yaz25 ODAK: en iyi 8 aile uyesi + isaret kararliligi ----
kar = P.loc[ad_list].copy()
kar["isaret_kararli"] = (np.sign(kar.yaz25) == np.sign(kar.guz25)) & \
                        (np.sign(kar.yaz25) == np.sign(kar.kis26))
print("\n=== yaz25'te en iyi 10, isaret kararliligi ile ===")
print(kar.sort_values("yaz25", ascending=False).head(10).to_string(
    float_format=lambda x: f"{x:+.5f}"))
R["05_yaz25_en_iyi"] = json.loads(kar.sort_values("yaz25", ascending=False).head(12).to_json())

# ---- KUTLENIN OLDUGU YERDE SINYAL VAR MI: yuksek p'li satirlarda AUC ----
print("\n=== MSE kutlesinin oldugu yerde (p>4) sifir sinyali var mi? ===")
hp = []
for b in BLOKLAR:
    d, w = D[b], W[b]; p = d.p.values; z = (d.tuketim.values <= 0).astype(int)
    for esik_p in (3.0, 4.0, 5.0):
        m = p > esik_p
        if z[m].sum() < 20 or (1 - z[m]).sum() < 20: continue
        hp.append(dict(blok=b, p_esigi=esik_p, n=int(m.sum()),
            sifir_n=int(z[m].sum()), sifir_orani=round(float(z[m].mean()), 5),
            AUC_q=round(float(roc_auc_score(z[m], Q[b]["q"][m])), 4),
            AUC_qw=round(float(roc_auc_score(z[m], Q[b]["qw"][m])), 4),
            bu_kumenin_MSE_payi=round(float(np.sum(w[m] * z[m] * p[m] ** 2)
                                            / np.sum(w * d.r.values ** 2)), 4)))
        print(hp[-1])
R["06_yuksek_p_auc"] = hp
with open(os.path.join(CIK, "p36_c.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1, default=str)
print("\nyazildi p36_c.json")
