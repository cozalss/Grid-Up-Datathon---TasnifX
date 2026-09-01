# -*- coding: utf-8 -*-
"""A yonunun TESHISI: blok-disi artik regresyonu rho'su (yaz25 -0.147) GERCEK mi,
   yoksa docs/82'nin 'takvim kapsami' YAPAYLIGI mi?
   1) rhat'i takvim-yapaylik altuzayina (sabit, ay, hg, ufuk, rejim) dik yap -> rho ne kalir
   2) rho'nun CEP DAGILIMI (hangi satirlar tasiyor)
   3) AYNI yonu TEST uzerinde kur -> 30-gonderim span'i disinda ne kaliyor (yenilik payi)
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import lightgbm as lgb

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
PK = os.path.join(KOK, "experiments/model29/p_kalici")
sys.path.insert(0, PK)
import p27_ortak as P
CIK = r"C:/Users/Cem/AppData/Local/Temp/claude/c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/d8509f77-6f9b-4e1d-b980-62e299ed4fc5/scratchpad"
ATIL = {"tuketim", "id", "_blok", "tanim", "tarih", "lokasyon", "p", "y", "r", "w",
        "sog_cat", "sog_xgb", "sog_lgbm", "ay", "hg", "tanim_num"}
PAR = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
           feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0,
           verbose=-1, num_threads=6, seed=7)
t0 = time.time()


def rho_olc(d, u):
    w = d["w"].values; r = d["r"].values; W = w.sum()
    u = np.asarray(u, float)
    nrm = np.sqrt(np.sum(w * u * u) / W)
    if nrm <= 0:
        return None
    un = u / nrm
    rho = float(np.sum(w * r * un) / W)
    v = w * (r * un - rho)
    gg = pd.DataFrame({"g": d["tanim"].values, "v": v}).groupby("g", observed=True)["v"].sum().values
    se = float(np.sqrt(np.sum(gg * gg)) / W)
    rms = float(np.sqrt(np.sum(w * r * r) / W))
    return dict(rho=rho, oran=rho / rms, se=se, t=rho / se if se > 0 else 0.0)


def dik_yap(d, u, Z):
    """u'yu Z sutun uzayina (agirlikli) dik yap."""
    w = d["w"].values
    G = Z.T @ (w[:, None] * Z)
    b = Z.T @ (w * u)
    c = np.linalg.lstsq(G, b, rcond=None)[0]
    return u - Z @ c


D = {b: None for b in ("yaz25", "guz25", "kis26")}
for b in D:
    d = P.blok(b, soguk_harman="cat", son_islem=True).reset_index(drop=True)
    d["w"] = P.agirlik(d)
    D[b] = d
print("bloklar kuruldu t=%.0f" % (time.time() - t0)); sys.stdout.flush()
kol = [c for c in D["yaz25"].columns if c not in ATIL and
       (pd.api.types.is_numeric_dtype(D["yaz25"][c]) or pd.api.types.is_bool_dtype(D["yaz25"][c]))]

hedef = "yaz25"
dh = D[hedef]
tr = pd.concat([D[b] for b in D if b != hedef], ignore_index=True)
m = lgb.train(PAR, lgb.Dataset(tr[kol].astype("float32"), label=tr["r"].values,
                               weight=tr["w"].values), num_boost_round=700)
rhat = m.predict(dh[kol].astype("float32"))
np.save(os.path.join(CIK, "k10_rhat_yaz25.npy"), rhat)
print("model egitildi t=%.0f" % (time.time() - t0)); sys.stdout.flush()

S = {}
S["ham"] = rho_olc(dh, rhat)
n = len(dh)
one = np.ones((n, 1))
ay = pd.get_dummies(dh["ay"]).values.astype(float)
hg = pd.get_dummies(dh["hg"]).values.astype(float)
uf = pd.get_dummies(np.minimum(dh["ufuk_gun"].values.astype(int) // 15, 8)).values.astype(float)
rj = dh["soguk_mu"].values.astype(float).reshape(-1, 1)
katman = [("sabit", one),
          ("sabit+ay", np.hstack([one, ay])),
          ("sabit+ay+hg", np.hstack([one, ay, hg])),
          ("sabit+ay+hg+ufuk", np.hstack([one, ay, hg, uf])),
          ("sabit+ay+hg+ufuk+rejim", np.hstack([one, ay, hg, uf, rj])),
          ("YAPAYLIK TAM (rejim x ay x ufuk)", np.hstack([one, ay, hg, uf, rj, rj * ay, rj * uf]))]
for ad, Z in katman:
    up = dik_yap(dh, rhat, Z)
    S["dik_" + ad] = rho_olc(dh, up)
    S["dik_" + ad]["kalan_norm_payi"] = float(
        np.sum(dh["w"].values * up * up) / np.sum(dh["w"].values * rhat * rhat))

# CEP DAGILIMI: rho'nun hangi satirlardan geldigi (birim yon sabit tutulur)
w = dh["w"].values; r = dh["r"].values; W = w.sum()
nrm = np.sqrt(np.sum(w * rhat * rhat) / W)
u = rhat / nrm
katki = w * r * u / W
tuk = dh["tuketim"].values.astype(float)
sg = dh["soguk_mu"].values == 1
cep = {"soguk&sifir": sg & (tuk == 0), "soguk&pozitif": sg & (tuk > 0),
       "sicak&sifir": (~sg) & (tuk == 0), "sicak&pozitif": (~sg) & (tuk > 0)}
S["cep_katkisi"] = {k: float(katki[v].sum()) for k, v in cep.items()}
S["cep_katkisi"]["TOPLAM(rho)"] = float(katki.sum())
for a in (4, 5, 6, 7):
    S["cep_katkisi"]["ay=%d" % a] = float(katki[dh["ay"].values == a].sum())

# --- TEST tarafi: ayni yonu kur, span-disi payini olc
tam = pd.concat([D[b] for b in D], ignore_index=True)
mt = lgb.train(PAR, lgb.Dataset(tam[kol].astype("float32"), label=tam["r"].values,
                                weight=tam["w"].values), num_boost_round=700)
te = pd.read_parquet(os.path.join(KOK, "data/interim/deney/test.parquet"))
rhat_te = mt.predict(te[kol].astype("float32"))
np.save(os.path.join(CIK, "k10_rhat_test.npy"), rhat_te)
V = np.load(os.path.join(PK, "p34_V30.npy"))
BAZ = np.load(os.path.join(PK, "p34_dik_baz.npy"))
A = np.hstack([V, BAZ.T])
N = A.shape[0]
G = (A.T @ A) / N
Ginv = np.linalg.pinv(G + 1e-10 * np.trace(G) / G.shape[0] * np.eye(G.shape[0]), rcond=1e-12)
u_te = rhat_te - rhat_te.mean() * 0
c = Ginv @ ((A.T @ u_te) / N)
up = u_te - A @ c
S["TEST_span_disi_pay"] = float((up * up).mean() / (u_te * u_te).mean())
S["TEST_span_disi_carpan"] = float(np.sqrt(S["TEST_span_disi_pay"]))
print("model2 + span t=%.0f" % (time.time() - t0))

with open(os.path.join(CIK, "k10_teshis.json"), "w", encoding="utf-8") as f:
    json.dump(S, f, indent=1)
print("\n%-42s %9s %9s %7s %9s" % ("A YONU TESHISI (yaz25)", "oran", "SE", "t", "kalan||"))
for k, v in S.items():
    if isinstance(v, dict) and "oran" in v:
        print("%-42s %+9.4f %9.4f %7.1f %9s" % (
            k, v["oran"], v["se"], v["t"],
            ("%.3f" % v["kalan_norm_payi"]) if "kalan_norm_payi" in v else "-"))
print("\nCEP KATKISI (rho'nun ayrisimi):")
for k, v in S["cep_katkisi"].items():
    print("   %-16s %+.5f" % (k, v))
print("\nTEST span-disi pay=%.4f  carpan=%.4f" % (S["TEST_span_disi_pay"], S["TEST_span_disi_carpan"]))
