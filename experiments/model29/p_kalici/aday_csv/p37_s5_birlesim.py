# -*- coding: utf-8 -*-
"""YON 4 s5: soguk gozlenebilirlerin EN IYI BIRLESIMI (blok-disi).
ridge + LightGBM; hedef blogun kendi verisi hicbir asamada kullanilmaz."""
import json
import os
import numpy as np
import pandas as pd
from ortak import blok, ezber_maskesi, rho_olc, BLOKLAR, KOK
import p27_ortak as P

SP = os.path.dirname(os.path.abspath(__file__))
T = pd.read_parquet(os.path.join(KOK, "data/interim/deney/test.parquet"))
sgT = (T.soguk_mu.values == 1)

D = {}
for b in BLOKLAR:
    d = blok(b)
    D[b] = dict(d=d, w=P.agirlik(d), sg=(d.soguk_mu.values == 1), tmz=None)
    D[b]["tmz"] = D[b]["sg"] & (~ezber_maskesi(b))

# TEST'te VE her uc blokta soguk satirlarda tanimli kolonlar
KOL = []
for c in T.columns:
    if c in ("id", "tanim", "tarih", "_blok", "soguk_mu", "tuketim") or c.startswith("t_"):
        continue
    if not pd.api.types.is_numeric_dtype(T[c]):
        continue
    if not np.isfinite(T[c].values[sgT]).all() or np.nanstd(T[c].values[sgT]) <= 0:
        continue
    ok = True
    for b in BLOKLAR:
        d, sg = D[b]["d"], D[b]["sg"]
        if c not in d.columns or not np.isfinite(pd.to_numeric(d[c], errors="coerce").values[sg]).all():
            ok = False
            break
    if ok:
        KOL.append(c)
print("uc blokta + TEST'te soguk tarafta TAM tanimli kolon:", len(KOL))

X = {b: np.c_[[pd.to_numeric(D[b]["d"][c], errors="coerce").values.astype(np.float64)
               for c in KOL]].T for b in BLOKLAR}
for b in BLOKLAR:
    X[b] = np.nan_to_num(X[b], nan=0.0, posinf=0.0, neginf=0.0)
hepsi = np.vstack([X[b][D[b]["sg"]] for b in BLOKLAR])
mu, sd = hepsi.mean(0), hepsi.std(0) + 1e-9

R = {"n_kolon": len(KOL)}
print("\n" + "=" * 96)
print("(A) RIDGE artik regresyonu -- egitim DIGER iki blogun SOGUK satirlari")
print("%-8s %-6s | %-24s | %-22s" % ("lam", "blok", "rho +- SE (t)", "TEMIZ alt kume"))
print("-" * 80)
for lam in (10.0, 100.0, 1000.0, 10000.0):
    hc = {}
    for b in BLOKLAR:
        Xtr, rtr, wtr = [], [], []
        for o in BLOKLAR:
            if o == b:
                continue
            m = D[o]["sg"]
            Xtr.append((X[o][m] - mu) / sd)
            rtr.append(D[o]["d"].r.values[m])
            wtr.append(D[o]["w"][m])
        Xtr = np.vstack(Xtr); rtr = np.concatenate(rtr); wtr = np.concatenate(wtr)
        Xtr = np.c_[Xtr, np.ones(len(Xtr))]
        A = (Xtr * wtr[:, None]).T @ Xtr + lam * np.eye(Xtr.shape[1])
        bta = np.linalg.solve(A, (Xtr * wtr[:, None]).T @ rtr)
        d, sg, w = D[b]["d"], D[b]["sg"], D[b]["w"]
        g = np.c_[(X[b] - mu) / sd, np.ones(len(X[b]))] @ bta
        delta = np.zeros(len(d))
        delta[sg] = g[sg] - np.average(g[sg], weights=w[sg])
        o1 = rho_olc(d, delta, w)
        o2 = rho_olc(d, np.where(D[b]["tmz"], delta, 0.0), w)
        hc[b] = dict(rho=round(o1["rho"], 5), se=round(o1["se"], 5),
                     rho_temiz=round(o2["rho"], 5), g_sd=round(float(np.std(g[sg])), 5))
        print("%-8.0f %-6s | %+.4f +- %.4f (t %+.1f)  | %+.4f   (g_sd %.4f)"
              % (lam, b, o1["rho"], o1["se"], o1["t"], o2["rho"], np.std(g[sg])))
    R["ridge_lam%d" % int(lam)] = hc

# ------------------------------------------------------------------ LightGBM
try:
    import lightgbm as lgb
    print("\n" + "=" * 96)
    print("(B) LightGBM artik regresyonu -- egitim DIGER iki blogun SOGUK satirlari")
    print("    DURUST erken durdurma yok (test etiketi yok); sabit agac sayisi")
    for nag in (100, 400):
        hc = {}
        for b in BLOKLAR:
            Xtr, rtr, wtr = [], [], []
            for o in BLOKLAR:
                if o == b:
                    continue
                m = D[o]["sg"]
                Xtr.append(X[o][m]); rtr.append(D[o]["d"].r.values[m]); wtr.append(D[o]["w"][m])
            Xtr = np.vstack(Xtr); rtr = np.concatenate(rtr); wtr = np.concatenate(wtr)
            ds = lgb.Dataset(Xtr, label=rtr, weight=wtr, feature_name=list(KOL))
            m_ = lgb.train(dict(objective="l2", learning_rate=0.05, num_leaves=31,
                                min_data_in_leaf=200, feature_fraction=0.8,
                                bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=7),
                           ds, num_boost_round=nag)
            d, sg, w = D[b]["d"], D[b]["sg"], D[b]["w"]
            g = m_.predict(X[b])
            delta = np.zeros(len(d))
            delta[sg] = g[sg] - np.average(g[sg], weights=w[sg])
            o1 = rho_olc(d, delta, w)
            o2 = rho_olc(d, np.where(D[b]["tmz"], delta, 0.0), w)
            hc[b] = dict(rho=round(o1["rho"], 5), se=round(o1["se"], 5),
                         rho_temiz=round(o2["rho"], 5))
            print("  agac=%-4d %-6s rho %+.4f +- %.4f (t %+.1f) | TEMIZ %+.4f"
                  % (nag, b, o1["rho"], o1["se"], o1["t"], o2["rho"]))
        R["lgbm_%d" % nag] = hc
except Exception as e:  # pragma: no cover
    print("LightGBM atlandi:", e)

with open(os.path.join(SP, "s5_birlesim.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, indent=1)
print("\nyazildi: s5_birlesim.json")
