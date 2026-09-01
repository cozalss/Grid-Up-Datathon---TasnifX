# -*- coding: utf-8 -*-
"""YON 4 s4: HAVUZLANMIS TEMIZ soguk olcumu (p30 doktrini) + ufuk/pencere gruplari
+ 'temiz soguk' uzerinde blok-disi artik regresyonu (soguk gozlenebilirlerin EN IYI birlesimi).
Hicbir esik/parametre yaz25'ten secilmez."""
import json
import os
import numpy as np
import pandas as pd
from ortak import blok, ezber_maskesi, rho_olc, BLOKLAR, KOK
import p27_ortak as P

SP = os.path.dirname(os.path.abspath(__file__))
T = pd.read_parquet(os.path.join(KOK, "data/interim/deney/test.parquet"))
sgT = (T.soguk_mu.values == 1)
KOL = []
for c in T.columns:
    if c in ("id", "tanim", "tarih", "_blok", "soguk_mu", "tuketim") or c.startswith("t_"):
        continue
    if not pd.api.types.is_numeric_dtype(T[c]):
        continue
    v = T[c].values[sgT]
    if np.isfinite(v).all() and np.nanstd(v) > 0:
        KOL.append(c)

# ---- havuz kur: 3 blok, her biri kendi kohort agirligiyla, blok kutleleri esitlenmis
parcalar = []
for b in BLOKLAR:
    d = blok(b)
    w = P.agirlik(d)
    ez = ezber_maskesi(b)
    sg = (d.soguk_mu.values == 1)
    parcalar.append(dict(b=b, d=d, w=w / w.mean(), sg=sg, temiz=sg & (~ez)))

W = np.concatenate([p["w"] for p in parcalar])
Rr = np.concatenate([p["d"].r.values for p in parcalar])
SG = np.concatenate([p["sg"] for p in parcalar])
TMZ = np.concatenate([p["temiz"] for p in parcalar])
KUME = np.concatenate([p["d"].tanim.values.astype(str) + "_" + p["b"] for p in parcalar])
BLK = np.concatenate([np.full(len(p["d"]), p["b"]) for p in parcalar])

hv = pd.DataFrame({"r": Rr})
hv["tanim"] = KUME
sw = W.sum()
print("havuz n=%d  soguk=%d  TEMIZ soguk=%d" % (len(W), SG.sum(), TMZ.sum()))


def rho_havuz(delta, maske):
    d = np.where(maske, delta, 0.0).astype(np.float64)
    m = np.sum(W[maske] * d[maske]) / np.sum(W[maske]) if maske.any() else 0.0
    d = np.where(maske, d - m, 0.0)
    nrm = np.sqrt(np.sum(W * d * d) / sw)
    if nrm <= 0 or not np.isfinite(nrm):
        return None
    u = d / nrm
    rho = float(np.sum(W * Rr * u) / sw)
    g = W * Rr * u
    df = pd.DataFrame({"k": KUME, "g": g, "w": W})
    ag = df.groupby("k", sort=False).sum()
    res = ag.g.values - rho * ag.w.values
    se = float(np.sqrt(np.sum(res * res)) / sw)
    return rho, se


# ---- (1) TEMIZ havuzda tekil tarama
print("\n" + "=" * 96)
print("(1) HAVUZLANMIS TEMIZ SOGUK (n=%d; yaz25 582 + guz25 907 + kis26 61918)" % TMZ.sum())
print("    'TUM soguk' sutunu kirlilik dahil karsilastirma icindir.")
sat = {}
for c in KOL:
    try:
        v = np.concatenate([p["d"][c].values.astype(np.float64) for p in parcalar])
    except Exception:
        continue
    if not np.isfinite(v[TMZ]).all() or np.std(v[TMZ]) <= 0:
        continue
    a = rho_havuz(v, TMZ)
    bq = rho_havuz(v, SG)
    if a is None:
        continue
    sat[c] = dict(temiz=a, tum=bq)
# sabit yon
sat["__SOGUK_SABIT__"] = dict(temiz=rho_havuz(np.ones(len(W)), TMZ),
                              tum=rho_havuz(np.ones(len(W)), SG))
# NOT: sabit icin merkezleme onu sifirlar; ayrica gostergeyi elle olc
def rho_gosterge(maske):
    d = maske.astype(np.float64)
    nrm = np.sqrt(np.sum(W * d * d) / sw)
    u = d / nrm
    rho = float(np.sum(W * Rr * u) / sw)
    g = W * Rr * u
    ag = pd.DataFrame({"k": KUME, "g": g, "w": W}).groupby("k", sort=False).sum()
    res = ag.g.values - rho * ag.w.values
    return rho, float(np.sqrt(np.sum(res * res)) / sw)

print("  GOSTERGE yonleri: TUM soguk %+.4f+-%.4f | TEMIZ soguk %+.4f+-%.4f"
      % (rho_gosterge(SG) + rho_gosterge(TMZ)))
del sat["__SOGUK_SABIT__"]

sirali = sorted([c for c in sat if sat[c]["temiz"]], key=lambda c: -abs(sat[c]["temiz"][0]))
print("\n  %-26s %-20s %-20s" % ("kolon", "TEMIZ havuz rho", "TUM soguk rho"))
print("  " + "-" * 68)
for c in sirali[:20]:
    a, b2 = sat[c]["temiz"], sat[c]["tum"]
    if b2 is None:
        b2 = (float("nan"), float("nan"))
    print("  %-26s %+.4f +- %.4f    %+.4f +- %.4f" % (c, a[0], a[1], b2[0], b2[1]))

# ---- (2) ufuk / pencere gruplari (blok-disi kayma)
print("\n" + "=" * 96)
print("(2) BLOK-DISI UFUK/PENCERE KOVA KAYMASI (soguk-only)")
D = {p["b"]: p for p in parcalar}
for gk, nk in (("ufuk_gun", 8), ("ozet_pencere_gun", 6), ("p_gun_sayisi", 6), ("p_doluluk", 5)):
    for b in BLOKLAR:
        pb = D[b]
        d, w, sg = pb["d"], pb["w"], pb["sg"]
        if gk not in d.columns:
            continue
        # kova sinirlari: DIGER bloklardan
        dis = np.concatenate([D[o]["d"][gk].values[D[o]["sg"]] for o in BLOKLAR if o != b])
        kes = np.unique(np.quantile(dis, np.linspace(0, 1, nk + 1)))
        kay = {}
        for o in BLOKLAR:
            if o == b:
                continue
            po = D[o]
            kv = np.digitize(po["d"][gk].values, kes)
            for kk in np.unique(kv[po["sg"]]):
                m = po["sg"] & (kv == kk)
                kay.setdefault(kk, [0.0, 0.0])
                kay[kk][0] += float(np.sum(po["w"][m] * po["d"].r.values[m]))
                kay[kk][1] += float(np.sum(po["w"][m]))
        mp = {k: (v[0] / v[1]) * (v[1] / (v[1] + 100.0)) for k, v in kay.items() if v[1] > 0}
        kvb = np.digitize(d[gk].values, kes)
        vv = np.array([mp.get(k, 0.0) for k in kvb], dtype=np.float64)
        delta = np.zeros(len(d))
        delta[sg] = vv[sg] - np.average(vv[sg], weights=w[sg])
        if np.std(delta[sg]) <= 0:
            print("  %-18s %-6s  ---" % (gk, b))
            continue
        o2 = rho_olc(d, delta, w)
        print("  %-18s %-6s  rho %+.4f +- %.4f (t %+.1f)" % (gk, b, o2["rho"], o2["se"], o2["t"]))

# ---- (3) TEMIZ soguk uzerinde blok-disi ARTIK REGRESYONU (en iyi birlesim)
print("\n" + "=" * 96)
print("(3) SOGUK GOZLENEBILIRLERIN EN IYI BIRLESIMI -- blok-disi ridge artik regresyonu")
print("    egitim: DIGER iki blogun soguk satirlari; olcum: hedef blogun soguk satirlari")
X = {}
for b in BLOKLAR:
    d = D[b]["d"]
    M = np.c_[[d[c].values.astype(np.float64) for c in KOL]].T
    X[b] = M
mu = np.concatenate([X[b] for b in BLOKLAR]).mean(0)
sd = np.concatenate([X[b] for b in BLOKLAR]).std(0) + 1e-9
son = {}
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
    A = (Xtr * wtr[:, None]).T @ Xtr + 50.0 * np.eye(Xtr.shape[1])
    bta = np.linalg.solve(A, (Xtr * wtr[:, None]).T @ rtr)
    d = D[b]["d"]; sg = D[b]["sg"]; w = D[b]["w"]
    Xte = np.c_[(X[b] - mu) / sd, np.ones(len(X[b]))]
    g = Xte @ bta
    delta = np.zeros(len(d))
    delta[sg] = g[sg] - np.average(g[sg], weights=w[sg])
    o2 = rho_olc(d, delta, w)
    tmz = D[b]["temiz"]
    o3 = rho_olc(d, np.where(tmz, delta, 0.0), w)
    son[b] = dict(rho=round(o2["rho"], 5), se=round(o2["se"], 5),
                  rho_temiz=round(o3["rho"], 5), n_temiz=int(tmz.sum()))
    print("  %-6s rho %+.4f +- %.4f (t %+.1f) | TEMIZ alt kume %+.4f (n=%d)"
          % (b, o2["rho"], o2["se"], o2["t"], o3["rho"], tmz.sum()))

out = dict(
    havuz_temiz={c: dict(temiz=[round(x, 5) for x in sat[c]["temiz"]],
                         tum=([round(x, 5) for x in sat[c]["tum"]] if sat[c]["tum"] else None))
                 for c in sirali[:40]},
    gosterge=dict(tum_soguk=[round(x, 5) for x in rho_gosterge(SG)],
                  temiz_soguk=[round(x, 5) for x in rho_gosterge(TMZ)]),
    artik_regresyonu=son,
)
with open(os.path.join(SP, "s4_temiz.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print("\nyazildi: s4_temiz.json")
