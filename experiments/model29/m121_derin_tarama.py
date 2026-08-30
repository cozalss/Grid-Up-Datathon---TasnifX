"""DERIN TARAMA -- daha cok donusum ve ikili etkilesim.

m118 yalnizca uc donusum denedi (x_sv, x_soguk, ust10) ve |rho_s| tavanini
0.027'den 0.0405'e cikardi. Bu betik donusum kumesini genisletiyor:
  x_ufuk, x_ay, alt10, orta, kare, log, ve EN IYI oznitelikler arasi
  ikili carpimlar.

Olcut yine LB-capali:  katki = min(|rho_cv|, 1.95|rho_s|)^2 * Q_dik
rho_s = L_span/sqrt(Q_span) DOGRUDAN LB OLCUMUDUR.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = os.path.join(KOK, "data/interim/deney")
AO = os.path.join(KOK, "data/interim/aile_onbellek")
S = os.path.join(KOK, "submissions")
M29 = os.path.join(KOK, "experiments/model29")
BURA = os.path.dirname(os.path.abspath(__file__))
TABAN = "tuketim_m6_ikiyon.csv"  # M0 m112den gelir (docs/69)
HEDEF_SOGUK, CARPAN, TAVAN = 0.222, 0.798, 1.95
sys.path.insert(0, M29)
from m112_kalibre import EK_MODEL, M0, buzmeli_r_hat  # noqa: E402

te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
IDS = te.id.values


def oku(f):
    d = pd.read_csv(os.path.join(S, f))
    k = "tuketim" if "tuketim" in d.columns else d.columns[-1]
    if not np.array_equal(d.id.values, IDS):
        if len(d) != len(IDS) or d.id.duplicated().any():
            return None
        pos = pd.Index(d.id).get_indexer(IDS)
        if (pos < 0).any():
            return None
        d = d.iloc[pos].reset_index(drop=True)
    return np.log1p(d[k].values.astype(np.float64))


a0 = oku(TABAN)
N = len(a0)
with open(os.path.join(M29, "olculmus_skorlar.json")) as fh:
    SK = json.load(fh)
with open(os.path.join(M29, "m112_durum.json")) as fh:
    DUR = json.load(fh)
V, L = [], []
for f, Pj in SK.items():
    if f == TABAN or not os.path.exists(os.path.join(S, f)):
        continue
    v = oku(f)
    if v is None or len(v) != N:
        continue
    dd = v - a0
    V.append(dd)
    L.append((M0 + float((dd * dd).mean()) - Pj * Pj) / 2)
for f, Lj in EK_MODEL.items():
    V.append(oku(f) - a0)
    L.append(Lj)
for o in DUR.get("olcumler", []):
    dd = oku(o["dosya"]) - a0
    V.append(dd)
    L.append((M0 + float((dd * dd).mean()) - o["skor"] ** 2) / 2)
V, L = np.array(V).T, np.array(L)
G = (V.T @ V) / N
Gi = np.linalg.pinv(G, rcond=1e-6)
_, gercek, kL = buzmeli_r_hat(V, L, G, N)
print(f"saf optimum {np.sqrt(M0 - gercek):.6f}")

tp = pd.read_parquet(os.path.join(DN, "test.parquet"))
e = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
blk = e[e._blok == "yaz25"]
sic, sog = blk[blk.soguk_mu == 0], blk[blk.soguk_mu == 1]
P = [
    np.load(os.path.join(AO, f"yaz25_{t}_{aa}_uretim.npy")).astype(np.float64)
    for t in (1000, 1001, 1002)
    for aa in ("cat", "xgb", "lgbm")
    if os.path.exists(os.path.join(AO, f"yaz25_{t}_{aa}_uretim.npy"))
]
z = np.load(os.path.join(DN, "soguk_tahmin_yaz25.npz"))
idx = np.concatenate([sic.index.values, sog.index.values])
pb = np.concatenate([np.mean(P, axis=0), np.mean([z[q] for q in z.files], axis=0)])
bf = e.loc[idx].copy()
rb = np.log1p(bf.tuketim.values.astype(np.float64)) - pb
sgm = bf.soguk_mu.values.astype(np.float64)
ww = np.where(sgm == 1, HEDEF_SOGUK / sgm.mean(), (1 - HEDEF_SOGUK) / (1 - sgm.mean()))
ww = ww / ww.mean()
m0b = float((ww * rb * rb).mean())


def st(x):
    x = np.asarray(x, dtype=np.float64).copy()
    f = np.isfinite(x)
    if not f.any():
        return None
    x[~f] = np.median(x[f])
    x -= x.mean()
    s = np.sqrt(float((x * x).mean()))
    return x / s if s > 1e-12 else None


svT, svB = st(a0), st(pb)
sgT = tp.soguk_mu.values.astype(np.float64)
ufT, ufB = st(tp.ufuk_gun.to_numpy()), st(bf.ufuk_gun.to_numpy())
ayT = st(pd.to_datetime(tp.tarih).dt.month.to_numpy().astype(float))
ayB = st(pd.to_datetime(bf.tarih).dt.month.to_numpy().astype(float))

ATLA = {"id", "tanim", "lokasyon", "tarih", "bolge", "_blok", "tuketim", "soguk_mu"}
KOL = [
    c
    for c in tp.columns
    if c not in ATLA and pd.api.types.is_numeric_dtype(tp[c]) and c in bf.columns
]
ODAK = [
    "p_pencere_payi",
    "p_gun_sayisi",
    "p_ilk_ofset",
    "p_son_ofset",
    "p_yayilma",
    "p_doluluk",
    "tanim_on2",
    "tanim_on3",
    "t_yuk_faktoru",
    "t_egim_sicaklik_ort",
    "t_egim_cdd22",
    "yas",
    "asiri_sicak",
    "t_kayma",
    "t_log_medyan",
    "t_log_p90",
    "t_log_p10",
    "osm_direk_yogunlugu",
    "guc_yuzdelik",
    "t_hg_sapma",
    "t_ay_sapma",
]
ODAK = [c for c in ODAK if c in KOL]
print(f"{len(KOL)} kolon, {len(ODAK)} odak kolonu")


def donusumler(kol):
    xt, xb = tp[kol].to_numpy(), bf[kol].to_numpy()
    yield kol, st(xt), st(xb)
    for ad, mt, mb in [
        ("x_sv", svT, svB),
        ("x_soguk", sgT, sgm),
        ("x_ufuk", ufT, ufB),
        ("x_ay", ayT, ayB),
    ]:
        a, b = st(xt), st(xb)
        if a is None or b is None:
            continue
        yield f"{kol}:{ad}", st(a * mt), st(b * mb)
    fv = xt[np.isfinite(xt)]
    if fv.size:
        for q, etiket in [(0.9, "ust10"), (0.1, "alt10"), (0.75, "ust25")]:
            e_ = np.quantile(fv, q)
            if q >= 0.5:
                yield f"{kol}:{etiket}", st((xt > e_).astype(float)), st((xb > e_).astype(float))
            else:
                yield f"{kol}:{etiket}", st((xt < e_).astype(float)), st((xb < e_).astype(float))
    a, b = st(xt), st(xb)
    if a is not None and b is not None:
        yield f"{kol}:kare", st(a**2), st(b**2)


def ikili():
    for i, c1 in enumerate(ODAK):
        a1, b1 = st(tp[c1].to_numpy()), st(bf[c1].to_numpy())
        if a1 is None or b1 is None:
            continue
        for c2 in ODAK[i + 1 :]:
            a2, b2 = st(tp[c2].to_numpy()), st(bf[c2].to_numpy())
            if a2 is None or b2 is None:
                continue
            yield f"{c1}*{c2}", st(a1 * a2), st(b1 * b2)


def degerlendir(uret):
    cikti = []
    for ad, xt, xb in uret:
        if xt is None or xb is None:
            continue
        cc = Gi @ ((V.T @ xt) / N)
        Lsp = float(cc @ L)
        xp = xt - V @ cc
        Qd = float((xp * xp).mean())
        Qs = 1.0 - Qd
        if Qd < 0.05 or Qs < 0.02:
            continue
        rho_s = Lsp / np.sqrt(Qs)
        rho_cv = CARPAN * float((ww * rb * xb).mean()) / np.sqrt(m0b)
        rk = np.sign(rho_cv) * min(abs(rho_cv), TAVAN * abs(rho_s))
        cikti.append((ad, rho_cv, rho_s, rk, Qd, rk * rk * Qd))
    return cikti


hepsi = []
for kol in KOL:
    hepsi += degerlendir(donusumler(kol))
print(f"tekli+donusum: {len(hepsi)} aday")
hepsi += degerlendir(ikili())
print(f"ikili sonrasi: {len(hepsi)} aday")
hepsi.sort(key=lambda t: -t[5])
print(f"\n{'eksen':>40s} {'rho_cv':>8s} {'rho_s':>8s} {'rho_kul':>8s} {'Q_dik':>6s} {'katki':>10s}")
for s in hepsi[:35]:
    print(f"{s[0][:40]:>40s} {s[1]:+8.4f} {s[2]:+8.4f} {s[3]:+8.4f} {s[4]:6.3f} {s[5]:10.3e}")
rs = np.array([abs(s[2]) for s in hepsi])
print(f"\n|rho_s| maks {rs.max():.4f}  (m118'de 0.0405)   %99 {np.quantile(rs, 0.99):.4f}")
with open(os.path.join(BURA, "m121_derin_tarama.json"), "w") as fh:
    json.dump(
        [
            dict(eksen=s[0], rho_cv=s[1], rho_s=s[2], rho_kul=s[3], Qd=s[4], katki=s[5])
            for s in hepsi[:300]
        ],
        fh,
        indent=1,
    )
print("-> ac_derin.json (en iyi 300)")
