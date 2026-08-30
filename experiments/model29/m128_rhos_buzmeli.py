"""rho_s'i BUZMELI tahminle kur ve kararliligini olc.

SORUN. rho_s = c'L / sqrt(Q_span),  c = pinv(G, rcond)(V'x/N).
G'nin tekil degerleri ... 3.9e-06, 5.3e-07, 1.9e-12 ... ve rcond=1e-6
kesimi (6.6e-07) tam aralarina dusuyor. c o neredeyse-tekil kipe buyuk
katsayi verince L'nin gurultusu buyuyor. 40 eksenin 12'si kirilgan;
t_yuk_faktoru'nde rho_s 1e-4'te -0.004, 1e-6'da -0.020 (5 KAT).

COZUM. L_span = <r_hat, x>/N. r_hat zaten kip basina optimal buzmeyle
kurulmus, gurultu-farkindalikli tahmindir (m112.buzmeli_r_hat); tekil
kipleri kendiliginden oldurur. Geometri (x_perp) icin pinv kullanmaya
devam edilir -- orada gurultu yok, yalnizca izdusum var.

    rho_s = (r_hat . x)/N / sqrt(Q_span)
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
TABAN = "tuketim_m6_ikiyon.csv"
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
r_hat, gercek, kL = buzmeli_r_hat(V, L, G, N)
print(f"span {V.shape[1]} yon   saf optimum {np.sqrt(M0 - gercek):.6f}")

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
pb = np.concatenate([np.mean(P, axis=0), np.mean([z[k] for k in z.files], axis=0)])
bf = e.loc[idx].copy()


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
sgm = bf.soguk_mu.values.astype(np.float64)
ufT, ufB = st(tp.ufuk_gun.to_numpy()), st(bf.ufuk_gun.to_numpy())
ayT = st(pd.to_datetime(tp.tarih).dt.month.to_numpy().astype(np.float64))
ayB = st(pd.to_datetime(bf.tarih).dt.month.to_numpy().astype(np.float64))
CARP = {"x_sv": (svT, svB), "x_soguk": (sgT, sgm), "x_ufuk": (ufT, ufB), "x_ay": (ayT, ayB)}
ESIK = {"ust10": (0.9, True), "ust25": (0.75, True), "alt10": (0.1, False)}


def kur(ad):
    if "*" in ad:
        k1, k2 = ad.split("*", 1)
        if k1 not in tp.columns or k2 not in tp.columns:
            return None, None
        a1, b1 = st(tp[k1].to_numpy()), st(bf[k1].to_numpy())
        a2, b2 = st(tp[k2].to_numpy()), st(bf[k2].to_numpy())
        if a1 is None or a2 is None or b1 is None or b2 is None:
            return None, None
        return st(a1 * a2), st(b1 * b2)
    kol, kip = ad.split(":", 1) if ":" in ad else (ad, "")
    if kol not in tp.columns or kol not in bf.columns:
        return None, None
    xt, xb = tp[kol].to_numpy(), bf[kol].to_numpy()
    if kip in CARP:
        mt, mb = CARP[kip]
        a_, b_ = st(xt), st(xb)
        return (None, None) if a_ is None or b_ is None else (st(a_ * mt), st(b_ * mb))
    if kip in ESIK:
        q, ust = ESIK[kip]
        fv = xt[np.isfinite(xt)]
        if fv.size == 0:
            return None, None
        v_ = np.quantile(fv, q)
        if ust:
            return st((xt > v_).astype(np.float64)), st((xb > v_).astype(np.float64))
        return st((xt < v_).astype(np.float64)), st((xb < v_).astype(np.float64))
    if kip == "kare":
        a_, b_ = st(xt), st(xb)
        return (None, None) if a_ is None else (st(a_**2), st(b_**2))
    return st(xt), st(xb)


with open(os.path.join(M29, "m122_nihai.json")) as fh:
    EKSENLER = json.load(fh)["eksenler"]
RC = [1e-4, 1e-5, 1e-6, 1e-8, 1e-10]
print(f"\n{'eksen':>30s} {'ESKI (c L)':>32s} {'YENI (<r_hat,x>)':>32s}")
print(
    f"{'':>30s} "
    + " ".join(f"{f'{rc:.0e}':>10s}" for rc in RC[:3])
    + "  "
    + " ".join(f"{f'{rc:.0e}':>10s}" for rc in RC[:3])
)
kotu_e, kotu_y = [], []
for ad in EKSENLER:
    xt, _ = kur(ad)
    if xt is None:
        continue
    eski, yeni = [], []
    for rc in RC:
        Gi = np.linalg.pinv(G, rcond=rc)
        c = Gi @ ((V.T @ xt) / N)
        xp = xt - V @ c
        Qs = 1.0 - float((xp * xp).mean())
        if Qs <= 1e-9:
            eski.append(np.nan)
            yeni.append(np.nan)
            continue
        eski.append(float(c @ L) / np.sqrt(Qs))
        yeni.append(float((r_hat * xt).mean()) / np.sqrt(Qs))
    eski, yeni = np.array(eski), np.array(yeni)
    se = float(np.nanmax(np.abs(eski - eski[2])))
    sy = float(np.nanmax(np.abs(yeni - yeni[2])))
    if se > 0.3 * abs(eski[2]) or np.ptp(np.sign(eski)) > 0:
        kotu_e.append(ad)
    if sy > 0.3 * abs(yeni[2]) or np.ptp(np.sign(yeni)) > 0:
        kotu_y.append(ad)
    print(
        f"{ad[:30]:>30s} "
        + " ".join(f"{v:+10.5f}" for v in eski[:3])
        + "  "
        + " ".join(f"{v:+10.5f}" for v in yeni[:3])
    )

print(
    f"\nKIRILGAN eksen sayisi:  ESKI {len(kotu_e)}/{len(EKSENLER)}   "
    f"YENI {len(kotu_y)}/{len(EKSENLER)}"
)
if kotu_y:
    print("  yeni yontemde hala kirilgan:", kotu_y)
else:
    print("  -> yeni yontem butun eksenlerde rcond'a KARARLI")
