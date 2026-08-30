"""KATSAYI FORMULU: sqrt(Q_dik) CARPANI OLMALI MI?

seviye kalibrasyonu IKI BIRIM YON arasindaydi:
    rho_s = L_span/sqrt(Q_span) = +0.0156   (span birim yonu)
    rho_u = L_dik /sqrt(Q_dik)  = -0.0304   (dik birim yonu)
    oran 1.95
Yani "1.95*|rho_s|" DIK BIRIM YONDEKI korelasyonun tahminidir ve u
yonundeki katsayi dogrudan o olmalidir.

m122 ise  katsayi = 1.95*|rho_s| * sqrt(Q_dik)  koyuyor. Bu, 1.95*|rho_s|'i
TUM eksenin korelasyonu (rho_x) sayip izotropiyle dik parcaya dagitmaya
denk gelir. seviye'de rho_x/rho_s aslinda 0.99, 1.95 degil.

Iki formul BILESIGIN YONUNU degistirir (kappa olcegi ayri secildigi icin
yalnizca goreli agirliklar onemli). Hangisi dogru: TARTISMA DEGIL OLCUM.

SINAV: her formulun urettigi bilesik yon icin, yaz25'te zaman-bolmeli
tutma ve blok korelasyonu. Yuksek olan kazanir.
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
Gi = np.linalg.pinv(G, rcond=1e-6)
r_hat, gercek, kL = buzmeli_r_hat(V, L, G, N)
MSE_OPT = M0 - gercek

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

# eksenleri bir kez kur: test tarafinda ardisik dik birim yonler, blok tarafinda ham
UT, UB, RHO_KUL, QD = [], [], [], []
ONC = []
for ad in EKSENLER:
    xt, xb = kur(ad)
    if xt is None:
        continue
    cc = Gi @ ((V.T @ xt) / N)
    Lsp = float(cc @ L)
    xp0 = xt - V @ cc
    Qs = 1.0 - float((xp0 * xp0).mean())
    rho_s = Lsp / np.sqrt(Qs)
    xp = xp0.copy()
    for u in ONC:
        xp -= float((xp * u).mean()) * u
    Qd = float((xp * xp).mean())
    kor = float((ww * rb * xb).mean()) / np.sqrt(m0b)
    UT.append(xp / np.sqrt(Qd))
    UB.append(xb)
    RHO_KUL.append(np.sign(CARPAN * kor) * TAVAN * abs(rho_s))
    QD.append(Qd)
    ONC.append(xp / np.sqrt(Qd))
print(f"{len(UT)} eksen kuruldu")

# --- iki formul ---
FORMUL = {
    "A (kod): rho_kul*sqrt(Qd)": [rk * np.sqrt(q) for rk, q in zip(RHO_KUL, QD)],
    "B (kalibrasyon): rho_kul": list(RHO_KUL),
}

uf = bf.ufuk_gun.to_numpy()
ZAMAN = [40, 50, 61, 72, 85]
print(f"\n{'formul':>28s} {'rho_pred':>9s} {'blok kor':>9s} {'zaman tut':>10s} {'2.sira f':>9s}")
for ad, betalar in FORMUL.items():
    # test tarafinda bilesik yon
    duz = np.zeros(N)
    for b, u in zip(betalar, UT):
        duz += b * u
    rho = float(np.sqrt(float((duz * duz).mean())))
    # blok tarafinda AYNI goreli agirliklarla yon kur ve korelasyonu olc
    duzB = np.zeros(len(rb))
    ONCB = []
    for b, xb in zip(betalar, UB):
        x = xb.copy()
        for u in ONCB:
            x -= float((ww * x * u).mean()) / float((ww * u * u).mean()) * u
        n = np.sqrt(float((ww * x * x).mean()))
        if n < 0.15:
            continue
        x /= n
        duzB += b * x
        ONCB.append(x)
    nB = np.sqrt(float((ww * duzB * duzB).mean()))
    korB = float((ww * rb * (duzB / nB)).mean()) / np.sqrt(m0b) if nB > 1e-12 else 0.0
    # zaman-bolmeli tutma: AGIRLIKLAR SABIT (LB'den), yalniz korelasyon olculur
    tut = []
    for kes in ZAMAN:
        m1, m2 = uf <= kes, uf > kes
        for f_, o_ in ((m1, m2), (m2, m1)):
            n1 = np.sqrt(float((ww[f_] * duzB[f_] ** 2).mean()))
            n2 = np.sqrt(float((ww[o_] * duzB[o_] ** 2).mean()))
            if n1 < 1e-12 or n2 < 1e-12:
                continue
            k1 = float((ww[f_] * rb[f_] * (duzB[f_] / n1)).mean()) / np.sqrt(
                float((ww[f_] * rb[f_] ** 2).mean())
            )
            k2 = float((ww[o_] * rb[o_] * (duzB[o_] / n2)).mean()) / np.sqrt(
                float((ww[o_] * rb[o_] ** 2).mean())
            )
            if abs(k1) > 1e-12:
                tut.append(k2 / k1)
    tz = float(np.median(tut)) if tut else 0.0
    kap = np.sqrt(max(MSE_OPT - 0.99790**2, 1e-12))
    print(f"{ad:>28s} {rho:9.4f} {korB:9.4f} {tz:10.3f} {kap / rho:9.3f}")

print("\nNOT: kappa ayri secildigi icin rho_pred'in OLCEGI dosyayi degistirmez;")
print("dosyayi degistiren GORELI agirliklardir. 'blok kor' o yonun yaz25")
print("artigiyla korelasyonu -- yuksek olan daha iyi yon demektir.")
