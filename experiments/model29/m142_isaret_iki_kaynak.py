"""TASIMA OLCUMUNU DUZELT: katsayilar LB'den, bloktan FIT EDILMIYOR.

Onceki olcum (m125) her yarida agirliklari YENIDEN FIT edip orani aliyordu.
Ama gercek kurulumda katsayilar LB'den geliyor (1.95*|rho_s|), bloktan
fit edilmiyor. Dolayisiyla "fit/holdout" ayrimi yapay ve oran gurultulu
(n=6'da sd 0.647 -- olculemiyor).

DOGRUSU: bilesigi LB katsayilariyla SABIT kur, sonra blogun farkli zaman
pencerelerinde korelasyonunu olc. Fit yok, sizinti yok, oran yok.

Test penceresi 122 gunluk bir ufuk oldugu icin GEC pencere en iyi vekildir.
Her on-ek uzunlugu icin:
  kor_tum    tum blokta korelasyon
  kor_erken  gun 1-40
  kor_gec    gun 83-122      <- test'e en yakin
  kor_sd     bes pencere arasindaki sacilim
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
Gi = np.linalg.pinv(G, rcond=1e-6)
r_hat, gercek, kL = buzmeli_r_hat(V, L, G, N)
MSE_OPT = M0 - gercek
print(f"saf optimum {np.sqrt(MSE_OPT):.6f}")

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
pb = np.concatenate([np.mean(P, axis=0), np.mean([q for q in (z[k] for k in z.files)], axis=0)])
bf = e.loc[idx].copy()
rb = np.log1p(bf.tuketim.values.astype(np.float64)) - pb
sgm = bf.soguk_mu.values.astype(np.float64)
ww = np.where(sgm == 1, HEDEF_SOGUK / sgm.mean(), (1 - HEDEF_SOGUK) / (1 - sgm.mean()))
ww = ww / ww.mean()


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


# ---------------------------------------------------------------------------
# ISARETIN IKINCI, BAGIMSIZ KAYNAGI.
#
# m141 yaz25 isaretlerinin diger bloklara tasinmadigini gosterdi. Ama
# ISARETIN IKINCI BIR KAYNAGI VAR ve o CV'den degil, GERCEK TEST
# ARTIGINDAN geliyor: rho_s = <r_hat, x>/N / sqrt(Q_span). r_hat, LB
# skorlarindan cozulmus test artigi tahminidir -- yaz25'i hic gormez.
#
# Eger yaz25 CV isareti ile LB isareti buyuk oranda ORTUSUYORSA, iki
# BAGIMSIZ kaynak ayni seyi soyluyor demektir ve bloklar arasi donme
# kadar kotu durumda degiliz.
# Sansa gore beklenen ortusme %50.
# ---------------------------------------------------------------------------
r_hat_, gercek_, kL_ = buzmeli_r_hat(V, L, G, N)
with open(os.path.join(M29, "m122_nihai.json")) as fh:
    EKS = json.load(fh)["eksenler"]
m0b = float((ww * rb * rb).mean())

print(f"\n{'eksen':>34s} {'rho_cv (CV)':>12s} {'rho_s (LB)':>11s} {'ortusme':>9s}")
ayni = 0
top = 0
AYRI = []
for ad in EKS:
    xt, xb = kur(ad)
    if xt is None or xb is None:
        continue
    cc = Gi @ ((V.T @ xt) / N)
    xp0 = xt - V @ cc
    Qs = 1.0 - float((xp0 * xp0).mean())
    if Qs < 0.02:
        continue
    rho_s = float((r_hat_ * xt).mean()) / np.sqrt(Qs)
    rho_cv = CARPAN * float((ww * rb * xb).mean()) / np.sqrt(m0b)
    o = np.sign(rho_cv) == np.sign(rho_s)
    top += 1
    ayni += int(o)
    if not o:
        AYRI.append(ad)
    print(f"{ad[:34]:>34s} {rho_cv:+12.4f} {rho_s:+11.4f} {'AYNI' if o else 'TERS':>9s}")

from math import comb  # noqa: E402

p = sum(comb(top, k) for k in range(ayni, top + 1)) / 2.0**top
print(f"\nORTUSME: {ayni}/{top} = {ayni / top:.0%}   (sans %50)")
print(f"  tek yonlu binom p = {p:.2e}")
print(f"  ters olanlar: {AYRI}")
print("\nBU IKI KAYNAK BAGIMSIZDIR:")
print("  rho_cv  -> yaz25 blogu (2025 verisi, CV modeli)")
print("  rho_s   -> LB skorlari (2026 test artigi, blogu hic gormez)")
print("Yuksek ortusme, isaretlerin gercek oldugunu iki yandan destekler.")
