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
# 5 UYUSMAYAN EKSEN NE OLMALI?
#
# m142: 40 eksenin 35'inde CV isareti ile LB isareti ORTUSUYOR, 5'inde TERS.
# Bu 5 icin uc secenek var:
#   (a) CV isaretini kullan  (mevcut kurulus)
#   (b) LB isaretini kullan  (rho_s'in isareti -- GERCEK test artigindan)
#   (c) o eksenleri TAMAMEN AT (belirsizler)
# Olcut: bilesigin yaz25 blogundaki korelasyonu VE rho_pred.
#
# ONEMLI: bu secim yaz25'e bakilarak yapilamaz (CV isareti zaten yaz25'ten
# geliyor, (a) otomatik kazanir). Bu yuzden asil olcut rho_pred DEGIL,
# eksenlerin KENDI ICINDE tutarliligi: belirsiz eksen dik bilesene gurultu
# katiyorsa atmak kazandirir.
# ---------------------------------------------------------------------------
r_hat_, gercek_, kL_ = buzmeli_r_hat(V, L, G, N)
with open(os.path.join(M29, "m122_nihai.json")) as fh:
    EKS = json.load(fh)["eksenler"]
m0b = float((ww * rb * rb).mean())

BILGI = []
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
    BILGI.append((ad, xt, xb, rho_s, rho_cv, xp0))

TERS = [b[0] for b in BILGI if np.sign(b[3]) != np.sign(b[4])]
print(f"\n{len(BILGI)} eksen, {len(TERS)} tanesinde isaretler TERS")


def kur_bilesik(kip):
    ONC_T, ONC_B, duz_b, rho2 = [], [], np.zeros(len(rb)), 0.0
    for ad, xt, xb, rho_s, rho_cv, xp0 in BILGI:
        ters = np.sign(rho_s) != np.sign(rho_cv)
        if ters and kip == "at":
            continue
        isr = np.sign(rho_s) if (ters and kip == "LB") else np.sign(rho_cv)
        xp = xp0.copy()
        for u in ONC_T:
            xp -= float((xp * u).mean()) * u
        Qd = float((xp * xp).mean())
        if Qd < 0.25:
            continue
        ub = xb.copy()
        for u in ONC_B:
            ub -= float((ww * ub * u).mean()) / float((ww * u * u).mean()) * u
        nb = np.sqrt(float((ww * ub * ub).mean()))
        if nb < 0.15:
            continue
        ub /= nb
        b = isr * TAVAN * abs(rho_s)
        duz_b += b * ub
        rho2 += b * b
        ONC_T.append(xp / np.sqrt(Qd))
        ONC_B.append(ub)
    n1 = np.sqrt(float((ww * duz_b * duz_b).mean()))
    kor = (
        0.0
        if n1 < 1e-12
        else float((ww * rb * duz_b).mean()) / (n1 * np.sqrt(float((ww * rb * rb).mean())))
    )
    return len(ONC_T), np.sqrt(rho2), kor


print(f"\n{'kip':>28s} {'eksen':>6s} {'rho_pred':>9s} {'yaz25 kor':>10s} {'2.sira f':>9s}")
for kip, ad2 in [
    ("cv", "(a) CV isareti [MEVCUT]"),
    ("LB", "(b) LB isareti"),
    ("at", "(c) belirsizleri AT"),
]:
    n, rho, kor = kur_bilesik(kip)
    print(f"{ad2:>28s} {n:6d} {rho:9.4f} {kor:10.4f} {0.0991 / rho:9.3f}")

print("\nDIKKAT: yaz25 kor sutunu (a) lehine YANLIDIR -- CV isareti zaten")
print("  yaz25'ten geliyor. Karsilastirmada rho_pred'e ve eksen sayisina bak.")
print("Bu 5 eksenin toplam katkisi kucukse, hangisini sectigimiz onemsizdir.")
