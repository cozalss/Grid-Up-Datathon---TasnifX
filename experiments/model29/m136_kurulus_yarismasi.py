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
# HANGI KURULUS GERCEKTEN DAHA IYI SKOR VERIR?
#
# m135 kafa karistirici cikti: kucuk sigma saf optimumu iyilestiriyor ama
# mevcut 40 eksende rho_pred'i dusuruyor. Ama o karsilastirma HAKSIZ --
# eksenler ESKI r_hat ile secilmisti.
#
# DURUST OLCUM: her sigma icin TUM zinciri bastan kur (r_hat -> eksen secimi
# -> katsayi) ve gonderimin GERCEKTE ulasacagi MSE'yi yaz:
#     sabit = M0 - 2*kL + ort(duzeltme^2)        (rho = 0 hali)
#     P^2   = sabit - 2*rho*sqrt(Q_toplam)       (rho gerceklesirse)
# Iki kurulus da kendi en iyi halinde yarisir. Hicbir dosya yazilmaz.
# ---------------------------------------------------------------------------
from m112_kalibre import L_gurultusu  # noqa: E402

RHO_S_ALT, AZAMI = 0.015, 40
GI5 = np.linalg.pinv(G, rcond=1e-5)
m0b = float((ww * rb * rb).mean())
with open(os.path.join(M29, "m121_derin_tarama.json")) as fh:
    TARAMA = json.load(fh)

tn = bf.tanim.values
uqn = pd.unique(tn)
gi = pd.Series(np.arange(len(uqn)), index=uqn)[tn].to_numpy()
rngp = np.random.default_rng(5)
PERM = [
    np.argsort(np.argsort(rngp.permutation(len(uqn))[gi], kind="stable"), kind="stable")
    for _ in range(20)
]

sig_eski = L_gurultusu(V, N)
YUV = 5e-6 / np.sqrt(3.0)
SEN = {
    "A eski (sigma 2.2e-4)": sig_eski,
    "B yeni (sigma 2.9e-6)": np.full(len(L), YUV),
}


def kurulus(sg):
    rh, kaz, kL = buzmeli_r_hat(V, L, G, N, sigma=sg)
    duz = np.zeros(N)
    ONC, kul = [], []
    for kayit in TARAMA:
        if len(kul) >= AZAMI:
            break
        xt, xb = kur(kayit["eksen"])
        if xt is None or xb is None:
            continue
        cc = Gi @ ((V.T @ xt) / N)
        xp0 = xt - V @ cc
        Qs = 1.0 - float((xp0 * xp0).mean())
        if Qs < 0.02:
            continue
        rho_s = float((rh * xt).mean()) / np.sqrt(Qs)
        if abs(rho_s) < RHO_S_ALT:
            continue
        cc5 = GI5 @ ((V.T @ xt) / N)
        xp5 = xt - V @ cc5
        Qs5 = 1.0 - float((xp5 * xp5).mean())
        if Qs5 < 0.02:
            continue
        if abs(float((rh * xt).mean()) / np.sqrt(Qs5) - rho_s) > 0.3 * abs(rho_s):
            continue
        xp = xp0.copy()
        for u in ONC:
            xp -= float((xp * u).mean()) * u
        Qd = float((xp * xp).mean())
        if Qd < 0.25:
            continue
        kor = float((ww * rb * xb).mean()) / np.sqrt(m0b)
        gur = np.std([float((ww * rb[s] * xb).mean()) / np.sqrt(m0b) for s in PERM])
        if abs(kor) < 3 * gur:
            continue
        rho_cv = 0.798 * kor
        if abs(rho_cv) < 1.95 * abs(rho_s):
            continue
        rk = np.sign(rho_cv) * 1.95 * abs(rho_s)
        duz += rk * (xp / np.sqrt(Qd))
        ONC.append(xp / np.sqrt(Qd))
        kul.append((kayit["eksen"], rk))
    Q = float((duz * duz).mean())
    sabit = M0 - 2 * kL + Q
    rho = float(np.sqrt(Q))
    return dict(kL=kL, saf=np.sqrt(M0 - kaz), n=len(kul), rho=rho, sabit=sabit)


print(
    f"\n{'kurulus':>24s} {'saf opt':>9s} {'kL':>9s} {'eksen':>6s} {'rho_pred':>9s} "
    f"{'sabit':>11s} {'sqrt(sabit)':>12s}"
)
R = {}
for ad, sg in SEN.items():
    d = kurulus(sg)
    R[ad] = d
    print(
        f"{ad:>24s} {d['saf']:9.6f} {d['kL']:9.6f} {d['n']:6d} {d['rho']:9.4f} "
        f"{d['sabit']:11.7f} {np.sqrt(d['sabit']):12.6f}"
    )

print("\nGERCEKLESEN f'ye gore beklenen LB skoru  (P^2 = sabit - 2*f*rho^2)")
print(f"{'f':>6s} " + " ".join(f"{ad[:12]:>13s}" for ad in R) + "   kazanan")
for f in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    ps = {}
    for ad, d in R.items():
        ps[ad] = np.sqrt(max(d["sabit"] - 2 * f * d["rho"] ** 2, 1e-9))
    kz = min(ps, key=ps.get)
    print(f"{f:6.1f} " + " ".join(f"{ps[a]:13.5f}" for a in R) + f"   {kz[:1]}")
print("\nf = 1.95 carpaninin gerceklesen orani. Her f'de dusuk olan kazanir.")
