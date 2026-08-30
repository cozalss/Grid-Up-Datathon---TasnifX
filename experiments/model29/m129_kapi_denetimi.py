"""DOKUZUNCU HATA ARAYISI -- iki kapinin kendisi saglam mi?

(a) PLASEBO KAPISI 20 permutasyonla kuruluyor. sd tahmininin kendi hatasi
    ~1/sqrt(2*19) = %16. |z|>=3 esiginin hemen ustundeki eksenler sansa gore
    gecmis olabilir. 100 permutasyonla yeniden sinanir.

(b) |rho_s| >= 0.015 ESIGI eski tahminleyicinin sigma'sina (3e-4) dayaniyordu.
    Yeni tahminleyici rho_s = <r_hat,x>/N / sqrt(Q_span); gurultusu r_hat'in
    gurultusundan geliyor. Monte Carlo ile olculur: L'yi kendi sigma_L'siyle
    bozup r_hat'i yeniden kur, rho_s'in sacilimina bak.
    Esik, olculen sigma'nin en az 5 katinda olmali.
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
from m112_kalibre import EK_MODEL, M0, L_gurultusu, buzmeli_r_hat  # noqa: E402

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
sigL = L_gurultusu(V, N)
print(f"span {V.shape[1]} yon   ort sigma_L {sigL.mean():.3e}")

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

# --- (a) plasebo 20 vs 100 permutasyon ---
rng = np.random.default_rng(5)
tn = bf.tanim.values
uqn = pd.unique(tn)
gi = pd.Series(np.arange(len(uqn)), index=uqn)[tn].to_numpy()
P20 = [
    np.argsort(np.argsort(rng.permutation(len(uqn))[gi], kind="stable"), kind="stable")
    for _ in range(20)
]
rng2 = np.random.default_rng(101)
P100 = [
    np.argsort(np.argsort(rng2.permutation(len(uqn))[gi], kind="stable"), kind="stable")
    for _ in range(100)
]

# --- (b) rho_s Monte Carlo ---
rng3 = np.random.default_rng(7)
R_HATLAR = []
for _ in range(60):
    Ln = L + rng3.normal(0, sigL)
    rh, _, _ = buzmeli_r_hat(V, Ln, G, N, sigma=sigL)
    R_HATLAR.append(rh)
print(f"rho_s Monte Carlo icin {len(R_HATLAR)} bozulmus r_hat kuruldu\n")

print(
    f"{'eksen':>30s} {'kor':>8s} {'z(20)':>7s} {'z(100)':>7s} {'plasebo':>8s} "
    f"{'rho_s':>8s} {'sigma(rho_s)':>13s} {'SNR':>6s} {'esik':>7s}"
)
plasebo_kotu, esik_kotu = [], []
for ad in EKSENLER:
    xt, xb = kur(ad)
    if xt is None:
        continue
    kor = float((ww * rb * xb).mean()) / np.sqrt(m0b)
    g20 = np.std([float((ww * rb[s] * xb).mean()) / np.sqrt(m0b) for s in P20])
    g100 = np.std([float((ww * rb[s] * xb).mean()) / np.sqrt(m0b) for s in P100])
    z20, z100 = kor / (g20 + 1e-12), kor / (g100 + 1e-12)
    pl = "GECER" if abs(z100) >= 3 else "KALIR"
    if abs(z100) < 3:
        plasebo_kotu.append(ad)
    cc = Gi @ ((V.T @ xt) / N)
    xp = xt - V @ cc
    Qs = 1.0 - float((xp * xp).mean())
    rho_s = float((r_hat * xt).mean()) / np.sqrt(Qs)
    orn = [float((rh * xt).mean()) / np.sqrt(Qs) for rh in R_HATLAR]
    sg = float(np.std(orn))
    snr = abs(rho_s) / (sg + 1e-15)
    ek = "GECER" if snr >= 5 else "ZAYIF"
    if snr < 5:
        esik_kotu.append(ad)
    print(
        f"{ad[:30]:>30s} {kor:+8.4f} {z20:+7.1f} {z100:+7.1f} {pl:>8s} "
        f"{rho_s:+8.4f} {sg:13.3e} {snr:6.1f} {ek:>7s}"
    )

print(f"\n(a) PLASEBO 100 permutasyonda KALAN: {len(plasebo_kotu)}/{len(EKSENLER)}")
if plasebo_kotu:
    print("   ", plasebo_kotu)
print(f"(b) rho_s SNR < 5 olan: {len(esik_kotu)}/{len(EKSENLER)}")
if esik_kotu:
    print("   ", esik_kotu)
