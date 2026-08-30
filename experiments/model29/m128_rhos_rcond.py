"""SEKIZINCI HATA ARAYISI -- rho_s'in rcond duyarliligi.

Bilesigin HER katsayisi 1.95*|rho_s| ile belirleniyor ve
    rho_s = L_span / sqrt(Q_span),   L_span = c'L,  c = pinv(G, rcond) (V'x/N)
Yani rcond secimi dogrudan katsayilari belirliyor. Bugun r_hat'in rcond'a
duyarli oldugu bulunmustu (rank 22->23 gecisinde nrm 0.003941 -> 0.004790).
rho_s ayni pinv'i kullaniyor -- ayni kirilganlik var mi?

Ayrica plasebo kapisi 20 permutasyonla kuruluyor; sd tahmininin kendi hatasi
~%16. Sinirdaki eksenler sansa gore gecip kalabilir. Marjinal olanlari
100 permutasyonla yeniden sinariz.
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
from m112_kalibre import EK_MODEL, M0  # noqa: E402

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
sv = np.linalg.svd(G, compute_uv=False)
print(f"span {V.shape[1]} yon, tekil degerler: en buyuk {sv[0]:.3e} en kucuk {sv[-1]:.3e}")
print("  " + " ".join(f"{s:.1e}" for s in sv[-6:]))

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
RC = [1e-4, 1e-5, 1e-6, 1e-8, 1e-10]
GI = {rc: np.linalg.pinv(G, rcond=rc) for rc in RC}
print(
    f"\n{'eksen':>34s} "
    + " ".join(f"{f'rho_s {rc:.0e}':>13s}" for rc in RC)
    + f" {'sacilim':>9s} {'karar':>8s}"
)
kotu = []
for ad in EKSENLER:
    xt, _ = kur(ad)
    if xt is None:
        continue
    vals = []
    for rc in RC:
        c = GI[rc] @ ((V.T @ xt) / N)
        Lsp = float(c @ L)
        xp = xt - V @ c
        Qs = 1.0 - float((xp * xp).mean())
        vals.append(Lsp / np.sqrt(Qs) if Qs > 1e-9 else np.nan)
    vals = np.array(vals)
    ref = vals[2]  # 1e-6
    sac = float(np.nanmax(np.abs(vals - ref)))
    # katsayi 1.95*|rho_s| oldugu icin sacilim katsayida 1.95 kat buyur
    kritik = sac > 0.3 * abs(ref) or np.ptp(np.sign(vals)) > 0
    if kritik:
        kotu.append(ad)
    print(
        f"{ad[:34]:>34s} "
        + " ".join(f"{v:+13.5f}" for v in vals)
        + f" {sac:9.2e} {'KIRILGAN' if kritik else 'kararli':>8s}"
    )

print(f"\n{len(EKSENLER)} eksenin {len(kotu)} tanesi rcond'a KIRILGAN")
if kotu:
    print("  ", kotu)
    print("  -> bu eksenlerin katsayisi regularizasyon secimine bagli, guvenilmez")
else:
    print("  -> rho_s butun eksenlerde rcond'a KARARLI; sekizinci hata burada degil")
