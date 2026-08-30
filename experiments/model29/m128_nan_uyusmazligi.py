"""SEKIZINCI HATA ARAYISI -- NaN doldurmasinin blok/test uyusmazligi.

t_* oznitelikleri SOGUK trafolarda NaN. Soguk pay: test %22.2, yaz25 %7.5.
st() NaN'lari MEDYANLA dolduruyor, yani test yonunun ~%25'i sabit bir blok,
blok yonunun ~%8'i. Iki farkli sekil.

Agirlik ww metrikte soguk payini %22.2'ye cekiyor AMA yonun kendi
standartlastirmasi (merkezleme + norm) AGIRLIKSIZ yapiliyor. Bu tutarsizlik
CV korelasyonunun -- yani ISARETIN -- kaymasina yol acabilir. Isaret
CV'den geldigi icin bu dogrudan bilesigin yonunu bozar.

SINAV: her ekseni iki turlu standartlastir ve blok korelasyonunu karsilastir.
  A  agirliksiz merkezleme/norm  (mevcut kod)
  B  ww-agirlikli merkezleme/norm (metrikle tutarli)
ISARET DEGISEN eksen varsa gercek bir hatadir.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = os.path.join(KOK, "data/interim/deney")
AO = os.path.join(KOK, "data/interim/aile_onbellek")
M29 = os.path.join(KOK, "experiments/model29")
HEDEF_SOGUK, CARPAN = 0.222, 0.798
sys.path.insert(0, M29)

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
tp = pd.read_parquet(os.path.join(DN, "test.parquet"))


def st_a(x, w=None):
    """MEVCUT: agirliksiz merkezleme ve norm, NaN -> medyan."""
    x = np.asarray(x, dtype=np.float64).copy()
    f = np.isfinite(x)
    if not f.any():
        return None
    x[~f] = np.median(x[f])
    x -= x.mean()
    s = np.sqrt(float((x * x).mean()))
    return x / s if s > 1e-12 else None


def st_b(x, w):
    """AGIRLIKLI: ww ile merkezleme ve norm, NaN -> AGIRLIKLI medyan."""
    x = np.asarray(x, dtype=np.float64).copy()
    f = np.isfinite(x)
    if not f.any():
        return None
    # agirlikli medyan
    xs = x[f]
    wsi = w[f]
    o = np.argsort(xs)
    c = np.cumsum(wsi[o]) / wsi.sum()
    med = xs[o][np.searchsorted(c, 0.5)]
    x[~f] = med
    mu = float((w * x).mean()) / float(w.mean())
    x -= mu
    s = np.sqrt(float((w * x * x).mean()) / float(w.mean()))
    return x / s if s > 1e-12 else None


svB_a, svB_b = st_a(pb), st_b(pb, ww)
ufB_a, ufB_b = st_a(bf.ufuk_gun.to_numpy()), st_b(bf.ufuk_gun.to_numpy(), ww)
ayv = pd.to_datetime(bf.tarih).dt.month.to_numpy().astype(np.float64)
ayB_a, ayB_b = st_a(ayv), st_b(ayv, ww)
CARP_A = {"x_sv": svB_a, "x_soguk": sgm, "x_ufuk": ufB_a, "x_ay": ayB_a}
CARP_B = {"x_sv": svB_b, "x_soguk": sgm, "x_ufuk": ufB_b, "x_ay": ayB_b}
ESIK = {"ust10": (0.9, True), "ust25": (0.75, True), "alt10": (0.1, False)}


def kurB(ad, mod):
    st_ = st_a if mod == "A" else (lambda x: st_b(x, ww))
    CARP = CARP_A if mod == "A" else CARP_B
    if "*" in ad:
        k1, k2 = ad.split("*", 1)
        if k1 not in bf.columns or k2 not in bf.columns:
            return None
        b1, b2 = st_(bf[k1].to_numpy()), st_(bf[k2].to_numpy())
        return None if b1 is None or b2 is None else st_(b1 * b2)
    kol, kip = ad.split(":", 1) if ":" in ad else (ad, "")
    if kol not in bf.columns:
        return None
    xb = bf[kol].to_numpy()
    if kip in CARP:
        b_ = st_(xb)
        return None if b_ is None else st_(b_ * CARP[kip])
    if kip in ESIK:
        q, ust = ESIK[kip]
        xt = tp[kol].to_numpy()
        fv = xt[np.isfinite(xt)]
        if fv.size == 0:
            return None
        v_ = np.quantile(fv, q)
        return st_(((xb > v_) if ust else (xb < v_)).astype(np.float64))
    if kip == "kare":
        b_ = st_(xb)
        return None if b_ is None else st_(b_**2)
    return st_(xb)


with open(os.path.join(M29, "m122_nihai.json")) as fh:
    EKSENLER = json.load(fh)["eksenler"]

print(f"{'eksen':>34s} {'kor A':>9s} {'kor B':>9s} {'isaret':>8s} {'oran':>7s}")
ters, hepsi = [], []
for ad in EKSENLER:
    xa, xb_ = kurB(ad, "A"), kurB(ad, "B")
    if xa is None or xb_ is None:
        continue
    ka = float((ww * rb * xa).mean()) / np.sqrt(m0b)
    kb = float((ww * rb * xb_).mean()) / np.sqrt(m0b)
    uy = np.sign(ka) == np.sign(kb)
    if not uy:
        ters.append(ad)
    hepsi.append((ad, ka, kb))
    print(
        f"{ad[:34]:>34s} {ka:+9.4f} {kb:+9.4f} {'UYDU' if uy else 'TERS':>8s} "
        f"{kb / ka if abs(ka) > 1e-12 else float('nan'):7.2f}"
    )

print(f"\nISARET UYUMU: {len(hepsi) - len(ters)}/{len(hepsi)}")
if ters:
    print("TERS DONENLER:", ters)
    print("-> bu eksenlerin CV isareti standartlastirma seciminden etkileniyor,")
    print("   yani guvenilmez. Bilesikten cikarilmalari degerlendirilmeli.")
else:
    print("-> hicbir eksenin isareti degismiyor. NaN/agirlik uyusmazligi")
    print("   bilesigin YONUNU bozmuyor; SEKIZINCI HATA BURADA DEGIL.")
oranlar = np.array([b / a for _, a, b in hepsi if abs(a) > 1e-12])
print(
    f"buyukluk orani (B/A): medyan {np.median(oranlar):.3f}  "
    f"%5-%95 {np.quantile(oranlar, 0.05):.3f}-{np.quantile(oranlar, 0.95):.3f}"
)
