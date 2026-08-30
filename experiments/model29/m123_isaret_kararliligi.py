"""ISARET KARARLILIGI -- bahsin can alici noktasi.

14 eksenin BUYUKLUGU LB'den (1.95*|rho_s|), ama ISARETI CV'den geliyor.
Isaretler test doneminde tutmazsa c negatif olur ve duseriz.

Trafo-bolmeli dogrulama (m120) kesitsel kararliligi gosterdi (%90.5 tutma).
Eksik olan ZAMANSAL kararlilik: yaz25'in ilk yarisindaki isaret, ikinci
yarisinda da duruyor mu? Bu, yaz25 -> yaz26 aktariminin en yakin vekili.

Uc sinav:
  A  yaz25 gun 1-61  vs  gun 62-122      (ayni mevsim, ileri zaman)
  B  yaz25 tek gunler vs cift gunler     (zaman etkisi yok -- ust sinir)
  C  bilesigin TAMAMI icin ayni ikisi    (eksenler arasi sadelesme dahil)
"""

import json
import os

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = os.path.join(KOK, "data/interim/deney")
AO = os.path.join(KOK, "data/interim/aile_onbellek")
M29 = os.path.join(KOK, "experiments/model29")
BURA = os.path.dirname(os.path.abspath(__file__))
HEDEF_SOGUK, CARPAN = 0.222, 0.798

with open(os.path.join(M29, "m122_nihai.json")) as fh:
    NIH = json.load(fh)
EKSENLER = NIH["eksenler"]
print(f"{len(EKSENLER)} eksen sinaniyor")

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


def st(x):
    x = np.asarray(x, dtype=np.float64).copy()
    f = np.isfinite(x)
    if not f.any():
        return None
    x[~f] = np.median(x[f])
    x -= x.mean()
    s = np.sqrt(float((x * x).mean()))
    return x / s if s > 1e-12 else None


svB = st(pb)
ufB = st(bf.ufuk_gun.to_numpy())
ayB = st(pd.to_datetime(bf.tarih).dt.month.to_numpy().astype(np.float64))
CARP = {"x_sv": svB, "x_soguk": sgm, "x_ufuk": ufB, "x_ay": ayB}
ESIK = {"ust10": (0.9, True), "ust25": (0.75, True), "alt10": (0.1, False)}


def kurB(ad):
    if "*" in ad:
        k1, k2 = ad.split("*", 1)
        if k1 not in bf.columns or k2 not in bf.columns:
            return None
        b1, b2 = st(bf[k1].to_numpy()), st(bf[k2].to_numpy())
        return None if b1 is None or b2 is None else st(b1 * b2)
    kol, kip = ad.split(":", 1) if ":" in ad else (ad, "")
    if kol not in bf.columns:
        return None
    xb = bf[kol].to_numpy()
    if kip in CARP:
        b_ = st(xb)
        return None if b_ is None else st(b_ * CARP[kip])
    if kip in ESIK:
        q, ust = ESIK[kip]
        fv = xb[np.isfinite(xb)]
        if fv.size == 0:
            return None
        v_ = np.quantile(fv, q)
        return st(((xb > v_) if ust else (xb < v_)).astype(np.float64))
    if kip == "kare":
        b_ = st(xb)
        return None if b_ is None else st(b_**2)
    return st(xb)


X = {a: kurB(a) for a in EKSENLER}
X = {a: v for a, v in X.items() if v is not None}
uf = bf.ufuk_gun.to_numpy()
gun = pd.to_datetime(bf.tarih).dt.dayofyear.to_numpy()
BOLME = {
    "A zaman (gun1-61 / 62-122)": (uf <= 61, uf > 61),
    "B tek/cift gun (zaman etkisi yok)": (gun % 2 == 1, gun % 2 == 0),
}


def kor(mask, x):
    w, r = ww[mask], rb[mask]
    return float((w * r * x[mask]).mean()) / np.sqrt(float((w * r * r).mean()))


for ad_b, (m1, m2) in BOLME.items():
    print(f"\n{'=' * 74}\n{ad_b}   ({int(m1.sum()):,} / {int(m2.sum()):,} satir)\n{'=' * 74}")
    print(f"{'eksen':>34s} {'yari-1':>8s} {'yari-2':>8s} {'isaret':>7s} {'oran':>7s}")
    ayni = 0
    oranlar = []
    for a, x in X.items():
        k1, k2 = kor(m1, x), kor(m2, x)
        uy = np.sign(k1) == np.sign(k2)
        ayni += int(uy)
        if abs(k1) > 1e-9:
            oranlar.append(k2 / k1)
        print(
            f"{a[:34]:>34s} {k1:+8.4f} {k2:+8.4f} {'UYDU' if uy else 'TERS':>7s} "
            f"{k2 / k1 if abs(k1) > 1e-9 else float('nan'):7.2f}"
        )
    print(f"\n  ISARET UYUMU: {ayni}/{len(X)}")
    print(
        f"  buyukluk orani (yari2/yari1): medyan {np.median(oranlar):+.2f}  "
        f"ortalama {np.mean(oranlar):+.2f}"
    )

# --- C: bilesigin tamami ---
print(f"\n{'=' * 74}\nC  BILESIGIN TAMAMI (eksenler arasi sadelesme dahil)\n{'=' * 74}")
for ad_b, (m1, m2) in BOLME.items():
    duz1 = np.zeros(len(rb))
    ONC = []
    for a, x in X.items():
        xx = x.copy()
        for u in ONC:
            xx -= float((ww * xx * u).mean()) / float((ww * u * u).mean()) * u
        n = np.sqrt(float((ww * xx * xx).mean()))
        if n < 0.15:
            continue
        xx /= n
        beta = CARPAN * kor(m1, xx)  # AGIRLIKLAR YARI-1'DEN
        duz1 += beta * xx
        ONC.append(xx)
    n = np.sqrt(float((ww * duz1 * duz1).mean()))
    if n < 1e-12:
        continue
    u1 = duz1 / n
    print(f"  {ad_b}")
    print(f"     yari-1 (fit)  kor = {kor(m1, u1):+.4f}")
    print(f"     yari-2 (sinav) kor = {kor(m2, u1):+.4f}   tutma {kor(m2, u1) / kor(m1, u1):.3f}")
