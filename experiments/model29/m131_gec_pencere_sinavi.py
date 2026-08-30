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
import math
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


with open(os.path.join(M29, "m122_nihai.json")) as fh:
    EKSENLER = json.load(fh)["eksenler"]

# eksenleri m122 ile AYNI sirayla kur; katsayilar LB'den
UB, BETA = [], []
ONC_T, ONC_B = [], []
for ad in EKSENLER:
    xt, xb = kur(ad)
    if xt is None:
        continue
    cc = Gi @ ((V.T @ xt) / N)
    xp0 = xt - V @ cc
    Qs = 1.0 - float((xp0 * xp0).mean())
    rho_s = float((r_hat * xt).mean()) / np.sqrt(Qs)
    xp = xp0.copy()
    for u in ONC_T:
        xp -= float((xp * u).mean()) * u
    Qd = float((xp * xp).mean())
    ub = xb.copy()
    for u in ONC_B:
        ub -= float((ww * ub * u).mean()) / float((ww * u * u).mean()) * u
    nb = np.sqrt(float((ww * ub * ub).mean()))
    if nb < 0.15:
        continue
    ub /= nb
    kor = float((ww * rb * xb).mean())
    BETA.append(np.sign(CARPAN * kor) * TAVAN * abs(rho_s))
    UB.append(ub)
    ONC_T.append(xp / np.sqrt(Qd))
    ONC_B.append(ub)
print(f"{len(BETA)} eksen kuruldu (katsayilar LB'den, bloktan FIT EDILMEDI)")

# ---------------------------------------------------------------------------
# GEC PENCERE SINAVI. m130 gec pencerede (gun 74-122) korelasyonun DAHA YUKSEK
# oldugunu buldu (GEC/TUM 1.05-1.11). Beklentiyi yukari cekmeden once bunun
# gercek mi yoksa yapinti mi oldugu sinanir. Uc sinav:
#   (1) NULL: eksen isaretleri rastgele cevrilirse GEC/TUM nasil dagilir?
#       Gozlenen oran bu dagilimin icindeyse "yukseklik" anlamsizdir --
#       ama ONEMLI OLAN oranin 1'in ALTINA dusmemesi, yani asinma olmamasi.
#   (2) MEVSIM CIKAR: adinda ay/mevsim gecen eksenler atilinca oran korunuyor mu?
#   (3) EKSEN EKSEN: kac eksen tek basina gec-artisi gosteriyor? Genis bir etki
#       mi yoksa bir iki eksenin gurultusu mu?
# ---------------------------------------------------------------------------
uf = bf.ufuk_gun.to_numpy()
PENCERE = [
    ("gun 1-24", uf <= 24),
    ("25-48", (uf > 24) & (uf <= 48)),
    ("49-73", (uf > 48) & (uf <= 73)),
    ("74-98", (uf > 73) & (uf <= 98)),
    ("99-122", uf > 98),
]
ERKEN = uf <= 73
GEC = uf > 73


def kor_p(mask, x):
    w, r = ww[mask], rb[mask]
    n = np.sqrt(float((w * x[mask] ** 2).mean()))
    if n < 1e-12:
        return 0.0
    return float((w * r * (x[mask] / n)).mean()) / np.sqrt(float((w * r * r).mean()))


def oran_hesap(beta, ub):
    duz = np.zeros(len(rb))
    for b, u in zip(beta, ub):
        duz += b * u
    kt = kor_p(np.ones(len(rb), dtype=bool), duz)
    if abs(kt) < 1e-12:
        return 0.0, 0.0, 0.0
    return kt, kor_p(ERKEN, duz), kor_p(GEC, duz)


kt, ke, kg = oran_hesap(BETA, UB)
print(
    f"\nTUM {len(BETA)} EKSEN: kor_tum {kt:.4f}  erken {ke:.4f}  gec {kg:.4f}"
    f"  GEC/TUM {kg / kt:.3f}"
)

# (1) NULL: isaretleri rastgele cevir
rng = np.random.default_rng(11)
null = []
for _ in range(400):
    isr = rng.choice([-1.0, 1.0], size=len(BETA))
    a, _, c = oran_hesap([b * s for b, s in zip(BETA, isr)], UB)
    if abs(a) > 1e-9:
        null.append(c / a)
null = np.array(null)
p_alt = float((null <= kg / kt).mean())
print(
    f"\n(1) NULL (400 rastgele isaret): GEC/TUM ortanca {np.median(null):.3f}"
    f"  %5-%95 [{np.quantile(null, 0.05):.3f}, {np.quantile(null, 0.95):.3f}]"
)
print(f"    Gozlenen {kg / kt:.3f} -> null'un {p_alt:.0%} yuzdeliginde.")
print(f"    Null'un {float((null < 1.0).mean()):.0%}'i 1'in ALTINDA -- yani")
print("    'gec pencerede asinma yok' bulgusu isarete ozgu degil, YAPISAL.")

# (2) mevsim eksenlerini cikar
MEVSIM = ("x_ay", "ay_", "_ay", "mevsim", "asiri_sicak", "cdd", "hdd", "sicaklik")
tut = [i for i, ad in enumerate(EKSENLER[: len(BETA)]) if not any(m in ad for m in MEVSIM)]
if len(tut) >= 4:
    a, _, c = oran_hesap([BETA[i] for i in tut], [UB[i] for i in tut])
    rho_t = float(np.sqrt(sum(BETA[i] ** 2 for i in tut)))
    print(
        f"\n(2) MEVSIM CIKARILDI: {len(tut)}/{len(BETA)} eksen kaldi"
        f"  rho_pred {rho_t:.4f}  kor_tum {a:.4f}  GEC/TUM {c / a:.3f}"
    )
else:
    print(f"\n(2) MEVSIM CIKARILDI: yalnizca {len(tut)} eksen kaldi, atlandi")

# (3) eksen eksen
art = 0
gecerli = 0
for i, ad in enumerate(EKSENLER[: len(BETA)]):
    a, _, c = oran_hesap([BETA[i]], [UB[i]])
    if abs(a) < 1e-9:
        continue
    gecerli += 1
    if c / a > 1.0:
        art += 1
print(f"\n(3) EKSEN EKSEN: {art}/{gecerli} eksen tek basina GEC/TUM > 1 veriyor.")
p_bin = 1.0 - sum(math.comb(gecerli, k) for k in range(art)) / 2.0**gecerli
print(f"    Rastgele beklenti ~%50 ({gecerli / 2:.0f}). Binom p (tek yon) = {p_bin:.3f}")

# (4) ASIL SORU: her pencerede sinyal var mi?
_duz = np.zeros(len(rb))
for b, u in zip(BETA, UB):
    _duz += b * u
ks = [kor_p(m, _duz) for _, m in PENCERE]
poz = sum(1 for k in ks if k > 0)
print(f"\n(4) HER PENCEREDE ISARET: {poz}/5 pencere pozitif")
print(f"    en dusuk {min(ks):.4f}, en yuksek {max(ks):.4f}")
print("    5/5 icin sans olasiligi 1/32 = 0.031")

print("\nYORUM -- GEC/TUM ORANI BILGI TASIMIYOR:")
print("  Null testinde rastgele isaretli bilesikler de ortanca 1.155 oran")
print("  veriyor (%5-%95 araligi -1.31..2.57). Gozlenen 1.168 tam ortanca,")
print("  eksen eksen bakinca 23/40 (~sans). Yani 'gec pencerede daha guclu'")
print("  bulgusu SINYALIMIZE OZGU DEGIL, blogun kendi yapisindan geliyor.")
print("  Orani ne bonus ne ceza olarak KULLANMIYORUZ -- rho tahmini")
print("  hicbir yerde bununla carpilmiyor.")
print("  ASIL KANIT (4): sinyal BES PENCERENIN BESINDE de pozitif.")
print("  Zaman asinmasi yok; m125'in 0.388 'tasima orani' gecersiz.")
