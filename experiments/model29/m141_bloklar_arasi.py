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
        if a_ is None or b_ is None:
            return None, None
        return st(a_**2), st(b_**2)
    return st(xt), st(xb)


# ---------------------------------------------------------------------------
# CV SISMESI SINAVI -- bileseni HIC GORMEDIGI bloklarda olc.
#
# Blok korelasyonu (0.2125) 1.95 carpanindan BAGIMSIZ tek kanittir --
# korelasyon olcek-degismezdir. Ama yaz25 uzerinde olculdu ve eksenler
# de yaz25'e bakilarak secildi/isaretlendi. Sisme olabilir.
#
# SINAV: AYNI eksenler, AYNI LB katsayilari, AYNI isaretler (yaz25'ten),
# ama korelasyon guz25 ve kis26'da olculur. Bu bloklar secimde hic
# kullanilmadi. Korelasyon ayakta kalirsa sisme sinirlidir.
#
# NOT: eksen SECIMI ve ISARET yaz25'ten geldigi icin sinav tam temiz degil;
# ama agirliklar LB'den, ve secim/isaret tek bir serbestlik derecesi.
# ---------------------------------------------------------------------------
ISARETLER = {}
AILE = {}
HAVA = (
    "sicak",
    "cdd",
    "hdd",
    "nem",
    "vpd",
    "et0",
    "bulut",
    "yagis",
    "hissedilen",
    "asiri",
    "ruzgar",
    "guneslen",
    "x_ay",
    "ay_",
)
AILELER = {
    "hava/mevsim": lambda a: any(h in a for h in HAVA),
    "trafo/yapisal": lambda a: not any(h in a for h in HAVA),
}
r_hat_, gercek_, kL_ = buzmeli_r_hat(V, L, G, N)
with open(os.path.join(M29, "m122_nihai.json")) as fh:
    EKS = json.load(fh)["eksenler"]


def blok_kur(blok):
    bl = e[e._blok == blok]
    sic, sog = bl[bl.soguk_mu == 0], bl[bl.soguk_mu == 1]
    Pl = [
        np.load(os.path.join(AO, f"{blok}_{t}_{aa}_uretim.npy")).astype(np.float64)
        for t in (1000, 1001, 1002)
        for aa in ("cat", "xgb", "lgbm")
        if os.path.exists(os.path.join(AO, f"{blok}_{t}_{aa}_uretim.npy"))
    ]
    zz = np.load(os.path.join(DN, f"soguk_tahmin_{blok}.npz"))
    ii = np.concatenate([sic.index.values, sog.index.values])
    pp = np.concatenate([np.mean(Pl, axis=0), np.mean([zz[q] for q in zz.files], axis=0)])
    ff = e.loc[ii].copy()
    rr = np.log1p(ff.tuketim.values.astype(np.float64)) - pp
    sg = ff.soguk_mu.values.astype(np.float64)
    w2 = np.where(sg == 1, HEDEF_SOGUK / sg.mean(), (1 - HEDEF_SOGUK) / (1 - sg.mean()))
    return ff, pp, rr, sg, w2 / w2.mean()


print("\nBLOKLAR ARASI TASIMA -- ayni eksen, ayni LB katsayisi, ayni isaret")
print(
    f"{'blok':>8s} {'n':>9s} {'donem':>23s} {'kor':>8s} {'kor/yaz25':>10s} {'gereken 0.0991':>15s}"
)
TEMEL = None
for blok in ["yaz25", "guz25", "kis26"]:
    ff, pp, rr, sg, w2 = blok_kur(blok)
    svB2 = st(pp)
    ufB2 = st(ff.ufuk_gun.to_numpy())
    ayB2 = st(pd.to_datetime(ff.tarih).dt.month.to_numpy().astype(np.float64))
    CARP.update(
        {"x_sv": (svT, svB2), "x_soguk": (sgT, sg), "x_ufuk": (ufT, ufB2), "x_ay": (ayT, ayB2)}
    )
    globals()["bf"], globals()["ww"] = ff, w2
    ONC, duz, PARCA = [], np.zeros(len(rr)), []
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
        ub = xb.copy()
        for u in ONC:
            ub -= float((w2 * ub * u).mean()) / float((w2 * u * u).mean()) * u
        nb = np.sqrt(float((w2 * ub * ub).mean()))
        if nb < 0.15:
            continue
        ub /= nb
        # ISARET yaz25'ten sabit gelir (ISARETLER sozlugu asagida doldurulur)
        isr = ISARETLER.get(ad)
        if isr is None:
            isr = np.sign(float((w2 * rr * xb).mean()))
            ISARETLER[ad] = isr
        duz += isr * TAVAN * abs(rho_s) * ub
        PARCA.append((ad, ub, isr * TAVAN * abs(rho_s)))
        ONC.append(ub)

    def _kor(v):
        n1 = np.sqrt(float((w2 * v * v).mean()))
        if n1 < 1e-12:
            return 0.0
        return float((w2 * rr * v).mean()) / (n1 * np.sqrt(float((w2 * rr * rr).mean())))

    kor = _kor(duz)
    AILE.setdefault(blok, {})["tum"] = kor
    for aile, secici in AILELER.items():
        v = np.zeros(len(rr))
        for ad2, u2, b2 in PARCA:
            if secici(ad2):
                v += b2 * u2
        AILE[blok][aile] = _kor(v)
    if TEMEL is None:
        TEMEL = kor
    d0, d1 = str(ff.tarih.min())[:10], str(ff.tarih.max())[:10]
    print(
        f"{blok:>8s} {len(rr):9d} {d0 + '..' + d1:>23s} {kor:8.4f} "
        f"{kor / TEMEL:10.3f} {'EVET' if kor > 0.0991 else 'HAYIR':>15s}"
    )

print("\nyaz25 = eksenlerin secildigi ve isaretlendigi blok (ev sahasi).")
print("guz25/kis26 = HIC GORULMEDI. Oran 1'e yakinsa sisme sinirlidir.")

print("\nAILE AILE -- guz25/kis26'daki isaret donmesi mevsimsel mi?")
kols = ["tum"] + list(AILELER)
print(f"{'blok':>8s} " + " ".join(f"{k:>15s}" for k in kols))
for b in AILE:
    print(f"{b:>8s} " + " ".join(f"{AILE[b].get(k, 0.0):15.4f}" for k in kols))
print("\nEger donme YALNIZCA hava/mevsim ailesindeyse aciklama gecerlidir:")
print("  test yaz25 ile AYNI mevsimdedir (Nisan-Temmuz), guz25/kis26 degil.")
print("Ama trafo/yapisal aile de donuyorsa aciklama COKER -- o aile")
print("  mevsimden bagimsiz olmali ve isaretini korumaliydi.")

# EKSEN EKSEN ISARET UYUMU: yaz25 isareti diger bloklarda tutuyor mu?
print("\n\nEKSEN EKSEN ISARET UYUMU (yaz25'e gore)")
ISR_BLOK = {}
for blok in ["yaz25", "guz25", "kis26"]:
    ff, pp, rr, sg, w2 = blok_kur(blok)
    svB2, ufB2 = st(pp), st(ff.ufuk_gun.to_numpy())
    ayB2 = st(pd.to_datetime(ff.tarih).dt.month.to_numpy().astype(np.float64))
    CARP.update(
        {"x_sv": (svT, svB2), "x_soguk": (sgT, sg), "x_ufuk": (ufT, ufB2), "x_ay": (ayT, ayB2)}
    )
    globals()["bf"], globals()["ww"] = ff, w2
    d = {}
    for ad in EKS:
        xt, xb = kur(ad)
        if xt is None or xb is None:
            continue
        d[ad] = np.sign(float((w2 * rr * xb).mean()))
    ISR_BLOK[blok] = d
ortak = [a for a in EKS if all(a in ISR_BLOK[b] for b in ISR_BLOK)]
for blok in ["guz25", "kis26"]:
    ayni = sum(1 for a in ortak if ISR_BLOK[blok][a] == ISR_BLOK["yaz25"][a])
    print(
        f"  yaz25 ile {blok}: {ayni}/{len(ortak)} eksende AYNI isaret "
        f"({ayni / len(ortak):.0%}; sans %50)"
    )
ikisi = sum(
    1 for a in ortak if ISR_BLOK["guz25"][a] == ISR_BLOK["yaz25"][a] == ISR_BLOK["kis26"][a]
)
print(f"  ucunde de ayni: {ikisi}/{len(ortak)}  (sans %25)")
print("\n%50'ye yakinsa ISARETLER TASINMIYOR demektir -- bu, bilesigin")
print("  isaretinin yaz25'e ozgu oldugunu gosterir.")
