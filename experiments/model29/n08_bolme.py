# ruff: noqa  -- YARIM KALDI: ajan kesildi, betik HIC KOSMADI, sonuc uretmedi.
# Sordugu soru sonradan baska olcumlerle cevaplandi (bkz. commit mesaji).
"""HANGI BOLME? -- 4 dik sonda yonunun secimini VEKIL ARTIKLA olcer.

m148_demet_plani.py BETA'yi 4 dik bloga boluyor ve her blogu bir LB sondasiyla
olcuyor. Ortonormal yonlerde  toplam_k rho_k^2 = ||P_altuzay r_gercek||^2,
yani sonuc YALNIZCA secilen 4 boyutlu ALT UZAYA baglidir. Soru: hangi bolme
daha cok gercek artik yakalar?

YONTEM (sizintisiz):
  * Eksenler, KATS (1.95*|rho_s|*isaret) ve RHO_CV m148'in dongusunun
    BIREBIR kopyasindan gelir. rho_s test uzayindaki r_hat'ten (LB bilgisi),
    rho_cv YAZ25 blogundan gelir.
  * "Gercek artik" VEKILI ise DIS BLOKLARDIR (guz25 + kis26). Boylece
    agirliklari belirleyen bilgi (yaz25 + LB) ile dogruluk olcen bilgi
    (guz25/kis26) AYRISIR. m144'un hedef kodlamasindaki ayni ilke.
  * Her bolme adayinda gruplar AYRIK oldugu ve eksenler vekil uzayda
    ortonormallestirildigi icin grup yonleri kendiliginden diktir:
        skor = toplam_g ( <w*r_vekil, v_g> / (||v_g|| sqrt(m0)) )^2
  * Guven araligi: tanim (musteri) duzeyinde KUME BOOTSTRAP.

HICBIR GONDERIM YAZILMAZ. submissions/ altina dokunulmaz.
"""

import gc
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
RHO_S_ALT = 0.015
AZAMI_EKSEN = 40
sys.path.insert(0, M29)
from m112_kalibre import EK_MODEL, M0, buzmeli_r_hat  # noqa: E402

te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
IDS = te.id.values
del te
gc.collect()


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
GI5 = np.linalg.pinv(G, rcond=1e-5)
r_hat, gercek, kL = buzmeli_r_hat(V, L, G, N)
print(f"saf optimum {np.sqrt(M0 - gercek):.6f}   V: {V.shape}", flush=True)

# --------------------------------------------------------------- veri
tp = pd.read_parquet(os.path.join(DN, "test.parquet"))
e = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
SIC_AILE = [(t, aa) for t in (1000, 1001, 1002) for aa in ("cat", "xgb", "lgbm")]


def blok_artik(eg, blok):
    blk = eg[eg._blok == blok]
    sic, sog = blk[blk.soguk_mu == 0], blk[blk.soguk_mu == 1]
    P = [
        np.load(os.path.join(AO, f"{blok}_{t}_{aa}_uretim.npy")).astype(np.float64)
        for t, aa in SIC_AILE
        if os.path.exists(os.path.join(AO, f"{blok}_{t}_{aa}_uretim.npy"))
    ]
    z = np.load(os.path.join(DN, f"soguk_tahmin_{blok}.npz"))
    idx = np.concatenate([sic.index.values, sog.index.values])
    pb = np.concatenate([np.mean(P, axis=0), np.mean([z[q] for q in z.files], axis=0)])
    return idx, np.log1p(eg.loc[idx].tuketim.values.astype(np.float64)) - pb, pb


idx, rb, pb = blok_artik(e, "yaz25")
bf = e.loc[idx].copy()
sgm = bf.soguk_mu.values.astype(np.float64)
ww = np.where(sgm == 1, HEDEF_SOGUK / sgm.mean(), (1 - HEDEF_SOGUK) / (1 - sgm.mean()))
ww = ww / ww.mean()
m0b = float((ww * rb * rb).mean())

# --- VEKIL: dis bloklar (guz25 + kis26). Agirlik kaynagi ile AYRIK. ---
d_idx, d_r, d_p = [], [], []
for b in ("guz25", "kis26"):
    i2, r2, p2 = blok_artik(e, b)
    d_idx.append(i2)
    d_r.append(r2)
    d_p.append(p2)
DIS_R = np.concatenate(d_r)
DIS_P = np.concatenate(d_p)
df = e.loc[np.concatenate(d_idx)].copy()
sgd = df.soguk_mu.values.astype(np.float64)
wd = np.where(sgd == 1, HEDEF_SOGUK / sgd.mean(), (1 - HEDEF_SOGUK) / (1 - sgd.mean()))
wd = wd / wd.mean()
m0d = float((wd * DIS_R * DIS_R).mean())
ND = len(DIS_R)
print(f"yaz25 {len(rb):,} satir | VEKIL dis blok {ND:,} satir", flush=True)
del e
gc.collect()


def st(x):
    x = np.asarray(x, dtype=np.float64).copy()
    f = np.isfinite(x)
    if not f.any():
        return None
    x[~f] = np.median(x[f])
    x -= x.mean()
    s = np.sqrt(float((x * x).mean()))
    return x / s if s > 1e-12 else None


svT, svB, svD = st(a0), st(pb), st(DIS_P)
sgT = tp.soguk_mu.values.astype(np.float64)
ufT, ufB, ufD = st(tp.ufuk_gun.to_numpy()), st(bf.ufuk_gun.to_numpy()), st(df.ufuk_gun.to_numpy())  # noqa: E501
ayT = st(pd.to_datetime(tp.tarih).dt.month.to_numpy().astype(np.float64))
ayB = st(pd.to_datetime(bf.tarih).dt.month.to_numpy().astype(np.float64))
ayD = st(pd.to_datetime(df.tarih).dt.month.to_numpy().astype(np.float64))
CARP = {
    "x_sv": (svT, svB, svD),
    "x_soguk": (sgT, sgm, sgd),
    "x_ufuk": (ufT, ufB, ufD),
    "x_ay": (ayT, ayB, ayD),
}
ESIK = {"ust10": (0.9, True), "ust25": (0.75, True), "alt10": (0.1, False)}
UC = (None, None, None)


def kur(ad):
    """m148'in kur()'u; UCUNCU cikti = vekil (dis blok) uzayindaki eksen."""
    if ad.startswith("M[") and "]x[" in ad and ad.endswith("]"):
        k1, k2 = ad[2:-1].split("]x[", 1)
        a1, b1, c1 = kur(k1)
        a2, b2, c2 = kur(k2)
        if any(q is None for q in (a1, a2, b1, b2, c1, c2)):
            return UC
        return st(a1 * a2), st(b1 * b2), st(c1 * c2)
    if "*" in ad:
        k1, k2 = ad.split("*", 1)
        if k1 not in tp.columns or k2 not in tp.columns:
            return UC
        a1, b1, c1 = st(tp[k1].to_numpy()), st(bf[k1].to_numpy()), st(df[k1].to_numpy())
        a2, b2, c2 = st(tp[k2].to_numpy()), st(bf[k2].to_numpy()), st(df[k2].to_numpy())
        if any(q is None for q in (a1, a2, b1, b2, c1, c2)):
            return UC
        return st(a1 * a2), st(b1 * b2), st(c1 * c2)
    kol, kip = ad.split(":", 1) if ":" in ad else (ad, "")
    if kol not in tp.columns or kol not in bf.columns or kol not in df.columns:
        return UC
    xt, xb, xd = tp[kol].to_numpy(), bf[kol].to_numpy(), df[kol].to_numpy()
    if kip in CARP:
        mt, mb, md = CARP[kip]
        a_, b_, c_ = st(xt), st(xb), st(xd)
        if a_ is None or b_ is None or c_ is None:
            return UC
        return st(a_ * mt), st(b_ * mb), st(c_ * md)
    if kip in ESIK:
        q, ust = ESIK[kip]
        fv = xt[np.isfinite(xt)]
        if fv.size == 0:
            return UC
        v_ = np.quantile(fv, q)
        if ust:
            return (
                st((xt > v_).astype(np.float64)),
                st((xb > v_).astype(np.float64)),
                st((xd > v_).astype(np.float64)),
            )
        return (
            st((xt < v_).astype(np.float64)),
            st((xb < v_).astype(np.float64)),
            st((xd < v_).astype(np.float64)),
        )
    if kip == "mnt75":
        fv = xt[np.isfinite(xt)]
        if fv.size == 0:
            return UC
        v_ = float(np.quantile(fv, 0.75))
        return (
            st(np.maximum(xt - v_, 0.0)),
            st(np.maximum(xb - v_, 0.0)),
            st(np.maximum(xd - v_, 0.0)),
        )
    if kip == "kare":
        a_, b_, c_ = st(xt), st(xb), st(xd)
        if a_ is None or b_ is None or c_ is None:
            return UC
        return st(a_**2), st(b_**2), st(c_**2)
    return st(xt), st(xb), st(xd)


with open(os.path.join(M29, "m121_derin_tarama.json")) as fh:
    TARAMA = json.load(fh)
rng = np.random.default_rng(5)
tn = bf.tanim.values
uqn = pd.unique(tn)
gi = pd.Series(np.arange(len(uqn)), index=uqn)[tn].to_numpy()
PERM = [
    np.argsort(np.argsort(rng.permutation(len(uqn))[gi], kind="stable"), kind="stable")
    for _ in range(20)
]

with open(os.path.join(M29, "m144_yeni_aileler.json"), encoding="utf-8") as fh:
    _M144 = json.load(fh)["kapidan_gecen"]
YENI_EKSENLER = [r["eksen"] for r in sorted(_M144, key=lambda r: -abs(r["rho_s"]))]
YENI_AILE = {r["eksen"]: r["aile"] for r in _M144}

# ------------------------------------------------------- EKSEN DONGUSU
kul, KAT_LISTE, RHO_CV_LISTE, RHO_S_LISTE, AILE_LISTE, YENI_MASKE = [], [], [], [], [], []
ONCEKI = []  # test uzayi, float32 (Gram-Schmidt sadece Qd kapisi icin)
VEKIL_U = []  # vekil uzayda ortonormallestirilmis eksenler, float32
for kayit in TARAMA + [{"eksen": a, "_yeni": True} for a in YENI_EKSENLER]:
    _yeni = bool(kayit.get("_yeni"))
    if not _yeni and len(kul) >= AZAMI_EKSEN:
        continue
    ad = kayit["eksen"]
    xt, xb, xd = kur(ad)
    if xt is None or xb is None or xd is None:
        continue
    cc = Gi @ ((V.T @ xt) / N)
    xp0 = xt - V @ cc
    Qs = 1.0 - float((xp0 * xp0).mean())
    if Qs < 0.02:
        continue
    rho_s = float((r_hat * xt).mean()) / np.sqrt(Qs)
    if abs(rho_s) < RHO_S_ALT:
        continue
    cc5 = GI5 @ ((V.T @ xt) / N)
    xp5 = xt - V @ cc5
    Qs5 = 1.0 - float((xp5 * xp5).mean())
    if Qs5 < 0.02:
        continue
    if abs(float((r_hat * xt).mean()) / np.sqrt(Qs5) - rho_s) > 0.3 * abs(rho_s):
        continue
    xp = xp0
    for u in ONCEKI:
        xp = xp - float((xp * u).mean()) * u.astype(np.float64)
    Qd = float((xp * xp).mean())
    if Qd < 0.25:
        continue
    kor = float((ww * rb * xb).mean()) / np.sqrt(m0b)
    gur = np.std([float((ww * rb[s] * xb).mean()) / np.sqrt(m0b) for s in PERM])
    if abs(kor) < 3 * gur:
        continue
    rho_cv = CARPAN * kor
    if abs(rho_cv) < TAVAN * abs(rho_s):
        continue
    # --- VEKIL uzayda ayni sirayla Gram-Schmidt (agirlikli ic carpim) ---
    yd = xd.copy()
    for u in VEKIL_U:
        uu = u.astype(np.float64)
        yd = yd - float((wd * yd * uu).mean()) * uu
    nd = np.sqrt(float((wd * yd * yd).mean()))
    if nd < 1e-6:
        continue
    yd = yd / nd

    ONCEKI.append((xp / np.sqrt(Qd)).astype(np.float32))
    VEKIL_U.append(yd.astype(np.float32))
    kul.append(ad)
    KAT_LISTE.append(float(np.sign(rho_cv) * TAVAN * abs(rho_s)))
    RHO_CV_LISTE.append(float(rho_cv))
    RHO_S_LISTE.append(float(rho_s))
    YENI_MASKE.append(_yeni)
    AILE_LISTE.append(YENI_AILE.get(ad, "m121_taban"))

n = len(kul)
KATS = np.array(KAT_LISTE)
RHO_CV = np.array(RHO_CV_LISTE)
RHO_S = np.array(RHO_S_LISTE)
AILE = np.array(AILE_LISTE)
ORAN = np.abs(RHO_CV) / np.abs(KATS)
print(f"\n{n} eksen kabul edildi. ||BETA|| = {np.sqrt((KATS**2).sum()):.4f}", flush=True)
print(
    f"oran = |rho_cv|/|KATS|: min {ORAN.min():.3f} med {np.median(ORAN):.3f} maks {ORAN.max():.3f}",
    flush=True,
)
del ONCEKI, V, r_hat, tp
gc.collect()

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
    "gunes",
    "x_ay",
    "ay_",
    "sicaklik",
)
HV = np.array([bool(any(h in a for h in HAVA)) for a in kul])
KOKEN_H = AILE == "H_carpim40"
ORAN_YUK = np.median(ORAN) <= ORAN
print(
    f"hava {HV.sum()} / yapi {(~HV).sum()} | H_carpim40 {KOKEN_H.sum()} / diger {(~KOKEN_H).sum()}",
    flush=True,
)

# --------------------------------------------- VEKIL PROJEKSIYON OLCUMU
UD = np.array(VEKIL_U)  # (n, ND) float32, vekil uzayda ortonormal
del VEKIL_U
gc.collect()
CH = 50_000  # (n, ND) float64 kopyasi bellege SIGMAZ -> parca parca


def gram():
    g = np.zeros((n, n))
    for s in range(0, ND, CH):
        blk = UD[:, s : s + CH].astype(np.float64)
        g += (blk * wd[s : s + CH]) @ blk.T
    return g / ND


def birlestir(idx, w):
    """sum_i w_i * UD[idx_i]  (float64, ND uzunlugunda)."""
    out = np.zeros(ND)
    for k_, i_ in enumerate(idx):
        out += float(w[k_]) * UD[i_].astype(np.float64)
    return out


dik_sapma = float(np.abs(gram() - np.eye(n)).max())
print(f"vekil uzayda diklik sapmasi: {dik_sapma:.2e}", flush=True)

wr = wd * DIS_R  # (ND,)
# c[i] = <w r, u_i> / sqrt(m0d)  -- her eksenin vekil artikla korelasyonu
C = np.array([float(UD[i].astype(np.float64) @ wr) for i in range(n)]) / ND / np.sqrt(m0d)
print(
    f"tek eksen |korelasyon| maks {np.abs(C).max():.4f}, "
    f"toplam C^2 (136 boyut tavani) = {(C**2).sum():.6f}",
    flush=True,
)


def bolme_skoru(gruplar):
    """gruplar: maske listesi (ayrik). Doner: (toplam rho^2, rho_g listesi)."""
    rr = []
    for m in gruplar:
        if m.sum() == 0:
            continue
        a = np.abs(KATS[m])
        s = np.sign(KATS[m])
        w = s * a
        nw = np.sqrt((w * w).sum())
        if nw < 1e-12:
            continue
        rr.append(float((w @ C[m]) / nw))
    return float(np.sum(np.square(rr))), rr


def carpraz(m1, m2):
    return [m1 & m2, m1 & ~m2, ~m1 & m2, ~m1 & ~m2]


ADAYLAR = {
    "taban_tek_yon": [np.ones(n, bool)],
    "A_aile_koken": carpraz(HV, KOKEN_H),
    "B_oran": carpraz(HV, ORAN_YUK),
    "C1_isaret": carpraz(HV, KATS > 0),
    "C2_rho_s_buyukluk": carpraz(HV, np.abs(RHO_S) >= np.median(np.abs(RHO_S))),
    "C3_yalniz_oran4": [
        np.quantile(ORAN, 0.75) <= ORAN,
        (np.median(ORAN) <= ORAN) & (np.quantile(ORAN, 0.75) > ORAN),
        (np.quantile(ORAN, 0.25) <= ORAN) & (np.median(ORAN) > ORAN),
        np.quantile(ORAN, 0.25) > ORAN,
    ],
    "C4_yalniz_aile4": None,  # asagida m144 aile etiketlerinden kurulur
}
# C4: m144 aile etiketleri -> en buyuk 3 aile + kalan
_ag = {f: float(np.sqrt((KATS[f == AILE] ** 2).sum())) for f in set(AILE_LISTE)}
_ilk3 = sorted(_ag, key=lambda k: -_ag[k])[:3]
_m3 = [f == AILE for f in _ilk3]
ADAYLAR["C4_yalniz_aile4"] = _m3 + [~np.any(_m3, axis=0)]

# ORACLE ust sinir (asiri uydurma, yalniz referans): en iyi 4 tek eksen
_ora = np.argsort(-np.abs(C))[:4]
ADAYLAR["ORACLE_en_iyi4_eksen"] = [np.arange(n) == i for i in _ora]

print()
print(f"{'aday':>22s} {'boyut':>5s} {'toplam rho^2':>13s} {'sqrt':>8s}", flush=True)
SONUC = {}
for ad, gr in ADAYLAR.items():
    s, rr = bolme_skoru(gr)
    SONUC[ad] = {
        "rho2": s,
        "rho_k": rr,
        "boyut": len(rr),
        "eksen_sayilari": [int(m.sum()) for m in gr],
    }
    print(f"{ad:>22s} {len(rr):5d} {s:13.6f} {np.sqrt(s):8.4f}", flush=True)

# ------------------------------------------------- KUME BOOTSTRAP (tanim)
tnd = df.tanim.values
uqd = pd.unique(tnd)
gid = pd.Series(np.arange(len(uqd)), index=uqd)[tnd].to_numpy()
NK = len(uqd)
print(f"\nbootstrap: {NK:,} tanim kumesi", flush=True)

# kume basina biriktirilecekler
PAY = {}  # aday -> (NK, boyut) kume basina <w r, v_g>*ND katkisi
NRM = {}  # aday -> (NK, boyut) kume basina agirlikli ||v_g||^2 katkisi
VEKS = {}
for ad, gr in ADAYLAR.items():
    vs = []
    for m in gr:
        if m.sum() == 0:
            continue
        w = KATS[m]
        nw = np.sqrt((w * w).sum())
        if nw < 1e-12:
            continue
        vs.append(birlestir(np.flatnonzero(m), w / nw))
    VEKS[ad] = vs
    P = np.zeros((NK, len(vs)))
    Q = np.zeros((NK, len(vs)))
    for j, v in enumerate(vs):
        P[:, j] = np.bincount(gid, weights=wr * v, minlength=NK)
        Q[:, j] = np.bincount(gid, weights=wd * v * v, minlength=NK)
    PAY[ad] = P
    NRM[ad] = Q
RR = np.bincount(gid, weights=wd * DIS_R * DIS_R, minlength=NK)
del UD, VEKS
gc.collect()

B = 2000
brng = np.random.default_rng(11)
BOOT = {ad: np.zeros(B) for ad in ADAYLAR}
for b in range(B):
    cnt = brng.multinomial(NK, np.full(NK, 1.0 / NK)).astype(np.float64)
    m0b_ = cnt @ RR
    for ad in ADAYLAR:
        p = cnt @ PAY[ad]
        q = cnt @ NRM[ad]
        BOOT[ad][b] = float(np.sum((p * p) / (q * m0b_)))

print()
print(f"{'aday':>22s} {'rho^2':>10s} {'%2.5':>10s} {'%97.5':>10s}", flush=True)
for ad in ADAYLAR:
    lo, hi = np.percentile(BOOT[ad], [2.5, 97.5])
    SONUC[ad]["boot_ort"] = float(BOOT[ad].mean())
    SONUC[ad]["boot_ci"] = [float(lo), float(hi)]
    print(f"{ad:>22s} {SONUC[ad]['rho2']:10.6f} {lo:10.6f} {hi:10.6f}", flush=True)

# --- ikili farklar (eslesmis bootstrap) ---
IKILI = {}
ANA = [
    "A_aile_koken",
    "B_oran",
    "C1_isaret",
    "C2_rho_s_buyukluk",
    "C3_yalniz_oran4",
    "C4_yalniz_aile4",
    "taban_tek_yon",
]
print()
print(f"{'fark':>44s} {'delta':>11s} {'%2.5':>11s} {'%97.5':>11s} {'P(>0)':>7s}", flush=True)
for i in range(len(ANA)):
    for j in range(i + 1, len(ANA)):
        a, b_ = ANA[i], ANA[j]
        d = BOOT[a] - BOOT[b_]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = float((d > 0).mean())
        IKILI[f"{a} - {b_}"] = {
            "delta": float(SONUC[a]["rho2"] - SONUC[b_]["rho2"]),
            "ci": [float(lo), float(hi)],
            "P_pozitif": p,
        }
        print(
            f"{a + ' - ' + b_:>44s} {SONUC[a]['rho2'] - SONUC[b_]['rho2']:+11.6f} "
            f"{lo:+11.6f} {hi:+11.6f} {p:7.3f}",
            flush=True,
        )

# ----------------------------------------------------- CAPRAZ TABLO A vs B
tab = np.zeros((2, 2), int)
for a_ in (0, 1):
    for b_ in (0, 1):
        tab[a_, b_] = int(((bool(a_) == KOKEN_H) & (bool(b_) == ORAN_YUK)).sum())
print("\nCAPRAZ TABLO  koken (satir) x oran (sutun)", flush=True)
print(f"{'':>16s} {'oran DUSUK':>11s} {'oran YUKSEK':>12s}", flush=True)
print(f"{'m121/diger':>16s} {tab[0, 0]:11d} {tab[0, 1]:12d}", flush=True)
print(f"{'H_carpim40':>16s} {tab[1, 0]:11d} {tab[1, 1]:12d}", flush=True)
_a, _b, _c, _d = tab[0, 0], tab[0, 1], tab[1, 0], tab[1, 1]
_den = np.sqrt(float((_a + _b) * (_c + _d) * (_a + _c) * (_b + _d)))
phi = float((_a * _d - _b * _c) / _den) if _den > 0 else 0.0
ortusme = float((KOKEN_H == ORAN_YUK).mean())
ortusme = max(ortusme, 1 - ortusme)
print(f"phi (Cramer V) = {phi:+.3f} | en iyi hizalamada ortusme = {ortusme:.1%}", flush=True)

# 4 hucrenin BIREBIR ortusmesi (A ve B bloklarinin kesisimi)
A_gr, B_gr = carpraz(HV, KOKEN_H), carpraz(HV, ORAN_YUK)
kes = np.zeros((4, 4), int)
for i in range(4):
    for j in range(4):
        kes[i, j] = int((A_gr[i] & B_gr[j]).sum())
print("A bloklari x B bloklari kesisim matrisi:", flush=True)
for i in range(4):
    print("   " + " ".join(f"{kes[i, j]:4d}" for j in range(4)), flush=True)

CIKTI = {
    "aciklama": "Hangi 4-yonlu bolme daha cok gercek artik yakalar? "
    "Vekil = guz25+kis26 artigi (agirlik kaynagi yaz25+LB ile ayrik).",
    "n_eksen": n,
    "vekil_satir": int(ND),
    "vekil_diklik_sapmasi": dik_sapma,
    "tek_eksen_C2_toplami": float((C**2).sum()),
    "oran_ozet": {
        "min": float(ORAN.min()),
        "medyan": float(np.median(ORAN)),
        "maks": float(ORAN.max()),
    },
    "adaylar": SONUC,
    "ikili_farklar": IKILI,
    "capraz_tablo_koken_x_oran": tab.tolist(),
    "phi": phi,
    "ortusme": ortusme,
    "A_x_B_kesisim": kes.tolist(),
    "eksenler": [
        {
            "eksen": kul[i],
            "aile": AILE_LISTE[i],
            "hava": bool(HV[i]),
            "kats": float(KATS[i]),
            "rho_cv": float(RHO_CV[i]),
            "rho_s": float(RHO_S[i]),
            "oran": float(ORAN[i]),
            "vekil_kor": float(C[i]),
        }
        for i in range(n)
    ],
}
with open(os.path.join(BURA, "n08_bolme.json"), "w", encoding="utf-8") as fh:
    json.dump(CIKTI, fh, indent=1, ensure_ascii=False)
print("\nyazildi: n08_bolme.json", flush=True)
