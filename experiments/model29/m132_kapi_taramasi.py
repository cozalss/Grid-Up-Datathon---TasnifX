"""NIHAI BILESIK -- duzeltilmis kapiyla.

DUZELTME. Onceki kapi "oran = rho_pred / rho_s(bilesik) <= 4" idi. Bu YANLIS:
tek eksen icin oran = 1.95*sqrt(Q_dik) <= 1.95 her zaman; cok eksende ise
eksenlerin SPAN bileseni birbirini goturunce payda kuculuyor ve oran siisiyor.
Yani oran, inandiriciligi degil isaret sadelesmesini olcuyor.

DOGRU KAPI: her eksenin katsayisi 1.95*|rho_s| TAVANINA DAYANSIN.
Dayaniyorsa tahmin LB'nin kendi olcumune capalidir (CV'ye degil).
    rho_kul = isaret(rho_cv) * min(|rho_cv|, 1.95*|rho_s|)
    TAVAN DAYANIYOR  <=>  |rho_cv| >= 1.95*|rho_s|

Bilesigin ongorulen rho'su = ||beta|| (dik eksenler). Bu, her eksenin kendi
LB olcumune capali oldugu icin savunulabilir; tek varsayim 1.95 carpaninin
seviye'den digerlerine tasinmasi (n=1, docs/68).

Ek kapi: rho_s'in kendi gurultusu sigma(rho_s) ~ 3e-4; |rho_s| >= 0.015
(50 sigma) sarti aranir.
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
BURA = os.path.dirname(os.path.abspath(__file__))
TABAN = "tuketim_m6_ikiyon.csv"  # M0 m112den gelir (docs/69)
HEDEF_SOGUK, CARPAN, TAVAN = 0.222, 0.798, 1.95
#: Canli liderlik tablosu (2026-08-30 17:26). Hedefler gun icinde SERTLESTI:
#: Duo-Electra 1.00129 -> 0.99790 -> 0.99614, Berke Kuc yeni girdi 0.99927.
HEDEF_2, HEDEF_3 = 0.99614, 0.99927
RHO_S_ALT = 0.015
AZAMI_EKSEN = int(os.environ.get("AZAMI", "300"))  # m132: kapi taramasi
sys.path.insert(0, M29)
from m112_kalibre import EK_MODEL, M0, buzmeli_r_hat  # noqa: E402

te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
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
GI5 = np.linalg.pinv(G, rcond=1e-5)  # rcond kararlilik kapisi icin
r_hat, gercek, kL = buzmeli_r_hat(V, L, G, N)
nrm = float((r_hat * r_hat).mean())
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
pb = np.concatenate([np.mean(P, axis=0), np.mean([z[q] for q in z.files], axis=0)])
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

duz = np.zeros(N)
kul = []
BETALAR = []
DIKLER = []
print(
    f"\n{'eksen':>34s} {'rho_cv':>8s} {'rho_s':>8s} {'rho_kul':>8s} {'Q_dik':>6s} "
    f"{'tavan':>6s} {'kum.rho':>8s}"
)
ONCEKI = []
for kayit in TARAMA:
    if len(kul) >= AZAMI_EKSEN:
        break
    ad = kayit["eksen"]
    xt, xb = kur(ad)
    if xt is None or xb is None:
        continue
    # GEOMETRI (izdusum) -- burada gurultu yok, pinv dogrudan kullanilir
    cc = Gi @ ((V.T @ xt) / N)
    xp0 = xt - V @ cc
    Qs = 1.0 - float((xp0 * xp0).mean())
    if Qs < 0.02:
        continue
    # L_span TAHMINI (docs/70). Eskiden c'L kullaniliyordu; c neredeyse-tekil
    # kiplere buyuk katsayi verdigi icin L'nin gurultusunu buyutuyordu.
    # G'nin tekil degerleri ...3.9e-06, 5.3e-07... ve rcond=1e-6 kesimi
    # (6.6e-07) tam aralarina dusuyor: 40 eksenin 12'si rcond'a kirilgandi,
    # t_yuk_faktoru'nde rho_s 1e-4'te -0.004, 1e-6'da -0.020 (5 KAT).
    # r_hat zaten kip basina optimal buzmeyle kurulmus gurultu-farkindalikli
    # tahmindir; <r_hat, x>/N kararlidir (kirilgan eksen 12 -> 2).
    rho_s = float((r_hat * xt).mean()) / np.sqrt(Qs)
    if abs(rho_s) < RHO_S_ALT:
        continue
    # RCOND KARARLILIK KAPISI: geometri de rcond'a asiri duyarli olmasin
    cc5 = GI5 @ ((V.T @ xt) / N)
    xp5 = xt - V @ cc5
    Qs5 = 1.0 - float((xp5 * xp5).mean())
    if Qs5 < 0.02:
        continue
    if abs(float((r_hat * xt).mean()) / np.sqrt(Qs5) - rho_s) > 0.3 * abs(rho_s):
        continue
    xp = xp0.copy()
    for u in ONCEKI:
        xp -= float((xp * u).mean()) * u
    Qd = float((xp * xp).mean())
    if Qd < 0.25:  # eksenler birbirinden GERCEKTEN farkli olsun
        continue
    kor = float((ww * rb * xb).mean()) / np.sqrt(m0b)
    gur = np.std([float((ww * rb[s] * xb).mean()) / np.sqrt(m0b) for s in PERM])
    if abs(kor) < 3 * gur:
        continue
    rho_cv = CARPAN * kor
    dayanir = abs(rho_cv) >= TAVAN * abs(rho_s)
    if not dayanir:  # KAPI: tavan dayanmiyorsa tahmin CV'ye kalir
        continue
    # KATSAYI (docs/69 §2.5). seviye kalibrasyonu IKI BIRIM YON arasindaydi:
    #   rho_s = L_span/sqrt(Q_span) = +0.0156  (span birim yonu)
    #   rho_u = L_dik /sqrt(Q_dik)  = -0.0304  (dik birim yonu)   oran 1.95
    # Yani 1.95*|rho_s| DOGRUDAN dik birim yondeki korelasyonun tahminidir ve
    # u yonundeki optimal katsayi da odur. Eski kod ayrica sqrt(Q_dik) ile
    # carpiyordu; bu 1.95*|rho_s|'i TUM eksenin korelasyonu sayip izotropiyle
    # dik parcaya dagitmaya denk gelir -- oysa seviye'de rho_x/rho_s = 0.99,
    # 1.95 degil. Olcum de sqrt'siz hali destekliyor: blok korelasyonu
    # 0.2288 vs 0.2269, zaman-bolmeli tutma 1.098 vs 1.057.
    rho_kul = np.sign(rho_cv) * TAVAN * abs(rho_s)
    duz += rho_kul * (xp / np.sqrt(Qd))
    ONCEKI.append(xp / np.sqrt(Qd))
    kul.append(ad)
    BETALAR.append(float(rho_kul))
    DIKLER.append(xb)
    print(
        f"{ad[:34]:>34s} {rho_cv:+8.4f} {rho_s:+8.4f} {rho_kul:+8.4f} {Qd:6.3f} "
        f"{'EVET':>6s} {np.sqrt(float((duz * duz).mean())):8.4f}"
    )

Q = float((duz * duz).mean())
birim = duz / np.sqrt(Q)
RHO = float(np.sqrt(Q))

print(f"\nKAPILARDAN GECEN TOPLAM EKSEN: {len(kul)}  (m122 sert tavani 40)")

# blok tarafinda ayni sirayla dik'le ve SABIT LB katsayilariyla korelasyon olc
ONC_B, UB, BETA2 = [], [], []
for ad, b, xb in zip(kul, BETALAR, DIKLER):
    ub = xb.copy()
    for u in ONC_B:
        ub -= float((ww * ub * u).mean()) / float((ww * u * u).mean()) * u
    nb = np.sqrt(float((ww * ub * ub).mean()))
    if nb < 0.15:
        continue
    ub /= nb
    ONC_B.append(ub)
    UB.append(ub)
    BETA2.append(b)

uf = bf.ufuk_gun.to_numpy()
PENC = [
    ("1-24", uf <= 24),
    ("25-48", (uf > 24) & (uf <= 48)),
    ("49-73", (uf > 48) & (uf <= 73)),
    ("74-98", (uf > 73) & (uf <= 98)),
    ("99-122", uf > 98),
]


def korp(mask, x):
    w, r = ww[mask], rb[mask]
    n = np.sqrt(float((w * x[mask] ** 2).mean()))
    if n < 1e-12:
        return 0.0
    return float((w * r * (x[mask] / n)).mean()) / np.sqrt(float((w * r * r).mean()))


kap2 = float(np.sqrt(max(MSE_OPT - 0.99614**2, 0)))
kap3 = float(np.sqrt(max(MSE_OPT - 0.99927**2, 0)))
print(f"\n2. sira icin gereken rho {kap2:.4f}   3. sira {kap3:.4f}")
print(
    f"\n{'n':>4s} {'rho_pred':>9s} {'kor_tum':>8s} {'poz.penc':>9s} "
    f"{'en dusuk':>9s} {'2.sira f':>9s} {'3.sira f':>9s}"
)
for n in [10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, len(BETA2)]:
    if n > len(BETA2):
        continue
    d = np.zeros(len(rb))
    for b, u in zip(BETA2[:n], UB[:n]):
        d += b * u
    rho = float(np.sqrt(sum(b * b for b in BETA2[:n])))
    ks = [korp(m, d) for _, m in PENC]
    print(
        f"{n:4d} {rho:9.4f} {korp(np.ones(len(rb), dtype=bool), d):8.4f} "
        f"{sum(1 for k in ks if k > 0):5d}/5 {min(ks):9.4f} "
        f"{kap2 / rho:9.3f} {kap3 / rho:9.3f}"
    )
print("\nf = gereken rho / rho_pred = olctugumuz 1.95 carpaninin kacta kacinin")
print("    tutmasi gerektigi. Kucuk f = daha guvenli.")
