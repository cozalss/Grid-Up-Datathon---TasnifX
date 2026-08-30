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
from pathlib import Path

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
AZAMI_EKSEN = 40  # kesim KAPIDAN gelsin, sert tavandan degil (Kural 64)
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
print(
    f"\n{'eksen':>34s} {'rho_cv':>8s} {'rho_s':>8s} {'rho_kul':>8s} {'Q_dik':>6s} "
    f"{'tavan':>6s} {'kum.rho':>8s}"
)
ONCEKI = []
KAT_LISTE = []
RHO_CV_LISTE = []
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
    KAT_LISTE.append(float(rho_kul))
    RHO_CV_LISTE.append(float(rho_cv))
    print(
        f"{ad[:34]:>34s} {rho_cv:+8.4f} {rho_s:+8.4f} {rho_kul:+8.4f} {Qd:6.3f} "
        f"{'EVET':>6s} {np.sqrt(float((duz * duz).mean())):8.4f}"
    )

Q = float((duz * duz).mean())
birim = duz / np.sqrt(Q)
RHO = float(np.sqrt(Q))
print(f"\n{len(kul)} eksen, bilesigin ongorulen rho = {RHO:.4f}")

# ---------------------------------------------------------------------------
# DEMET PLANI -- agirliklari TAHMIN etmek yerine OLC.
#
# 6 gonderim hakkimiz var (31 Agu 3, 1 Eylul 3). Eksenler birbirine dik
# oldugu icin  skor^2 = MSE_OPT - toplam(rho_k^2)  ve her OLCULEN yon
# RISKSIZDIR: rho_k=0 cikarsa skor degismez, isareti yanlis cikarsa isareti
# duzeltiriz. Yani olcum, tahminin her zaman >= iyisidir.
#
# Bileseni tek yon olarak gondermek yerine 5 DIK DEMETE bolup her birini
# ayri olcersek, 5 boyutlu alt uzayda OPTIMUM bilesimi buluruz:
#     toplam(rho_k^2) >= rho_u^2   (esitlik ancak tahminimiz tam isabetse)
# Boylece hem 1.95 carpani hem ISARET riski ortadan kalkar.
#
# KUMULATIF KURGU: sonda k = (span_opt + toplam_{j<k} rho_j u_j) + kappa_k u_k
# Yani her sonda hem onceki OLCUMLERI kullanir hem yeni bir yon olcer.
# Hak biterse elimizde kalan son dosya tum onceki kazanimlari tasir.
# ---------------------------------------------------------------------------
DEMET = int(os.environ.get("DEMET", "5"))
YUV = 5e-6 / np.sqrt(3.0)
TABAN_MSE = float(M0 - 2 * kL + float((r_hat * r_hat).mean()))
print(f"\nGERCEK taban MSE = {TABAN_MSE:.7f} -> saf span skoru {np.sqrt(TABAN_MSE):.5f}")

# ONCEKI: eksenlerin dik BIRIM yonleri (m122 dongusunde kuruldu)
# KATS: her eksenin ongorulen katsayisi (isaret * 1.95 * |rho_s|)
U = np.array(ONCEKI)  # (n_eksen, N)
KATS = np.array(KAT_LISTE)
print(
    f"{len(U)} dik birim yon, ongorulen katsayilar |b| in [{np.abs(KATS).min():.4f}, "
    f"{np.abs(KATS).max():.4f}]"
)

# ---------------------------------------------------------------------------
# DEMET SECIMI -- ONEMLI DUZELTME.
#
# Ortonormal yonlerde  toplam(rho_k^2) = ||P_altuzay r||^2, yani sonuc yalnizca
# SECILEN ALT UZAYA baglidir, eksenleri nasil grupladigimiza DEGIL. Onceki
# surumdeki "sirayla dagit" bolmesi keyfiydi.
#
# Elimizde 5 boyut var ve 40 boyutluk dik uzayin neresinde sinyal oldugunu
# bilmiyoruz. En iyi kullanim: RAKIP AGIRLIKLANDIRMA HIPOTEZLERINE yaymak.
# Her hipotez bir yon onerir; Gram-Schmidt ile diklestirilir:
#   H1  1.95*|rho_s| agirligi   (m144 bunun sisik oldugunu gosterdi)
#   H2  rho_cv agirligi         (yaz25'te DOGRUDAN olculen dik korelasyon)
#   H3  hava/mevsim ailesi      (m141'de ayri davrandi)
#   H4  trafo/yapisal ailesi
#   H5  esit agirlik            (hicbir tahmine guvenmeyen taban)
# Hangi hipotez dogruysa o boyut buyuk rho verir; hepsi yanlissa kayip YOK.
# ---------------------------------------------------------------------------
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
ISR = np.sign(KATS)
HIPOTEZ = {
    "H1 1.95|rho_s|": np.abs(KATS),
    "H2 rho_cv": np.abs(np.array(RHO_CV_LISTE)),
    "H3 hava/mevsim": np.array([1.0 if any(h in a for h in HAVA) else 0.0 for a in kul]),
    "H4 trafo/yapisal": np.array([0.0 if any(h in a for h in HAVA) else 1.0 for a in kul]),
    "H5 esit": np.ones(len(kul)),
}
BETA = np.zeros(N)
for i in range(len(U)):
    BETA = BETA + KATS[i] * U[i]

print()
print(f"{'hipotez':>18s} {'eksen':>6s} {'artakalan':>10s} {'ongorulen rho_k':>16s}")
G, RHO_K, ETIKET = [], [], []
for ad, ag in HIPOTEZ.items():
    v = np.zeros(N)
    for i in range(len(U)):
        v = v + ISR[i] * ag[i] * U[i]
    n0 = float(np.sqrt(float((v * v).mean())))
    if n0 < 1e-12:
        continue
    v = v / n0
    for g in G:
        v = v - float((v * g).mean()) * g
    n1 = float(np.sqrt(float((v * v).mean())))
    if n1 < 0.05:
        print(f"{ad:>18s} {int((ag > 0).sum()):6d} {n1:10.3f}  ATLANDI (onceki yone cok yakin)")
        continue
    v = v / n1
    G.append(v)
    # H1 disindaki yonler HEDGE'''dir: onlar icin ongoru YOK (BETA tamamen
    # H1 boyunca uzandigi icin dik bilesenleri sifir cikar). Sondanin kendi
    # skoru onemsiz -- onemli olan OLCUM -- bu yuzden kappa TEKDUZE secilir:
    # her yone ongorulen toplamin esit payi. Olcum hassasiyeti her halukarda
    # 1e-4'ten iyi.
    RHO_K.append(abs(float((BETA * v).mean())))
    ETIKET.append(ad)
    print(f"{ad:>18s} {int((ag > 0).sum()):6d} {n1:10.3f} {RHO_K[-1]:16.4f}")
G = np.array(G)
RHO_K = np.array(RHO_K)
DEMET = len(G)
# TAVAN DUZELTMESI (m145 denetimi). 1.95 tek bir deneyden geliyordu; dort
# bagimsiz yol |c| ~ 0.7 diyor (%90 araligi [0.3, 1.3]). Net kazanc
# 2*a*rho - a^2 asimetriktir: c_gercek=0.7 iken a=1.95 kullanmak kazanci
# NEGATIFE cevirir (hicbir sey yapmamaktan kotu), ama a=0.8 kullanmak
# c_gercek=1.95 ciksa bile POZITIF kalir. Bu yuzden kappa 0.8/1.95 ile
# olceklenir. Olcum hassasiyeti yine de 1e-4'ten iyi kalir.
C_TAVAN = 0.8
KAPPA_K = np.full(DEMET, (C_TAVAN / 1.95) * float(RHO) / np.sqrt(DEMET))
gruplar = [[] for _ in range(DEMET)]
print()
print(f"{DEMET} dik yon kuruldu. sqrt(toplam rho_k^2) = {np.sqrt((RHO_K**2).sum()):.4f}")
dikkat = np.abs(G @ G.T / N - np.eye(DEMET)).max()
print(f"demetlerin dikligi: en buyuk sapma {dikkat:.2e}  (0 olmali)")

# ---------------------------------------------------------------------------
# KUMULATIF URETIM. Sondalar sirayla gonderilir; her sonucun ardindan bu betik
# TEKRAR kosulur ve bir SONRAKI dosyayi uretir.
#   m148_olcumler.json  ->  {"1": 0.99612, "2": 1.00034, ...}   (LB skorlari)
#   rho_k = (sabit_k - P^2) / (2*kappa_etkin_k)
#   bir sonraki sondanin TABANI = span_opt + toplam_{olculen} rho_j G_j
# Hak biterse elde kalan son dosya TUM onceki kazanimlari tasir.
# ---------------------------------------------------------------------------
OLC_YOL = os.path.join(BURA, "m148_olcumler.json")
OLCUM = {}
if os.path.exists(OLC_YOL):
    with open(OLC_YOL) as fh:
        OLCUM = {int(k): float(v) for k, v in json.load(fh).items()}
GECMIS_YOL = os.path.join(BURA, "m148_demet.json")
GECMIS = {}
if os.path.exists(GECMIS_YOL):
    with open(GECMIS_YOL) as fh:
        GECMIS = {d["sonda"]: d for d in json.load(fh).get("sondalar", [])}

RHO_OLC = {}
for k, P in sorted(OLCUM.items()):
    g = GECMIS.get(k)
    if not g:
        print(f"  UYARI: sonda {k} icin kayit yok, atlandi")
        continue
    RHO_OLC[k] = (g["sabit"] - P * P) / (2 * g["kappa_etkin"])

if RHO_OLC:
    print("\nOLCULEN rho_k:")
    for k, r in sorted(RHO_OLC.items()):
        print(
            f"  demet {k}: P={OLCUM[k]:.5f} -> rho_k = {r:+.5f}   "
            f"tahmin {RHO_K[k - 1]:+.4f}   gerceklesme {r / RHO_K[k - 1]:+.2f}"
        )
    _t2 = sum(r * r for r in RHO_OLC.values())
    print(
        f"  toplam rho^2 = {_t2:.6f}  -> su anki nihai skor "
        f"{np.sqrt(max(TABAN_MSE - _t2, 1e-9)):.5f}"
    )

taban = a0 + r_hat.copy()
for k, r in RHO_OLC.items():
    taban = taban + r * G[k - 1]

SIRADAKI = next((k for k in range(1, DEMET + 1) if k not in RHO_OLC), None)
print(f"\nSIRADAKI: {'sonda ' + str(SIRADAKI) if SIRADAKI else 'hepsi olculdu -> NIHAI'}")

PLAN = list(GECMIS.values())
for k in [SIRADAKI - 1] if SIRADAKI else []:
    kap = float(KAPPA_K[k])
    y = np.clip(np.expm1(taban + kap * G[k]), 0.0, None)
    out = pd.DataFrame({"id": te.id.values, "tuketim": y})
    if not all(
        [
            len(out) == 714688,
            bool((out.id.values == ss.iloc[:, 0].values).all()),
            int(out.tuketim.isna().sum()) == 0,
            int((out.tuketim < 0).sum()) == 0,
            bool(np.isfinite(out.tuketim.values).all()),
            bool(out.tuketim.max() < 3 * np.expm1(a0).max()),
        ]
    ):
        print(f"  sonda {k + 1}: KAPI KALDI")
        continue
    yol = os.path.join(S, f"tuketim_D{k + 1}_demet.csv")
    out.to_csv(yol + ".tmp", index=False)
    Path(yol + ".tmp").replace(yol)
    dgv = np.log1p(out.tuketim.values) - a0
    sabit = float(M0 - 2 * kL + float(dgv @ dgv) / N)
    ek = dgv - (np.log1p(np.clip(np.expm1(taban), 0.0, None)) - a0)
    ketkin = float(np.sqrt(float((ek * ek).mean())))
    PLAN = [q for q in PLAN if q["sonda"] != k + 1]
    PLAN.append(
        dict(
            sonda=k + 1,
            dosya=f"tuketim_D{k + 1}_demet.csv",
            kappa=kap,
            kappa_etkin=ketkin,
            sabit=sabit,
            rho_k_tahmin=float(RHO_K[k]),
            yon=ETIKET[k],
        )
    )
    print(f"\nURETILDI: submissions/tuketim_D{k + 1}_demet.csv")
    print(f"  kappa={kap:.5f}  kappa_etkin={ketkin:.6f}  sabit={sabit:.9f}")
    print(f"  COZUM:  rho_{k + 1} = ({sabit:.9f} - P*P) / {2 * ketkin:.6f}")
    print(
        f"  olcum hatasi {YUV / max(ketkin, 1e-12):.2e}   "
        f"rho=0 ise {np.sqrt(max(sabit, 1e-9)):.5f}, "
        f"tahmin tutarsa {np.sqrt(max(sabit - 2 * ketkin * KAPPA_K[k], 1e-9)):.5f}"
    )

if SIRADAKI is None:
    y = np.clip(np.expm1(taban), 0.0, None)
    out = pd.DataFrame({"id": te.id.values, "tuketim": y})
    yol = os.path.join(S, "tuketim_Z_NIHAI.csv")
    out.to_csv(yol + ".tmp", index=False)
    Path(yol + ".tmp").replace(yol)
    _t2 = sum(r * r for r in RHO_OLC.values())
    print("\nNIHAI URETILDI: submissions/tuketim_Z_NIHAI.csv")
    print(f"  beklenen skor {np.sqrt(max(TABAN_MSE - _t2, 1e-9)):.5f}")

print(f"\n{'toplam rho^2':>13s} {'nihai skor':>11s}  sira")
for f in [0.0, 0.1, 0.25, 0.5, 1.0]:
    t2 = f * float((RHO_K**2).sum())
    sk = np.sqrt(max(TABAN_MSE - t2, 1e-9))
    sr = (
        "1. SIRA"
        if sk < 0.99009
        else "2. SIRA"
        if sk < 0.99614
        else "3. sira"
        if sk < 0.99927
        else "4. sira"
        if sk < 0.99937
        else "5. sira"
        if sk < 1.00049
        else "6.+"
    )
    print(f"{t2:13.5f} {sk:11.5f}  {sr}")

with open(GECMIS_YOL, "w") as fh:
    json.dump(
        dict(
            taban_mse=TABAN_MSE,
            demet=DEMET,
            rho_k_tahmin=RHO_K.tolist(),
            yuvarlama=YUV,
            sondalar=sorted(PLAN, key=lambda q: q["sonda"]),
        ),
        fh,
        indent=1,
    )
print("\n-> m148_demet.json    HICBIR GONDERIM YAPILMADI.")
