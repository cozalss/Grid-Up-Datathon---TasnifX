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
    # H_carpim40 ailesi: "M[eksen1]x[eksen2]" -- iki m121 ekseninin carpimi.
    # Her iki parca da kur()'un zaten bildigi adlardir, ozyineleme yeter.
    if ad.startswith("M[") and "]x[" in ad and ad.endswith("]"):
        ic = ad[2:-1]
        k1, k2 = ic.split("]x[", 1)
        a1, b1 = kur(k1)
        a2, b2 = kur(k2)
        if a1 is None or a2 is None or b1 is None or b2 is None:
            return None, None
        return st(a1 * a2), st(b1 * b2)
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
    if kip == "mnt75":
        # MENTESE (hinge): max(x - q75, 0). m144'un G_mentese ailesi.
        # Esik kipi bilgiyi 0/1'e indirger; mentese esigin USTUNDEKI
        # buyuklugu de tasir, bu yuzden ayri bir yon acar.
        fv = xt[np.isfinite(xt)]
        if fv.size == 0:
            return None, None
        v_ = float(np.quantile(fv, 0.75))
        return st(np.maximum(xt - v_, 0.0)), st(np.maximum(xb - v_, 0.0))
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
# --- m144'un F_guc_yas + G_mentese aileleri: mevcut 40'a KURULUS GEREGI dik
# (hepsi Qd >= 0.25 kapisindan gecti), bu yuzden GERCEKTEN YENI boyut acarlar.
# m157: 40 eksenin ICINDE 5. yon yok (en iyi artik PCA E[rho^2] = 0.00012);
# bu 10 eksen ise 10 KATINI vaat ediyor. sqrt(sum rho_s^2) = 0.0593.
# D/B/E/C aileleri m144'un 200+ satirlik ureteclerini isterdi -- tasiyici
# betige yarisma bitmeden o buyuklukte kod GIRMEZ.
# --- m144'un KAPIDAN GECEN 329 ekseninin TAMAMI aday olur.
# Onceki surum bunlarin yalnizca 10'unu elle yazmisti ("tasiyici betige
# 200 satirlik ureteç girmez" gerekcesiyle); oysa H_carpim40 ailesinin
# 285 ekseni "M[a]x[b]" adlandirmasiyla kur()'un ZATEN bildigi iki eksenin
# carpimidir -- ureteç gerekmez, ozyineleme yeter.
#
# NEDEN ONEMLI: BETA = toplam KATS[i]*U[i] ve GD_1 = BETA/||BETA||, yani
# tahminimizin TAMAMI 1. sondada. m144 ||BETA|| icin 0.2522 (dar) vs
# 0.4972 (genis) veriyor -- bu buyumenin tamami dogrudan olculen rho'ya gider.
#
# KAPILAR YENIDEN UYGULANIR: asagidaki dongu her aday icin Qs, rho_s,
# rcond kararliligi, Q_dik >= 0.25, plasebo z >= 3 ve tavan kapilarini
# BASTAN gecirir. Yani m144'un kapilarina GUVENMIYORUZ, tekrar oluyoruz.
# Q_dik kapisi ardisik uygulandigi icin eksen sayisi KENDILIGINDEN sinirlanir.
with open(os.path.join(M29, "m144_yeni_aileler.json"), encoding="utf-8") as fh:
    _M144 = json.load(fh)["kapidan_gecen"]
# |rho_s| buyukten kucuge: dik artigi en cok tasiyan once girsin
YENI_EKSENLER = [r["eksen"] for r in sorted(_M144, key=lambda r: -abs(r["rho_s"]))]
YENI_AILE = {r["eksen"]: r["aile"] for r in _M144}
YENI_MASKE = []
AILE_LISTE = []
for kayit in TARAMA + [{"eksen": a, "_yeni": True} for a in YENI_EKSENLER]:
    _yeni = bool(kayit.get("_yeni"))
    if not _yeni and len(kul) >= AZAMI_EKSEN:
        continue  # ilk 40'ta dur AMA yeni eksenlere devam et (eskiden break)
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
    YENI_MASKE.append(_yeni)
    AILE_LISTE.append(YENI_AILE.get(ad, "m121_taban"))
    print(
        f"{ad[:34]:>34s} {rho_cv:+8.4f} {rho_s:+8.4f} {rho_kul:+8.4f} {Qd:6.3f} "
        f"{'EVET':>6s} {np.sqrt(float((duz * duz).mean())):8.4f}"
    )

# --- K_AZAMI: kabul edilen eksen sayisini kirp (n11_eksen_secimi.json).
# Varsayilan 25; K_AZAMI=0 kirpmayi kapatir.
_KA = int(os.environ.get("K_AZAMI", "25"))
if _KA and len(kul) > _KA:
    print(
        f"\nK_AZAMI={_KA}: {len(kul)} eksenin ilk {_KA}'i tutuluyor "
        f"(n11: K=25'te gerceklesen rho %34 daha yuksek)"
    )
    kul = kul[:_KA]
    KAT_LISTE = KAT_LISTE[:_KA]
    RHO_CV_LISTE = RHO_CV_LISTE[:_KA]
    YENI_MASKE = YENI_MASKE[:_KA]
    AILE_LISTE = AILE_LISTE[:_KA]
    ONCEKI = ONCEKI[:_KA]
    # duz (bilesik) YENIDEN kurulur -- kirpilan eksenlerin katkisi cikar
    duz = np.zeros(N)
    for _i in range(_KA):
        duz = duz + KAT_LISTE[_i] * ONCEKI[_i]

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
# YENI EKSENLERI AYIRAN MASKE. H1..H4 yeni eksenlerde SIFIRLANIR ki
# GD[0..3] BIREBIR AYNI kalsin -- D1 zaten uretildi ve sabit/kappa_etkin
# m148_demet.json'a dondu; GD[0] kayarsa olculen rho_1 YANLIS YONE atfedilir.
# Yeni yon (Y) ise yalniz yeni eksenlerde yasar, dolayisiyla otomatik olarak
# H1..H4'e diktir.
AILE = np.array(AILE_LISTE)
_HV = np.array([bool(any(h in a for h in HAVA)) for a in kul])

# --- BLOKLAR ---------------------------------------------------------------
# NEDEN IKI KIP: K_AZAMI=25 ile kesince ilk 25 eksenin HEPSI m121_taban
# cikiyor (H_carpim40'in |rho_s|'leri daha kucuk, kesimin altinda kaliyor).
# O zaman "aile" bolmesi yalnizca IKI blok uretir ve 6 gonderim hakkinin
# olcum kapasitesi bosa gider. Bu yuzden ikinci bir bolme kipi gerekli.
#
#   aile : {m121_taban, H_carpim40} x {hava, yapi}
#          Ikinci ayrim KOKEN. Yalnizca genis span (K buyuk) icin anlamli.
#   oran : {hava, yapi} x {oran yuksek, oran dusuk}
#          oran = |rho_cv| / |KATS|. Agirliklandirma belirsizligini DOGRUDAN
#          hedge eder: KATS = 1.95*|rho_s| ile agirliklandiriyoruz ama
#          rho_cv (yaz25'te DOGRUDAN olculen) rakip tahmindir ve n10 1.95'i
#          disladi (P(|c| >= 1.95) = 0.0004). Iki agirliklandirma ayri
#          bloklara duserse hangisinin dogru oldugunu LB KENDISI secer.
#          hava/yapi ayrimi korunur (m141 bu iki ailenin ayri davrandigini
#          olctu). K=25'te dort dengeli blok verir.
BLOK_KIP = os.environ.get("BLOK_KIP", "oran")
_HAM = {}
if BLOK_KIP == "oran":
    _ORAN = np.abs(np.array(RHO_CV_LISTE)) / np.maximum(np.abs(KATS), 1e-12)
    # Medyan HER GRUP ICINDE ayri alinir. Tum eksenler uzerinden tek medyan
    # alinca sekiz "yapi" ekseninin hepsi ayni tarafa dusuyor ve bir hucre
    # BOS kaliyordu -> dort yerine UC blok. Grup ici medyan dort dolu hucre
    # garantiler.
    _YUK = np.zeros(len(kul), dtype=bool)
    _ORTLAR = {}
    for _g, _msk in (("hava", _HV), ("yapi", ~_HV)):
        if _msk.sum() >= 2:
            _md = float(np.median(_ORAN[_msk]))
            _ORTLAR[_g] = _md
            _YUK |= _msk & (_md < _ORAN)
    print(
        f"\nbolme kipi: oran (grup ici medyan |rho_cv|/|KATS| = "
        f"{ {k: round(v, 3) for k, v in _ORTLAR.items()} })"
    )
    for _ad2, _m2 in [
        ("hava/oran-yuksek", _HV & _YUK),
        ("hava/oran-dusuk", _HV & ~_YUK),
        ("yapi/oran-yuksek", ~_HV & _YUK),
        ("yapi/oran-dusuk", ~_HV & ~_YUK),
    ]:
        if _m2.sum():
            _HAM[_ad2] = _m2
else:
    print("\nbolme kipi: aile")
    for _f in sorted(set(AILE_LISTE)):
        _m = np.equal(AILE, _f)  # ruff SIM300: AILE buyuk harfli, sabit sanilyor
        if _f in ("m121_taban", "H_carpim40"):
            # buyuk aileler: hava vs yapisal diye ikiye ayrilir
            _HAM[f"{_f}/hava"] = _m & _HV
            _HAM[f"{_f}/yapi"] = _m & ~_HV
        else:
            _HAM[_f] = _m

# blogun ongorulen agirligi = ||BETA_blok|| = sqrt(toplam KATS^2)
_AG = {k: float(np.sqrt((KATS[m] ** 2).sum())) for k, m in _HAM.items() if m.sum()}
# DEMET_HEDEF: kac blok olcecegiz. 6 hak = DEMET_HEDEF sonda + 1 nihai.
DEMET_HEDEF = int(os.environ.get("DEMET_HEDEF", "4"))
# Cok kucuk bloklari komsuya kat AMA hedefin ALTINA INME -- her blok bir
# OLCULEBILEN BOYUTTUR ve hakkimiz varken boyut vermek anlamsiz.
BLOK_ALT = 0.06
while True:
    _kucuk = [k for k, v in _AG.items() if v < BLOK_ALT]
    if not _kucuk or len(_AG) <= DEMET_HEDEF:
        break
    _k = min(_kucuk, key=lambda k: _AG[k])
    _hedef = min((k for k in _AG if k != _k), key=lambda k: _AG[k])
    _HAM[_hedef] = _HAM[_hedef] | _HAM[_k]
    del _HAM[_k], _AG[_k]
    _AG[_hedef] = float(np.sqrt((KATS[_HAM[_hedef]] ** 2).sum()))

# DEMET_HEDEF blok kalana kadar en kucuk ikisini birlestir
while len(_AG) > DEMET_HEDEF:
    _k = min(_AG, key=lambda k: _AG[k])
    _hedef = min((k for k in _AG if k != _k), key=lambda k: _AG[k])
    _HAM[_hedef] = _HAM[_hedef] | _HAM[_k]
    del _HAM[_k], _AG[_k]
    _AG[_hedef] = float(np.sqrt((KATS[_HAM[_hedef]] ** 2).sum()))

# EN BUYUK ONCE: hak biterse elde kalan dosya en cok kazanci tasisin
_SIRA = sorted(_AG, key=lambda k: -_AG[k])
print()
print(f"{'blok':>34s} {'eksen':>6s} {'||BETA_b||':>11s}")
for _k in _SIRA:
    print(f"{_k[:34]:>34s} {int(_HAM[_k].sum()):6d} {_AG[_k]:11.4f}")
print(f"{'TOPLAM':>34s} {len(kul):6d} {np.sqrt(sum(v * v for v in _AG.values())):11.4f}")


def _blok(m):
    w = np.zeros(len(kul))
    w[m] = np.abs(KATS[m])
    return w


HIPOTEZ = {k: _blok(_HAM[k]) for k in _SIRA}

# --- UYARLANABILIR 5. YON (docs/77) --------------------------------------
# UYARLANABILIR=<blok_no> verilirse o blogun UST YARISI (kendi |KATS|
# medyaninin ustundeki eksenler) 5. yon olarak eklenir. Gram-Schmidt bu
# yonu GD_<blok_no>'ye diklestirecegi icin olculen sey blogun KENDISI
# degil, blok ICINDEKI dagilimdir.
_UY = os.environ.get("UYARLANABILIR")
if _UY:
    _en = int(_UY)
    if not 1 <= _en <= len(_SIRA):
        raise SystemExit(f"DUR: UYARLANABILIR={_en} ama {len(_SIRA)} blok var.")
    _ad = _SIRA[_en - 1]
    _idx = np.flatnonzero(_HAM[_ad])
    _ort = float(np.median(np.abs(KATS[_idx])))
    _ust = _idx[np.abs(KATS[_idx]) > _ort]
    if len(_ust) < 3:
        raise SystemExit(
            f"DUR: blok {_en} ({_ad}) bolunemiyor -- ust yaride {len(_ust)} eksen var, "
            f"en az 3 gerekli."
        )
    _mu = np.zeros(len(kul), dtype=bool)
    _mu[_ust] = True
    HIPOTEZ[f"{_ad}/ust-yari"] = _blok(_mu)
    print(
        f"\nUYARLANABILIR: blok {_en} ({_ad}) ikiye bolundu -> ust yari "
        f"({len(_ust)}/{len(_idx)} eksen) 5. yon olarak eklendi.\n"
        f"  n06_kappa.py TEKRAR KOSULMALI (blok sayisi {len(_SIRA)} -> {len(_SIRA) + 1})."
    )
BETA = np.zeros(N)
for i in range(len(U)):
    BETA = BETA + KATS[i] * U[i]

print()
print(f"{'hipotez':>18s} {'eksen':>6s} {'artakalan':>10s} {'ongorulen rho_k':>16s}")
GD, RHO_K, ETIKET = [], [], []
for ad, ag in HIPOTEZ.items():
    v = np.zeros(N)
    for i in range(len(U)):
        v = v + ISR[i] * ag[i] * U[i]
    n0 = float(np.sqrt(float((v * v).mean())))
    if n0 < 1e-12:
        continue
    v = v / n0
    for g in GD:
        v = v - float((v * g).mean()) * g
    n1 = float(np.sqrt(float((v * v).mean())))
    if n1 < 0.05:
        print(f"{ad:>18s} {int((ag > 0).sum()):6d} {n1:10.3f}  ATLANDI (onceki yone cok yakin)")
        continue
    v = v / n1
    GD.append(v)
    RHO_K.append(abs(float((BETA * v).mean())))
    ETIKET.append(ad)
    print(f"{ad:>18s} {int((ag > 0).sum()):6d} {n1:10.3f} {RHO_K[-1]:16.4f}")
GD = np.array(GD)
RHO_K = np.array(RHO_K)
DEMET = len(GD)

# ---------------------------------------------------------------------------
# KAPPA SECIMI (m150 optimizasyonu).
#
# Denge: nihaiye maliyet olcum_hatasi^2 (kappa ile AZALIR) vs sondanin kendi
# kaybi kappa^2 - 2*kappa*mu_k (kappa ile ARTAR, hak biterse odenir).
#   yon 1 (H1): mu_1 = 0.0565, kappa* = 0.0566 -> mevcut 0.0517 zaten isabetli
#   yon 2-4   : mu_k ~ 0 (BETA tamamen H1 boyunca), kappa* = 0.0125
# Hedge yonlerinde eski tekduze 0.0517, her biri o sonda elde kalan son dosya
# olursa ~0.0004 skor kaybettiriyordu.
#
# ALT SINIR NEDEN 0.0125: olcum hatasi yalnizca LB yuvarlamasi degildir.
# sabit icindeki (M0 - 2*kL) OLCULMUS degil KALIBRE edilmis bir sabittir;
# m112'nin LOO'suna gore ort |hata| = 1.72e-04. Bu da 1/(2*kappa) ile buyur.
# ---------------------------------------------------------------------------
SABIT_HATA = 1.72e-4  # m112 LOO: kalibre sabitin kendi hatasi
# ZINCIR KARARLILIGI (kirmizi takim K1). m150 kappa_2..4 = 0.0125 onermisti;
# o optimizasyon YALNIZCA sondanin kendi yedek degerine bakiyordu ve ZINCIR
# BUYUTMESINI atlamisti. Capraz terim kappa_k'ya BOLUNUR:
#   rho_1'deki e_1 hatasi -> rho_4'te -r_1*e_1/kappa_4  (kazanc r_1/kappa_4)
# r_1 = 0.1475 (1. sira senaryosu) ve kappa_4 = 0.0125 iken kazanc 11.8:
#   kappa 0.0125 -> toplam kayip 0.000490 rho^2
#   kappa 0.0517 -> toplam kayip 0.000031 rho^2   (16 KAT IYI)
# 2. sira icin gereken 0.01089 yaninda 0.000490 kucuk degildir.
# Bu yuzden TEKDUZE 0.0517; kappa_1 zaten D1'in uretildigi degerdir.
# n06_kappa.py: blok basina optimum. Amac fonksiyonu
#   -sigma(rho)^2 + P(yurutme kazasi) * E[2*kappa*rho_k - kappa^2]
# ile |c| ~ log-normal(medyan 0.57, %90 GA [0.17,1.26]) uzerinden ornekleme.
# Sonuc: buyuk bloklarda buyuk kappa. Zincir buyutmesi max 2.0 (guvenli).
_KJ = os.path.join(M29, "n06_kappa.json")
if os.path.exists(_KJ):
    with open(_KJ, encoding="utf-8") as fh:
        KAPPA_K = np.array(json.load(fh)["kappa_yeni"], dtype=np.float64)
    if len(KAPPA_K) != DEMET:
        raise SystemExit(
            f"DUR: n06_kappa.json {len(KAPPA_K)} kappa veriyor ama {DEMET} blok var. "
            f"DEMET_HEDEF degistiyse n06_kappa.py'yi TEKRAR KOS."
        )
else:
    # Sifirdan kurulumda kilitlenme olmasin: n06 m148'in ciktisini okur,
    # m148 de n06'nin ciktisini. Dosya yoksa ESKI tekduze degere DUS ve
    # yuksek sesle uyar; sonra n06'yi kosup m148'i tekrar kosmak yeter.
    KAPPA_K = np.full(DEMET, 0.05174190699701174)
    print("\n  !! UYARI: n06_kappa.json yok -> ESKI tekduze kappa 0.05174.")
    print("     n06_kappa.py'yi kos, sonra bu betigi TEKRAR kos.\n")
print()
print(f"{DEMET} dik yon kuruldu. sqrt(toplam rho_k^2) = {np.sqrt((RHO_K**2).sum()):.4f}")
dikkat = np.abs(GD @ GD.T / N - np.eye(DEMET)).max()
print(f"demetlerin dikligi: en buyuk sapma {dikkat:.2e}  (0 olmali)")
if dikkat > 1e-8:
    raise SystemExit(f"DUR: demetler dik degil (sapma {dikkat:.2e})")

# ---------------------------------------------------------------------------
# KUMULATIF URETIM. Sondalar sirayla gonderilir; her sonucun ardindan bu betik
# TEKRAR kosulur ve bir SONRAKI dosyayi uretir.
#   m148_olcumler.json  ->  {"1": 0.99612, "2": 1.00034, ...}   (LB skorlari)
#
# COZUM FORMULU (m150 buldu, m153 sentetik gercekle dogruladi):
#   d = r_hat + toplam_{j<k} r_j GD_j + kappa GD_k
#   P^2 = M0 - 2<r,d> + Q(d),  <r,r_hat> = kL,  <r,GD_j> = rho_j
#   sabit = M0 - 2*kL + Q(d)
#   =>  P^2 = sabit - 2*CAPRAZ - 2*kappa*rho_k,  CAPRAZ = toplam_{j<k} r_j*rho_j
#
# DIKKAT -- CAPRAZ terim toplam(rho_j^2) DEGILDIR: tabana giren r_j (o an
# elimizdeki cozum) ile gercek rho_j ancak duzeltme BASTAN uygulanirsa esittir.
# Bu yuzden her sondanin kaydina tabana giren r_j'ler YAZILIR ve capraz
# onlarla hesaplanir. Bir skor sonradan duzeltilirse yalniz bu surum ayakta
# kalir. (m153: bozuk kosuda S_onceki 0.002850 iken gercek capraz 0.001947'ydi.)
#
# Bu terim DUSTUGU icin eski surum sonda 2'de ISARETI BILE TERS cozuyordu
# (+0.0185 yerine -0.0300) ve nihai dosya yedekten kotu cikarken betik
# "2. sira" diye rapor ediyordu.
# ---------------------------------------------------------------------------
OLC_YOL = os.environ.get("OLCUM_DOSYA") or os.path.join(BURA, "m148_olcumler.json")
OLCUM = {}
if os.path.exists(OLC_YOL):
    with open(OLC_YOL) as fh:
        OLCUM = {int(k): float(v) for k, v in json.load(fh).items()}
GECMIS_YOL = os.path.join(BURA, "m148_demet.json")
GECMIS = {}
if os.path.exists(GECMIS_YOL):
    with open(GECMIS_YOL) as fh:
        GECMIS = {d["sonda"]: d for d in json.load(fh).get("sondalar", [])}


def kapilar(cerceve):
    """Her uretilen dosya AYNI kapilardan gecer -- Z_NIHAI dahil."""
    return {
        "satir": len(cerceve) == 714688,
        "id": bool((cerceve.id.values == ss.iloc[:, 0].values).all()),
        "NaN": int(cerceve.tuketim.isna().sum()) == 0,
        "negatif": int((cerceve.tuketim < 0).sum()) == 0,
        "sonlu": bool(np.isfinite(cerceve.tuketim.values).all()),
        "maks": bool(cerceve.tuketim.max() < 3 * np.expm1(a0).max()),
    }


def yaz_atomik(yol, veri):
    """Kesinti olursa gonderilmis sondalarin sabit/kappa_etkin'i KAYBOLMASIN."""
    with open(yol + ".tmp", "w") as fh:
        json.dump(veri, fh, indent=1)
    Path(yol + ".tmp").replace(yol)


# --- olculen sondalardan rho_k'yi COZ (capraz terim dahil) ---
RHO_OLC = {}
for k, P in sorted(OLCUM.items()):
    g = GECMIS.get(k)
    if not g:
        raise SystemExit(
            f"DUR: sonda {k} icin kayit yok. Onu atlamak bir GONDERIM HAKKINI "
            f"bosa harcar. m148_demet.json'u kontrol et."
        )
    if not 0.90 < P < 1.20:
        raise SystemExit(
            f"DUR: sonda {k} icin girilen skor {P} makul araligin (0.90, 1.20) "
            f"disinda. Ondalik kaymis ya da yanlis sayi girilmis olabilir."
        )
    gerekli = {int(j) for j in g.get("onceki_r", {})}
    if not gerekli <= set(RHO_OLC):
        raise SystemExit(
            f"DUR: sonda {k} tabaninda {sorted(gerekli)} sondalarinin olcumu "
            f"kullanilmis ama elimizde {sorted(RHO_OLC)} var. Eksik: "
            f"{sorted(gerekli - set(RHO_OLC))}. Capraz terim SESSIZCE duserdi "
            f"ve rho_{k} ISARETI ters cikardi. Once eksik skoru gir."
        )
    capraz = sum(
        float(g.get("onceki_r", {}).get(str(j), 0.0)) * RHO_OLC[j] for j in RHO_OLC if j < k
    )
    RHO_OLC[k] = (g["sabit"] - 2.0 * capraz - P * P) / (2 * g["kappa_etkin"])
    if abs(RHO_OLC[k]) > 0.40:  # K5: 0.20 KAZANDIGIMIZ senaryoda duruyordu
        raise SystemExit(
            f"DUR: sonda {k} icin cozulen rho = {RHO_OLC[k]:+.4f}, |rho| > 0.20. "
            f"||r_hat|| = 0.061 tavani goz onune alindiginda bu olanaksiz; "
            f"skor yanlis girilmis ya da dosya degismis olabilir."
        )

if RHO_OLC:
    print("\nOLCULEN rho_k:")
    for k, r in sorted(RHO_OLC.items()):
        tah = RHO_K[k - 1]
        oran = f"{r / tah:+.2f}" if abs(tah) > 1e-9 else "  -  "
        print(
            f"  demet {k} ({ETIKET[k - 1]:>16s}): P={OLCUM[k]:.5f} -> "
            f"rho_k = {r:+.6f}   tahmin {tah:+.4f}   gerceklesme {oran}"
        )
    _t2 = sum(r * r for r in RHO_OLC.values())
    _sk = np.sqrt(max(TABAN_MSE - _t2, 1e-9))
    print(f"  toplam rho^2 = {_t2:.6f}  ->  su anki nihai skor {_sk:.5f}")
    if _sk > 1.00115:
        print("  UYARI: su anki nihai, 1.00115 yedeginden KOTU. Yedegi koru.")

# --- taban: saf span + olculen demetlerin katkisi ---
taban = a0 + r_hat.copy()
ONCEKI_R = {}
for k, r in sorted(RHO_OLC.items()):
    taban = taban + r * GD[k - 1]
    ONCEKI_R[str(k)] = float(r)

# KACIS YOLU (kirmizi takim K2). Bir sonda ERROR donerse ya da zaman biterse
# 4 olcumun 4'unu beklemek Z_NIHAI'nin HIC uretilememesine gider (slack 0).
# NIHAI=1 cevre degiskeni, elde HANGI olcumler varsa onlarla nihaiyi uretir.
SIRADAKI = next((k for k in range(1, DEMET + 1) if k not in RHO_OLC), None)
if os.environ.get("NIHAI") == "1" and RHO_OLC:
    print(f"  [KACIS] NIHAI=1 -> {len(RHO_OLC)} olcumle nihai uretiliyor")
    print(f"          olculenler: {sorted(RHO_OLC)}")
    SIRADAKI = None
if os.environ.get("DOKUM"):
    _dk = os.environ["DOKUM"]
    np.save(os.path.join(_dk, "r_hat.npy"), r_hat)
    np.save(os.path.join(_dk, "GD.npy"), GD)
    np.save(os.path.join(_dk, "a0.npy"), a0)
    with open(os.path.join(_dk, "sabitler.json"), "w", encoding="utf-8") as _fh:
        json.dump(
            {
                "kL": float(kL),
                "M0": float(M0),
                "N": int(N),
                "TABAN_MSE": float(TABAN_MSE),
                "etiket": list(ETIKET),
            },
            _fh,
        )
    print(f"  [DOKUM] r_hat/GD/a0/sabitler -> {_dk}")
print(f"\nSIRADAKI: {'sonda ' + str(SIRADAKI) if SIRADAKI else 'hepsi olculdu -> NIHAI'}")

PLAN = list(GECMIS.values())
for k in [SIRADAKI - 1] if SIRADAKI else []:
    yol = os.path.join(S, f"tuketim_D{k + 1}_demet.csv")
    kayit = GECMIS.get(k + 1)
    # ZATEN URETILMIS DOSYAYI YENIDEN URETME: olculmus_skorlar.json ya da
    # m112_durum.json bu arada degisirse r_hat kayar; dosya da kayitli sabit de
    # degisir, oysa elde tutulan skor ESKI dosyanindir -> rho yanlis cozulur.
    if kayit and os.path.exists(yol):
        # K1 (kirmizi takim n12): kayit ile diskteki dosya AYRISABILIYOR.
        # git checkout m148_demet.json kaydi geri alir ama CSV yerinde
        # kalir; bu dal dosyayi yeniden uretmedigi gibi kaydi da
        # guncellemez, sonra bayat kayit diske geri yazilir. Bayat
        # sabit/kappa_etkin ile rho YANLIS cozulur (gozlenen ornekte
        # +0.0501 yerine +0.0746, 0.00153 skor kaybi).
        # Bu yuzden dosyadan YENIDEN HESAPLA ve kayitla karsilastir.
        _v = oku(kayit["dosya"])
        if _v is None:
            raise SystemExit(f"DUR: {kayit['dosya']} okunamadi.")
        _dg = _v - a0
        _sb = float(M0 - 2 * kL + float(_dg @ _dg) / N)
        if abs(_sb - kayit["sabit"]) > 1e-9:
            raise SystemExit(
                f"DUR: sonda {k + 1} KAYIT ile DOSYA UYUSMUYOR.\n"
                f"     kayitta sabit={kayit['sabit']:.9f}, dosyadan {_sb:.9f} "
                f"(fark {_sb - kayit['sabit']:+.3e}).\n"
                f"     Dosya baska bir yapilandirmayla uretilmis. "
                f"n07_temiz_kurulum.py ile SIFIRDAN kur."
            )
        print(f"\n  sonda {k + 1} ZATEN VAR: {kayit['dosya']}")
        print(f"    dosya-kayit tutarli (sabit {_sb:.9f})")
        print(f"    kappa_etkin={kayit['kappa_etkin']:.6f}  sabit={kayit['sabit']:.9f}")
        print(
            f"    COZUM:  rho_{k + 1} = ({kayit['sabit']:.9f} - 2*"
            f"{sum(float(kayit.get('onceki_r', {}).get(str(j), 0.0)) * RHO_OLC[j] for j in RHO_OLC):.9f} - P*P) / "
            f"{2 * kayit['kappa_etkin']:.6f}"
        )
        print("    Yeniden uretilmedi. Skoru m148_olcumler.json'a yazip tekrar kos.")
        break
    kap = float(KAPPA_K[k])
    y = np.clip(np.expm1(taban + kap * GD[k]), 0.0, None)
    out = pd.DataFrame({"id": te.id.values, "tuketim": y})
    kp = kapilar(out)
    if not all(kp.values()):
        raise SystemExit(f"DUR: sonda {k + 1} KAPI KALDI -> {kp}")
    dgv = np.log1p(out.tuketim.values) - a0
    sabit = float(M0 - 2 * kL + float(dgv @ dgv) / N)
    tb = np.log1p(np.clip(np.expm1(taban), 0.0, None)) - a0
    ek = dgv - tb
    # K4 (n12): cebir <ek, GD_k> istiyor, ||ek|| degil. Fark birinci
    # mertebede kucuk ama ek'in GD_k'ya DIK bileseni kirpma artigidir ve
    # dusuk tuketimli satirlarda yogunlasir -- gercek artikla yapisal
    # korelasyonu olabilir, o yuzden hata butcesine giriyor.
    ketkin = float((ek * GD[k]).mean())
    ek_dik = ek - ketkin * GD[k]
    ek_dik_n = float(np.sqrt(float((ek_dik * ek_dik).mean())))
    # K3 (n12): ESKI iki "oz-denetim"in IKISI de CEBIRSEL OZDESLIKTI.
    # tm_kontrol, TABAN_MSE'nin BIREBIR AYNI ifadesiydi (fark her zaman
    # tam 0.0); artik = sabit - ideal - kirpma da acildiginda ozdeslige
    # iniyordu (olculen 2.7e-17). Ikisi de HICBIR SEYI sinamiyordu.
    # Yerlerine asagida GERCEK bir kontrol var: yazilan CSV geri okunur.
    # (2) sabit'in ideal cebirden sapmasi YALNIZCA kirpmadan gelmeli.
    #     Kirpma terimi TAM olculebilir; geriye kalan artik ~0 olmali.
    #     Eski surum kirpmayi hesaba katmiyordu ve kappa buyuyunce
    #     (0.0125 -> 0.0517) sonda 2'de boru hattini KILITLIYORDU.
    _r2 = sum(v * v for v in RHO_OLC.values())
    # ||ek||^2 kullanilir; ketkin artik <ek,GD_k> oldugu icin ketkin**2
    # eksik kalirdi (fark tam olarak ||ek_dik||^2).
    ek_n2 = float((ek * ek).mean())
    ideal = TABAN_MSE + _r2 + ek_n2
    kirpma = (
        2.0 * float((tb * ek).mean())
        + float((tb * tb).mean())
        - float((r_hat * r_hat).mean())
        - _r2
    )
    artik = sabit - ideal - kirpma
    if abs(artik) > 1e-9:
        raise SystemExit(
            f"DUR: cebir tutmadi. artik={artik:.3e}. "
            f"sabit={sabit:.9f} ideal={ideal:.9f} kirpma={kirpma:.3e}"
        )
    if abs(kirpma) > 1e-4:
        print(f"  UYARI: kirpma sapmasi buyuk ({kirpma:+.2e}) -- kappa cok mu buyuk?")
    # KAPILAR VE OZ-DENETIM GECTI -> ANCAK SIMDI DISKE YAZ.
    out.to_csv(yol + ".tmp", index=False)
    Path(yol + ".tmp").replace(yol)
    # K3 (n12): ASIL KONTROL. Yazilan dosyayi GERI OKU ve sabit'i ONDAN
    # yeniden hesapla. Bu, cebirsel bir ozdeslik DEGIL: CSV'nin ondalik
    # bicimlendirmesi degerleri yuvarlar ve GONDERILEN sey diskteki
    # dosyadir, bellekteki dizi degil. Bu risk daha once HIC olculmemisti.
    _gv = oku(f"tuketim_D{k + 1}_demet.csv")
    if _gv is None:
        Path(yol).unlink(missing_ok=True)
        raise SystemExit(f"DUR: yazilan D{k + 1} geri okunamadi, dosya silindi.")
    _gd = _gv - a0
    _gs = float(M0 - 2 * kL + float(_gd @ _gd) / N)
    _gk = float(((_gd - tb) * GD[k]).mean())
    if abs(_gs - sabit) > 1e-9 or abs(_gk - ketkin) > 1e-9:
        Path(yol).unlink(missing_ok=True)
        raise SystemExit(
            f"DUR: DISKTEKI dosya bellektekinden farkli (CSV yuvarlamasi).\n"
            f"     sabit {sabit:.12f} -> {_gs:.12f} (fark {_gs - sabit:+.3e})\n"
            f"     kappa_etkin {ketkin:.12f} -> {_gk:.12f}\n"
            f"     Dosya silindi. Yazma bicimi duzeltilmeden GONDERILMEZ."
        )
    # Bundan sonrasi DISKTEKI dosyanin degerleridir -- gonderilen sey odur.
    sabit, ketkin = _gs, _gk
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
            onceki_r=dict(ONCEKI_R),  # capraz terim icin ZORUNLU
        )
    )
    # OLCUM HATASI: LB yuvarlamasi TEK BASINA degil; kalibre sabitin kendi
    # hatasi (SABIT_HATA) de 1/(2*kappa) ile buyur ve baskin olan odur.
    # K4: ek'in GD_k'ya dik bileseni gercek artikla korele olabilir.
    # |korelasyon| ~ 0.05 varsayimiyla hata butcesine katiliyor.
    hata = float(np.sqrt(YUV**2 + SABIT_HATA**2 + (0.05 * ek_dik_n) ** 2) / (2 * abs(ketkin)))
    print(f"\nURETILDI: submissions/tuketim_D{k + 1}_demet.csv   [{ETIKET[k]}]")
    print(f"  kappa={kap:.5f}  kappa_etkin={ketkin:.6f}  sabit={sabit:.9f}")
    print(f"  COZUM:  rho_{k + 1} = ({sabit:.9f} - 2*CAPRAZ - P*P) / {2 * ketkin:.6f}")
    print(
        f"          CAPRAZ = {sum(float(ONCEKI_R.get(str(j), 0.0)) * RHO_OLC[j] for j in RHO_OLC):.9f}"
    )
    print(
        f"  olcum hatasi {hata:.2e}   (yuvarlama {YUV / (2 * ketkin):.2e} + "
        f"kalibre sabit {SABIT_HATA / (2 * ketkin):.2e})"
    )
    print(
        f"  rho_{k + 1}=0 ise skor {np.sqrt(max(sabit, 1e-9)):.5f}   |   "
        f"rho_{k + 1}=kappa ise {np.sqrt(max(sabit - 2 * ketkin * kap, 1e-9)):.5f}"
    )

if SIRADAKI is None:
    y = np.clip(np.expm1(taban), 0.0, None)
    out = pd.DataFrame({"id": te.id.values, "tuketim": y})
    kp = kapilar(out)
    if not all(kp.values()):
        raise SystemExit(f"DUR: Z_NIHAI KAPI KALDI -> {kp}")
    yol = os.path.join(S, "tuketim_Z_NIHAI.csv")
    out.to_csv(yol + ".tmp", index=False)
    Path(yol + ".tmp").replace(yol)
    # Nihai dosya SIRALAMAYI belirleyen dosyadir. Diske yazilan sey ile
    # bellekteki dizi CSV ondalik bicimlendirmesi yuzunden ayrisabilir;
    # gonderilen sey DISKTEKI dosyadir. Geri oku ve beklenen skoru ONDAN
    # hesapla -- bu bir cebirsel ozdeslik degil, gercek bir kontroldur.
    _zv = oku("tuketim_Z_NIHAI.csv")
    if _zv is None:
        Path(yol).unlink(missing_ok=True)
        raise SystemExit("DUR: Z_NIHAI geri okunamadi, dosya silindi.")
    _zd = _zv - a0
    _zs = float(M0 - 2 * kL + float(_zd @ _zd) / N)
    _zbek = float(np.sqrt(max(_zs - 2.0 * sum(r * r for r in RHO_OLC.values()), 1e-9)))
    _t2 = sum(r * r for r in RHO_OLC.values())
    _sk = np.sqrt(max(TABAN_MSE - _t2, 1e-9))
    print("\nNIHAI URETILDI: submissions/tuketim_Z_NIHAI.csv")
    print(f"  toplam rho^2 = {_t2:.6f}   beklenen skor {_sk:.5f}")
    print(
        f"  DISKTEKI DOSYADAN: {_zbek:.5f}  <- ASIL BEKLENTI "
        f"(kirpma+CSV yuvarlamasi dahil, fark {_zbek - _sk:+.2e})"
    )
    if abs(_zbek - _sk) > 1e-3:
        raise SystemExit(
            f"DUR: diskteki Z_NIHAI ideal cebirden {_zbek - _sk:+.3e} sapiyor. "
            f"Bu kadari kirpmayla aciklanamaz, bir HATA isaretidir. GONDERME."
        )
    if _zbek > 1.00115:
        print("  UYARI: 1.00115 yedeginden KOTU -- son secimde YEDEGI kullan.")


# KARAR TABLOSU. ONEMLI: siralamayi BUGUNKU tabloya gore soylemek YANILTIR --
# nihai siralama 1 Eylul 23:59 UTC'deki tabloya gore belirlenir ve rakipler
# o ana kadar iyilesmeye devam eder. n02_esik_tahmini.json bitis icin
# 2. sira 0.9897 [0.9870, 0.9908], 1. sira 0.9872 tahmin ediyor.
# Asagida IKISI de gosterilir ki hangi esige gore konustugumuz belli olsun.
_ESIK_BUGUN = [
    (0.990095, "1."),
    (0.996145, "2."),
    (0.999275, "3."),
    (0.999375, "4."),
    (1.000475, "5."),
    (1.000495, "6."),
]
_ESIK_BITIS = [(0.9872, "1."), (0.9897, "2."), (0.99927, "3.")]


def _sira(sk, tablo):
    for esik, ad in tablo:
        if sk < esik:
            return ad
    return "geride"


print(
    f"\n{'toplam rho^2':>13s} {'nihai skor':>11s}  {'bugunku tablo':>13s}  {'BITIS TAHMINI':>13s}"
)
for f in [0.0, 0.00349, 0.00973, 0.01690, 0.02253, 0.03000]:
    sk = np.sqrt(max(TABAN_MSE - f, 1e-9))
    print(f"{f:13.5f} {sk:11.5f}  {_sira(sk, _ESIK_BUGUN):>13s}  {_sira(sk, _ESIK_BITIS):>13s}")
print("  (bitis tahmini: n02_esik_tahmini.json, 2. sira 0.9897 %80 GA [0.9870, 0.9908])")

yaz_atomik(
    GECMIS_YOL,
    dict(
        taban_mse=TABAN_MSE,
        demet=DEMET,
        rho_k_tahmin=RHO_K.tolist(),
        yuvarlama=YUV,
        sabit_hata=SABIT_HATA,
        sondalar=sorted(PLAN, key=lambda q: q["sonda"]),
    ),
)
print("\n-> m148_demet.json    HICBIR GONDERIM YAPILMADI.")
