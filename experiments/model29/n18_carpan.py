"""n18 -- CARPAN'i n=1 yerine COK NOKTAYLA olcmeye calis.

HICBIR GONDERIM YAPILMAZ. submissions/ altina YAZILMAZ. m148 DEGISTIRILMEZ.

=== SORU ===
m148:  rho_cv = CARPAN * kor      (kor = yaz25 blok korelasyonu)
       CARPAN = 0.798             (m113_yon_kurucu.py:199 -> -0.0304 / kor_seviye)
Yani CARPAN TEK bir eksenden (seviye) geliyor, n=1.

=== GOREVIN ONERDIGI YOL VE NEDEN CALISMIYOR (BOLUM 1) ===
29 gonderimin LB korelasyonu cebirsel olarak cikar; ama BLOK korelasyonu
icin gonderim farkinin yaz25 KARSILIGI gerekir. Bolum 1 bunun kurulup
kurulamayacagini VERIYLE sinar (onbellekte hangi vekiller var).

=== YERINE YAPILABILEN OLCUM (BOLUM 2-5) ===
CARPAN'in tanimi CARPILARAK ikiye ayrilir:
    CARPAN = rho_dik^LB(seviye) / kor(seviye)
           = [rho_dik^LB/rho_s](seviye) * [rho_s/kor](seviye)
           =        |c|                 *        T
  |c| : n10 bunu 29 LB olcumuyle OLCTU  -> 0.434, %90 GA [0.184, 0.798]
  T   : rho_s (LB-capali span korelasyonu) / kor (blok korelasyonu).
        T HER EKSEN ICIN OLCULEBILIR -- cunku rho_s LB'den, kor bloktan
        dogrudan gelir. n=1 yerine n=629 aday eksen.
Boylece CARPAN = |c| * T ve HER IKI carpan da coklu olcume dayanir.

DIKKAT (ve raporda soylenecek): bu ayrisim dogruysa YOL B (CARPAN yolu)
YOL A/C'den BAGIMSIZ DEGILDIR -- ikisi de |c|'ye dayanir. YOL B'nin
"bagimsiz" gorunmesi T'nin yalnizca seviye ekseninde degerlendirilmis
olmasindan geliyor.
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
HEDEF_SOGUK, CARPAN_ESKI, TAVAN_ESKI = 0.222, 0.798, 1.95
SIGMA_L = 2.9377803611172106e-06  # n10, olculdu (G'nin tam sifir kipleri)
sys.path.insert(0, M29)
from m112_kalibre import EK_MODEL, M0, buzmeli_r_hat  # noqa: E402

np.seterr(all="ignore")
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
V, L, ADV, PSK = [], [], [], []
for f, Pj in SK.items():
    if f == TABAN or not os.path.exists(os.path.join(S, f)):
        continue
    v = oku(f)
    if v is None or len(v) != N:
        continue
    dd = v - a0
    V.append(dd)
    L.append((M0 + float((dd * dd).mean()) - Pj * Pj) / 2)
    ADV.append(f)
    PSK.append(Pj)
for f, Lj in EK_MODEL.items():
    V.append(oku(f) - a0)
    L.append(Lj)
    ADV.append(f)
    PSK.append(None)
for o in DUR.get("olcumler", []):
    dd = oku(o["dosya"]) - a0
    V.append(dd)
    L.append((M0 + float((dd * dd).mean()) - o["skor"] ** 2) / 2)
    ADV.append(o["dosya"])
    PSK.append(o["skor"])
V, L = np.array(V).T, np.array(L)
K_V = V.shape[1]
G = (V.T @ V) / N
Gi = np.linalg.pinv(G, rcond=1e-6)
SIG_VEC = np.full(K_V, SIGMA_L)
r_hat, gercek, kL = buzmeli_r_hat(V, L, G, N, sigma=SIG_VEC)
print("=" * 78)
print("n18  CARPAN'IN COK-NOKTALI OLCUMU")
print("=" * 78)
print(f"span V: {V.shape}   ||r_hat|| = {np.sqrt(float((r_hat * r_hat).mean())):.5f}")

# ---------------------------------------------------------------------------
# BOLUM 1 -- GOREVIN ISTEDIGI DOGRUDAN YOL: gonderim farkinin yaz25 karsiligi
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("BOLUM 1  Gonderim farki d_j'nin yaz25 KARSILIGI kurulabilir mi?")
print("=" * 78)
_test_onbellek = [f for f in os.listdir(AO) if f.startswith("test")]
_npz = os.listdir(DN)
print(f"aile_onbellek'te 'test' onekli dosya sayisi : {len(_test_onbellek)}")
print(f"deney/ icerigi                              : {_npz}")
_z = np.load(os.path.join(DN, "sicak_tahmin.npz"))
print(
    f"sicak_tahmin.npz anahtarlari (blok onekleri): {sorted({k.split('_')[0] for k in _z.files})}"
)
print("""
HUKUM 1. Gonderim farkinin yaz25 karsiligi KURULAMAZ. Uc bagimsiz neden:
  (a) TEST TARAFINDA MODEL ONBELLEGI YOK. aile_onbellek ve sicak_tahmin.npz
      YALNIZCA yaz25/guz25/kis26 bloklarini tasiyor; test icin tek bir
      aile tahmini bile onbellekte degil. Yani "ayni modelin iki tarafi"
      eslemesi ancak a0 <-> pb ciftinde (m148'in x_sv ekseni) vardir.
  (b) 27 SKORLU GONDERIMIN COGU ESKI BORU HATLARINDAN. v2/v7/v15/.../v109,
      m4, p51, s3y40 haftalar icinde farkli betiklerle uretildi; hicbirinin
      yaz25 tahmini diske yazilmadi ve uretim yapilandirmalari (harman
      agirliklari, kirpma, kapasite kurallari) tek bir bildirimde degil.
  (c) KALAN 2 GONDERIM (YP_seviye, K_yenibas) a0 + r_hat + kappa*xp
      bicimindedir; xp span(V)'ye DIK bilesendir ve span(V) yine
      gonderimlerden kuruludur -- (a)+(b) yuzunden onun da blok karsiligi yok.
Dolayisiyla "29 nokta uzerinde CARPAN_j = rho_j^LB / rho_j^blok" HESAPLANAMAZ.
""")

# ---------------------------------------------------------------------------
# BLOK KURULUSU -- m148_demet_plani.py 95-113 satirlarinin AYNISI
# ---------------------------------------------------------------------------
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
zs = np.load(os.path.join(DN, "soguk_tahmin_yaz25.npz"))
idx = np.concatenate([sic.index.values, sog.index.values])
pb = np.concatenate([np.mean(P, axis=0), np.mean([zs[q] for q in zs.files], axis=0)])
bf = e.loc[idx].copy()
rb = np.log1p(bf.tuketim.values.astype(np.float64)) - pb
sgm = bf.soguk_mu.values.astype(np.float64)
ww = np.where(sgm == 1, HEDEF_SOGUK / sgm.mean(), (1 - HEDEF_SOGUK) / (1 - sgm.mean()))
ww = ww / ww.mean()
m0b = float((ww * rb * rb).mean())
del P, blk, sic, sog


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
        fv = xt[np.isfinite(xt)]
        if fv.size == 0:
            return None, None
        v_ = float(np.quantile(fv, 0.75))
        return st(np.maximum(xt - v_, 0.0)), st(np.maximum(xb - v_, 0.0))
    if kip == "kare":
        a_, b_ = st(xt), st(xb)
        return (None, None) if a_ is None else (st(a_**2), st(b_**2))
    return st(xt), st(xb)


# ---------------------------------------------------------------------------
# BOLUM 2 -- seviye CAPASINI YENIDEN URET (0.798'in kendisi dogrulanir)
# ---------------------------------------------------------------------------
print("=" * 78)
print("BOLUM 2  CARPAN = 0.798 CAPASININ YENIDEN URETIMI")
print("=" * 78)
xt_sv, xb_sv = svT, svB
cc = Gi @ ((V.T @ xt_sv) / N)
xp = xt_sv - V @ cc
Qs_sv = 1.0 - float((xp * xp).mean())
rho_s_sv = float((r_hat * xt_sv).mean()) / np.sqrt(Qs_sv)
kor_sv = float((ww * rb * xb_sv).mean()) / np.sqrt(m0b)
print(f"  seviye ekseni: Q_span = {Qs_sv:.4f}")
print(f"  rho_s(seviye) = {rho_s_sv:+.5f}   (LB'den; m112/docs-69 degeri +0.0156)")
print(f"  kor (seviye)  = {kor_sv:+.5f}   (yaz25 blok)")
print(f"  m113'un tanimi: CARPAN = -0.0304 / kor  = {-0.0304 / kor_sv:+.4f}   (belgedeki 0.798)")
print(f"  |c|(seviye)   = -0.0304 / rho_s = {-0.0304 / rho_s_sv:+.4f}   (belgedeki 1.95)")
T_SV = rho_s_sv / kor_sv
print(f"  T(seviye) = rho_s/kor = {T_SV:+.4f}")
print(f"  KONTROL: |c|*T = {abs(-0.0304 / rho_s_sv) * abs(T_SV):.4f}  ~ 0.798 olmali")

# ---------------------------------------------------------------------------
# BOLUM 3 -- T = rho_s/kor'u TUM ADAY EKSEN HAVUZUNDA olc
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("BOLUM 3  T = rho_s / kor -- n=1 yerine tum aday havuzu")
print("=" * 78)
with open(os.path.join(M29, "m121_derin_tarama.json")) as fh:
    TARAMA = json.load(fh)
with open(os.path.join(M29, "m144_yeni_aileler.json"), encoding="utf-8") as fh:
    M144 = json.load(fh)["kapidan_gecen"]
ADAY = [(r["eksen"], "m121") for r in TARAMA] + [(r["eksen"], r.get("aile", "m144")) for r in M144]
gor = set()
ADAY = [(a, f) for a, f in ADAY if not (a in gor or gor.add(a))]
print(f"aday eksen (tekil): {len(ADAY)}")

rng = np.random.default_rng(5)
tn = bf.tanim.values
uqn = pd.unique(tn)
gi = pd.Series(np.arange(len(uqn)), index=uqn)[tn].to_numpy()
PERM = [
    np.argsort(np.argsort(rng.permutation(len(uqn))[gi], kind="stable"), kind="stable")
    for _ in range(20)
]

# rho_s'in olcum gurultusu: L'yi sigma_L ile sars, r_hat'i yeniden kur.
NSIM = 24
rng2 = np.random.default_rng(11)
RHAT_SIM = []
for _ in range(NSIM):
    Lp = L + rng2.normal(0.0, SIGMA_L, K_V)
    rh, _, _ = buzmeli_r_hat(V, Lp, G, N, sigma=SIG_VEC)
    RHAT_SIM.append(rh)
print(f"rho_s gurultusu icin {NSIM} L-sarsimi kuruldu (sigma_L = {SIGMA_L:.2e})")

HAVA = (
    "sicak", "cdd", "hdd", "nem", "vpd", "et0", "bulut", "yagis", "hissedilen",
    "asiri", "ruzgar", "guneslen", "gunes", "x_ay", "ay_", "sicaklik",
)  # fmt: skip

AD, RS, KO, GUR, SRS, QS, AILE, HV = [], [], [], [], [], [], [], []
for ad, aile in ADAY:
    xt, xb = kur(ad)
    if xt is None or xb is None:
        continue
    cc = Gi @ ((V.T @ xt) / N)
    xp = xt - V @ cc
    Qs = 1.0 - float((xp * xp).mean())
    if Qs < 0.02:
        continue
    ip = float((r_hat * xt).mean())
    rho_s = ip / np.sqrt(Qs)
    srs = float(np.std([float((rh * xt).mean()) / np.sqrt(Qs) for rh in RHAT_SIM]))
    kor = float((ww * rb * xb).mean()) / np.sqrt(m0b)
    gur = float(np.std([float((ww * rb[s] * xb).mean()) / np.sqrt(m0b) for s in PERM]))
    AD.append(ad)
    RS.append(rho_s)
    KO.append(kor)
    GUR.append(gur)
    SRS.append(srs)
    QS.append(Qs)
    AILE.append(aile)
    HV.append(bool(any(h in ad for h in HAVA)))
RS, KO, GUR, SRS, QS = map(np.array, (RS, KO, GUR, SRS, QS))
HV = np.array(HV)
AILE = np.array(AILE)
n = len(AD)
print(f"kurulabilen ve Qs>=0.02 gecen eksen: {n}")
print(f"ort sigma(rho_s) = {SRS.mean():.2e}   ort gurultu(kor) = {GUR.mean():.2e}")


def egim(rs, ko, srs, gur, ad=""):
    """T tahmincileri.

    DIKKAT -- BURADA HATA-ICINDE-DEGISKEN (Deming) DUZELTMESI YAPILMAZ.
    Once denendi ve REDDEDILDI: lam = var(e_y)/var(e_x) ~ 1.2e-3 gibi cok
    kucuk oldugu icin Deming cozumu Syy/Sxy'ye (ters baglanim) degeniyor ve
    sacilmayi TAMAMEN olcum hatasina yaziyor. Oysa SNR_kor medyani ~20;
    olcum hatasi sacilmanin ancak binde ikisini aciklar. Sacilma GERCEK
    HETEROJENLIKTIR (eksenden eksene T farki), gurultu degil. Bu durumda
    dogru tahminci sifirdan gecen OLS'tir: "kor verildiginde rho_s'in
    beklenen degeri" tam olarak kullanmak istedigimiz seydir.
    """
    sxx, syy, sxy = float((ko * ko).sum()), float((rs * rs).sum()), float((ko * rs).sum())
    ols = sxy / sxx if sxx > 0 else np.nan
    med = float(np.nanmedian(rs / np.where(np.abs(ko) > 1e-12, ko, np.nan)))
    l1 = float(np.abs(rs).sum() / np.abs(ko).sum()) if np.abs(ko).sum() > 0 else np.nan
    return dict(
        ols=float(ols),
        medyan=med,
        mutlak_oran=l1,
        korelasyon=float(sxy / np.sqrt(sxx * syy)) if sxx * syy > 0 else np.nan,
        ad=ad,
    )


SNR_K = np.abs(KO) / np.maximum(GUR, 1e-30)
SNR_R = np.abs(RS) / np.maximum(SRS, 1e-30)
TJ = RS / np.where(np.abs(KO) > 1e-12, KO, np.nan)

print(
    f"\n{'alt kume':>28s} {'n':>4s} {'OLS_T':>7s} {'medyan':>7s} {'|.|oran':>8s} {'r(rs,kor)':>11s}"
)
KUMELER = {}
for etk, msk in [
    ("TUM", np.ones(n, bool)),
    ("SNR_kor>=5", SNR_K >= 5),
    ("SNR_kor>=5 & SNR_rs>=3", (SNR_K >= 5) & (SNR_R >= 3)),
    ("hava", HV),
    ("yapisal", ~HV),
    ("m121 havuzu", AILE == "m121"),
    ("m144 H_carpim40", AILE == "H_carpim40"),
    ("|rho_s|>=0.015 (m148 kapisi)", np.abs(RS) >= 0.015),
    ("Q_span>=0.5", QS >= 0.5),
]:
    if msk.sum() < 5:
        continue
    r = egim(RS[msk], KO[msk], SRS[msk], GUR[msk], etk)
    med = float(np.nanmedian(TJ[msk]))
    KUMELER[etk] = dict(
        n=int(msk.sum()), **{k: r[k] for k in ("ols", "medyan", "mutlak_oran", "korelasyon")}
    )
    print(
        f"{etk:>28s} {int(msk.sum()):4d} {r['ols']:7.3f} {r['medyan']:7.3f} "
        f"{r['mutlak_oran']:8.3f} {r['korelasyon']:11.3f}"
    )


# --- KUME ONYUKLEME: eksenler birbirine cok korele; oznitelik KOKUNE gore blokla
def kok(ad):
    a = ad
    if a.startswith("M["):
        a = a[2:-1].split("]x[")[0]
    a = a.split("*")[0].split(":")[0]
    return a


KOK_ET = np.array([kok(a) for a in AD])
UK = np.unique(KOK_ET)
print(f"\nonyukleme kumesi (oznitelik koku) sayisi: {len(UK)}")
rng3 = np.random.default_rng(2026)
NB = 2000
boot = []
msk_ana = SNR_K >= 5
for _ in range(NB):
    sec = rng3.choice(UK, len(UK), replace=True)
    ix = np.concatenate([np.flatnonzero(msk_ana & (u == KOK_ET)) for u in sec])
    if len(ix) < 5:
        continue
    boot.append(egim(RS[ix], KO[ix], SRS[ix], GUR[ix])["ols"])
boot = np.array([b for b in boot if np.isfinite(b)])
T_NOKTA = float(egim(RS[msk_ana], KO[msk_ana], SRS[msk_ana], GUR[msk_ana])["ols"])
T_LO, T_HI = (float(x) for x in np.quantile(boot, [0.05, 0.95]))
print(f"T (sifirdan OLS, SNR_kor>=5) = {T_NOKTA:+.4f}   %90 GA [{T_LO:+.4f}, {T_HI:+.4f}]")
print(f"T(seviye) tek nokta    = {T_SV:+.4f}   <- 0.798'in dayandigi deger")

# ---------------------------------------------------------------------------
# BOLUM 4 -- T bilesen sayisina gore degisiyor mu? (bilesik kurup K'yi tara)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("BOLUM 4  T BILESEN SAYISINA (K) gore degisiyor mu?")
print("=" * 78)
# m148'in kendi secim sirasini kullan: m121 sirasi + m144 |rho_s| sirasi
SIRA = [i for i, (a, f) in enumerate(zip(AD, AILE)) if f == "m121"]
SIRA += sorted([i for i in range(n) if AILE[i] != "m121"], key=lambda i: -abs(RS[i]))
KDEG = [1, 2, 3, 5, 10, 15, 25, 40, 60, 100, 150, 250, len(SIRA)]
KDEG = sorted({k for k in KDEG if k <= len(SIRA)})
print(f"{'K':>6s} {'Q_span':>8s} {'rho_s':>10s} {'kor':>12s} {'T':>10s}")
K_TABLO = []
vt = np.zeros(N)
vb = np.zeros(len(rb))
for pos, i in enumerate(SIRA, start=1):
    xt, xb = kur(AD[i])
    w = np.sign(KO[i]) * abs(RS[i])
    vt += w * xt
    vb += w * xb
    if pos not in KDEG:
        continue
    nt = np.sqrt(float((vt * vt).mean()))
    nb = np.sqrt(float((ww * vb * vb).mean()))
    if nt < 1e-12 or nb < 1e-12:
        continue
    ut, ub = vt / nt, vb / nb
    ccx = Gi @ ((V.T @ ut) / N)
    xpx = ut - V @ ccx
    Qsx = 1.0 - float((xpx * xpx).mean())
    rsx = float((r_hat * ut).mean()) / np.sqrt(max(Qsx, 1e-12))
    korx = float((ww * rb * ub).mean()) / np.sqrt(m0b)
    K_TABLO.append(dict(K=pos, Q_span=Qsx, rho_s=rsx, kor=korx, T=rsx / korx))
    print(f"{pos:6d} {Qsx:8.3f} {rsx:10.4f} {korx:12.4f} {rsx / korx:10.3f}")

# ---------------------------------------------------------------------------
# BOLUM 6 -- T KARARLI MI? span'den BIR GONDERIM CIKARIP yeniden olc.
#
# GEREKCE: seviye ekseninin rho_s'i belgelerde +0.0156, SIMDI -0.0153.
# Isaret DEGISMIS. Aradaki fark L gurultusu degil, span'in BUYUMUS olmasi.
# Yani rho_s span'in bilesimine duyarli. T bu duyarliliktan sagkalmiyorsa
# olcum GECERSIZDIR ve boyle raporlanmalidir.
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("BOLUM 6  T KARARLILIGI -- span'den bir gonderim cikarilarak")
print("=" * 78)
# BELLEK: 600 eksenin test vektorunu onbelleklemek 3.4 GB eder. 80 eksenlik
# rastgele altorneklem yeter -- olculen sey EGIM, tek eksen degil.
_rs6 = np.random.default_rng(404)
IXA = np.flatnonzero(msk_ana)
if len(IXA) > 80:
    IXA = np.sort(_rs6.choice(IXA, 80, replace=False))
XT_ONB = {int(i): kur(AD[i])[0] for i in IXA}
print(f"altorneklem: {len(IXA)} eksen (bellek icin), tam kume {int(msk_ana.sum())}")
T_TAM_ALT = egim(RS[IXA], KO[IXA], SRS[IXA], GUR[IXA])["ols"]
print(f"altornekle tam-span T = {T_TAM_ALT:+.4f}  (tum kumede {T_NOKTA:+.4f})")
T_LOO, TSV_LOO = [], []
for j in range(K_V):
    keep = [q for q in range(K_V) if q != j]
    Vj, Lj = V[:, keep], L[keep]
    Gj = (Vj.T @ Vj) / N
    Gij = np.linalg.pinv(Gj, rcond=1e-6)
    rj, _, _ = buzmeli_r_hat(Vj, Lj, Gj, N, sigma=SIG_VEC[keep])
    rsj = np.empty(len(IXA))
    for q, i in enumerate(IXA):
        xt = XT_ONB[int(i)]
        cx = Gij @ ((Vj.T @ xt) / N)
        xpx = xt - Vj @ cx
        Qj = max(1.0 - float((xpx * xpx).mean()), 1e-6)
        rsj[q] = float((rj * xt).mean()) / np.sqrt(Qj)
    T_LOO.append(egim(rsj, KO[IXA], SRS[IXA], GUR[IXA])["ols"])
    cx = Gij @ ((Vj.T @ svT) / N)
    xpx = svT - Vj @ cx
    Qj = max(1.0 - float((xpx * xpx).mean()), 1e-6)
    TSV_LOO.append((float((rj * svT).mean()) / np.sqrt(Qj)) / kor_sv)
T_LOO, TSV_LOO = np.array(T_LOO), np.array(TSV_LOO)
del XT_ONB
print(
    f"havuz T   : tam span {T_NOKTA:+.4f} | LOO ort {T_LOO.mean():+.4f} sd {T_LOO.std():.4f} "
    f"aralik [{T_LOO.min():+.4f}, {T_LOO.max():+.4f}]"
)
print(
    f"seviye T  : tam span {T_SV:+.4f} | LOO ort {TSV_LOO.mean():+.4f} sd {TSV_LOO.std():.4f} "
    f"aralik [{TSV_LOO.min():+.4f}, {TSV_LOO.max():+.4f}]"
)
print(
    f"isaret kararliligi: havuz T pozitif oran {float((T_LOO > 0).mean()):.2f}, "
    f"seviye T pozitif oran {float((TSV_LOO > 0).mean()):.2f}"
)
KARARLI = bool(T_LOO.std() < 0.25 * abs(T_LOO.mean()) and (T_LOO > 0).all())
print(
    f"HUKUM 6: havuz T {'KARARLI' if KARARLI else 'KARARSIZ'} "
    f"(dalgalanma / ortalama = {T_LOO.std() / max(abs(T_LOO.mean()), 1e-12):.2f})"
)

# ---------------------------------------------------------------------------
# BOLUM 5 -- CARPAN = |c| * T dagilimi
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("BOLUM 5  CARPAN = |c| * T")
print("=" * 78)
with open(os.path.join(M29, "n10_c_carpani.json"), encoding="utf-8") as fh:
    CN = json.load(fh)["c_nihai"]
C_MID = float(CN["nokta"])
C_LO, C_HI = (float(x) for x in CN["ga90"])
rng4 = np.random.default_rng(77)
SN = 200000
c = np.exp(rng4.normal(np.log(C_MID), (np.log(C_HI) - np.log(C_LO)) / (2 * 1.6449), SN))
print(f"|c| (n10, n=29)           : {C_MID:.3f}  %90 GA [{C_LO:.3f}, {C_HI:.3f}]")
print(f"T(seviye)  (n=1, LOO sd {TSV_LOO.std():.3f}) : {T_SV:.3f}")
print(
    f"T(havuz)   (n={int(msk_ana.sum())}, LOO sd {T_LOO.std():.3f}) : {abs(T_NOKTA):.3f}  "
    f"%90 GA [{abs(T_LO):.3f}, {abs(T_HI):.3f}]"
)

# IKI OKUMA -- ve aralarindaki fark KAPATILMIYOR.
#  A) EN AZ VARSAYIMLI: T'yi OLDUGU GIBI birak (seviye, n=1), yalnizca
#     n=1 olan |c| bacagini n10'un n=29 olcumuyle degistir. Bu, HICBIR
#     yeni tasima varsayimi gerektirmez -- 0.798'in ICINDEKI 1.986'yi
#     n10'un 0.434'uyle degistirmekten ibarettir.
#  B) HAVUZ T: T'yi de 605 eksende olculen degerle degistir. Daha cok
#     veri kullanir AMA yeni bir tasima varsayimi ekler: havuz eksenlerinin
#     span PAYI kucuk (medyan Q_span dusuk), CARPAN ise span'a TAMAMEN DIK
#     bir yone uygulanacak. Bu tasima GOSTERILMEMISTIR.
Tsv_b = rng4.normal(abs(T_SV), TSV_LOO.std(), SN)
Tsv_b = np.abs(Tsv_b)
carp_A = c * Tsv_b
Tp_b = np.abs(rng4.choice(np.abs(boot), SN, replace=True))
carp_B = c * Tp_b
print()
print(f"{'okuma':>44s} {'medyan':>8s} {'%90 GA':>20s} {'YOL B rho':>10s}")
RHO_BLOK_B4 = 0.1814
TABAN_MSE = 1.00202690323433
OKUMA = {}
for ad, cp in [
    ("A  |c|_n10 x T(seviye)  -- EN AZ VARSAYIMLI", carp_A),
    ("B  |c|_n10 x T(havuz)   -- EK TASIMA VARSAYIMI", carp_B),
]:
    q = [float(np.quantile(cp, x)) for x in (0.05, 0.25, 0.5, 0.75, 0.95)]
    rho = RHO_BLOK_B4 * cp
    sk = np.sqrt(np.maximum(TABAN_MSE - np.minimum(rho**2, TABAN_MSE - 1e-6), 1e-9))
    OKUMA[ad] = dict(
        medyan=q[2],
        ga50=[q[1], q[3]],
        ga90=[q[0], q[4]],
        rho_medyan=float(np.median(rho)),
        skor_medyan=float(np.median(sk)),
        skor_ga90=[float(np.quantile(sk, 0.05)), float(np.quantile(sk, 0.95))],
    )
    print(f"{ad:>44s} {q[2]:8.3f} [{q[0]:7.3f},{q[4]:7.3f}] {np.median(rho):10.4f}")
print(
    f"{'m148 su an kullaniyor':>44s} {CARPAN_ESKI:8.3f} {'(n=1, |c|=1.986 gomulu)':>20s} "
    f"{RHO_BLOK_B4 * CARPAN_ESKI:10.4f}"
)
carp = carp_A  # nokta tahmin olarak EN AZ VARSAYIMLI okuma
CQ = [float(np.quantile(carp, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)]
rho_b = RHO_BLOK_B4 * carp
sk = np.sqrt(np.maximum(TABAN_MSE - np.minimum(rho_b**2, TABAN_MSE - 1e-6), 1e-9))
print(f"\nNOKTA TAHMIN (okuma A): CARPAN = {CQ[2]:.3f}  %90 GA [{CQ[0]:.3f}, {CQ[4]:.3f}]")
print(
    f"  0.798 bu dagilimin %{100 * float((carp < CARPAN_ESKI).mean()):.1f} yuzdeliginde "
    f"(yani neredeyse tum kutlenin USTUNDE)"
)
print(
    f"  YOL B yeniden: rho medyan {np.median(rho_b):.4f} (eski 0.1448), "
    f"nihai skor {np.median(sk):.5f}"
)
print(
    f"  okuma B alinirsa: rho {np.median(RHO_BLOK_B4 * carp_B):.4f}, "
    f"skor {float(np.median(np.sqrt(np.maximum(TABAN_MSE - (RHO_BLOK_B4 * carp_B) ** 2, 1e-9)))):.5f}"
)

CIKTI = dict(
    aciklama="n18: CARPAN'in cok-noktali olcumu. CARPAN = |c| * T ayrismasi.",
    tarih="2026-08-31",
    betik="experiments/model29/n18_carpan.py",
    dogrudan_yol="OLCULEMEDI -- gonderim farkinin yaz25 karsiligi kurulamiyor (BOLUM 1)",
    seviye_capasi=dict(
        rho_s=rho_s_sv,
        kor=kor_sv,
        Q_span=Qs_sv,
        T=T_SV,
        carpan_yeniden=float(-0.0304 / kor_sv),
        c_yeniden=float(-0.0304 / rho_s_sv),
    ),
    T=dict(
        nokta=T_NOKTA,
        ga90=[T_LO, T_HI],
        n=int(msk_ana.sum()),
        kume_sayisi=int(len(UK)),
        kumeler=KUMELER,
        seviye=T_SV,
    ),
    K_tablosu=K_TABLO,
    LOO_kararlilik=dict(
        havuz_T_ort=float(T_LOO.mean()),
        havuz_T_sd=float(T_LOO.std()),
        havuz_T_aralik=[float(T_LOO.min()), float(T_LOO.max())],
        seviye_T_ort=float(TSV_LOO.mean()),
        seviye_T_sd=float(TSV_LOO.std()),
        seviye_T_aralik=[float(TSV_LOO.min()), float(TSV_LOO.max())],
        kararli=KARARLI,
    ),
    c_n10=dict(nokta=C_MID, ga90=[C_LO, C_HI]),
    CARPAN=dict(
        medyan=CQ[2],
        ga50=[CQ[1], CQ[3]],
        ga90=[CQ[0], CQ[4]],
        eski=CARPAN_ESKI,
        eski_yuzdeligi=float((carp < CARPAN_ESKI).mean()),
        okuma="A (|c|_n10 x T(seviye)) -- en az varsayimli",
    ),
    okumalar=OKUMA,
    yol_B_yeniden=dict(
        rho_medyan=float(np.median(rho_b)),
        skor_medyan=float(np.median(sk)),
        skor_ga90=[float(np.quantile(sk, 0.05)), float(np.quantile(sk, 0.95))],
    ),
    uyarilar=[
        "CARPAN = |c|*T ayrisimi dogruysa YOL B, YOL A/C'den BAGIMSIZ DEGILDIR; "
        "ucu de |c|'ye dayanir. YOL B'nin bagimsiz gorunmesi T'nin yalnizca "
        "seviye ekseninde degerlendirilmis olmasindandir.",
        "T havuzu m148/m121/m144 tarafindan bloga BAKARAK secildi; T'nin kendisi "
        "IC-ORNEKTIR ve secim yanliligi tasir.",
        "|c| gonderim FARKI yonlerinde olculdu (dik pay kucuk); oznitelik "
        "eksenlerine tasindigi LB ile GOSTERILMEMISTIR (n10'un kendi uyarisi).",
        "T(havuz) = 0.116 span PAYI kucuk eksenlerde olculdu; CARPAN ise span'a "
        "TAM DIK bir yone uygulaniyor. Okuma B bu tasimayi VARSAYAR, GOSTERMEZ. "
        "Bu yuzden nokta tahmin okuma A'dir.",
        "m148'in TAVAN kapisi (|rho_cv| >= 1.95|rho_s|) bu ayrisim altinda "
        "|c|*T_j*|kor_j| >= |c|*|rho_s_j|, yani T_j <= T(seviye) demektir. "
        "Kapi BAGIMSIZ bir kanit tasimaz; n=1 kalibrasyonu tekrar eder.",
    ],
)
with open(os.path.join(M29, "n18_carpan.json"), "w", encoding="utf-8") as fh:
    json.dump(CIKTI, fh, indent=1, ensure_ascii=False)
print("\n-> experiments/model29/n18_carpan.json    HICBIR GONDERIM YAPILMADI.")
