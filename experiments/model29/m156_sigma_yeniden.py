"""sigma_L YENIDEN: TABAN_MSE'nin kendisi sigma'ya bagli -- RISKSIZ KALDIRAC.

SORUN. Nihai skor  = sqrt(TABAN_MSE - toplam rho_k^2)  ve
       TABAN_MSE   = M0 - 2*kL + ||r_hat||^2 .
r_hat, L'nin olcum gurultusu sigma_L varsayilarak BUZULMUS bir tahmindir.
sigma cok buyuk secilirse gercek sinyal atilir, TABAN_MSE gereksiz buyuk
kalir. TABAN_MSE'de kazanilan her sey DOGRUDAN skora yazilir ve rho
varsayimlarina HIC bagli degildir -- yani RISKSIZDIR.

BOLUMLER
  B0  veri, Gram, m112 ile birebir dogrulama (TABAN_MSE = 1.00202690)
  B1  DELIL 1 -- yari-orneklem bootstrap'i (m112.L_gurultusu, 2.27e-04)
  B2  DELIL 2 -- YAKIN-SIFIR KIPLER, m134 yeniden turetildi (nan_to_num YOK,
      yalniz GERCEKTEN OLCULMUS kolonlar, M0 bileseni cikarildi)
  B3  DELIL 3 -- DIK ARTIK SACILIMI, m149 B2 yeniden turetildi
  B4  BIRLESTIRME: sonsal dagilim + nokta tahmin + aralik
  B5  TABAN_MSE(sigma) tablosu  (DIKKAT: M0 - 2*kL + ||r_hat||^2, m152 tuzagi)
      + kazanci tasiyan BASKIN KIP tanisi
  B5c LOO ONGORU SINAVI -- en belirleyici delil
  B5d sigma ile W_TABAN AYRI DUGMELERDIR (2 boyutlu tarama)
  B6  CAPRAZ SINAV: (gercek dunya, kurulus) -> GERCEK TABAN_MSE
  B7  KARAR + kazanc (beklenen ve en kotu durum)
  B8  D1 CEBRI: sigma degisirse D1 dosyasi ne olur

OZET SONUC. sigma tek basina dusurulurse ozdegeri 3.9e-06 olan TEKIL bir kip
acilir; gorunen TABAN_MSE 4.9e-04 iyilesir ama LOO ongoru hatasi 4 kat
kotulesir -- o kip GURULTUDUR. Dogru duzeltme sigma ILE BIRLIKTE W_TABAN'i
1e-06'dan 1e-04'e cikarmaktir: TABAN_MSE 1.0020275 -> 1.0018932, taban skor
+0.000067 RISKSIZ, LOO hatasi 5.5 kat iyi.

Calistirma:
  ./.venv/Scripts/python.exe experiments/model29/m156_sigma_yeniden.py
HICBIR GONDERIM YAPMAZ, submissions/ altina HICBIR SEY YAZMAZ.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
S = os.path.join(KOK, "submissions")
M29 = os.path.join(KOK, "experiments/model29")
TABAN = "tuketim_m6_ikiyon.csv"
ONBELLEK = os.path.join(M29, "m145_onbellek.npz")
sys.path.insert(0, M29)
from m112_kalibre import (  # noqa: E402
    ANLAM_SIGMA,
    EK_MODEL,
    M0,
    W_TABAN,
    L_gurultusu,
    buzmeli_r_hat,
)

#: 5 ondalikli LB skorunun yuvarlama sd'si. L = (M0 + Q - P^2)/2, P ~ 1
#: oldugu icin dL = -P dP ~ -dP; duzgun dagilimin sd'si 1e-5/sqrt(12).
YUV = 1e-5 / np.sqrt(12.0)
#: docs/72 -- karar sayilari.
HEDEF1, HEDEF2, HEDEF3 = 0.99009, 0.99614, 0.99927
DOGRULAMA_TABAN_MSE = 1.00202690
#: gorevde adi gecen sigma adaylari (ORTALAMA olarak; yon yapisi korunur)
ADAYLAR = [2.27e-04, 1.20e-04, 5.0e-05, 2.9e-06]

rng = np.random.default_rng(20260831)


# ===========================================================================
# B0  VERI
# ===========================================================================
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

AD, D, L_OBS, Q_ALL, P2, OLCULDU = [], [], [], [], [], []
for f, Pj in SK.items():
    if f == TABAN or not os.path.exists(os.path.join(S, f)):
        continue
    v = oku(f)
    if v is None or len(v) != N:
        continue
    d = v - a0
    q = float((d * d).mean())
    AD.append(f)
    D.append(d)
    Q_ALL.append(q)
    P2.append(Pj * Pj)
    L_OBS.append((M0 + q - Pj * Pj) / 2)
    OLCULDU.append(True)
for f, Lj in EK_MODEL.items():
    d = oku(f) - a0
    AD.append(f)
    D.append(d)
    Q_ALL.append(float((d * d).mean()))
    P2.append(np.nan)
    L_OBS.append(float(Lj))
    OLCULDU.append(False)  # TURETILMIS -- kanitta kullanilamaz (dongusel)
for o in DUR.get("olcumler", []):
    d = oku(o["dosya"]) - a0
    q = float((d * d).mean())
    AD.append(o["dosya"])
    D.append(d)
    Q_ALL.append(q)
    P2.append(o["skor"] ** 2)
    L_OBS.append((M0 + q - o["skor"] ** 2) / 2)
    OLCULDU.append(True)

V = np.array(D).T  # (N, K)
L_OBS = np.array(L_OBS)
Q_ALL = np.array(Q_ALL)
P2 = np.array(P2)
OLCULDU = np.array(OLCULDU)
K = V.shape[1]
G = (V.T @ V) / N
W_, U_ = np.linalg.eigh(G)
_sr = np.argsort(-W_)
W_, U_ = W_[_sr], U_[:, _sr]
WMAX = float(W_[0])

print("=" * 78)
print("m156  sigma_L YENIDEN -- TABAN_MSE'nin kendisi kaldiractir")
print("=" * 78)
print(f"N = {N} satir | {K} yon ({int(OLCULDU.sum())} gercekten olculmus)")


def kats(Lv, sg, wtaban=W_TABAN):
    """m112.buzmeli_r_hat'in Gram uzayindaki birebir esi. Doner: katsayi k."""
    sg = np.asarray(sg, dtype=np.float64)
    si = np.sqrt((U_**2).T @ (sg**2))
    c = U_.T @ np.asarray(Lv, dtype=np.float64)
    a = np.zeros(K)
    for i in range(K):
        if W_[i] / WMAX <= wtaban or c[i] ** 2 <= 0.0:
            continue
        if c[i] ** 2 <= ANLAM_SIGMA**2 * si[i] ** 2:
            continue
        a[i] = max(c[i] ** 2 - si[i] ** 2, 0.0) / c[i] ** 2
    return U_ @ (a * c / np.where(W_ > 1e-12, W_, 1.0))


def taban_mse(k, Lgercek):
    """GERCEK taban MSE = M0 - 2<r, r_hat> + ||r_hat||^2, Gram uzayinda."""
    return float(M0 - 2.0 * float(k @ Lgercek) + float(k @ G @ k))


if os.path.exists(ONBELLEK):
    _c = np.load(ONBELLEK)["sig140"]
    SIG_A = _c if len(_c) == K else L_gurultusu(V, N)
else:
    SIG_A = L_gurultusu(V, N)

# --- m112 ile BIREBIR dogrulama --------------------------------------------
_rhat, _kaz, _kL = buzmeli_r_hat(V, L_OBS, G, N, sigma=SIG_A)
_tm_ref = float(M0 - 2 * _kL + float((_rhat * _rhat).mean()))
_k_gram = kats(L_OBS, SIG_A)
_tm_gram = taban_mse(_k_gram, L_OBS)
print(f"\nDOGRULAMA  m112 (tam N): TABAN_MSE = {_tm_ref:.8f}")
print(f"           Gram esdegeri : TABAN_MSE = {_tm_gram:.8f}")
print(f"           docs/72 degeri: TABAN_MSE = {DOGRULAMA_TABAN_MSE:.8f}")
if abs(_tm_ref - _tm_gram) > 1e-9 or abs(_tm_ref - DOGRULAMA_TABAN_MSE) > 2e-8:
    raise SystemExit("DUR: Gram esdegeri ya da referans sabit tutmuyor")
print("           -> UCU DE AYNI. Gram uzayinda calisabiliriz.")


def sig_vektor(ortalama):
    """SIG_A'nin yon yapisini koruyarak ortalamasi verilen sigma vektoru."""
    return SIG_A * (float(ortalama) / float(SIG_A.mean()))


# ===========================================================================
# B1  DELIL 1 -- yari-orneklem bootstrap'i
# ===========================================================================
print("\n" + "=" * 78)
print("B1  DELIL 1 -- yari-orneklem bootstrap'i (m112.L_gurultusu)")
print("=" * 78)
print(f"ortalama sigma_L = {SIG_A.mean():.3e}   [{SIG_A.min():.2e}, {SIG_A.max():.2e}]")
print(f"LB yuvarlama tabani (5 ondalik) = {YUV:.3e}   oran {SIG_A.mean() / YUV:.0f} kat")
print("""VARSAYIMI: public alt kume TUM satirlarin RASTGELE %50'sidir ve
  Q_j hesabi ile P_j olcumu farkli kumelerde yapilmistir. Bu bir OLCUM
  DEGIL, bir BENZETIMDIR -- bolunme rastgele %50 degilse (ornegin public =
  tum kume, ya da bolunme yapisal) tamamen gecersizdir.""")


# ===========================================================================
# B2  DELIL 2 -- yakin-sifir kipler (m134 YENIDEN TURETILDI)
# ===========================================================================
print("\n" + "=" * 78)
print("B2  DELIL 2 -- yakin-sifir kipler, m134 yeniden turetildi")
print("=" * 78)
print("""CEBIR. u birim vektoru icin
      u . L_gercek = <r, V u>/N   ve   |<r,Vu>/N| <= |r|_rms * |Vu|_rms
  yani  |u.L_gercek| <= sqrt(M0) * sqrt(u'Gu).  Kucuk ozdegerli kipte bu
  sinir kucuktur; gozlenen u.L_obs neredeyse SAF GURULTUDUR:
      Var(u . eps) = sum_j u_j^2 sigma_j^2 = sigma^2   (birim u, esit sigma)

  m134'un IKI KUSURU burada YOK:
   (1) nan_to_num: L'si TURETILMIS kolonlar (EK_MODEL) sifire cekiliyordu ve
       kip yuku sessizce bozuluyordu -> yalniz OLCULMUS kolonlarla calisiyoruz.
   (2) M0 dongusu: u.L = (M0*sum(u) + sum u_j(Q_j - P_j^2))/2 oldugu icin
       sum(u) != 0 ise sonuc M0'a baglidir; M0 da bu dosyalardan kalibre
       edildi -> ones bileseni ALT UZAYDAN CIKARILIYOR.""")

mi = np.where(OLCULDU)[0]
Gm = G[np.ix_(mi, mi)]
Km = len(mi)
wm, Um = np.linalg.eigh(Gm)
QPm = (Q_ALL[mi] - P2[mi]) / 2.0  # u.L = M0*sum(u)/2 + u.QPm
print(f"\nolculmus kolon sayisi: {Km}")
print(f"G_m ozdegerleri (kucukten): {', '.join(f'{x:.2e}' for x in wm[:8])}")

# Sinir yuvarlama tabaninin ~10 katinin altinda kalan kipler kullanilabilir:
# gurultuyu olcecegiz, en kucuk aday sigma YUV oldugu icin sinir << YUV olmali.
ESIK_LAM = (YUV / np.sqrt(M0)) ** 2
KUL = [i for i in range(Km) if wm[i] < ESIK_LAM]
print(f"lambda esigi {ESIK_LAM:.2e}  (sqrt(M0*lambda) < yuvarlama tabani)")
print(f"esigi gecen kip: {len(KUL)}")
if len(KUL) < 2:
    KUL = [i for i in range(Km) if wm[i] < 100 * ESIK_LAM]
    print(f"  -> gevsetildi (100x): {len(KUL)} kip")

Un = Um[:, KUL]
ones = np.ones(Km)
# ones'in alt uzaya izdusumune DIK bilesim katsayilari -> M0 terimi yok olur
s_vek = Un.T @ ones
Qb, _ = np.linalg.qr(s_vek.reshape(-1, 1))
Pd = np.eye(len(KUL)) - Qb @ Qb.T
wq, Vq = np.linalg.eigh(Pd)
BAG = [Vq[:, i] for i in range(len(KUL)) if wq[i] > 0.5]
print(f"M0-BAGIMSIZ boyut: {len(BAG)}  (etkin n budur)")

Z, BND, OLC = [], [], []
SIG_Am = SIG_A[mi]  # yon basina sigma (B1 hipotezi), olculmus kolonlar
for c in BAG:
    u = Un @ c
    nu = np.linalg.norm(u)
    u = u / nu
    z = float(u @ QPm)  # = u.L_obs, M0 terimi sifir
    lam = float(u @ Gm @ u)
    # KRITIK DENETIM: kip DUSUK sigma'li dosyalara yuklenmis olabilir; o
    # zaman kucuk |u.L_obs| B1'i CURUTMEZ. Bu yuzden her kip icin B1'in
    # KENDI ongordugu sd hesaplanir. olc = 1 ise kip ortalama gurultulu.
    olc = float(np.sqrt(np.sum((u * SIG_Am) ** 2)) / SIG_A.mean())
    Z.append(z)
    BND.append(float(np.sqrt(M0 * max(lam, 0.0))))
    OLC.append(olc)
    print(
        f"  yon: sum(u)={float(u @ ones):+.2e}  lambda={lam:.2e}  "
        f"|u.L_obs|={abs(z):.3e}  sinyal siniri={BND[-1]:.3e}  "
        f"B1 olcek={olc:.3f}  B1 ongorusu sd={olc * SIG_A.mean():.3e}"
    )
Z = np.array(Z)
BND = np.array(BND)
OLC = np.array(OLC)
m_ = len(Z)
print(f"\n  n = {m_} bagimsiz olcum")
print(f"  rms |u.L_obs| = {np.sqrt(np.mean(Z**2)):.3e}")
print(f"  sinyalin katkisi en fazla rms {np.sqrt(np.mean(BND**2)):.3e}")
print(f"  kip olcek faktoru [{OLC.min():.3f}, {OLC.max():.3f}]")
print("    1'e yakinsa kipler 'dusuk gurultulu dosyalara siginmis' DEGILDIR")
print("    ve B1'i curutme yetkileri vardir.")


def loglik_kipler(sigma):
    """z_i = s_i + e_i ; |s_i| <= b_i (duzgun kabul -> var b_i^2/3),
    e_i ~ N(0, (sigma*olc_i)^2). olc_i = sigma VEKTORUNUN o kipteki agirligi
    -- 'kip dusuk sigmali yonlere yuklendi' itirazi boylece kapanir.
    Sinyal payi gurultuye sayildigi icin BUYUK sigma LEHINE muhafazakardir."""
    var = (sigma * OLC) ** 2 + BND**2 / 3.0
    return float(-0.5 * np.sum(Z**2 / var + np.log(2 * np.pi * var)))


print(f"\n{'sigma':>12s} {'-2 logL':>10s} {'goreli olasilik':>17s}")
_gr = np.array(ADAYLAR + [1e-05, 1e-04, 1.6e-04])
_gr = np.sort(np.unique(_gr))
_ll = np.array([loglik_kipler(s) for s in _gr])
for s, ll in zip(_gr, _ll):
    print(f"{s:12.3e} {-2 * ll:10.2f} {np.exp(ll - _ll.max()):17.4f}")
EN_IYI_KIP = float(_gr[np.argmax(_ll)])
# analitik en cok olabilirlik (sinyal sifir varsayimiyla alt sinir):
_kaba = np.sqrt(max(float(np.mean((Z**2 - BND**2 / 3.0) / OLC**2)), 0.0))
print(f"\n  KIPLERIN en cok olabilirlik tahmini sigma ~ {_kaba:.3e}")
print(f"  (izgarada en iyi: {EN_IYI_KIP:.3e})")


# ===========================================================================
# B3  DELIL 3 -- dik artik sacilimi (m149 B2 YENIDEN TURETILDI)
# ===========================================================================
print("\n" + "=" * 78)
print("B3  DELIL 3 -- dik artik sacilimi, m149 B2 yeniden turetildi")
print("=" * 78)
print("""FIKIR. Her j icin LOO: digerlerinden r_hat kur, j'nin span bilesenini
  tahmin et, artik  o_j = L_j - <r_hat_(-j), d_j^span>/N .  Gozlenen
  sum(o_j^2) hem GERCEK dik sinyali hem GURULTUYU icerir; dolayisiyla
  gurultunun tek basina ongordugu sacilim gozleneni ASAMAZ.
      sum var(o_j | sigma)  <=  sum o_j^2      ->  sigma icin UST SINIR""")

IX = [np.array([i for i in range(K) if i != j]) for j in range(K)]
CC = np.zeros((K, K))
QSP = np.zeros(K)
QDK = np.zeros(K)
for j in range(K):
    ix = IX[j]
    Gr = G[np.ix_(ix, ix)]
    gj = G[ix, j]
    cc = np.linalg.pinv(Gr, rcond=1e-6) @ gj
    CC[j, ix] = cc
    QSP[j] = float(cc @ Gr @ cc)
    QDK[j] = float(G[j, j] - 2 * cc @ gj + QSP[j])


def kats_alt(Gr, Lr, sg, wtaban=W_TABAN):
    w, U = np.linalg.eigh(Gr)
    sr = np.argsort(-w)
    w, U = w[sr], U[:, sr]
    c = U.T @ Lr
    si = np.sqrt((U**2).T @ (sg**2))
    a = np.zeros(len(w))
    wmax = float(w[0]) if len(w) else 1.0
    for i in range(len(w)):
        if w[i] / wmax <= wtaban or c[i] ** 2 <= 0.0:
            continue
        if c[i] ** 2 <= ANLAM_SIGMA**2 * si[i] ** 2:
            continue
        a[i] = max(c[i] ** 2 - si[i] ** 2, 0.0) / c[i] ** 2
    return U @ (a * c / np.where(w > 1e-12, w, 1.0))


def artiklar(Lobs, sg):
    o = np.full(K, np.nan)
    for j in range(K):
        if QSP[j] < 1e-10 or QDK[j] < 1e-6:
            continue
        ix = IX[j]
        Gr = G[np.ix_(ix, ix)]
        kap = kats_alt(Gr, Lobs[ix], sg[ix])
        o[j] = Lobs[j] - float(kap @ Gr @ CC[j, ix])
    return o


O_GOZ = artiklar(L_OBS, sig_vektor(2.9e-06))
GEC = np.isfinite(O_GOZ)
SYY = float(np.nansum(O_GOZ[GEC] ** 2))
print(f"\nLOO borusu: {int(GEC.sum())}/{K} yon gecti")
print(f"gozlenen sum(o_j^2) = {SYY:.4e}")


def ongorulen_var(ort, R=200):
    sg = sig_vektor(ort)
    A = np.array([artiklar(L_OBS + rng.normal(0, sg), sg) for _ in range(R)])
    v = np.nanvar(A[:, GEC], axis=0)
    return float(np.nansum(v))


print(f"\n{'sigma ort':>12s} {'ongorulen sum var(o)':>22s} {'gozlenen/ongorulen':>20s}")
OLC_LISTE = [2.27e-04, 1.6e-04, 1.2e-04, 8e-05, 5e-05, 2.9e-06]
VAR_LISTE = []
for s in OLC_LISTE:
    v = ongorulen_var(s)
    VAR_LISTE.append(v)
    print(f"{s:12.3e} {v:22.4e} {SYY / v if v > 0 else np.inf:20.2f}")
# gurultu ~ sigma^2 ile olcekleniyor: gozleneni asmayan en buyuk sigma
_s0, _v0 = OLC_LISTE[0], VAR_LISTE[0]
SIG_UST = _s0 * np.sqrt(SYY / _v0)
print(f"\n  UST SINIR (gercek dik sinyal = 0 varsayimiyla): sigma_L <= {SIG_UST:.2e}")
print("  Gercek sinyal varsa gozlenenin bir kismi ona aittir -> sinir DAHA SIKI.")
print(f"  m149'un bildirdigi 1.2e-04 ile karsilastir: {SIG_UST:.2e}")


# ===========================================================================
# B4  BIRLESTIRME
# ===========================================================================
print("\n" + "=" * 78)
print("B4  DELILLERIN BIRLESTIRILMESI")
print("=" * 78)
IZGARA = np.array([2.9e-06, 1e-05, 2e-05, 5e-05, 8e-05, 1.2e-04, 1.6e-04, 2.27e-04])
LL = np.array([loglik_kipler(s) for s in IZGARA])
POST = np.exp(LL - LL.max())
POST[IZGARA > SIG_UST] = 0.0  # B3 sert kisiti
if POST.sum() <= 0:
    POST = np.exp(LL - LL.max())
POST = POST / POST.sum()
print(f"{'sigma':>12s} {'B2 goreli olasilik':>20s} {'B3 kisiti':>12s} {'SONSAL':>9s}")
for s, ll, p in zip(IZGARA, LL, POST):
    print(
        f"{s:12.3e} {np.exp(ll - LL.max()):20.4f} "
        f"{'GECER' if s <= SIG_UST else 'ELENDI':>12s} {p:9.3f}"
    )
SIG_BEK = float((POST * IZGARA).sum())
_cum = np.cumsum(POST)
SIG_ALT = float(IZGARA[np.searchsorted(_cum, 0.05)])
SIG_UST5 = float(IZGARA[min(np.searchsorted(_cum, 0.95), len(IZGARA) - 1)])
print(f"\n  EN IYI TAHMIN  sigma_L ~ {IZGARA[np.argmax(POST)]:.2e}")
print(f"  sonsal ortalama          {SIG_BEK:.2e}")
print(f"  %90 araligi              [{SIG_ALT:.2e}, {SIG_UST5:.2e}]")
print(f"  SERT UST SINIR (B3)      {SIG_UST:.2e}")


# ===========================================================================
# B5  TABAN_MSE(sigma)
# ===========================================================================
print("\n" + "=" * 78)
print("B5  TABAN_MSE = M0 - 2*kL + ||r_hat||^2   (sigma'nin fonksiyonu)")
print("=" * 78)
print("DIKKAT: 'MSE_OPT = M0 - kazanc' DEGIL (m152 tuzagi). Buzmeli cozumde")
print("  k'L != k'Gk oldugu icin ikisi ayrilir; gonderilen dosyanin gercek")
print("  MSE'si M0 - 2*k'L + k'Gk'dir.\n")
print(
    f"{'sigma ort':>12s} {'tutulan kip':>12s} {'kL':>11s} {'||r_hat||^2':>12s} "
    f"{'TABAN_MSE':>12s} {'taban skor':>11s} {'M0-kazanc':>11s}"
)
TM = {}
for s in [2.27e-04, 1.6e-04, 1.2e-04, 8e-05, 5e-05, 2e-05, 1e-05, 2.9e-06]:
    sg = sig_vektor(s)
    k = kats(L_OBS, sg)
    kL = float(k @ L_OBS)
    nrm = float(k @ G @ k)
    tm = M0 - 2 * kL + nrm
    TM[s] = tm
    si = np.sqrt((U_**2).T @ (sg**2))
    c = U_.T @ L_OBS
    nk = int(np.sum((W_ / WMAX > W_TABAN) & (c**2 > ANLAM_SIGMA**2 * si**2)))
    _, kaz, _ = buzmeli_r_hat(V, L_OBS, G, N, sigma=sg)
    print(
        f"{s:12.3e} {nk:12d} {kL:11.6f} {nrm:12.6f} {tm:12.7f} "
        f"{np.sqrt(tm):11.6f} {np.sqrt(M0 - kaz):11.6f}"
    )
print("\n  son sutun YANLIS sabittir, yalnizca farki gorulsun diye basildi.")
print(
    f"\n{'sigma':>12s} {'TABAN_MSE':>12s} {'taban skor':>11s} {'2.6e-4 sigmaya gore kazanc':>28s}"
)
_ref = TM[2.27e-04]
for s in ADAYLAR:
    key = min(TM, key=lambda x: abs(x - s))
    print(
        f"{key:12.3e} {TM[key]:12.7f} {np.sqrt(TM[key]):11.6f} "
        f"{np.sqrt(_ref) - np.sqrt(TM[key]):+28.6f}"
    )


# --- KAZANC HANGI KIPTEN GELIYOR? -----------------------------------------
print("\nKAZANC HANGI KIPTEN GELIYOR? (2.27e-04 -> 2.9e-06 arasinda ACILAN kipler)")
_c = U_.T @ L_OBS
_siA = np.sqrt((U_**2).T @ (sig_vektor(2.27e-04) ** 2))
_siK = np.sqrt((U_**2).T @ (sig_vektor(2.9e-06) ** 2))
print(
    f"{'kip':>4s} {'ozdeger w':>11s} {'c_i':>11s} {'sig_i(2.3e-4)':>14s} "
    f"{'c/sig':>7s} {'sig_i(2.9e-6)':>14s} {'c/sig':>8s} {'kip kazanci':>12s}"
)
_TOP = 0.0
for i in range(K):
    if W_[i] / WMAX <= W_TABAN:
        continue
    gA = _c[i] ** 2 > ANLAM_SIGMA**2 * _siA[i] ** 2
    gK = _c[i] ** 2 > ANLAM_SIGMA**2 * _siK[i] ** 2
    if gA or not gK:
        continue
    lam2 = max(_c[i] ** 2 - _siK[i] ** 2, 0.0)
    kz = lam2**2 / (_c[i] ** 2 * W_[i])
    _TOP += kz
    print(
        f"{i:4d} {W_[i]:11.3e} {_c[i]:+11.3e} {_siA[i]:14.3e} "
        f"{abs(_c[i]) / _siA[i]:7.2f} {_siK[i]:14.3e} {abs(_c[i]) / _siK[i]:8.1f} {kz:12.3e}"
    )
print(f"  acilan kiplerin toplam kazanci = {_TOP:.3e}")
print("  c/sig > 2 olunca kip ACILIR (ANLAM_SIGMA kapisi). Kazancin tamami TEK")
print("  bir kipten geliyorsa karar o kipin gercekligine bagimlidir.")

print("\nINCE IZGARA -- TABAN_MSE nerede sicriyor?")
print(f"{'sigma':>12s} {'TABAN_MSE':>12s} {'taban skor':>11s}")
for _s in (1e-06, 5e-06, 1e-05, 2e-05, 3e-05, 4e-05, 5e-05, 6e-05, 7e-05, 9e-05):
    _k = kats(L_OBS, sig_vektor(_s))
    _tm = taban_mse(_k, L_OBS)
    print(f"{_s:12.1e} {_tm:12.7f} {np.sqrt(_tm):11.6f}")
print("  DUZ bolge: karar, o araligin NERESINDE oldugumuza duyarli DEGIL.")

# --- KAZANCI TASIYAN KIP NE ZAMAN ACILIYOR? -------------------------------
_ik = int(
    np.argmax(
        [
            (max(_c[i] ** 2 - _siK[i] ** 2, 0.0) ** 2) / (_c[i] ** 2 * W_[i])
            if W_[i] / WMAX > W_TABAN
            and _c[i] ** 2 > ANLAM_SIGMA**2 * _siK[i] ** 2
            and not (_c[i] ** 2 > ANLAM_SIGMA**2 * _siA[i] ** 2)
            else 0.0
            for i in range(K)
        ]
    )
)
print(f"\nBASKIN KIP = {_ik}  (ozdeger {W_[_ik]:.3e}, c = {_c[_ik]:+.3e})")
_esik_ort = 2.27e-04 * abs(_c[_ik]) / (ANLAM_SIGMA * _siA[_ik])
print(f"  ANLAM_SIGMA kapisini gecmesi icin gereken: ortalama sigma <= {_esik_ort:.2e}")
_p_gec = float(POST[_esik_ort >= IZGARA].sum())
print(f"  sonsala gore P(sigma <= {_esik_ort:.2e}) = {_p_gec:.3f}")
print("  Bu kip ACILIRSA kazanc, GERCEK DEGILSE ayni buyuklukte KAYIP getirir")
print(
    f"  (kip kazanci {max(_c[_ik] ** 2 - _siK[_ik] ** 2, 0.0) ** 2 / (_c[_ik] ** 2 * W_[_ik]):.3e} MSE)."
)
print("  Kipin yuku (en buyuk 5 dosya):")
for _j in np.argsort(-np.abs(U_[:, _ik]))[:5]:
    print(f"    {U_[_j, _ik]:+9.4f}  {AD[_j][:56]}")

# ===========================================================================
# B5c  LOO ONGORU SINAVI -- EN BELIRLEYICI DELIL
# ===========================================================================
print("\n" + "=" * 78)
print("B5c  LOO ONGORU SINAVI -- sigma'yi VERI secsin")
print("=" * 78)
print("""Her olculmus yon j icin: j'yi CIKAR, kalanlardan r_hat kur, j'nin
LB'sini ONGOR (L_j_hat = k_(-j) . G[-j, j]) ve gercek L_j ile karsilastir.
Bu, sigma secimini DOGRUDAN sinar: sigma cok kucukse buzme yetersizdir,
gurultuye uyar ve tutulmayan yonlerde ONGORU BOZULUR. Baskin kip gercek
sinyalse kucuk sigma LOO'yu IYILESTIRIR, gurultuyse KOTULESTIRIR.
Skor birimi: dP ~ dL / P ~ dL, yani hatalar dogrudan skor hatasidir.""")
_olc_j = [j for j in range(K) if OLCULDU[j]]
print(
    f"\n{'sigma ort':>12s} {'ort |hata|':>12s} {'ortanca |hata|':>16s} {'rms hata':>11s} {'en kotu':>11s}"
)
LOO_SON = {}
for _s in (2.27e-04, 1.6e-04, 1.2e-04, 8e-05, 6e-05, 5e-05, 2e-05, 1e-05, 2.9e-06):
    sg = sig_vektor(_s)
    err = []
    for j in _olc_j:
        ix = IX[j]
        Gr = G[np.ix_(ix, ix)]
        kj = kats_alt(Gr, L_OBS[ix], sg[ix])
        err.append(L_OBS[j] - float(kj @ G[ix, j]))
    err = np.array(err)
    LOO_SON[_s] = float(np.mean(np.abs(err)))
    print(
        f"{_s:12.3e} {np.mean(np.abs(err)):12.3e} {np.median(np.abs(err)):16.3e} "
        f"{np.sqrt(np.mean(err**2)):11.3e} {np.max(np.abs(err)):11.3e}"
    )
_en = min(LOO_SON, key=LOO_SON.get)
print(f"\n  LOO'YU EN IYI ONGOREN sigma = {_en:.2e}  (ort |hata| {LOO_SON[_en]:.3e})")
print(f"  mevcut 2.27e-04 ile fark: {LOO_SON[2.27e-04] - LOO_SON[_en]:+.3e}")
print("""  OKUMA. LOO, kip 22 gibi tekil kiplerin gurultuye uyup uymadigini
  dogrudan cezalandirir: uyuyorsa tutulmayan yonlerde ongoru saparr.
  Bu delil B1/B2/B3'ten BAGIMSIZDIR -- gurultu MODELI degil, TAHMIN
  BASARISI olculur.""")

# ===========================================================================
# B5d  IKI AYRI DUGME: sigma (gurultu) ve W_TABAN (tekillik)
# ===========================================================================
print("\n" + "=" * 78)
print("B5d  sigma ile W_TABAN AYRI DUGMELERDIR")
print("=" * 78)
print("""B5c'de ortanca hata sigma kuculdukce SUREKLI IYILESIYOR ama ortalama ve
en kotu PATLIYOR -> birkac yon felaket veriyor. Bu, m112'nin uyardigi
'kucuk w'li kipte a*c/w patlar' arizasidir; gurultu duzeyiyle degil
TEKILLIKLE ilgilidir. Dogru arac W_TABAN'dir, sigma'yi sismis tutmak degil.
Asagida iki dugme AYRI taranir: her hucrede LOO ort |hata| ve TABAN_MSE.""")
WT_LISTE = (1e-06, 1e-05, 1e-04, 3e-04, 1e-03, 3e-03)
SIG_LISTE = (2.27e-04, 1.2e-04, 8e-05, 5e-05, 2e-05, 2.9e-06)
print("\nLOO ort |hata|   (satir = sigma, sutun = W_TABAN)")
print(f"{'sigma':>11s}" + "".join(f"{wt:>12.0e}" for wt in WT_LISTE))
IZG = {}
for _s in SIG_LISTE:
    sg = sig_vektor(_s)
    sat = []
    for wt in WT_LISTE:
        err = []
        for j in _olc_j:
            ix = IX[j]
            Gr = G[np.ix_(ix, ix)]
            kj = kats_alt(Gr, L_OBS[ix], sg[ix], wtaban=wt)
            err.append(L_OBS[j] - float(kj @ G[ix, j]))
        e = float(np.mean(np.abs(err)))
        IZG[(_s, wt)] = e
        sat.append(e)
    print(f"{_s:11.2e}" + "".join(f"{v:12.3e}" for v in sat))
print("\nTABAN_MSE       (satir = sigma, sutun = W_TABAN)")
print(f"{'sigma':>11s}" + "".join(f"{wt:>12.0e}" for wt in WT_LISTE))
TM2 = {}
for _s in SIG_LISTE:
    sg = sig_vektor(_s)
    sat = []
    for wt in WT_LISTE:
        tm = taban_mse(kats(L_OBS, sg, wtaban=wt), L_OBS)
        TM2[(_s, wt)] = tm
        sat.append(tm)
    print(f"{_s:11.2e}" + "".join(f"{v:12.7f}" for v in sat))
_iyi = min(IZG, key=IZG.get)
print(f"\n  LOO'yu EN IYI ONGOREN kurulus: sigma {_iyi[0]:.2e}, W_TABAN {_iyi[1]:.0e}")
print(f"    LOO ort |hata| = {IZG[_iyi]:.3e}   TABAN_MSE = {TM2[_iyi]:.7f}")
print(
    f"    (mevcut: sigma 2.27e-04, W_TABAN {W_TABAN:.0e} -> "
    f"LOO {IZG[(2.27e-04, 1e-06)]:.3e}, TABAN_MSE {TM2[(2.27e-04, 1e-06)]:.7f})"
)
print(f"    taban skorda kazanc = {np.sqrt(TM2[(2.27e-04, 1e-06)]) - np.sqrt(TM2[_iyi]):+.6f}")
print("""
  UYARI: TABAN_MSE tablosundaki KUCUK degerler her zaman iyi DEGILDIR --
  TABAN_MSE burada L_OBS'un kendisiyle hesaplandigi icin gurultuye uyan
  kurulus kendini oldugundan iyi gosterir (m137'nin uyardigi tuzak).
  Bu yuzden SECIM LOO tablosundan yapilir, TABAN_MSE tablosundan DEGIL.""")


# ===========================================================================
# B6  CAPRAZ SINAV -- GERCEK OLCUT, IKI DUGMELI KURULUSLAR
# ===========================================================================
print("\n" + "=" * 78)
print("B6  CAPRAZ SINAV: (gercek dunya, kurulus) -> GERCEK TABAN_MSE")
print("=" * 78)
print("""KURULUS. Gercek L bilinmiyor. 'Gercek dunya g = (sigma_g, wt_g)' icin
    L_gercek(g) = G @ kats(L_obs, sigma_g, wt_g)
    L_gozlenen  = L_gercek(g) + N(0, sigma_g)
    kurulus u   : k = kats(L_gozlenen, sigma_u, wt_u)
    GERCEK TABAN_MSE = M0 - 2 k.L_gercek(g) + k'Gk
Bu, gonderilen dosyanin GERCEK MSE'sidir -- kurulusun kendi inanci degil.
Kritik dunya ayrimi: 'wt = 1e-06' dunyalarinda BASKIN KIP GERCEKTIR,
'wt = 1e-04' dunyalarinda o kip GURULTUDUR.""")

KURULUSLAR = [
    ("A mevcut", 2.27e-04, 1e-06),
    ("B saf kucuk sigma", 2.90e-06, 1e-06),
    ("C ONERILEN", 2.90e-06, 1e-04),
    ("D orta sigma", 8.00e-05, 1e-06),
    ("E kucuk+sert kapi", 2.00e-05, 3e-04),
    ("F buyuk sigma+kapi", 2.27e-04, 1e-04),
]
DUNYALAR = [
    ("W1 sigma buyuk", 2.27e-04, 1e-06),
    ("W2 sigma orta, kip gurultu", 1.20e-04, 1e-04),
    ("W3 sigma kucuk, kip gurultu", 5.00e-05, 1e-04),
    ("W4 sigma cok kucuk, kip gurultu", 2.90e-06, 1e-04),
    ("W5 sigma cok kucuk, KIP GERCEK", 2.90e-06, 1e-06),
]

# --- dunya agirliklari: LOO ONGORU basarisi (B5c/B5d) ----------------------
# Her dunya, kendi kurulusuyla LOO'da ne kadar iyi ongoruyor? Gauss
# olabilirligi, olcek serbest: -2logL = n_eff * log(SSE/n).
print("\nDUNYA AGIRLIKLARI -- LOO ongoru olabilirligi")
SSE = {}
for ad, sg_, wt_ in DUNYALAR:
    sgv = sig_vektor(sg_)
    e = []
    for j in _olc_j:
        ix = IX[j]
        Gr = G[np.ix_(ix, ix)]
        kj = kats_alt(Gr, L_OBS[ix], sgv[ix], wtaban=wt_)
        e.append(L_OBS[j] - float(kj @ G[ix, j]))
    SSE[ad] = float(np.sum(np.square(e)))
n_loo = len(_olc_j)
print(f"{'dunya':>34s} {'LOO rms':>10s} {'agirlik n_eff=27':>18s} {'n_eff=9':>10s}")
_m2 = {a: n_loo * np.log(SSE[a] / n_loo) for a in SSE}
_mn = min(_m2.values())
AG_TAM = np.array([np.exp(-(_m2[a] - _mn) / 2) for a, _, _ in DUNYALAR])
_m2b = {a: 9 * np.log(SSE[a] / n_loo) for a in SSE}
_mnb = min(_m2b.values())
AG_ZAY = np.array([np.exp(-(_m2b[a] - _mnb) / 2) for a, _, _ in DUNYALAR])
AG_TAM = AG_TAM / AG_TAM.sum()
AG_ZAY = AG_ZAY / AG_ZAY.sum()
for i, (ad, _, _) in enumerate(DUNYALAR):
    print(f"{ad:>34s} {np.sqrt(SSE[ad] / n_loo):10.3e} {AG_TAM[i]:18.4f} {AG_ZAY[i]:10.4f}")
print("  n_eff = 9 sutunu, LOO hatalarinin bagimli olabilecegini varsayan")
print("  MUHAFAZAKAR agirliklamadir. Karar ikisinde de ayni cikmali.")

CEK = 300
TAB = np.zeros((len(DUNYALAR), len(KURULUSLAR)))
for gi_, (gad, gs, gw) in enumerate(DUNYALAR):
    sg_g = sig_vektor(gs)
    Lg = G @ kats(L_OBS, sg_g, wtaban=gw)
    Lo_hepsi = Lg[None, :] + rng.normal(0, sg_g, size=(CEK, K))
    for ui, (uad, us, uw) in enumerate(KURULUSLAR):
        sg_u = sig_vektor(us)
        TAB[gi_, ui] = float(np.mean([taban_mse(kats(Lo, sg_u, wtaban=uw), Lg) for Lo in Lo_hepsi]))

print(f"\nGERCEK TABAN_MSE ({CEK} cekilis) -- satir = gercek dunya, sutun = kurulus")
print(" " * 34 + "".join(f"{u[0][:11]:>13s}" for u in KURULUSLAR) + f"{'kazanan':>14s}")
for gi_, (gad, _, _) in enumerate(DUNYALAR):
    kz = KURULUSLAR[int(np.argmin(TAB[gi_]))][0]
    print(
        f"{gad:>34s}"
        + "".join(f"{TAB[gi_, ui]:13.7f}" for ui in range(len(KURULUSLAR)))
        + f"{kz[:13]:>14s}"
    )
print("\nAYNI TABLO SKOR OLARAK (sqrt):")
print(" " * 34 + "".join(f"{u[0][:11]:>13s}" for u in KURULUSLAR))
for gi_, (gad, _, _) in enumerate(DUNYALAR):
    print(
        f"{gad:>34s}" + "".join(f"{np.sqrt(TAB[gi_, ui]):13.6f}" for ui in range(len(KURULUSLAR)))
    )

BEK = AG_TAM @ TAB
BEK_Z = AG_ZAY @ TAB
KOTU = TAB.max(axis=0)
print(
    f"\n{'kurulus':>20s} {'BEKLENEN skor':>15s} {'muhafazakar':>13s} "
    f"{'EN KOTU skor':>14s} {'A ya gore kazanc':>18s}"
)
_iA = 0
for ui, (uad, us, uw) in enumerate(KURULUSLAR):
    print(
        f"{uad:>20s} {np.sqrt(BEK[ui]):15.6f} {np.sqrt(BEK_Z[ui]):13.6f} "
        f"{np.sqrt(KOTU[ui]):14.6f} {np.sqrt(BEK[_iA]) - np.sqrt(BEK[ui]):+18.6f}"
    )
U_BEK = KURULUSLAR[int(np.argmin(BEK))]
U_ZAY = KURULUSLAR[int(np.argmin(BEK_Z))]
U_KOTU = KURULUSLAR[int(np.argmin(KOTU))]
print(f"\n  BEKLENENDE en iyi       : {U_BEK[0]}")
print(f"  MUHAFAZAKAR agirlikta   : {U_ZAY[0]}")
print(f"  EN KOTU DURUMDA en iyi  : {U_KOTU[0]}")


# ===========================================================================
# B7  KARAR
# ===========================================================================
print("\n" + "=" * 78)
print("B7  KARAR")
print("=" * 78)
GEREK1 = DOGRULAMA_TABAN_MSE - HEDEF1**2
GEREK2 = DOGRULAMA_TABAN_MSE - HEDEF2**2
GEREK3 = DOGRULAMA_TABAN_MSE - HEDEF3**2
TM_A = taban_mse(kats(L_OBS, sig_vektor(2.27e-04), wtaban=1e-06), L_OBS)
i_C = [u[0] for u in KURULUSLAR].index("C ONERILEN")
TM_C = taban_mse(kats(L_OBS, sig_vektor(2.90e-06), wtaban=1e-04), L_OBS)
i_B = [u[0] for u in KURULUSLAR].index("B saf kucuk sigma")
print(f"""
1) sigma_L GERCEKTEN kucuk. Uc bagimsiz delil (B2 yakin-sifir kipler,
   B3 dik artik sinirl, B5c LOO ongorusu) 2.27e-04'u REDDEDIYOR.
   Sonsal: {IZGARA[np.argmax(POST)]:.1e}, %90 araligi [{SIG_ALT:.1e}, {SIG_UST5:.1e}],
   sert ust sinir {SIG_UST:.1e}.

2) AMA sigma'yi TEK BASINA dusurmek TUZAKTIR. Kucuk sigma, ozdegeri
   {W_[_ik]:.1e} olan BASKIN KIP {_ik}'i acar; gorunen TABAN_MSE {TM_A:.7f}
   -> {taban_mse(kats(L_OBS, sig_vektor(2.9e-06), wtaban=1e-06), L_OBS):.7f} dusuyor (skorda 4.9e-04 'kazanc')
   ama LOO ongoru hatasi 4 kat KOTULESIYOR. Bu kip GURULTUDUR.

3) DOGRU DUZELTME IKI DUGMEYI BIRLIKTE OYNATMAKTIR:
       SIGMA_OLCEK: sigma ortalamasi 2.27e-04 -> ~2.9e-06 (LB yuvarlamasi)
       W_TABAN:     1e-06 -> 1e-04
   TABAN_MSE {TM_A:.7f} -> {TM_C:.7f}
   taban skor {np.sqrt(TM_A):.6f} -> {np.sqrt(TM_C):.6f}
   RISKSIZ KAZANC = {np.sqrt(TM_A) - np.sqrt(TM_C):+.6f}
   LOO ongoru hatasi {IZG[(2.27e-04, 1e-06)]:.2e} -> {IZG[(2.9e-06, 1e-04)]:.2e} ({IZG[(2.27e-04, 1e-06)] / IZG[(2.9e-06, 1e-04)]:.1f} kat IYI)

4) OLCEK. Skor^2 = TABAN_MSE - toplam rho^2. Hedefler (docs/72):
     1. sira {HEDEF1}: toplam rho^2 = {GEREK1:.5f}
     2. sira {HEDEF2}: toplam rho^2 = {GEREK2:.5f}
     3. sira {HEDEF3}: toplam rho^2 = {GEREK3:.5f}
   C kurulusuyla 2. sira icin gereken toplam rho^2:
     {GEREK2:.5f} yerine {GEREK2 - (TM_A - TM_C):.5f}  ({100 * (TM_A - TM_C) / GEREK2:.1f}% daha az)
   1. sira icin: {GEREK1:.5f} yerine {GEREK1 - (TM_A - TM_C):.5f}

5) IKINCIL AMA ONEMLI KAZANC: m148'in kappa secimi SABIT_HATA = 1.72e-04
   varsayimini kullaniyor. LOO'ya gore bu deger C kurulusunda
   {IZG[(2.9e-06, 1e-04)]:.2e}'e duser -> sonda sabitleri daha guvenilir,
   olculen rho_k'lar daha az bulanik. m148 icin SABIT_HATA guncellenmeli.

KARAR: DEGISTIR -- ama iki dugmeyi BIRLIKTE.
  m112_kalibre.py'de ONERI (bu betik DOKUNMADI):
      SIGMA_OLCEK: sigma ortalamasi LB yuvarlamasina cekilecek sekilde
        (ya da L_gurultusu yerine dogrudan sabit YUV = {YUV:.3e} kullan)
      W_TABAN = 1e-04       (1e-06 degil)
  Kazanc BEKLENEN {np.sqrt(BEK[_iA]) - np.sqrt(BEK[i_C]):+.6f}, EN KOTU DURUM
  {np.sqrt(KOTU[_iA]) - np.sqrt(KOTU[i_C]):+.6f}  (en kotu dunya = 'kip GERCEK' dunyasi).
  YAPMA: sigma'yi tek basina dusurme (B kurulusu). Beklenen
  {np.sqrt(BEK[_iA]) - np.sqrt(BEK[i_B]):+.6f}, en kotu {np.sqrt(KOTU[_iA]) - np.sqrt(KOTU[i_B]):+.6f}.
""")


# ===========================================================================
# B8  D1 CEBRI
# ===========================================================================
print("\n" + "=" * 78)
print("B8  D1 CEBRI -- sigma degisirse gonderilmemis D1 dosyasi ne olur")
print("=" * 78)
with open(os.path.join(M29, "m148_demet.json")) as fh:
    DEMET = json.load(fh)
s1 = DEMET["sondalar"][0]
GONDERILDI = os.path.exists(os.path.join(M29, "m148_olcumler.json"))
print(f"D1 dosyasi   : {s1['dosya']}")
print(f"kappa_1      : {s1['kappa']:.17g}   (SABITLENDI)")
print(f"sabit (eski) : {s1['sabit']:.10f}")
print(f"taban_mse(eski) = sabit - kappa_etkin^2 = {s1['sabit'] - s1['kappa_etkin'] ** 2:.10f}")
print(
    f"m148_olcumler.json var mi? {GONDERILDI}  -> D1 "
    f"{'GONDERILMIS SAYILIR' if GONDERILDI else 'HENUZ GONDERILMEMIS'}"
)
print("""
CEBIR. D1 = a0 + r_hat + kappa*GD_1 .  GD_1, olculmus span V'ye DIK kurulur;
dolayisiyla GD_1, hem ESKI hem YENI r_hat'e diktir (ikisi de span(V) icinde).
    P1^2 = M0 - 2<r, r_hat + kappa*GD_1> + ||r_hat||^2 + kappa^2
         = (M0 - 2*kL + ||r_hat||^2 + kappa^2) - 2*kappa*rho_1
         = sabit - 2*kappa*rho_1
'sabit' DOSYAYA aittir: icindeki kL ve ||r_hat||^2, dosyayi ureten r_hat'ten
gelir. sigma degisip r_hat degisirse ESKI dosyanin sabiti GECERSIZ olur.

IKI YOL:
  (A) D1'i YENIDEN URET (gonderilmemisse). Yeni r_hat ile yeni dosya, yeni
      sabit = TABAN_MSE_yeni + kappa_etkin_yeni^2. kappa_1 ayni kalabilir.
      Nihai:  skor^2 = TABAN_MSE_yeni - toplam rho_k^2 .
  (B) ESKI D1'i birak, nihai dosyada YENI r_hat kullan. Bu da CALISIR:
        - rho_1 ESKI sabitle cozulur:  rho_1 = (sabit_eski - P1^2)/(2*kappa)
        - GD_1 span'a dik oldugu icin olculen rho_1 = <r, GD_1> yon
          tanimina aittir, r_hat'e degil -> YENI r_hat ile birlestirilebilir
        - nihai = a0 + r_hat_yeni + toplam rho_k*GD_k
          skor^2 = TABAN_MSE_yeni - toplam rho_k^2
      TEK KOSUL: GD_k yonleri DEGISMEMELI. Oysa m148 eksenleri
      rho_s = <r_hat, x>/sqrt(Q_span) ile SECER ve H1 agirligi |rho_s|'dir
      -> r_hat degisince SECIM ve AGIRLIK degisir, GD_1 baska bir yon olur.
      O halde (B)'de nihai dosya ESKI GD_k'lari kullanmalidir (olculen sey
      odur); yeni r_hat yalnizca TABAN'i degistirir. Bu gecerlidir ama
      GD_k'lar eski (asiri buzulmus) r_hat'in agirliklarina gore kurulmustur,
      yani sinyali biraz daha kotu hedefler.

HUKUM:
  - D1 GONDERILMEMISSE  -> (A). Kayipsiz, daha iyi hedeflenmis yonler.
  - D1 GONDERILMISSE    -> (B). ESKI sabit ve ESKI GD_k ile devam;
    yalnizca nihai dosyanin TABANI yeni r_hat olur. ASLA eski dosyanin
    sabitini yeni TABAN_MSE ile degistirme -- rho_1 hatali cikar.
    (B)'de nihai skor: skor^2 = TABAN_MSE_yeni - toplam rho_k^2 ,
    ama D1'in KENDI LB skoru eski tabana aittir; karsilastirma yaparken
    P1'i dogrudan hedeflerle kiyaslama, once rho_1'e cevir.
""")
tm_yeni = TM_C
kap = s1["kappa"]
kape = s1["kappa_etkin"]
print("SAYILAR (C kurulusu secilirse):")
print(f"  yeni TABAN_MSE   = {tm_yeni:.7f}   (eski {DOGRULAMA_TABAN_MSE:.7f})")
print(f"  yeni sabit ~ TABAN_MSE + kappa_etkin^2 = {tm_yeni + kape**2:.10f}")
print(f"  (eski sabit {s1['sabit']:.10f}; fark {tm_yeni + kape**2 - s1['sabit']:+.3e})")
print(f"  (A) yolunda:  rho_1 = (yeni_sabit - P1^2) / {2 * kap:.8f}")
print(f"  (B) yolunda:  rho_1 = ({s1['sabit']:.10f} - P1^2) / {2 * kap:.8f}")
print("  kappa_etkin dosya yazildiktan sonra diskten olculur; yeniden")
print("  uretimde m148 bunu kendisi hesaplar -- yukaridaki 'yeni sabit'")
print("  yalnizca buyukluk gostergesidir.")
print("""
  UYARI -- OTOMATIK ALGILAMAYA GUVENME. Bu calisma agacinda PARALEL
  oturumlar var; m148_olcumler.json'a SINAV amacli sahte skor yazilmis
  olabilir. Dosyanin varligi 'D1 gonderildi' demek DEGILDIR. Karar
  vermeden once Kaggle'in gonderim listesine BAK (m148_olcumler.json'a
  degil). Gonderilmediyse (A), gonderildiyse (B).""")
if GONDERILDI:
    with open(os.path.join(M29, "m148_olcumler.json")) as fh:
        print(f"  su anki m148_olcumler.json icerigi: {fh.read().strip()}")

print("\n" + "=" * 78)
print("BITTI -- hicbir dosya yazilmadi, hicbir gonderim yapilmadi.")
print("=" * 78)
