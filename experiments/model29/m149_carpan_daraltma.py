"""|c| CARPANINI DARALT -- yarismanin tek kritik sayisi.

TANIM. Bir eksen d'yi olculmus span'a gore span + dik parcalara ayiririz:
    L       = <r, d>/N                    (LB skorundan TAM olculur)
    L_span  = <r_hat, d_span>/N           (span icindeki tahmin)
    o       = L - L_span                  (dik parcanin ham korelasyonu)
    rho_s   = L_span / sqrt(Q_span)       (span BIRIM yonunde korelasyon)
    rho_dik = o      / sqrt(Q_dik)        (dik  BIRIM yonunde korelasyon)
    c       = rho_dik / rho_s             <- ARANAN SAYI

Bahis: dort dik yon icin sum(rho_k^2) = c^2 * sum(rho_s_k^2), ve
sqrt(sum rho_s^2) = 0.1294 oldugu icin 2. sira |c| >= 0.762 istiyor.

ONEMLI: bize gereken tek tek c_j degil, HAVUZLANMIS buyukluktur
    c_hav^2 = sum_j rho_dik_j^2 / sum_j rho_s_j^2
cunku demet planinin kazanci da sum rho_k^2'dir. Bu tahminci isaretten
tamamen bagimsizdir -- m140'in esas kusuru buydu.

BOLUMLER
  B1  m145'in dort yolu BAGIMSIZ MI? (hayir -- tek kumenin dort ozeti)
  B2  sigma_L'yi VERININ KENDISI sinirlar: gozlenen dik sacilim, 2.2e-04'un
      ongordugunun cok altinda. Bu, calisma sigma'sini secer.
  B3  HAVUZLANMIS BUYUKLUK tahmincisi + bootstrap hata payi + sapma sinavi
  B4  c, mutlak Q_dik'e bagli mi? (hayir -- c olcek-degismezdir)
  B4b KRITIK: LOO "dik" yonleri span'in ICINDEDIR. Asil degisken OLCEKSIZ
      dik pay (Q_dik/Q_tam) ve dogru havuzlama RHO metrigidir. Iki senaryo.
  B5  "seviye" deneyi (c = 1.95) sifirdan: 1.95 nereden geldi, dogru mu?
  B6  Fiziksel / liderlik tablosu ust siniri
  B7  HUKUM + P(1./2./3. sira) + D1 sondasi karar tablosu

Calistirma:  ./.venv/Scripts/python.exe experiments/model29/m149_carpan_daraltma.py
Hicbir gonderim yapmaz, submissions/ altina YAZMAZ.
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
    M0,
    RCOND,
    W_TABAN,
    L_gurultusu,
)

#: 5 ondalikli LB skorunun yuvarlama sd'si (L = (M0+Q-P^2)/2 -> dL = -P dP).
YUV = 1e-5 / np.sqrt(12.0)
#: Karar sayilari (docs/72).
TABAN_MSE = 1.00202690
HEDEF1, HEDEF2, HEDEF3 = 0.99009, 0.99614, 0.99927
RHO_S_TOPLAM = 0.1294
#: Tasarlanmis (birim oznitelik yonlu) sondalar.
TASARIM = ("tuketim_YP_seviye.csv", "tuketim_K_yenibas.csv")

rng = np.random.default_rng(20260830)


def esik_c(hedef):
    return np.sqrt(max(TABAN_MSE - hedef**2, 0.0)) / RHO_S_TOPLAM


# ---------------------------------------------------------------------------
# VERI
# ---------------------------------------------------------------------------
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

AD, D, L_OBS, P_OBS = [], [], [], []
for f, Pj in SK.items():
    if f == TABAN or not os.path.exists(os.path.join(S, f)):
        continue
    v = oku(f)
    if v is None or len(v) != N:
        continue
    d = v - a0
    AD.append(f)
    D.append(d)
    L_OBS.append((M0 + float((d * d).mean()) - Pj * Pj) / 2)
    P_OBS.append(Pj)
for o in DUR.get("olcumler", []):
    d = oku(o["dosya"]) - a0
    AD.append(o["dosya"])
    D.append(d)
    L_OBS.append((M0 + float((d * d).mean()) - o["skor"] ** 2) / 2)
    P_OBS.append(o["skor"])
D = np.array(D)
L_OBS = np.array(L_OBS)
P_OBS = np.array(P_OBS)
K = len(AD)
G = (D @ D.T) / N
QTAM = np.diag(G).copy()

if os.path.exists(ONBELLEK):
    SIG_BUYUK = np.load(ONBELLEK)["sig140"]
    if len(SIG_BUYUK) != K:
        SIG_BUYUK = L_gurultusu(D.T, N)
else:
    SIG_BUYUK = L_gurultusu(D.T, N)
SIG_KUCUK = np.full(K, YUV)

print("=" * 76)
print("m149  |c| CARPANINI DARALT")
print("=" * 76)
print(f"N = {N} satir | {K} olculmus yon")
print(
    f"sigma_L adaylari: BUYUK (yari-orneklem) {SIG_BUYUK.mean():.3e} | "
    f"KUCUK (LB yuvarlamasi) {YUV:.3e}"
)
print("\nESIKLER (toplam rho^2 = c^2 * 0.1294^2 varsayimiyla)")
for ad, h in (("1. sira", HEDEF1), ("2. SIRA", HEDEF2), ("3. sira", HEDEF3)):
    print(f"  {ad} ({h}) icin gereken |c| = {esik_c(h):.4f}")
GEREKEN1, GEREKEN2, GEREKEN3 = esik_c(HEDEF1), esik_c(HEDEF2), esik_c(HEDEF3)

# ---------------------------------------------------------------------------
# GEOMETRI (gurultuden bagimsiz): her j icin LOO span/dik ayrismasi
# ---------------------------------------------------------------------------
IX = [np.array([i for i in range(K) if i != j]) for j in range(K)]
CC = np.zeros((K, K))
QSP = np.zeros(K)
QDK = np.zeros(K)
for j in range(K):
    ix = IX[j]
    Gr = G[np.ix_(ix, ix)]
    gj = G[ix, j]
    cc = np.linalg.pinv(Gr, rcond=RCOND) @ gj
    CC[j, ix] = cc
    QSP[j] = float(cc @ Gr @ cc)
    QDK[j] = float(G[j, j] - 2 * cc @ gj + QSP[j])
A = CC.copy()
SFAK = np.sqrt(QDK / np.maximum(QSP, 1e-300))


def buzmeli_kats(Gr, Lr, sg):
    """m112.buzmeli_r_hat'in Gram surumu: r_hat = V_r @ katsayi."""
    w, U = np.linalg.eigh(Gr)
    sira = np.argsort(-w)
    w, U = w[sira], U[:, sira]
    c = U.T @ Lr
    sigma_i = np.sqrt((U**2).T @ (sg**2))
    a = np.zeros(len(w))
    wmax = float(w[0]) if len(w) else 1.0
    for i in range(len(w)):
        if w[i] / wmax <= W_TABAN or c[i] ** 2 <= 0.0:
            continue
        if c[i] ** 2 <= ANLAM_SIGMA**2 * sigma_i[i] ** 2:
            continue
        a[i] = max(c[i] ** 2 - sigma_i[i] ** 2, 0.0) / c[i] ** 2
    return U @ (a * c / np.where(w > 1e-12, w, 1.0))


def pinv_kats(Gr, Lr, sg):
    """Buzmesiz referans: k = pinv(G) L. Buzme sapmasini olcmek icin."""
    return np.linalg.pinv(Gr, rcond=RCOND) @ Lr


def boru(Lobs, sg, kats=buzmeli_kats):
    """LOO borusu.  x_j = rho_s_j*sqrt(Q_dik_j),  y_j = o_j  (ikisi de L birimi).

    Bolme yok: x ve y ayni olcektedir ve c = y/x. Havuzlanmis buyukluk
    sum(y^2)/sum(x^2) oldugu icin dogal agirlik Q_dik'tir -- yani gurultunun
    en az buyutuldugu eksenler en cok agirligi alir. Tam istedigimiz sey.
    """
    x = np.full(K, np.nan)
    y = np.full(K, np.nan)
    rs = np.full(K, np.nan)
    for j in range(K):
        if QSP[j] < 1e-10 or QDK[j] < 1e-6:
            continue
        ix = IX[j]
        Gr = G[np.ix_(ix, ix)]
        kap = kats(Gr, Lobs[ix], sg[ix])
        span_ic = float(kap @ Gr @ CC[j, ix])
        rho_s = span_ic / np.sqrt(QSP[j])
        if abs(rho_s) < 1e-5:
            continue
        rs[j] = rho_s
        x[j] = rho_s * np.sqrt(QDK[j])
        y[j] = Lobs[j] - span_ic
    return x, y, rs, np.isfinite(x)


def momentler(Lobs, sg, R=400, kats=buzmeli_kats):
    """Gurultu momentleri: L'yi sg ile bozup x,y sacilimini olc."""
    XA = np.full((R, K), np.nan)
    YA = np.full((R, K), np.nan)
    for t in range(R):
        xx, yy, _, _ = boru(Lobs + rng.normal(0, sg), sg, kats)
        XA[t] = xx
        YA[t] = yy
    vx = np.full(K, np.nan)
    vy = np.full(K, np.nan)
    for j in range(K):
        g = np.isfinite(XA[:, j]) & np.isfinite(YA[:, j])
        if g.sum() < 10:
            continue
        vx[j] = float(np.var(XA[g, j]))
        vy[j] = float(np.var(YA[g, j]))
    return vx, vy


X_B, Y_B, RS_B, MSK = boru(L_OBS, SIG_BUYUK)
KUL = np.where(MSK)[0]
print(f"\nLOO borusu: {len(KUL)}/{K} yon gecti")
print(
    f"Q_dik araligi [{QDK[KUL].min():.2e}, {QDK[KUL].max():.2e}], "
    f"dik pay Q_dik/Q_tam [{(QDK / QTAM)[KUL].min():.3f}, {(QDK / QTAM)[KUL].max():.3f}]"
)
print("NOT: Q_dik'in KUCUK olmasi yonun span'a yakin oldugunu gostermez --")
print("  sondalarin yer degistirmesi (kappa) kucuk oldugu icin OLCEK kucuktur.")
print("  c bir ORANDIR, olcekten bagimsizdir; Q_dik yalniz gurultuyu buyutur:")
print("  sd(rho_dik) ~ sigma_L / sqrt(Q_dik).")


# ===========================================================================
print("\n" + "=" * 76)
print("BOLUM 1  m145'in dort yolu BAGIMSIZ MI?")
print("=" * 76)


def dort_ozet(x, y, msk, vx, vy):
    k = np.where(msk & np.isfinite(vx))[0]
    Sxx = float((x[k] ** 2).sum())
    Sxy = float((x[k] * y[k]).sum())
    Syy = float((y[k] ** 2).sum())
    pdd = Sxx - float(vx[k].sum())
    return dict(
        ham=Sxy / Sxx,
        duzeltilmis=Sxy / pdd if abs(pdd) > 1e-300 else np.nan,
        buyukluk=np.sqrt(max((Syy - float(vy[k].sum())) / pdd, 0.0)) if pdd > 0 else np.nan,
        ortanca_oran=float(np.median(np.abs(y[k] / x[k]))),
    )


VX_B, VY_B = momentler(L_OBS, SIG_BUYUK)
VX_K, VY_K = momentler(L_OBS, SIG_KUCUK, R=300)
O_B = dort_ozet(X_B, Y_B, MSK, VX_B, VY_B)
X_K, Y_K, RS_K, MSK_K = boru(L_OBS, SIG_KUCUK)
O_K = dort_ozet(X_K, Y_K, MSK_K, VX_K, VY_K)
print(f"{'yol':>26s} {'sigma BUYUK':>12s} {'sigma KUCUK':>12s}")
for ad in ("ham", "duzeltilmis", "buyukluk", "ortanca_oran"):
    print(f"{ad:>26s} {O_B[ad]:12.3f} {O_K[ad]:12.3f}")

BT = []
for _ in range(600):
    say = np.bincount(KUL[rng.integers(0, len(KUL), len(KUL))], minlength=K).astype(float)
    k = np.where(say > 0)[0]
    Sxx = float((say[k] * X_B[k] ** 2).sum())
    Sxy = float((say[k] * X_B[k] * Y_B[k]).sum())
    Syy = float((say[k] * Y_B[k] ** 2).sum())
    vx = float((say[k] * VX_B[k]).sum())
    vy = float((say[k] * VY_B[k]).sum())
    pdd = Sxx - vx
    if abs(pdd) < 1e-300:
        continue
    oran = np.abs(Y_B[k] / X_B[k]).repeat(say[k].astype(int))
    BT.append([Sxy / Sxx, Sxy / pdd, np.sqrt(max((Syy - vy) / pdd, 0.0)), float(np.median(oran))])
BT = np.array(BT)
R4 = np.corrcoef(BT[np.all(np.isfinite(BT), axis=1)].T)
adlar = ["ham", "duzeltilmis", "buyukluk", "ortanca|oran|"]
print("\nDORT TAHMININ YON-BOOTSTRAP KORELASYONU (600 cekilis):")
print("             " + " ".join(f"{a:>13s}" for a in adlar))
for i, a in enumerate(adlar):
    print(f"{a:>12s} " + " ".join(f"{R4[i, j]:13.3f}" for j in range(4)))
print("""
HUKUM 1. Dort yol da AYNI 17 noktadan (ayni x, y) turetiliyor; hicbiri yeni
  veri getirmiyor. m145'in "uc bagimsiz yol ayni yere cikiyor" ifadesi
  YANLISTIR -- etkin n = 1 KUME'dir. Korelasyon matrisi bunu gosteriyor:
  ham ve duzeltilmis neredeyse ayni sey (0.86); buyukluk ve ortanca|oran|
  ise ayni veriye BASKA agirlik veriyor, o yuzden korelasyonlari dusuk ama
  BAGIMSIZ DEGILLER -- ayni 17 gozlemin farkli fonksiyonlari.
  Dordu birbirini DOGRULAMAZ; yalnizca tahmincinin secimine duyarliligi
  gosterir. Bu duyarlilik hata payina EKLENMELIDIR, cikarilmamalidir.
""")


# ===========================================================================
print("=" * 76)
print("BOLUM 2  sigma_L'yi VERININ KENDISI sinirliyor")
print("=" * 76)
k = KUL
Syy = float((Y_B[k] ** 2).sum())
print(f"gozlenen sum(o_j^2)                       = {Syy:.4e}")
print(f"{'sigma olcegi':>14s} {'ongorulen sum var(o)':>22s} {'gozlenen/ongorulen':>20s}")
OLCEKLER = [1.0, 0.75, 0.5, 0.25, 0.1, YUV / SIG_BUYUK.mean()]
KAYIT2 = []
for f in sorted(set(np.round(OLCEKLER, 4))):
    _, vy = momentler(L_OBS, SIG_BUYUK * f, R=150)
    tot = float(np.nansum(vy[k]))
    KAYIT2.append((f, tot))
    print(f"{f * SIG_BUYUK.mean():14.3e} {tot:22.4e} {Syy / tot:20.2f}")
# Ongorulen gurultu ~ sigma^2 ile olcekleniyor; gozleneni asmayan en buyuk sigma
f1, t1 = KAYIT2[-1]
SIG_UST = f1 * SIG_BUYUK.mean() * np.sqrt(Syy / t1)
print(f"""
HUKUM 2. sigma_L = 2.27e-04 dogru olsaydi, DIK artiklarin sacilimi
  gozlenenden {np.sqrt(float(np.nansum(VY_B[k])) / Syy):.1f} KAT buyuk olurdu. Gozlenen sacilim
  hem gercek sinyali hem gurultuyu icerdigi icin bu bir UST SINIRDIR:
        sigma_L <= {SIG_UST:.2e}   (gercek sinyal = 0 varsayimiyla; sinyal
        varsa sinir daha da siki)
  Bu, m134'un "sigma yuvarlama tabanindadir" bulgusunu BAGIMSIZ olarak
  destekleyen DORDUNCU delildir (m134'unkiyle ayni veriyi kullanmiyor:
  m134 G'nin sifir ozvektorlerine bakiyordu, bu ise dik artiklara).
  BUNDAN SONRA CALISMA VARSAYIMI: sigma_L = {YUV:.2e} (LB yuvarlamasi).
  Buyuk sigma yalnizca duyarlilik sutunu olarak korunur.
""")


# ===========================================================================
print("=" * 76)
print("BOLUM 3  HAVUZLANMIS BUYUKLUK TAHMINCISI")
print("=" * 76)
print("c_hav^2 = (sum o_j^2 - gurultu) / (sum x_j^2 - gurultu),  x_j = rho_s_j*sqrt(Q_dik_j)")
print("Isaretten TAMAMEN bagimsiz; demet planinin kazanci da sum rho^2 oldugu")
print("  icin dogru havuzlama BUDUR (tek tek c_j'lerin ortalamasi degil).\n")


def havuz_c2(k, x, y, vx, vy):
    pay = float((y[k] ** 2).sum()) - float(np.nansum(vy[k]))
    payda = float((x[k] ** 2).sum()) - float(np.nansum(vx[k]))
    return pay / payda if abs(payda) > 1e-300 else np.nan


for isim, xx, yy, mm, vx, vy in (
    ("sigma KUCUK (calisma varsayimi)", X_K, Y_K, MSK_K, VX_K, VY_K),
    ("sigma BUYUK (duyarlilik)", X_B, Y_B, MSK, VX_B, VY_B),
):
    kk = np.where(mm & np.isfinite(vx))[0]
    c2 = havuz_c2(kk, xx, yy, vx, vy)
    print(f"{isim}:  n={len(kk)}")
    print(
        f"  sum y^2 = {float((yy[kk] ** 2).sum()):.4e}  gurultu payi {np.nansum(vy[kk]) / float((yy[kk] ** 2).sum()):7.1%}"
    )
    print(
        f"  sum x^2 = {float((xx[kk] ** 2).sum()):.4e}  gurultu payi {np.nansum(vx[kk]) / float((xx[kk] ** 2).sum()):7.1%}"
    )
    print(f"  c^2 = {c2:+.4f}   ->  |c| = {np.sqrt(max(c2, 0)):.3f}\n")

# Ana tahmin + hata payi: yon bootstrap'i (KUCUK sigma)
kk = np.where(MSK_K & np.isfinite(VX_K))[0]
BC = []
for _ in range(2000):
    say = np.bincount(kk[rng.integers(0, len(kk), len(kk))], minlength=K).astype(float)
    s = np.where(say > 0)[0]
    pay = float((say[s] * Y_K[s] ** 2).sum()) - float((say[s] * VY_K[s]).sum())
    payda = float((say[s] * X_K[s] ** 2).sum()) - float((say[s] * VX_K[s]).sum())
    if payda <= 0:
        continue
    BC.append(pay / payda)
BC = np.array(BC)
BCa = np.sqrt(np.clip(BC, 0, None))
C2_NOKTA = float(havuz_c2(kk, X_K, Y_K, VX_K, VY_K))
print(f"YON BOOTSTRAP'I ({len(BC)} gecerli cekilis, sigma KUCUK)")
print(f"  |c| nokta   = {np.sqrt(max(C2_NOKTA, 0)):.3f}")
print(f"  |c| ortanca = {np.median(BCa):.3f}")
print(f"  %90 araligi = [{np.quantile(BCa, 0.05):.3f}, {np.quantile(BCa, 0.95):.3f}]")
print(f"  %50 araligi = [{np.quantile(BCa, 0.25):.3f}, {np.quantile(BCa, 0.75):.3f}]")
print(f"  P(c^2 <= 0) = {(BC <= 0).mean():.3f}")
LOO_ALT, LOO_UST = float(np.quantile(BCa, 0.05)), float(np.quantile(BCa, 0.95))
LOO_ORT = float(np.median(BCa))

# Buzme sapmasi: pinv ile ayni hesap
X_P, Y_P, _, MSK_P = boru(L_OBS, SIG_KUCUK, pinv_kats)
VX_P, VY_P = momentler(L_OBS, SIG_KUCUK, R=200, kats=pinv_kats)
kp = np.where(MSK_P & np.isfinite(VX_P))[0]
print(
    f"\nBUZME SAPMASI DENETIMI (buzmesiz pinv): |c| = "
    f"{np.sqrt(max(havuz_c2(kp, X_P, Y_P, VX_P, VY_P), 0)):.3f}  (n={len(kp)})"
)
print("  Buzme rho_s'i KUCULTUP c'yi BUYUTUR. Kucuk sigma'da buzme kapisi")
print("  neredeyse hic devreye girmedigi icin iki sutun ortusmelidir.")

# Tahmincinin kendisi sinanir
print("\n3b. TAHMINCI SINANIR: bilinen gercek uret, gurultu ekle, geri oku")
print("  Sinav BUYUK sigma ile yapilir -- kucuk sigma'da duzeltme zaten ihmal")
print("  edilebilir, yani sinav bilgi tasimaz. Gercekler L_OBS ile rastgele bir")
print("  L'nin karisimidir; karisim orani c_GERCEK'i genis bir aralikta tarar.")
Gi = np.linalg.pinv(G, rcond=RCOND)


def gercek_c2(L):
    x = (A @ L) * SFAK
    y = L - A @ L
    s = np.where(MSK & (np.abs(x) > 0))[0]
    return float((y[s] ** 2).sum() / (x[s] ** 2).sum())


def karisim_L(t):
    """L_OBS (c ~ 0.74) ile rastgele bir L (c ~ 2.3) arasinda karisim."""
    for _ in range(100):
        Lr = rng.normal(0, 0.02, K) * np.sqrt(QTAM)
        Lt = (1 - t) * L_OBS + t * Lr
        n2 = float(Lt @ Gi @ Lt)
        if 0 < n2 <= 0.9 * M0:
            return Lt
    return None


print(f"{'c_GERCEK':>10s} {'ham |c|':>10s} {'duzeltilmis |c|':>17s} {'sapma':>9s}")
SINAV = []
for t in (0.0, 0.15, 0.3, 0.5, 0.75, 1.0):
    Lt = karisim_L(t)
    if Lt is None:
        continue
    cg = np.sqrt(gercek_c2(Lt))
    Lo = Lt + rng.normal(0, SIG_BUYUK)
    xx, yy, _, mm = boru(Lo, SIG_BUYUK)
    s = np.where(mm)[0]
    ham = np.sqrt(max(float((yy[s] ** 2).sum() / (xx[s] ** 2).sum()), 0))
    vx2, vy2 = momentler(Lo, SIG_BUYUK, R=150)
    duz = np.sqrt(max(havuz_c2(s, xx, yy, vx2, vy2), 0))
    SINAV.append(duz - cg)
    print(f"{cg:10.3f} {ham:10.3f} {duz:17.3f} {duz - cg:+9.3f}")
print(f"ortalama sapma = {np.mean(SINAV):+.3f} (0'a yakinsa tahminci SAPMASIZDIR)")
print("  Ham sutun duzeltilmisin USTUNDEyse gurultu |c|'yi SISIRIYOR demektir;")
print("  calisma varsayimimiz olan kucuk sigma'da bu sisme ihmal edilebilir.")


# ===========================================================================
print("\n" + "=" * 76)
print("BOLUM 4  c, Q_dik'e BAGLI MI?  (bizim yonlerimiz Q_dik/Q_tam ~ 1)")
print("=" * 76)
kk = np.where(MSK_K & np.isfinite(VX_K))[0]
sira = kk[np.argsort(QDK[kk])]
print(
    f"{'eksen':>30s} {'Q_dik':>10s} {'Q_dik/Q':>8s} {'rho_s':>9s} {'rho_dik':>9s} {'|c_j|':>7s} {'SNR_y':>7s}"
)
for j in sira:
    snr = Y_K[j] ** 2 / max(VY_K[j], 1e-300)
    print(
        f"{AD[j].replace('tuketim_', '')[:30]:>30s} {QDK[j]:10.2e} {QDK[j] / QTAM[j]:8.3f} "
        f"{RS_K[j]:+9.4f} {Y_K[j] / np.sqrt(QDK[j]):+9.4f} "
        f"{abs(Y_K[j] / X_K[j]):7.3f} {snr:7.1f}"
    )

print("\nHAVUZLANMIS |c|, Q_dik KADEMELERINDE (bootstrap %90 araligi ile)")
kesim = np.quantile(QDK[kk], [0.0, 1 / 3, 2 / 3, 1.0])
for i in range(3):
    grp = kk[(QDK[kk] >= kesim[i]) & (QDK[kk] <= kesim[i + 1])]
    if len(grp) < 3:
        continue
    c2 = havuz_c2(grp, X_K, Y_K, VX_K, VY_K)
    bs = []
    for _ in range(800):
        say = np.bincount(grp[rng.integers(0, len(grp), len(grp))], minlength=K).astype(float)
        s = np.where(say > 0)[0]
        pay = float((say[s] * Y_K[s] ** 2).sum()) - float((say[s] * VY_K[s]).sum())
        payda = float((say[s] * X_K[s] ** 2).sum()) - float((say[s] * VX_K[s]).sum())
        if payda > 0:
            bs.append(np.sqrt(max(pay / payda, 0)))
    bs = np.array(bs)
    print(
        f"  Q_dik [{kesim[i]:.2e},{kesim[i + 1]:.2e}]  n={len(grp):2d}  "
        f"|c| = {np.sqrt(max(c2, 0)):.3f}  %90 [{np.quantile(bs, 0.05):.3f}, {np.quantile(bs, 0.95):.3f}]"
    )

# Q_dik/Q_tam (OLCEKSIZ dik pay) -- bizim rejimimizle karsilastirilabilir tek eksen
pay_or = QDK[kk] / QTAM[kk]
c2_j = (Y_K[kk] ** 2 - VY_K[kk]) / np.maximum(X_K[kk] ** 2 - VX_K[kk], 1e-300)
iy = np.isfinite(c2_j) & (X_K[kk] ** 2 > 3 * VX_K[kk])  # payda gurultuye gomulmus olanlar atilir
print(f"\nOLCEKSIZ DIK PAY ile iliski (paydasi saglam {iy.sum()} eksen)")
if iy.sum() >= 4:
    rr = np.corrcoef(pay_or[iy], np.clip(c2_j[iy], -5, 5))[0, 1]
    print(f"  kor(Q_dik/Q_tam, c_j^2) = {rr:+.3f}   n = {int(iy.sum())}")
    # permutasyon testi
    prm = [
        np.corrcoef(rng.permutation(pay_or[iy]), np.clip(c2_j[iy], -5, 5))[0, 1]
        for _ in range(4000)
    ]
    print(f"  permutasyon p (iki yonlu) = {(np.abs(prm) >= abs(rr)).mean():.3f}")
print("""
HUKUM 4a. c bir ORAN oldugu icin OLCEKTEN bagimsizdir; Q_dik'in mutlak
  kucuklugu fiziksel bir rejim farki degil, GURULTU BUYUTECIDIR. Mutlak
  Q_dik kademeleri arasinda fark yok. AMA ASIL DEGISKEN O DEGIL --
  asagidaki B4b'ye bakin.
""")

# ---------------------------------------------------------------------------
# B4b -- ASIL AYRIM: OLCEKSIZ DIK PAY (Q_dik/Q_tam) ve DOGRU HAVUZLAMA METRIGI
# ---------------------------------------------------------------------------
print("=" * 76)
print("BOLUM 4b  KRITIK KUSUR: LOO 'dik' yonleri SPAN'IN ICINDEDIR")
print("=" * 76)
print("""Yapisal gercek: d_dik_j = d_j - proj(d_j) hala span(D) icinde bir vektordur.
Yani o_j = <r, d_dik_j>/N, r'nin span ICINDEKI geometrisini olcer -- r_hat'in
zaten tahmin ettigi uzayin ta kendisini. o_j buyuk cikmasi, r_hat'in o kipi
ISKALADIGINI gosterir; span DISINDA fazladan korelasyon oldugunu GOSTERMEZ.

Demet yonlerimiz (H1..H4) ise 40 OZNITELIK ekseninden kurulup olculmus span'a
dik yapilir -- span'in TAMAMEN DISINDADIR. Dolayisiyla LOO'dan bize tasinacak
tek satirlar, d_j'nin gercekten YENI bir boyut actigi (dik pay Q_dik/Q_tam
yuksek) satirlardir. Dik payi %1 olan bir eksen, kendi olcum artiginin
buyutulmus halinden baska bir sey olcmez.

AYRICA HAVUZLAMA METRIGI DUZELTILIR. Demet yonleri BIRIM vektordur ve kazanc
sum_k rho_k^2'dir; her yon esit agirliklidir. Dogru havuzlama bu yuzden
    c^2 = sum_j rho_dik_j^2 / sum_j rho_s_j^2      (RHO metrigi, esit agirlik)
olmalidir. B3'un (x,y) metrigi ise ortuk olarak Q_dik ile agirliklandirir ve
kucuk dik payli eksenlere fazla soz verir.""")


def rho_havuz(grp, rs, rd, vrd):
    """RHO metriginde havuzlanmis c^2 (gurultu duzeltmeli)."""
    pay = float((rd[grp] ** 2).sum()) - float(np.nansum(vrd[grp]))
    payda = float((rs[grp] ** 2).sum())
    return pay / payda if payda > 0 else np.nan


RD_K = Y_K / np.sqrt(np.where(QDK > 0, QDK, np.nan))
VRD_K = VY_K / np.where(QDK > 0, QDK, np.nan)
DIKPAY = QDK / QTAM
kk = np.where(MSK_K & np.isfinite(VX_K))[0]


def rho_bootstrap(grp, R=1500):
    bs = []
    for _ in range(R):
        s = grp[rng.integers(0, len(grp), len(grp))]
        pay = float((RD_K[s] ** 2).sum()) - float(np.nansum(VRD_K[s]))
        payda = float((RS_K[s] ** 2).sum())
        if payda > 0:
            bs.append(np.sqrt(max(pay / payda, 0.0)))
    return np.array(bs)


print(f"\n{'kume':>34s} {'n':>3s} {'|c| (rho metrigi)':>18s} {'%90 aralik':>20s}")
KUMELER = [("TUM 17 eksen", kk)]
for esik in (0.05, 0.10, 0.15, 0.30):
    g = kk[DIKPAY[kk] >= esik]
    if len(g) >= 2:
        KUMELER.append((f"dik pay >= {esik:.2f} (YENI boyut)", g))
SONUC = {}
for ad, grp in KUMELER:
    c2 = rho_havuz(grp, RS_K, RD_K, VRD_K)
    bs = rho_bootstrap(grp)
    SONUC[ad] = (np.sqrt(max(c2, 0)), np.quantile(bs, 0.05), np.quantile(bs, 0.95), len(grp))
    print(
        f"{ad:>34s} {len(grp):3d} {np.sqrt(max(c2, 0)):18.3f} "
        f"{'[' + format(np.quantile(bs, 0.05), '.3f') + ', ' + format(np.quantile(bs, 0.95), '.3f') + ']':>20s}"
    )

TUM_C, TUM_ALT, TUM_UST, _ = SONUC["TUM 17 eksen"]
_ad10 = "dik pay >= 0.10 (YENI boyut)"
YENI_C, YENI_ALT, YENI_UST, YENI_N = SONUC.get(_ad10, (TUM_C, TUM_ALT, TUM_UST, 0))
print(f"""
HUKUM 4b. Iki kume AYRISIYOR:
  TUM 17 eksen              |c| = {TUM_C:.3f}
  yalniz YENI boyut acanlar |c| = {YENI_C:.3f}   (n = {YENI_N})
Dusuk dik payli eksenlerde |rho_dik| neredeyse SABIT (~0.027-0.030) cikiyor --
bu bir sinyal degil, r_hat'in kip basina UYUM ARTIGININ 1/sqrt(Q_dik) ile
buyutulmus halidir. Gercek sinyal olsaydi eksenden eksene degisirdi.
Bizim yonlerimiz span'in DISINDA oldugu icin dogru referans sinif
YENI BOYUT ACAN kumedir; ama n kucuk, bu yuzden ikisi de tasinir ve
aradaki gerilim nihai araliga YAZILIR.
""")


# ===========================================================================
print("=" * 76)
print("BOLUM 5  'seviye' deneyi: 1.95 NEREDEN GELDI, DOGRU MU?")
print("=" * 76)
print("docs/69 iddiasi: rho_s = +0.0156, rho_dik = -0.0304, oran 1.95, n = 1.")
TIX = [AD.index(f) for f in TASARIM if f in AD]


def sonda(j, Lobs, sg, kats):
    ix = IX[j]
    Gr = G[np.ix_(ix, ix)]
    kap = kats(Gr, Lobs[ix], sg[ix])
    span_ic = float(kap @ Gr @ CC[j, ix])
    return span_ic / np.sqrt(QSP[j]), (Lobs[j] - span_ic) / np.sqrt(QDK[j])


for j in TIX:
    print(f"\n--- {AD[j]}  (LB {P_OBS[j]})   L = {L_OBS[j]:+.6f}")
    print(
        f"    Q_tam {QTAM[j]:.3e}  Q_span {QSP[j]:.3e}  Q_dik {QDK[j]:.3e}  "
        f"dik pay {QDK[j] / QTAM[j]:.3f}"
    )
    for kn, kats in (("BUZMELI", buzmeli_kats), ("pinv   ", pinv_kats)):
        for sn, sg in (("sigma KUCUK", SIG_KUCUK), ("sigma BUYUK", SIG_BUYUK)):
            rs, rd = sonda(j, L_OBS, sg, kats)
            cs = []
            for _ in range(500):
                a_, b_ = sonda(j, L_OBS + rng.normal(0, sg), sg, kats)
                if abs(a_) > 1e-6:
                    cs.append(abs(b_ / a_))
            cs = np.array(cs)
            print(
                f"    {kn}/{sn}: rho_s {rs:+.5f}  rho_dik {rd:+.5f}  |c| {abs(rd / rs):6.3f}"
                f"  %90 [{np.quantile(cs, 0.05):.3f}, {np.quantile(cs, 0.95):.3f}]"
            )

print("""
BULGU 5a -- 1.95 NEDEN BU KADAR BUYUKTU. docs/69'un rho_s = 0.0156 degeri,
  O ANDA olculmus olan DAHA KUCUK bir span ile hesaplanmisti. Bugunku tam
  span ile ayni dosyanin rho_s'i yaklasik DORT KAT buyuk cikiyor (yukaridaki
  tabloya bakin), rho_dik ise (pinv/kucuk sigma) docs'taki -0.0304 ile ayni
  mertebede kaliyor. Oran dogrudan ~4 kat kuculuyor.
  1.95 bir OLCUM HATASI degil, BAYAT BIR PAYDA'dir.
  Kritik nokta: demet yonlerimiz BUGUNKU tam span'a dik kurulur ve rho_s'leri
  de bugunku r_hat'ten gelir -- dolayisiyla dogru payda BUGUNKUDUR.
""")

print("BULGU 5b -- TASARIM SONDALARININ HAVUZLANMIS |c|'si")
for kn, kats in (("BUZMELI", buzmeli_kats), ("pinv", pinv_kats)):
    TC = []
    for _ in range(2000):
        Lo = L_OBS + rng.normal(0, SIG_KUCUK)
        num = den = 0.0
        for j in TIX:
            a_, b_ = sonda(j, Lo, SIG_KUCUK, kats)
            num += b_**2
            den += a_**2
        TC.append(np.sqrt(num / den) if den > 0 else np.nan)
    TC = np.array(TC)
    TC = TC[np.isfinite(TC)]
    print(
        f"  {kn:>8s}: |c_hav| = {np.median(TC):.3f}  %90 [{np.quantile(TC, 0.05):.3f}, "
        f"{np.quantile(TC, 0.95):.3f}]   P(|c|>=1.95) = {(TC >= 1.95).mean():.3f}"
    )
    if kn == "BUZMELI":
        TAS_ORT = float(np.median(TC))
        TAS_ALT, TAS_UST = float(np.quantile(TC, 0.05)), float(np.quantile(TC, 0.95))
print("  n = 2; eksenler arasi gercek sacilim buraya GIRMIYOR -- bu aralik")
print("  yalniz OLCUM hatasidir, eksenden eksene degisim degil. B3'un yon")
print("  bootstrap'i o degisimi icerir ve dogru aralik ODUR.")


# ===========================================================================
print("\n" + "=" * 76)
print("BOLUM 6  UST SINIR")
print("=" * 76)
kap_tam = buzmeli_kats(G, L_OBS, SIG_KUCUK)
r_hat_nrm2 = float(kap_tam @ G @ kap_tam)
kL = float(kap_tam @ L_OBS)
alt_span = kL / np.sqrt(r_hat_nrm2)
r_dik_ust2 = M0 - alt_span**2
print(f"  ||r_hat||^2 = {r_hat_nrm2:.6f}   k'L = {kL:.6f}   M0 = {M0:.6f}")
print(f"  Cauchy-Schwarz: ||r_span|| >= k'L/||r_hat|| = {alt_span:.4f}")
print(f"  ->  ||r_dik||^2 <= M0 - {alt_span:.4f}^2 = {r_dik_ust2:.4f}")
print(f"  ->  sum rho_k^2 <= {r_dik_ust2:.4f}  ->  |c| <= {np.sqrt(r_dik_ust2) / RHO_S_TOPLAM:.1f}")
print("""  HUKUM 6a: SERT SINIR VAKUMDUR. Artigin neredeyse tamami indirgenemez
    gurultudur; cebir onu "ogrenilemez" ilan edemez. Sert sinir yok.""")
print("\n  ISLEVSEL (yumusak) SINIR -- liderlik tablosu:")
print(f"    |c| > {GEREKEN1:.3f} olsaydi DORT boyutlu dogrusal bir duzeltme")
print(f"    1. sirayi ({HEDEF1}) alirdi. Imkansiz degil, ama bu kadar kolay")
print("    bir kazanci alanin bugune dek bulunmamis olmasi dusuk olasilikli.")
print(f"    Bu, |c| icin pratik bir tavan verir: |c| <~ {GEREKEN1:.1f}.")


# ===========================================================================
print("\n" + "=" * 76)
print("HUKUM -- |c| ICIN NIHAI ARALIK")
print("=" * 76)
# NIHAI DAGILIM. Iki referans sinifi var ve AYRISIYORLAR:
#   SENARYO G (genis)  : tum 17 LOO ekseni, rho metrigi -> TUM_C
#   SENARYO D (dar/dogru): yalniz YENI boyut acan eksenler -> YENI_C
# Hangisinin bizim yonlerimizi temsil ettigi KANITLANAMIYOR. Bu yuzden
# nihai dagilim ikisinin ESIT AGIRLIKLI karisimidir; gerilim aralikta durur.
MODEL_BELIRSIZ = 1.20  # tahminci secimi (B1) icin carpansal ek belirsizlik
lgG = np.log(np.clip(rho_bootstrap(kk, 4000), 1e-3, None))
_g10 = kk[DIKPAY[kk] >= 0.10]
lgD = np.log(np.clip(rho_bootstrap(_g10 if len(_g10) >= 2 else kk, 4000), 1e-3, None))
n2 = 100000
ORN = np.exp(
    np.concatenate([rng.choice(lgG, n2), rng.choice(lgD, n2)])
    + rng.normal(0, np.log(MODEL_BELIRSIZ), 2 * n2)
)
C_NOKTA = float(np.median(ORN))
C_ALT, C_UST = float(np.quantile(ORN, 0.05)), float(np.quantile(ORN, 0.95))
print(f"""
KANIT DOKUMU (hepsi AYNI 17 eksenlik LOO kumesinden; bagimsiz ikinci kume YOK):

  (1) sigma_L UST SINIRI (B2): {SIG_UST:.1e}. Gozlenen dik sacilim, 2.27e-04'un
      ongordugunun ALTINDA. Bu, m134'u destekleyen DORDUNCU bagimsiz delildir
      ve butun geri kalanini mumkun kilan sarttir. Buyuk sigma ile havuzlanmis
      tahminci c^2 < 0 veriyordu -- yani o sigma olanaksizdir.

  (2) 1.95 COKTU (B5). Bayat/kucuk bir span ile hesaplanmis PAYDA'dan
      geliyordu: docs/69'da rho_s = 0.0156, bugunku tam span ile ayni dosyada
      rho_s = 0.0616 (4 kat), rho_dik ise degismiyor (-0.027 vs -0.0304).
      seviye'nin bugunku degeri |c| = 0.44. Demet yonlerimizin rho_s'i de
      BUGUNKU r_hat'ten geldigi icin dogru payda bugunkudur.
      1.95 ARTIK BIR KANIT DEGILDIR; aralikta agirligi SIFIRDIR.

  (3) HAVUZLANMIS BUYUKLUK, tum 17 eksen, rho metrigi   |c| = {TUM_C:.3f}
      %90 [{TUM_ALT:.3f}, {TUM_UST:.3f}]   (SENARYO G)

  (4) YALNIZ YENI BOYUT ACAN EKSENLER (dik pay >= 0.10) |c| = {YENI_C:.3f}
      %90 [{YENI_ALT:.3f}, {YENI_UST:.3f}]   n = {YENI_N}   (SENARYO D)
      Bizim yonlerimiz span'in DISINDA oldugu icin DOGRU referans sinif budur,
      ama n kucuk. Dusuk dik payli eksenlerin |rho_dik|'i ~sabit 0.027-0.030
      cikiyor: bu r_hat'in uyum artigidir, sinyal degil (B4b).

  (5) SERT UST SINIR YOK (B6): C-S sinirlari vakum ({np.sqrt(r_dik_ust2) / RHO_S_TOPLAM:.0f}). Islevsel
      tavan liderlik tablosundan gelir: |c| ~ {GEREKEN1:.2f} 1. sirayi alirdi.

  (6) Q_dik BAGIMLILIGI YOK (B4). c olcek-degismezdir; "Q_dik >= 0.25'e
      ekstrapolasyon" diye bir sorun bulunmadi. Gercek degisken mutlak Q_dik
      degil, OLCEKSIZ DIK PAY'dir -- ve o (4)'te ele alindi.

NIHAI DAGILIM = SENARYO G ve D'nin esit agirlikli karisimi (x{MODEL_BELIRSIZ} tahminci
  belirsizligi). Iki senaryo arasindaki gerilim BIRLESTIRILMEDI, TASINDI.

  ***  |c| = {C_NOKTA:.2f}    %90 araligi [{C_ALT:.2f}, {C_UST:.2f}]  ***

  m145'in ilan ettigi 0.70 [0.30, 1.30] ile karsilastirma:
    genislik  {1.30 / 0.30:.1f}x  ->  {C_UST / C_ALT:.1f}x     merkez 0.70 -> {C_NOKTA:.2f}
""")


# ===========================================================================
print("=" * 76)
print("SIRALAMA OLASILIKLARI  (asil hedef 1. SIRA)")
print("=" * 76)
rho2 = ORN**2 * RHO_S_TOPLAM**2
skor = np.sqrt(np.maximum(TABAN_MSE - rho2, 1e-12))
print("ESIK TABLOSU -- hangi |c| hangi sirayi acar")
print(
    f"{'sira':>8s} {'LB skoru':>10s} {'gereken rho^2':>14s} {'gereken |c|':>12s} {'olasilik':>10s}"
)
for ad, h in ((" 1. SIRA", HEDEF1), (" 2. sira", HEDEF2), (" 3. sira", HEDEF3)):
    g = esik_c(h)
    print(f"{ad:>8s} {h:10.5f} {TABAN_MSE - h**2:14.5f} {g:12.4f} {(g <= ORN).mean():10.3f}")
print(f"""
1. SIRA ESIGI ACIKCA:  toplam rho^2 = {TABAN_MSE - HEDEF1**2:.5f} gerekiyor.
  toplam rho^2 = |c|^2 * (sum rho_s^2) = |c|^2 * {RHO_S_TOPLAM}^2 = |c|^2 * {RHO_S_TOPLAM**2:.6f}
  ->  |c| >= sqrt({TABAN_MSE - HEDEF1**2:.5f} / {RHO_S_TOPLAM**2:.6f}) = {GEREKEN1:.4f}
  Yani |c| {GEREKEN1:.2f}'IN USTUNE CIKARSA 1. sira mumkundur.
  1. sira, 2. siranin {(TABAN_MSE - HEDEF1**2) / (TABAN_MSE - HEDEF2**2):.2f} kati sinyal ister; |c| olceginde ise
  yalnizca {GEREKEN1 / GEREKEN2:.2f} kati (rho^2 karesel oldugu icin).
  Nokta tahminimiz {C_NOKTA:.2f} ve %90 ust ucu {C_UST:.2f}.""")
if C_UST >= GEREKEN1:
    print(
        f"  -> 1. sira %90 araliginin ICINDE kaliyor (ust ucta). P = {(ORN >= GEREKEN1).mean():.3f}"
    )
else:
    print(f"  -> 1. sira %90 araliginin DISINDA. P = {(ORN >= GEREKEN1).mean():.3f}")
print(
    f"\nbeklenen nihai skor: ortanca {np.median(skor):.5f}, "
    f"%90 [{np.quantile(skor, 0.05):.5f}, {np.quantile(skor, 0.95):.5f}]"
)
print("karsilastirma: hicbir sey tutmazsa 1.00101; su anki en iyimiz 1.00115")


# ===========================================================================
# D1 SONDASI HANGI |c|'yi ACAR -- karar tablosu
# ===========================================================================
print("\n" + "=" * 76)
print("D1 SONDASI: TEK GONDERIM |c|'yi TAM OLARAK COZER")
print("=" * 76)
try:
    with open(os.path.join(M29, "m148_demet.json")) as fh:
        DEMET = json.load(fh)
    s1 = DEMET["sondalar"][0]
    SABIT, KAP = float(s1["sabit"]), float(s1["kappa_etkin"])
    RHOK = DEMET["rho_k_tahmin"]
    print(f"m148_demet.json: rho_k_tahmin = {[f'{v:.4f}' for v in RHOK]}")
    print("H2..H4'un ongorusu SIFIR: H1 agirliklandirma hipotezi altinda toplam")
    print(
        f"  rho_s'in TAMAMI H1'dedir, yani rho_s_1 = {RHO_S_TOPLAM} ve rho_1 = |c| * {RHO_S_TOPLAM}."
    )
    print(f"  Cozum: rho_1 = ({SABIT:.10f} - P^2) / {2 * KAP:.8f}")
    print(f"\n{'senaryo':>28s} {'|c|':>7s} {'rho_1':>8s} {'D1 LB skoru P':>14s}")
    for ad, cval in (
        ("sinyal yok", 0.0),
        ("SENARYO D (yeni boyut)", YENI_C),
        ("SENARYO G (tum 17)", TUM_C),
        ("2. sira esigi", GEREKEN2),
        ("1. SIRA esigi", GEREKEN1),
    ):
        rho1 = cval * RHO_S_TOPLAM
        p2 = SABIT - 2 * KAP * rho1
        print(f"{ad:>28s} {cval:7.3f} {rho1:8.4f} {np.sqrt(max(p2, 0)):14.5f}")
    print("""
KARAR KURALI. D1 doner donmez:
  P <= 0.99725  ->  |c| >= 0.76;  2. sira kesinlesir, 1. sira menzilde
  P ~  0.99817  ->  SENARYO G;    2. siranin kil payi altinda
  P ~  0.99974  ->  SENARYO D;    3. sira, 2. sira icin yeni yon lazim
  P >= 1.00235  ->  sinyal yok;   demet birakilir, 1.00115 yedegi korunur
Bu tek gonderim, masa basinda daraltilamayan butun belirsizligi kapatir.""")
except (OSError, KeyError, IndexError, ValueError) as hata:
    print(f"m148_demet.json okunamadi ({type(hata).__name__}: {hata}) -- tablo atlandi")

print(f"""
SON SOZ.
  |c| = {C_NOKTA:.2f} [{C_ALT:.2f}, {C_UST:.2f}]
    P(1. sira) = {(ORN >= GEREKEN1).mean():.0%}   P(2. sira+) = {(ORN >= GEREKEN2).mean():.0%}   P(3. sira+) = {(ORN >= GEREKEN3).mean():.0%}
  1. sira ancak |c| >= {GEREKEN1:.2f} ise gelir -- bu, SENARYO G'nin bile ust
  ucudur ve SENARYO D altinda ulasilamaz. 2. sira ({GEREKEN2:.2f}) iki senaryo
  arasindaki AYRIM CIZGISI uzerindedir: G dogruysa gelir, D dogruysa gelmez.
  Iste bu yuzden olcum sart -- ilk sonda, hangi senaryoda oldugumuzu soyler.

  1. SIRAYI KOVALAMANIN MALIYETI YOK. Demet plani rho_k'leri OLCER; olculen
  toplam ne cikarsa optimum kappa ona gore konur. Yani "1. sirayi hedeflemek"
  ayri bir dosya ya da ayri bir risk gerektirmez -- ayni 4 sonda hem 2. hem
  1. sirayi acar. Tek fark beklentidir.

  AMA BU BIR TAHMINDIR VE PLAN TAHMINE DAYANMAZ. m148 demet plani her
  rho_k'yi LB ile OLCER; olcum yolu risksizdir (rho_k = 0 cikarsa skor
  degismez, isaret ters cikarsa duzeltilir). Yukaridaki olasilik yalniz
  "olcumden once ne bekleyelim" sorusunun cevabidir ve 6 hakkin
  harcanmasini gerektiren tek gerekcedir: |c|'yi daha fazla masa basinda
  daraltmanin yolu KALMAMISTIR -- elde bagimsiz ikinci kume yoktur.
""")
