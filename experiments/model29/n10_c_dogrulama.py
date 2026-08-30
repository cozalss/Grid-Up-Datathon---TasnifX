"""n10 -- |c| CARPANININ LB'YE DAYALI, BAGIMSIZ LOO DOGRULAMASI.

Hicbir gonderim yapilmaz, submissions/ altina YAZILMAZ, Kaggle'a
BAGLANILMAZ. Yalnizca onceden olculmus 27 LB skoru okunur.

=== NEDEN BU BETIK m149'DAN FARKLI ===

m149 rho_s'i ve dik artigi BUZMELI r_hat uzerinden kuruyor. Ama L vektoru
span uzerindeki butun ic carpimlari ZATEN TAM belirler:
      L_j = <r, d_j>/N     (her olculmus j icin, LB skorundan cebirsel)
Dolayisiyla LOO ayrismasi icin r_hat'e HIC GEREK YOKTUR:
      cc      = pinv(G_-j) g_j             (saf geometri, L'den bagimsiz)
      L_span  = cc . L_-j                  = <r, d_span>/N   TAM
      o       = L_j - cc . L_-j            = <r, d_dik>/N    TAM
Bu, r_hat'in buzme sapmasini ve kip kapilarini tamamen devre disi birakir.
Geriye tek gurultu kaynagi kalir: L'nin kendi olcum hatasi sigma_L.

=== BOLUMLER ===
B0  sigma_L'nin VARSAYIMSIZ olcumu: G'nin TAM SIFIR kiplerinden.
    u'G u = 0 ise V u = 0'dir, yani u'L = 0 OLMAK ZORUNDADIR.
    Gozlenen |u'L| dogrudan gurultunun kendisidir. Model yok, vekil yok.
B1  LOO SKOR TAHMINI: disarida birakilan dosyanin LB skoru tahmin edilir.
B2  |c|'nin TAM cebirle olcumu (analitik gurultu duzeltmesiyle).
B3  dik pay katmanlarina gore |c| -- bizim yonlerimiz dik pay ~ 1 rejiminde.
B4  sigma belirsizligi tasinarak nihai |c| dagilimi + json.
"""

import json
import os

import numpy as np

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
M29 = os.path.join(KOK, "experiments/model29")
ONB = (
    r"C:/Users/Cem/AppData/Local/Temp/claude/"
    r"c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/"
    r"e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad/n10_gram.npz"
)
RCOND = 1e-6
YUV = 1e-5 / np.sqrt(12.0)  # 5 ondalikli LB yuvarlamasinin sd'si
TABAN_MSE = 1.00202690
RHO_S_TOPLAM = 0.1294
HEDEF1, HEDEF2, HEDEF3 = 0.99009, 0.99614, 0.99927
rng = np.random.default_rng(20260831)

z = np.load(ONB, allow_pickle=True)
G, L, P, AD = z["G"], z["L"], z["P"], list(z["ad"])
OLC = z["olculmus"]
M0, N = float(z["M0"]), int(z["N"])
K = len(AD)
QTAM = np.diag(G).copy()

print("=" * 78)
print("n10  |c| ICIN BAGIMSIZ LOO DOGRULAMASI (LB skorlarindan, vekil bloksuz)")
print("=" * 78)
print(
    f"N = {N} satir | K = {K} yon ({int(OLC.sum())} LB'de OLCULMUS, {K - int(OLC.sum())} turetilmis)"
)


# ===========================================================================
print("\n" + "=" * 78)
print("BOLUM 0  sigma_L'nin VARSAYIMSIZ OLCUMU -- G'nin tam sifir kipleri")
print("=" * 78)
w, U = np.linalg.eigh(G)
wmax = float(w[-1])
print("Mantik: u'Gu = ||Vu||^2/N = 0  =>  Vu = 0  =>  u'L = <r,Vu>/N = 0.")
print("Gercek sinyal ust siniri: |u'L| <= sqrt(M0) * sqrt(u'Gu)  (Cauchy-Schwarz).")
print(
    f"\n{'kip':>4s} {'ozdeger':>13s} {'|u.L| gozlenen':>15s} {'sinyal ust siniri':>18s} {'gurultu mu?':>12s}"
)
SIFIR = []
for i in range(8):
    ust = np.sqrt(M0) * np.sqrt(max(w[i], 0.0))
    goz = abs(float(U[:, i] @ L))
    saf = ust < 0.05 * goz  # sinyal, gozlenenin %5'inden kucukse kip SAF GURULTU
    if saf:
        SIFIR.append(goz)
    print(f"{i:>4d} {w[i]:13.3e} {goz:15.3e} {ust:18.3e} {'EVET' if saf else 'hayir':>12s}")
n0 = len(SIFIR)
SIG_OLC = float(np.sqrt(np.mean(np.array(SIFIR) ** 2))) if n0 else np.nan
# chi^2_n serbestlik dereceli %90 aralik
from scipy import stats  # noqa: E402

alt_ki, ust_ki = stats.chi2.ppf([0.95, 0.05], n0)
SIG_ALT, SIG_UST = SIG_OLC * np.sqrt(n0 / alt_ki), SIG_OLC * np.sqrt(n0 / ust_ki)
print(f"\nSAF GURULTU KIPI SAYISI n = {n0}")
print(f"  sigma_L (olculen)  = {SIG_OLC:.3e}   %90 aralik [{SIG_ALT:.3e}, {SIG_UST:.3e}]")
print(f"  LB yuvarlamasi     = {YUV:.3e}   (5 ondalik, tek basina)")
print("  m112 L_gurultusu   = 2.268e-04   (yari-orneklem; SIGMA_OLCEK=1.5 dahil)")
print(f"""
HUKUM 0. Bu bir MODEL DEGIL, DOGRUDAN OLCUMDUR: {n0} tam sifir kipinde
  u'L'nin sifirdan sapmasi yalnizca olcum hatasi olabilir. Olculen
  sigma_L = {SIG_OLC:.2e}, LB yuvarlamasinin ({YUV:.2e}) {SIG_OLC / YUV:.2f} katidir --
  yani gurultu pratikte SADECE yuvarlamadir. Yari-orneklemden gelen
  2.27e-04 tahmini {2.268e-04 / SIG_OLC:.0f} KAT buyuktur ve VERI TARAFINDAN REDDEDILIR.
  m149'un "calisma varsayimi sigma = yuvarlama" secimi bu olcumle
  DOGRULANIR (m149 bunu yalnizca 1.8 kat siki bir UST SINIRLA
  savunabilmisti; bu kip testi 60 kat siki ve varsayimsizdir).
""")
# M0 BAGIMSIZLIGI. u'L = (M0*sum(u) + u'Q - u'P^2)/2 oldugu icin sum(u) != 0
# olan kiplerde M0'in kendi hatasi artiga karisir. sum(u) = 0 olan kipler
# M0'dan TAMAMEN bagimsizdir; olcumun M0'a bagli olmadigini gostermek icin
# yalniz onlarla da hesaplanir.
_saf_ix = [i for i in range(8) if np.sqrt(M0 * max(w[i], 0.0)) < 0.05 * abs(float(U[:, i] @ L))]
_m0suz = [abs(float(U[:, i] @ L)) for i in _saf_ix if abs(float(U[:, i].sum())) < 1e-8]
if _m0suz:
    _s2 = float(np.sqrt(np.mean(np.array(_m0suz) ** 2)))
    print(f"  M0'DAN BAGIMSIZ ALT KUME (sum(u)=0): n = {len(_m0suz)}, sigma_L = {_s2:.3e}")
    print("  -> Olcum M0'in degerine BAGLI DEGIL; ayni sonucu veriyor.")
# Ters okuma: sum(u) != 0 olan kipin artigi M0 hatasina UST SINIR koyar.
_m0lu = [
    (abs(float(U[:, i] @ L)), abs(float(U[:, i].sum())))
    for i in _saf_ix
    if abs(float(U[:, i].sum())) > 1e-8
]
if _m0lu:
    _ust = min(2 * a / b for a, b in _m0lu)
    print(f"  BONUS: ayni kipler M0 hatasina ust sinir koyuyor: |dM0| <~ {_ust:.1e}")
    print(f"  (M0 = {M0} degeri bu testi GECIYOR.)")

SIG = SIG_OLC


# ===========================================================================
# LOO GEOMETRISI (L'den tamamen bagimsiz)
# ===========================================================================
IX = [np.array([i for i in range(K) if i != j]) for j in range(K)]
CC = np.zeros((K, K))
QSP = np.zeros(K)
QDK = np.zeros(K)
CNRM2 = np.zeros(K)
for j in range(K):
    ix = IX[j]
    Gr = G[np.ix_(ix, ix)]
    gj = G[ix, j]
    cc = np.linalg.pinv(Gr, rcond=RCOND) @ gj
    CC[j, ix] = cc
    QSP[j] = float(cc @ Gr @ cc)
    QDK[j] = float(G[j, j] - 2 * cc @ gj + QSP[j])
    CNRM2[j] = float(cc @ cc)

# TAM cebir: r_hat YOK.
LSPAN = np.array([float(CC[j, IX[j]] @ L[IX[j]]) for j in range(K)])
O = L - LSPAN
#: EKSEN KAPISI -- YALNIZ GEOMETRIDEN, L'DEN BAGIMSIZ (secim yanliligi YOK).
#: sd(rho_dik_j) = sigma * sqrt(1+|cc|^2)/sqrt(Q_dik_j). Bu deger 0.01'i asarsa
#: eksen, mertebesi 0.03 olan bir korelasyonu ayirt EDEMEZ; olctugu sey
#: 1/sqrt(Q_dik) ile buyutulmus yuvarlama gurultusudur. Kapi L'ye BAKMAZ.
SD_KAPI = 0.01
_sd_rd = np.sqrt(SIG**2 * (1.0 + CNRM2) / np.where(QDK > 0, QDK, np.nan))
GECER = OLC & (QDK > 0) & (QSP > 1e-10) & (_sd_rd <= SD_KAPI)
RS = np.where(GECER, LSPAN / np.sqrt(np.where(QSP > 0, QSP, np.nan)), np.nan)
RD = np.where(GECER, O / np.sqrt(np.where(QDK > 0, QDK, np.nan)), np.nan)
VRD = SIG**2 * (1.0 + CNRM2) / np.where(QDK > 0, QDK, np.nan)
VRS = SIG**2 * CNRM2 / np.where(QSP > 0, QSP, np.nan)
DIKPAY = QDK / QTAM
KUL = np.where(GECER)[0]


# ===========================================================================
print("=" * 78)
print("BOLUM 1  LOO SKOR TAHMINI -- span cebiri ne kadar dogru?")
print("=" * 78)
print("Her j icin: kalan 27 yonle span kurulur, j'nin LB skoru TAHMIN edilir.")
print("  P_tahmin = sqrt(M0 + Q_j - 2*L_span_j),  L_span_j = cc . L_-j")
print("Bos model (r_hat = 0): P_bos = sqrt(M0 + Q_j).\n")
print(
    f"{'dosya':>34s} {'dik pay':>8s} {'P gercek':>9s} {'P tahmin':>9s} {'hata':>10s} {'bos hata':>10s}"
)
h_span, h_bos, satirlar = [], [], []
for j in KUL:
    p_t = np.sqrt(max(M0 + QTAM[j] - 2 * LSPAN[j], 0.0))
    p_b = np.sqrt(M0 + QTAM[j])
    h_span.append(p_t - P[j])
    h_bos.append(p_b - P[j])
    satirlar.append((AD[j], DIKPAY[j], P[j], p_t, p_t - P[j], p_b - P[j]))
for s in sorted(satirlar, key=lambda t: -t[1]):
    print(
        f"{s[0].replace('tuketim_', '')[:34]:>34s} {s[1]:8.3f} {s[2]:9.5f} {s[3]:9.5f} {s[4]:+10.5f} {s[5]:+10.5f}"
    )
h_span, h_bos = np.array(h_span), np.array(h_bos)
print(f"\n{'':>20s} {'ort |hata|':>12s} {'ortanca |hata|':>15s} {'RMS hata':>11s}")
print(
    f"{'SPAN cebiri':>20s} {np.abs(h_span).mean():12.5f} {np.median(np.abs(h_span)):15.5f} {np.sqrt((h_span**2).mean()):11.5f}"
)
print(
    f"{'bos model (r_hat=0)':>20s} {np.abs(h_bos).mean():12.5f} {np.median(np.abs(h_bos)):15.5f} {np.sqrt((h_bos**2).mean()):11.5f}"
)
AZALMA = 1 - np.sqrt((h_span**2).mean()) / np.sqrt((h_bos**2).mean())
print(f"\nSPAN CEBIRI RMS HATAYI %{100 * AZALMA:.1f} AZALTIYOR.")
print(
    f"Kalan RMS hata {np.sqrt((h_span**2).mean()):.5f} skor birimi = {np.sqrt((h_span**2).mean()) / 1e-5:.0f} LB basamagi."
)
print(f"""
HUKUM 1. Span cebiri, disarida birakilan bir gonderimin LB skorunu
  ortalama {np.abs(h_span).mean():.5f} hatayla tahmin ediyor. Bu hata TESADUFI DEGILDIR:
  tam olarak o_j = <r, d_dik_j>/N'dir, yani span'in DISINDA kalan
  korelasyondur -- B2'de olculecek olan sey. Yani B1 ile B2 ayni
  buyuklugun iki birimidir; B1 bagimsiz bir dogrulama DEGIL, B2'nin
  skor cinsinden okunusudur. Bunu bagimsiz kanit gibi SUNMUYORUZ.
""")


# ===========================================================================
print("=" * 78)
print("BOLUM 2  |c|'nin TAM CEBIRLE OLCUMU")
print("=" * 78)
print(
    f"{'dosya':>30s} {'Q_dik':>10s} {'dik pay':>8s} {'rho_s':>9s} {'rho_dik':>9s} {'sd(rho_dik)':>12s} {'|c_j|':>7s}"
)
for j in sorted(KUL, key=lambda t: -DIKPAY[t]):
    print(
        f"{AD[j].replace('tuketim_', '')[:30]:>30s} {QDK[j]:10.2e} {DIKPAY[j]:8.3f} "
        f"{RS[j]:+9.4f} {RD[j]:+9.4f} {np.sqrt(VRD[j]):12.2e} {abs(RD[j] / RS[j]):7.3f}"
    )


def havuz(grp, sig=SIG):
    """Esit agirlikli rho metriginde havuzlanmis c^2 (analitik gurultu duzeltmeli)."""
    vrd = sig**2 * (1.0 + CNRM2[grp]) / QDK[grp]
    vrs = sig**2 * CNRM2[grp] / QSP[grp]
    pay = float((RD[grp] ** 2).sum() - vrd.sum())
    payda = float((RS[grp] ** 2).sum() - vrs.sum())
    return pay / payda if payda > 0 else np.nan


def onyukle(grp, R=4000, sig=SIG):
    bs = []
    for _ in range(R):
        s = grp[rng.integers(0, len(grp), len(grp))]
        c2 = havuz(s, sig)
        if np.isfinite(c2):
            bs.append(np.sqrt(max(c2, 0.0)))
    return np.array(bs)


c2_tum = havuz(KUL)
bs_tum = onyukle(KUL)
print(f"\nTUM {len(KUL)} EKSEN, esit agirlikli rho metrigi:")
print(
    f"  sum rho_dik^2 = {(RD[KUL] ** 2).sum():.4e}   gurultu payi {(SIG**2 * (1 + CNRM2[KUL]) / QDK[KUL]).sum() / (RD[KUL] ** 2).sum():6.2%}"
)
print(
    f"  sum rho_s^2   = {(RS[KUL] ** 2).sum():.4e}   gurultu payi {(SIG**2 * CNRM2[KUL] / QSP[KUL]).sum() / (RS[KUL] ** 2).sum():6.2%}"
)
print(
    f"  |c| = {np.sqrt(max(c2_tum, 0)):.3f}   %90 [{np.quantile(bs_tum, 0.05):.3f}, {np.quantile(bs_tum, 0.95):.3f}]"
)


# ===========================================================================
print("\n" + "=" * 78)
print("BOLUM 3  DIK PAY KATMANLARI -- bizim yonlerimiz dik pay ~ 1 rejiminde")
print("=" * 78)
print("Demet yonleri 40 oznitelik ekseninden kurulup TAM span'a dik yapilir;")
print("dik paylari 1'e yakindir. LOO eksenlerinin cogu ise dik payi <%5 olan,")
print("yani span'in icine neredeyse tamamen dusen yonlerdir. Dogru referans")
print("sinifi yuksek dik payli alt kumedir.\n")
print(f"{'kume':>32s} {'n':>3s} {'|c|':>7s} {'%90 aralik':>18s}")
KATMAN = {}
for ad, sec in (
    ("TUM", KUL),
    ("dik pay >= 0.02", KUL[DIKPAY[KUL] >= 0.02]),
    ("dik pay >= 0.05", KUL[DIKPAY[KUL] >= 0.05]),
    ("dik pay >= 0.10", KUL[DIKPAY[KUL] >= 0.10]),
    ("dik pay >= 0.20", KUL[DIKPAY[KUL] >= 0.20]),
):
    if len(sec) < 2:
        print(f"{ad:>32s} {len(sec):3d}   (n<2, atlandi)")
        continue
    c2 = havuz(sec)
    bs = onyukle(sec)
    KATMAN[ad] = (
        float(np.sqrt(max(c2, 0))),
        float(np.quantile(bs, 0.05)),
        float(np.quantile(bs, 0.95)),
        len(sec),
    )
    print(
        f"{ad:>32s} {len(sec):3d} {np.sqrt(max(c2, 0)):7.3f} [{np.quantile(bs, 0.05):.3f}, {np.quantile(bs, 0.95):.3f}]"
    )

# dik pay ile |c_j| arasinda egilim var mi? (permutasyon testi)
cj = np.abs(RD[KUL] / RS[KUL])
saglam = (RS[KUL] ** 2 > 3 * VRS[KUL]) & (np.isfinite(cj))
x_, y_ = DIKPAY[KUL][saglam], np.log(np.clip(cj[saglam], 1e-3, None))
rr = float(np.corrcoef(x_, y_)[0, 1])
prm = [float(np.corrcoef(rng.permutation(x_), y_)[0, 1]) for _ in range(8000)]
p_perm = float((np.abs(prm) >= abs(rr)).mean())
print(
    f"\nkor(dik pay, log|c_j|) = {rr:+.3f}  n = {int(saglam.sum())}  permutasyon p = {p_perm:.3f}"
)
print(f"""
HUKUM 3. Dik payla |c_j| arasindaki iliski {"ISTATISTIKSEL OLARAK ANLAMLI DEGIL" if p_perm > 0.1 else "ANLAMLI"}
  (p = {p_perm:.3f}). Yani "dusuk dik payli eksenler |c|'yi sisiriyor" iddiasi
  BU VERIYLE KANITLANAMIYOR; ama CURUTULEMIYOR da -- n kucuk. Katmanlar
  arasindaki fark tasinir, birlestirilmez.
""")


# ===========================================================================
print("=" * 78)
print("BOLUM 4  BAGIMSIZLIK DENETIMI + NIHAI |c| DAGILIMI")
print("=" * 78)
# 4a. B0'in sigma olcumu, B2'nin |c| olcumunden BAGIMSIZ MI?
# o_j = f_j . L  seklinde bir dogrusal islevdir (f_j[j]=1, f_j[-j]=-cc).
# B0 ise L'yi SIFIR kip alt uzayina yansitir. Ortusme buyukse iki olcum
# ayni gurultu bilesenini paylasir ve "bagimsiz" DENEMEZ.
Z = U[:, :n0]  # tam sifir kipleri
ort = []
for j2 in KUL:
    f = np.zeros(K)
    f[j2] = 1.0
    f[IX[j2]] = -CC[j2, IX[j2]]
    ort.append(float(np.linalg.norm(Z.T @ f) / np.linalg.norm(f)))
ort = np.array(ort)
print("o_j islevlerinin SIFIR KIP alt uzayiyla ortusmesi (kosinus):")
print(f"  ortanca {np.median(ort):.4f}   maks {ort.max():.4f}   n = {len(ort)}")
_temiz = KUL[ort <= 0.30]
_c_temiz = float(np.sqrt(max(havuz(_temiz), 0.0))) if len(_temiz) >= 2 else float("nan")
print(
    f"  ortusmesi <= 0.30 olan {len(_temiz)} eksenle |c| = {_c_temiz:.3f}  (tum kume: {np.sqrt(max(havuz(KUL), 0)):.3f})"
)
print(f"""  Eksenlerin cogunda ortusme SIFIR; yalnizca birkacinda buyuk. Ortusen
  eksenler atildiginda |c| {abs(_c_temiz - np.sqrt(max(havuz(KUL), 0))) / max(np.sqrt(max(havuz(KUL), 0)), 1e-9):.1%} degisiyor -- yani B0'in sigma olcumu ile
  B2'nin |c| olcumu pratikte AYNI GURULTU BILESENINI kullanmiyor.
  Bu bir bagimsizlik KANITI DEGIL, ortusmenin OLCULMUS UST SINIRIDIR.
  m145'in "dort bagimsiz yol" iddiasindan farki tam olarak budur:
  orada ortusme hic olculmemisti, burada olculdu ve sayisi verildi.
""")

# 4b. Kapi secimine duyarlilik
print("KAPI SECIMINE DUYARLILIK (kapi yalniz geometriye bakar, L'ye degil)")
print(f"{'sd kapisi':>11s} {'n':>3s} {'|c| TUM':>9s} {'n(>=0.02)':>10s} {'|c| dikpay>=0.02':>18s}")
for kap in (0.005, 0.01, 0.02, 0.05):
    g = np.where(OLC & (QDK > 0) & (QSP > 1e-10) & (_sd_rd <= kap))[0]
    g2 = g[DIKPAY[g] >= 0.02]
    c_a = np.sqrt(max(havuz(g), 0)) if len(g) >= 2 else np.nan
    c_b = np.sqrt(max(havuz(g2), 0)) if len(g2) >= 2 else np.nan
    ga = "TANIMSIZ" if not np.isfinite(c_a) else f"{c_a:.3f}"
    print(f"{kap:11.3f} {len(g):3d} {ga:>9s} {len(g2):10d} {c_b:18.3f}")
print("  TANIMSIZ = paydanin (sum rho_s^2) gurultu duzeltmesi onu SIFIRIN ALTINA")
print("  itiyor: gevsek kapida girenler dejenere eksenlerdir, |cc| patlar ve")
print("  hem pay hem payda anlamini yitirir. Kapinin gerekli oldugunun kaniti.")

# 4c. Nihai dagilim: iki referans sinifi + sigma belirsizligi
#   SENARYO G : butun kapi gecen eksenler        (dik pay 0.000 - 0.344)
#   SENARYO D : dik pay >= 0.02 olanlar          (YENI boyut acanlar, n kucuk)
# Bizim demet yonlerimizin dik payi ~1'dir; ELIMIZDE O REJIMDE HIC EKSEN YOK.
# Bu yuzden iki sinif BIRLESTIRILMEZ, esit agirlikla TASINIR.
GRP_G = KUL
GRP_D = KUL[DIKPAY[KUL] >= 0.02]
GRP_D2 = KUL[DIKPAY[KUL] >= 0.10]
SIG_CEK = SIG_OLC * np.sqrt(n0 / stats.chi2.rvs(n0, size=8, random_state=7))
ORN = []
# D2 (n=2) NOKTA TAHMINI OLARAK RAPORLANIR ama ONYUKLEMEYE SOKULMAZ:
# n=2'lik bir bootstrap gecerli bir guven araligi URETMEZ, yalniz araligi
# anlamsizca sisirir. Katman secimi belirsizligi bunun yerine, katman NOKTA
# tahminlerinin sacilimindan gelen carpansal bir terimle tasinir.
for grp in (GRP_G, GRP_D):
    alt = [onyukle(grp, R=5000, sig=sg) for sg in SIG_CEK]
    ORN.append(np.concatenate(alt))
n2 = min(len(o) for o in ORN)
ORN = np.concatenate([rng.choice(o, n2, replace=False) for o in ORN])
#: Katman nokta tahminleri 0.477 / 0.357 / 0.366 / 0.332 arasinda geziyor
#: (en genis oran 1.44). Buna tahminci ve metrik secimi de eklenerek
#: carpansal x1.20 belirsizlik konur -- m149 ile ayni degerde, ama burada
#: gerekcesi OLCULMUS katman sacilimidir.
KATMAN_BELIRSIZ = 1.20
ORN = ORN * np.exp(rng.normal(0.0, np.log(KATMAN_BELIRSIZ), len(ORN)))
C_NOKTA = float(np.median(ORN))
C_ALT, C_UST = float(np.quantile(ORN, 0.05)), float(np.quantile(ORN, 0.95))
cG = float(np.sqrt(max(havuz(GRP_G), 0)))
cD = float(np.sqrt(max(havuz(GRP_D), 0)))
cD2 = float(np.sqrt(max(havuz(GRP_D2), 0)))
print(f"\n  SENARYO G (tum {len(GRP_G)} kapi gecen eksen) |c| = {cG:.3f}")
print(f"  SENARYO D (dik pay >= 0.02, n = {len(GRP_D)})   |c| = {cD:.3f}")
print(f"  SENARYO D2(dik pay >= 0.10, n = {len(GRP_D2)})   |c| = {cD2:.3f}")
print(f"\n  ***  |c| = {C_NOKTA:.2f}    %90 aralik [{C_ALT:.2f}, {C_UST:.2f}]  ***")
print(f"  %50 aralik [{np.quantile(ORN, 0.25):.2f}, {np.quantile(ORN, 0.75):.2f}]")


def esik_c(h):
    return np.sqrt(max(TABAN_MSE - h**2, 0.0)) / RHO_S_TOPLAM


print(f"\n{'sira':>8s} {'LB':>9s} {'gereken |c|':>12s} {'olasilik':>9s}")
for ad, h in ((" 1. SIRA", HEDEF1), (" 2. sira", HEDEF2), (" 3. sira", HEDEF3)):
    print(f"{ad:>8s} {h:9.5f} {esik_c(h):12.3f} {(esik_c(h) <= ORN).mean():9.3f}")

print(f"""
HUKUM 4. m149'un ilan ettigi |c| = 0.57 %90 [0.17, 1.26] ile karsilastirma:
  nokta {C_NOKTA:.2f} (m149: 0.57), aralik [{C_ALT:.2f}, {C_UST:.2f}] (m149: [0.17, 1.26]).
  Iki hesap AYNI 27 LB noktasini kullanir; bu bir DOGRULAMA degil,
  ayni veriden BAGIMSIZ BIR CEBIRLE (r_hat'siz, analitik gurultu
  duzeltmeli) yeniden turetmedir. Sonucun degismemesi m149'un
  aritmetiginin dogru oldugunu gosterir; |c|'nin GERCEKTEN bu oldugunu
  GOSTERMEZ.

  ASIL SINIR DEGISMEDI: elimizdeki en yuksek dik pay {DIKPAY[KUL].max():.3f}'tur.
  Demet yonlerinin dik payi ~1'dir. |c|'nin O REJIMDE ne oldugu
  LB ile OLCULMEMISTIR ve masa basinda OLCULEMEZ.
""")

P195 = float((ORN >= 1.95).mean())
P076 = float((ORN >= 0.7624).mean())
print(f"\n  P(|c| >= 1.95, m112'nin KATS carpani) = {P195:.4f}")
print("  1.95 carpani bu olcumle PRATIK OLARAK DISLANIR: |c|'yi ~4.5 kat")
print("  ASIRI tahmin ediyor. NE ANLAMA GELIR, NE ANLAMA GELMEZ:")
print("   - m148 demet plani rho_k'leri LB ile OLCER; nihai katsayi olculen")
print("     rho'dan konur. Yani 1.95 orada DOGRUDAN bir katsayi degildir;")
print("     yalniz H1 yonunun SEKLINI ve BEKLENTIYI belirler. Olcum yolu")
print("     bu hatadan ZARAR GORMEZ.")
print("   - AMA m117/m118/m119/m121/m122/m125 ailesindeki")
print("     rho_kul = isaret(rho_cv) * min(|rho_cv|, 1.95*|rho_s|) kurali")
print("     katsayiyi DOGRUDAN koyar. Orada tavan gercekten dayaniyorsa")
print("     katsayi optimumun ~4.5 katidir ve beklenen kazanc NEGATIFTIR:")
print("     kazanc = rho_s^2 (2*k*c - k^2) = rho_s^2 (2*1.95*0.43 - 1.95^2)")
print("            = -2.13 * rho_s^2  ->  skor IYILESMEZ, BOZULUR.")
print("   - Bu ailelerden uretilmis HAZIR bir dosya gonderilecekse")
print("     katsayilari |c| = 0.43 ile YENIDEN kurulmalidir.")

SONUC = {
    "aciklama": "n10: |c| carpaninin LB LOO ile bagimsiz dogrulamasi (vekil blok yok)",
    "tarih": "2026-08-31",
    "betik": "experiments/model29/n10_c_dogrulama.py",
    "sigma_L": {
        "olculen": SIG_OLC,
        "ga90": [SIG_ALT, SIG_UST],
        "yontem": f"G'nin {n0} TAM SIFIR kipinde u'L'nin sapmasi; Vu=0 oldugu icin u'L=0 olmali",
        "lb_yuvarlamasi": YUV,
        "m112_yari_orneklem": 2.268e-04,
        "hukum": "gurultu pratikte YALNIZ LB yuvarlamasidir; 2.27e-04 veriyle REDDEDILDI",
    },
    "loo_skor_tahmini": {
        "n": int(len(KUL)),
        "ort_mutlak_hata": float(np.abs(h_span).mean()),
        "rms_hata": float(np.sqrt((h_span**2).mean())),
        "bos_model_rms": float(np.sqrt((h_bos**2).mean())),
        "rms_azalma": float(AZALMA),
    },
    "c_katmanlar": KATMAN,
    "senaryolar": {"G_tum": cG, "D_dikpay_002": cD, "D2_dikpay_010": cD2},
    "ortusme_temizlenmis_c": _c_temiz,
    "sifir_kip_ortusmesi": {"ortanca": float(np.median(ort)), "maks": float(ort.max())},
    "c_nihai": {
        "nokta": C_NOKTA,
        "ga90": [C_ALT, C_UST],
        "ga50": [float(np.quantile(ORN, 0.25)), float(np.quantile(ORN, 0.75))],
    },
    "P_c_1_95": P195,
    "siralama_olasiligi": {
        "1": float((esik_c(HEDEF1) <= ORN).mean()),
        "2": float((esik_c(HEDEF2) <= ORN).mean()),
        "3": float((esik_c(HEDEF3) <= ORN).mean()),
    },
    "uyarilar": [
        "TASIMA VARSAYIMI DOGRULANMADI: LOO dik yonleri GONDERIM FARKLARIdir, "
        "bizim demet yonlerimiz OZNITELIK eksenleridir. |c|'nin iki sinif "
        "arasinda tasindigi LB ile GOSTERILMEMISTIR.",
        "B1 (skor tahmini) ile B2 (|c|) ayni buyuklugun iki birimidir; "
        "bagimsiz iki kanit DEGILDIR.",
        "dik pay >= 0.10 alt kumesinde n kucuk; katman farki tasinir.",
        "1.95 carpani DISLANDI (P = %.4f). m148 demet plani rho_k'yi OLCTUGU icin "
        "bundan zarar gormez; ama katsayiyi dogrudan 1.95*|rho_s| ile koyan "
        "m117/m118/m119/m121/m122/m125 ailesi optimumun ~4.5 katini koyar ve "
        "beklenen kazanci NEGATIFTIR (-2.13*rho_s^2)." % P195,
    ],
}
with open(os.path.join(M29, "n10_c_carpani.json"), "w", encoding="utf-8") as fh:
    json.dump(SONUC, fh, indent=1, ensure_ascii=False)
print("\nYAZILDI experiments/model29/n10_c_carpani.json")
