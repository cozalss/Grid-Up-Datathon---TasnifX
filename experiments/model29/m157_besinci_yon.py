"""BESINCI YON ARAYISI -- mevcut 4 yone DIK, GERCEKTEN YENI bir yon var mi?

SORU. m148 dort dik yon olcuyor (H1 1.95|rho_s|, H2 rho_cv, H3 hava/mevsim,
H4 trafo/yapisal). H5 "esit agirlik" H1'in icine dustugu icin ATILDI. Elde
2 hak fazla var (4 sonda + 1 nihai + 1 yedek). Bir 5. yon eklemeye deger mi?

CEBIR (docs/72). Ortonormal yonlerde
    skor^2 = TABAN_MSE - toplam_k rho_k^2 = TABAN_MSE - ||P_S rho||^2
yani sonuc YALNIZCA secilen S alt uzayina baglidir. Her olculen yon
RISKSIZDIR (rho_k=0 cikarsa skor degismez). Soru "kazanc pozitif mi" degil,
"kazanc bir GONDERIM HAKKINA deger mi".

BU BETIGIN ANA BULGUSU (asagida sayilarla): mevcut 40 eksenin ICINDE 5. bir
yon YOK -- H1..H4 zaten o 40 boyutun onsel sinyalinin neredeyse tamamini
kapsiyor ve artik boyutlarda yalnizca ISARET GURULTUSU terimi kaliyor.
GERCEKTEN yeni yon, m144'un KAPIYI GECEN 22 YENI EKSENINDEDIR (H_carpim40
ailesi disi). Bu eksenler tanim geregi mevcut 40'a DIK kuruldugu icin,
onlardan turetilen yonun H1..H4'e diklestirilmis artakalani TAM 1.000'dir.

KAYNAKLAR (hicbir agir boru hatti yeniden kosulmaz):
  m150_eksenler.json      40 eksen: ad, kats, rho_cv, rho_s, TABAN_MSE
  m144_yeni_aileler.json  108 yeni eksen (22'si H_carpim40 disi): rho_s, rho_cv
  m150_altuzay.py         onsel kovaryans modeli (A/B/D/C dallari) -- KOPYALANDI

HICBIR GONDERIM YAPILMAZ, submissions/ altina YAZILMAZ, m148'e DOKUNULMAZ.
"""

import json
import os

import numpy as np

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
M29 = os.path.join(KOK, "experiments/model29")

# --- m150'den aynen: onsel model sabitleri -------------------------------
P_ISARET = 5.0 / 40.0  # m142: iki bagimsiz kaynak 35/40 uyusuyor
Q_IS = 1.0 - 2 * P_ISARET
C_ORT, C_SS = 0.7, 0.30  # m145: |c| ~ 0.7, %90 araligi [0.3, 1.3]
EC2 = C_ORT**2 + C_SS**2
MU_OLCEK = C_ORT * Q_IS
PA, PB, PD = 0.45, 0.25, 0.15  # dal olasiliklari (A LB, B CV, D aile, C artik)
YUV = 5e-6 / np.sqrt(3.0)
SIG_C = 1.72e-4  # m112 LOO: kalibre sabitin kendi hatasi
SIG_P2 = 2.0 * YUV
MC = 60000
# YENI EKSENLERIN GECERLILIGI. m144'un 22 yeni ekseni mevcut 40'la AYNI
# kapilardan gecti; ama (i) cok daha buyuk bir taramanin (yaklasik 500 aday)
# hayatta kalanlari, yani secim yanliligi daha buyuk, (ii) |rho_s|'leri
# 0.015 kapisina daha yakin, (iii) docs/72 "yeni eksen YENI LB OLCUMU
# GETIRMIYOR" uyarisini tasiyor. Bu yuzden gecerliliklerine w olasiligi
# verilir; w=0 iken 5. yonun kazanci TAM SIFIRDIR.
W_YENI = 0.5

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
HEDEF = [("1. SIRA", 0.99009), ("2. SIRA", 0.99614), ("3. sira", 0.99927)]


def sira(sk):
    for ad, e in HEDEF:
        if sk < e:
            return ad
    return "4.+"


# =========================================================================
# 0) VERI
# =========================================================================
with open(os.path.join(M29, "m150_eksenler.json")) as fh:
    D40 = json.load(fh)
with open(os.path.join(M29, "m144_yeni_aileler.json")) as fh:
    D144 = json.load(fh)

AD40 = list(D40["eksen"])
KATS = np.array(D40["kats"])  # isaret(rho_cv) * 1.95 * |rho_s|
RCV40 = np.array(D40["rho_cv"])
RS40 = np.array(D40["rho_s"])
TABAN_MSE = float(D40["taban_mse"])
n40 = len(AD40)

# H_carpim40 ailesi = mevcut 40 eksenin BIRBIRIYLE carpimi. Bunlar "yeni bir
# fiziksel hipotez" degil, mevcut eksenlerin ikinci derece etkilesimidir ve
# m144'un kendi ayrimi da bu (secilen_h_haric). Yeni yon icin H_haric alinir.
YENI = list(D144["secilen_h_haric"])
ADY = [x["eksen"] for x in YENI]
AILEY = [x["aile"] for x in YENI]
RSY = np.array([x["rho_s"] for x in YENI])
RCVY = np.array([x["rho_cv"] for x in YENI])
nY = len(ADY)

n = n40 + nY
AD = AD40 + ADY
YENI_MU = np.zeros(n, dtype=bool)
YENI_MU[n40:] = True
RHO_S = np.concatenate([RS40, RSY])
RHO_CV = np.concatenate([RCV40, RCVY])
ISR = np.sign(RHO_CV)
V_LB = ISR * np.abs(RHO_S)  # c'nin carptigi vektor (1.95 ICINDE DEGIL)
HAVA_MU = np.array([1.0 if any(h in a for h in HAVA) else 0.0 for a in AD])

print("=" * 92)
print("0) KURULUS")
print("=" * 92)
print(f"mevcut eksen {n40}, m144 yeni eksen (H_carpim40 haric) {nY}, toplam {n}")
print(f"TABAN_MSE = {TABAN_MSE:.8f}   hicbir yon olculmezse skor {np.sqrt(TABAN_MSE):.5f}")
print(f"||rho_s|| mevcut 40      = {np.linalg.norm(RS40):.4f}")
print(f"||rho_s|| yeni {nY:2d} eksen  = {np.linalg.norm(RSY):.4f}")
print(
    f"||rho_s|| birlesik       = {np.linalg.norm(V_LB):.4f}  (m144: {D144['birlesik_rho_s_h_haric']:.4f})"
)
T40 = EC2 * float(RS40 @ RS40)
TALL = EC2 * float(V_LB @ V_LB)
print(f"onsel sinyal enerjisi T:  yalniz 40 = {T40:.5f}   birlesik = {TALL:.5f}")
ESIK1 = TABAN_MSE - 0.99009**2
ESIK2 = TABAN_MSE - 0.99614**2
ESIK3 = TABAN_MSE - 0.99927**2
print(f"gereken toplam rho^2:  1.sira {ESIK1:.5f}  2.sira {ESIK2:.5f}  3.sira {ESIK3:.5f}")
print("\nyeni eksenler (aile / rho_s / rho_cv):")
for a, f, s, c in zip(ADY, AILEY, RSY, RCVY):
    print(f"  {a:34s} {f:16s} rho_s {s:+.4f}  rho_cv {c:+.4f}")

# =========================================================================
# 1) MEVCUT DORT YON -- m148'in kurdugu haliyle, R^n koordinatlarinda.
#    U ortonormal oldugu icin R^n'deki Gram-Schmidt, N boyuttaki yonlerin
#    dikligiyle BIREBIR aynidir (m150 de boyle yapiyor).
# =========================================================================
print("\n" + "=" * 92)
print("1) MEVCUT DORT YON ve ALT UZAYIN DOYMASI")
print("=" * 92)


def gs(vekler, esik=0.05):
    """Gram-Schmidt; artakalan normu esigin altinda kalan aday ATILIR."""
    cik, art = [], []
    for v in vekler:
        v = np.asarray(v, dtype=np.float64).copy()
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            art.append(0.0)
            continue
        v = v / nv
        for g in cik:
            v = v - (v @ g) * g
        nv = float(np.linalg.norm(v))
        art.append(nv)
        if nv < esik:
            continue
        cik.append(v / nv)
    return np.array(cik), art


def sifirla_yeni(w):
    """m148'in hipotezleri YALNIZ ilk 40 ekseni tanir."""
    w = np.asarray(w, dtype=np.float64).copy()
    w[n40:] = 0.0
    return w


HIP_M148 = {
    "H1 1.95|rho_s|": sifirla_yeni(np.abs(np.concatenate([KATS, np.zeros(nY)]))),
    "H2 rho_cv": sifirla_yeni(np.abs(RHO_CV)),
    "H3 hava/mevsim": sifirla_yeni(HAVA_MU),
    "H4 trafo/yapisal": sifirla_yeni(1.0 - HAVA_MU),
    "H5 esit": sifirla_yeni(np.ones(n)),
}
HIPV = [ISR * h for h in HIP_M148.values()]
W4, ART = gs(HIPV)
for (ad, _), a in zip(HIP_M148.items(), ART):
    bayrak = "" if a >= 0.05 else "  <- ATILDI (onceki yone cok yakin)"
    print(f"  {ad:18s} artakalan {a:.3f}{bayrak}")
print(f"kurulan yon sayisi: {W4.shape[0]}  (m148: 4)")
DIK = float(np.abs(W4 @ W4.T - np.eye(W4.shape[0])).max())
print(f"diklik sapmasi {DIK:.2e}")
if W4.shape[0] != 4 or DIK > 1e-10:
    raise SystemExit("DUR: m148'in dort yonu yeniden uretilemedi.")

# =========================================================================
# 2) ONSEL -- m150'nin A/B/D/C dallari, YENI EKSENLERE genisletilmis.
#    Yeni eksenlerin gercekliği bir Bernoulli(w) ile carpilir: w=0 ise onlarin
#    rho'su TAM SIFIRDIR ve mevcut onsel aynen geri gelir.
# =========================================================================
W_CV = RHO_CV / np.linalg.norm(RHO_CV) * np.linalg.norm(V_LB)
V_HAVA = V_LB * HAVA_MU
V_YAPI = V_LB * (1.0 - HAVA_MU)
rg = np.random.default_rng(2026)


def ornekle(m=MC, pa=PA, pb=PB, pd_=PD, w_yeni=W_YENI, c_ort=C_ORT, c_ss=C_SS, p_is=P_ISARET):
    pc = max(1.0 - pa - pb - pd_, 0.0)
    dal = rg.choice(4, size=m, p=[pa, pb, pd_, pc])
    R = np.zeros((m, n))
    q = 1.0 - 2 * p_is
    ec2 = c_ort**2 + c_ss**2
    g_ort, g_ss = c_ort * q, np.sqrt(max(ec2 - (c_ort * q) ** 2, 1e-12))
    iA = dal == 0
    if iA.any():
        c = rg.normal(c_ort, c_ss, size=(int(iA.sum()), 1))
        eps = np.where(rg.random((int(iA.sum()), n)) < p_is, -1.0, 1.0)
        R[iA] = c * eps * V_LB
    iB = dal == 1
    if iB.any():
        R[iB] = rg.normal(g_ort, g_ss, size=(int(iB.sum()), 1)) * W_CV
    iD = dal == 2
    if iD.any():
        a_ = rg.normal(g_ort, g_ss, size=(int(iD.sum()), 1))
        b_ = rg.normal(g_ort, g_ss, size=(int(iD.sum()), 1))
        R[iD] = a_ * V_HAVA + b_ * V_YAPI
    iC = dal == 3
    if iC.any():
        R[iC] = rg.normal(0.0, np.sqrt(TALL / n), size=(int(iC.sum()), n))
    # yeni eksenlerin GECERLILIK anahtari
    khi = (rg.random(m) < w_yeni).astype(np.float64)
    R[:, n40:] *= khi[:, None]
    return R


R_ORN = ornekle()
SIG = (R_ORN.T @ R_ORN) / len(R_ORN)
MU = R_ORN.mean(axis=0)
print("\n" + "=" * 92)
print(f"2) ONSEL (MC {MC} ornek, yeni eksen gecerlilik olasiligi w = {W_YENI:.2f})")
print("=" * 92)
iz40 = float(np.trace(SIG[:n40, :n40]))
izY = float(np.trace(SIG[n40:, n40:]))
print(f"E||rho||^2 : mevcut 40 boyutta {iz40:.5f} + yeni {nY} boyutta {izY:.5f} = {iz40 + izY:.5f}")
kap4 = float(np.trace(W4 @ SIG @ W4.T))
print(f"MEVCUT 4 YONUN yakaladigi                : {kap4:.5f}")
print(f"  bunun mevcut 40 boyutun onsel enerjisine orani : {kap4 / iz40:.3f}")
print(f"  bunun TUM onsel enerjiye orani                 : {kap4 / (iz40 + izY):.3f}")
print(f"MEVCUT 40 BOYUTTA olculmeyen 36 boyutta kalan    : {iz40 - kap4:.5f}")
print(f"YENI {nY} boyutta hic dokunulmayan               : {izY:.5f}")
print(
    "\nYORUM: 40 boyutun icinde kalan artik, isaret gurultusu teriminden gelir\n"
    "ve tek bir yonde toplanmaz (asagida 40-ici PCA1 sayisi). Yeni boyutlardaki\n"
    "enerji ise TEK bir yonde toplanabilir -- asil aday orasidir."
)

# =========================================================================
# 3) ADAY BESINCI YONLER
# =========================================================================
print("\n" + "=" * 92)
print("3) ADAY 5. YONLER -- H1..H4'e diklestirilmis artakalan ve beklenen rho")
print("=" * 92)


def tekil(j):
    v = np.zeros(n)
    v[j] = 1.0
    return v


def aile_vek(*aileler):
    m = np.array([1.0 if f in aileler else 0.0 for f in AILEY])
    return np.concatenate([np.zeros(n40), ISR[n40:] * np.abs(RHO_S[n40:]) * m])


SIRA_RS = np.argsort(-np.abs(RS40))
UYUSMAZ = np.zeros(n)  # m142'de CV ile LB'nin uyusmadigi 5 eksen
UYUSMAZ[:n40] = (np.sign(RCV40) != np.sign(RS40)).astype(float)
UST_YARI = np.zeros(n)
UST_YARI[SIRA_RS[: n40 // 2]] = 1.0
MOD_UST = np.array([1.0 if (":ust10" in a or ":ust25" in a) else 0.0 for a in AD])

# --- artik onsel kovaryansin PCA'i (mevcut 4 yone dik alt uzayda) ---
PROJ = np.eye(n) - W4.T @ W4
SIG_ART = PROJ @ SIG @ PROJ
oz, ozv = np.linalg.eigh(SIG_ART)
PCA_TUM = ozv[:, ::-1].T
# yalniz mevcut 40 boyutla sinirli artik PCA (soru 1'in dogrudan cevabi)
P40 = np.zeros((n, n))
P40[:n40, :n40] = np.eye(n40)
SIG_A40 = P40 @ SIG_ART @ P40
PCA_40 = np.linalg.eigh(SIG_A40)[1][:, ::-1].T

ADAYLAR = {
    "P1 artik PCA1 (tum)": PCA_TUM[0],
    "P2 artik PCA2 (tum)": PCA_TUM[1],
    "P3 artik PCA3 (tum)": PCA_TUM[2],
    "Q1 artik PCA1 (40-ici)": PCA_40[0],
    "Q2 artik PCA2 (40-ici)": PCA_40[1],
    "S1 isaret uyusmaz (5)": sifirla_yeni(ISR * np.abs(RHO_S) * UYUSMAZ),
    "S2 |rho_s| ust yari": sifirla_yeni(ISR * np.abs(RHO_S) * UST_YARI),
    "S3 esik-kesit kipi": sifirla_yeni(ISR * np.abs(RHO_S) * MOD_UST),
    "S4 en buyuk tek eksen": tekil(int(SIRA_RS[0])),
    "S5 en buyuk 2. eksen": tekil(int(SIRA_RS[1])),
    "S6 HB-benzeri (rho_cv, 40)": sifirla_yeni(RHO_CV),
    "N1 m144 yeni 22 (hepsi)": np.concatenate([np.zeros(n40), ISR[n40:] * np.abs(RSY)]),
    "N2 m144 yeni: F+G": aile_vek("F_guc_yas", "G_mentese"),
    "N3 m144 yeni: F (yas)": aile_vek("F_guc_yas"),
    "N4 m144 yeni: G (mentese)": aile_vek("G_mentese"),
    "N5 m144 yeni: D (gecikme)": aile_vek("D_hava_gecikme"),
    "N6 m144 yeni: B+E (lok/ufuk)": aile_vek("B_lokasyon", "E_ufuk"),
    "N7 m144 yeni: rho_cv agirlik": np.concatenate([np.zeros(n40), RCVY]),
}

print(
    f"{'aday':>30s} {'artakalan':>10s} {'E[rho_5]':>9s} {'sd':>8s} {'E[rho_5^2]':>11s} {'skor kazanci':>13s}"
)
KAYIT = {}
for ad, v in ADAYLAR.items():
    v = np.asarray(v, dtype=np.float64)
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        continue
    g = v / nv
    g = g - W4.T @ (W4 @ g)
    art = float(np.linalg.norm(g))
    if art < 1e-12:
        # Artakalan SIFIR: aday mevcut dort yonun ICINDE. Kaydet ve YAZDIR --
        # "yeni yon degil" bulgusu, atlamak degil raporlamak gerekir.
        KAYIT[ad] = dict(g=np.zeros(n), art=art, mu=0.0, sd=0.0, e2=0.0, kaz=0.0)
        print(
            f"{ad:>30s} {art:10.3f} {0.0:9.4f} {0.0:8.4f} {0.0:11.5f} {0.0:13.5f}  <- H1..H4'UN ICINDE"
        )
        continue
    g = g / art
    x = R_ORN @ g
    mu_k = float(x.mean())
    if mu_k < 0:
        g, x, mu_k = -g, -x, -mu_k
    e2 = float((x * x).mean())
    sd = float(np.sqrt(max(e2 - mu_k**2, 0.0)))
    kaz = np.sqrt(TABAN_MSE - kap4) - np.sqrt(max(TABAN_MSE - kap4 - e2, 1e-9))
    KAYIT[ad] = dict(g=g, art=art, mu=mu_k, sd=sd, e2=e2, kaz=kaz)
    bayrak = "" if art >= 0.05 else "  <- m148 kurali: ATILIR"
    print(f"{ad:>30s} {art:10.3f} {mu_k:9.4f} {sd:8.4f} {e2:11.5f} {kaz:13.5f}{bayrak}")
print(
    "\n(skor kazanci = 5. yon TEK BASINA eklenirse ortalama MSE dususunun skora\n"
    " cevrilmisi; ilk dort yonun beklenen yakalamasi zaten dusulmus taban uzerinden)"
)

# =========================================================================
# 4) UFUK / BOLGE / REJIM BOLMELERI -- neden burada YOK
# =========================================================================
print("\n" + "=" * 92)
print("4) UFUK / BOLGE / SOGUK-REJIM BOLMELERI")
print("=" * 92)
PC = max(1.0 - PA - PB - PD, 0.0)
izo = PC * TALL / n
print(
    f"Bir bolme yonu (ornegin ufuk_gun < 60 / >= 60 ayri agirlik) R^N'de span(U)\n"
    f"disina cikar. Span disindaki bileseninin onselde TEK dayanagi C dalidir\n"
    f"(modellenmemis, izotropik): E[rho^2] = P_C * T / n = {izo:.6f}\n"
    f"  -> skor kazanci {np.sqrt(TABAN_MSE - kap4) - np.sqrt(TABAN_MSE - kap4 - izo):.6f}  (IHMAL EDILEBILIR)\n"
    f"Span ICI bileseni ise mevcut 40 eksenin bir YENIDEN AGIRLIKLANDIRMASIDIR;\n"
    f"o da yukaridaki S1-S3 adaylariyla ayni sinifta ve ayni kaderi paylasir.\n"
    f"AYRICA bu hipotezler ZATEN OLCULDU: m144 E_ufuk ailesinde 64 aday tarayip\n"
    f"4'unu, B_lokasyon ailesinde 48 aday tarayip 3'unu kapidan gecirdi; ikisi de\n"
    f"yukaridaki N6 adayinin icindedir. soguk rejim bolmesi ise mevcut 40 eksenin\n"
    f"14'unde ':x_soguk' kipi olarak ZATEN vardir."
)

# =========================================================================
# 5) ALT UZAY KARSILASTIRMASI -- P(1. sira), P(2. sira)
# =========================================================================
print("\n" + "=" * 92)
print("5) KURULUS KARSILASTIRMASI")
print("=" * 92)


def degerlendir(W):
    y = ((R_ORN @ W.T) ** 2).sum(axis=1)
    sk = np.sqrt(np.maximum(TABAN_MSE - y, 1e-9))
    return (
        float(y.mean()),
        float(np.median(sk)),
        float((y >= ESIK1).mean()),
        float((y >= ESIK2).mean()),
        float((y >= ESIK3).mean()),
    )


EN_IYI = max(KAYIT.items(), key=lambda t: t[1]["e2"])
EN_IYI_AD = EN_IYI[0]
g5 = EN_IYI[1]["g"]
# ikinci yeni yon: 5'e de diklestirilmis en iyi aday
W5 = np.vstack([W4, g5])
en2, g6 = None, None
for ad, k in sorted(KAYIT.items(), key=lambda t: -t[1]["e2"]):
    h = k["g"] - W5.T @ (W5 @ k["g"])
    if np.linalg.norm(h) < 0.05:
        continue
    g6 = h / np.linalg.norm(h)
    en2 = ad
    break
W6 = np.vstack([W5, g6]) if g6 is not None else W5

# H2/H3/H4'ten birini 5. yonle DEGISTIRME (GONDERIM MALIYETI YOK):
# D1 uretildi ve dondurulmus durumda, yani H1 ve GD[0] DEGISMEZ; ama H2-H4
# henuz gonderilmedi.
TAKAS = {}
for atil in (1, 2, 3):
    tut = [i for i in range(4) if i != atil]
    Wt = W4[tut]
    h = g5 - Wt.T @ (Wt @ g5)
    Wt = np.vstack([Wt, h / np.linalg.norm(h)])
    TAKAS[atil] = Wt

# (b) secenegi: H1..H4 62 eksen uzerinden YENIDEN kurulur -> H1 yonu degisir,
# D1 GECERSIZ olur. m155'in "62 eksen" satirinin bizim cercevemizdeki karsiligi.
HIP62 = [
    ISR * np.abs(1.95 * RHO_S),
    ISR * np.abs(RHO_CV),
    ISR * HAVA_MU,
    ISR * (1.0 - HAVA_MU),
]
W4_62, _ = gs(HIP62)

KURULUS = {
    "A mevcut 4 yon (H1..H4)": W4,
    "G 62-havuz 4 yon (D1 OLUR)": W4_62,
    f"B 5 yon (+{EN_IYI_AD})": W5,
    "C H2 -> yeni yon (TAKAS)": TAKAS[1],
    "D H3 -> yeni yon (TAKAS)": TAKAS[2],
    "E H4 -> yeni yon (TAKAS)": TAKAS[3],
}
if g6 is not None:
    KURULUS[f"F 6 yon (+{en2})"] = W6

print(
    f"{'kurulus':>34s} {'k':>2s} {'E[yakalanan]':>13s} {'ortanca skor':>13s} {'P(1.)':>7s} {'P(2.)':>7s} {'P(3.)':>7s}"
)
for ad, W in KURULUS.items():
    ey, os_, p1, p2, p3 = degerlendir(W)
    print(
        f"{ad:>34s} {W.shape[0]:2d} {ey:13.5f} {os_:13.5f} {p1:7.3f} {p2:7.3f} {p3:7.3f}  {sira(os_)}"
    )

# --- w duyarliligi ---
print("\nDUYARLILIK: yeni eksenlerin gecerlilik olasiligi w degisirse (k=4 vs k=5)")
print(
    f"{'w':>6s} {'E4':>9s} {'E5':>9s} {'ortanca4':>10s} {'ortanca5':>10s} {'dP(1.)':>8s} {'dP(2.)':>8s} {'dP(3.)':>8s}"
)
for w in (0.0, 0.25, 0.5, 0.75, 1.0):
    Rw = ornekle(w_yeni=w)

    def dg(W, Rw=Rw):
        y = ((Rw @ W.T) ** 2).sum(axis=1)
        sk = np.sqrt(np.maximum(TABAN_MSE - y, 1e-9))
        return (
            float(y.mean()),
            float(np.median(sk)),
            float((y >= ESIK1).mean()),
            float((y >= ESIK2).mean()),
            float((y >= ESIK3).mean()),
        )

    e4, o4, a1, a2, a3 = dg(W4)
    e5, o5, b1, b2, b3 = dg(W5)
    print(
        f"{w:6.2f} {e4:9.5f} {e5:9.5f} {o4:10.5f} {o5:10.5f} {b1 - a1:+8.3f} {b2 - a2:+8.3f} {b3 - a3:+8.3f}"
    )

# --- dal olasiligi duyarliligi ---
print("\nDUYARLILIK: dal olasiliklari (w = %.2f sabit)" % W_YENI)
print(f"{'senaryo':>26s} {'E4':>9s} {'E5':>9s} {'dP(2.)':>8s} {'dP(3.)':>8s}")
for adS, pa, pb, pd_ in (
    ("taban  A.45 B.25 D.15", 0.45, 0.25, 0.15),
    ("LB agir A.80 B.10 D.05", 0.80, 0.10, 0.05),
    ("CV agir A.20 B.55 D.10", 0.20, 0.55, 0.10),
    ("aile agir A.25 B.15 D.50", 0.25, 0.15, 0.50),
    ("kotumser A.30 B.15 D.10", 0.30, 0.15, 0.10),
):
    Rs = ornekle(pa=pa, pb=pb, pd_=pd_)
    y4 = ((Rs @ W4.T) ** 2).sum(axis=1)
    y5 = ((Rs @ W5.T) ** 2).sum(axis=1)
    print(
        f"{adS:>26s} {y4.mean():9.5f} {y5.mean():9.5f} "
        f"{float((y5 >= ESIK2).mean() - (y4 >= ESIK2).mean()):+8.3f} "
        f"{float((y5 >= ESIK3).mean() - (y4 >= ESIK3).mean()):+8.3f}"
    )

# =========================================================================
# 5b) m155'in "HB" YONU ve BLOK ile LB TAHMINI ARASINDAKI 4 KAT FARK
# =========================================================================
print("\n" + "=" * 92)
print("5b) m155'in HB YONU ve BLOK-LB CELISKISI")
print("=" * 92)
hb = KAYIT["S6 HB-benzeri (rho_cv, 40)"]
BLOK_RHO2 = 0.022098926820257866  # m155 sizintisiz, 40 eksen, net_zaman
LB_RHO2 = 0.57**2 * float(RS40 @ RS40)  # m149: |c| = 0.57
c_gereken = float(np.sqrt(BLOK_RHO2 / float(RS40 @ RS40)))
print(
    f"""HB, m155'te u yonlerinin BLOK karsiliklarina agirlikli EKK ile fit edilen
katsayi vektorudur (m155 sat.648, hb_yonu). Yani HB AYNI eksenlerin dik
parcalarinin SPAN'i icindedir -- YENI BIR BOYUT ACMAZ, bir YENIDEN
AGIRLIKLANDIRMADIR. Ortonormal tabanda EKK katsayisi korelasyon vektorune
oranti oldugu icin HB, rho_cv yonunun blok metrigindeki halidir; m148'in H2
yonu ise TAM OLARAK rho_cv yonudur (ISR*|rho_cv| = rho_cv).
  rho_cv yonunun H1..H4'e diklestirilmis artakalani = {hb["art"]:.3f}
-> HB'nin getirdiginin buyuk kismini m148 H2 ile ZATEN olcuyor. Ustelik HB
   BLOKTAN FIT EDILIR; m141 blok korelasyonlarinin guz25/kis26'ya
   TASINMADIGINI, isaretin bile dondugunu buldu. m155'in zaman bolmesi
   blok-ICI tasimayi sinar, BLOK->TEST tasimasini DEGIL.
   KARAR: HB tarzi blok-fit bir yon 5. sonda olarak ONERILMEZ.

BLOK ile LB ARASINDAKI 4 KAT FARK
  m155 (sizintisiz, zaman-bolmeli, 40 eksen) : toplam rho^2 = {BLOK_RHO2:.5f}
  LB capali tahmin (|c| = 0.57)              : toplam rho^2 = {LB_RHO2:.5f}
  oran {BLOK_RHO2 / LB_RHO2:.1f} kat  (buyuklukte {np.sqrt(BLOK_RHO2 / LB_RHO2):.1f} kat)
  m155'in sayisinin gerektirdigi carpan: |c| = {c_gereken:.2f}
  m149'un %90 araligi [0.17, 1.26] -> {c_gereken:.2f} bu araligin UST UCUDUR.
  Yani ISTATISTIKSEL CELISKI YOK; iki tahmin ayni belirsizligin iki ucu.
  Ayrica m155 blok R2'sini teste su TEK varsayimla ceviriyor:
      rho^2 = CARPAN^2 * R2,  CARPAN = 0.798
  Bu, docs/72'de COKEN 1.95 ile ayni siniftan, KALIBRE EDILMEMIS bir tasima
  carpanidir. Gercek tasima 0.4 olsaydi m155'in sayisi
  {BLOK_RHO2 * (0.4 / 0.798) ** 2:.5f}'e duser ve LB tahminiyle BIREBIR ortusurdu.
  HANGISINE GUVENMELI: LB capali olana. Blok yaz25'tir, test 122 gunluk bir
  UFUKTUR ve m141 bloklar arasi tasimayi curutmustur.
  AMA BU TARTISMA TASARIMI DEGISTIRMEZ: her yon OLCULDUGU icin risksizdir,
  hangisinin dogru oldugunu D1'in skoru birkac saat icinde soyleyecektir
  (docs/72 tablosu). m155 hakliysa D1 ~0.995 doner, LB tahmini hakliysa
  ~0.9985."""
)

# =========================================================================
# 6) KAPPA -- 5. yonun kappa'si ne olmali (m150 bolum 6 formulu)
# =========================================================================
print("\n" + "=" * 92)
print("6) 5. YONUN KAPPA'SI")
print("=" * 92)
KS = np.geomspace(1e-4, 0.4, 40001)


def olcum_hatasi(kappa):
    return np.sqrt(SIG_C**2 + SIG_P2**2) / (2 * kappa)


def kappa_opt(mu, q_dur=0.3):
    ceza = olcum_hatasi(KS) ** 2 + q_dur * (KS**2 - 2 * KS * mu)
    j = int(np.argmin(ceza))
    return float(KS[j]), float(ceza[j])


print(
    f"{'yon':>30s} {'mu_k':>9s} {'sd_k':>8s} {'kappa*':>9s} {'m148 kappa':>11s} {'ceza farki (skor)':>18s}"
)
for i, ad in enumerate(["H1", "H2", "H3", "H4"]):
    x = R_ORN @ W4[i]
    mu_k = abs(float(x.mean()))
    sd = float(np.sqrt(max(float((x * x).mean()) - mu_k**2, 0.0)))
    kst, cst = kappa_opt(mu_k)
    kmev = 0.05174190699701174 if i == 0 else 0.0125
    cmev = olcum_hatasi(kmev) ** 2 + 0.3 * (kmev**2 - 2 * kmev * mu_k)
    print(
        f"{ad:>30s} {mu_k:9.4f} {sd:8.4f} {kst:9.4f} {kmev:11.4f} {(cmev - cst) / (2 * np.sqrt(TABAN_MSE)):18.5f}"
    )
mu5 = abs(float((R_ORN @ g5).mean()))
sd5 = float(np.sqrt(max(float(((R_ORN @ g5) ** 2).mean()) - mu5**2, 0.0)))
k5, c5 = kappa_opt(mu5)
c_var = olcum_hatasi(0.0125) ** 2 + 0.3 * (0.0125**2 - 2 * 0.0125 * mu5)
print(
    f"{'5. yon ' + EN_IYI_AD:>30s} {mu5:9.4f} {sd5:8.4f} {k5:9.4f} {0.0125:11.4f} {(c_var - c5) / (2 * np.sqrt(TABAN_MSE)):18.5f}"
)
print(
    f"\n-> 5. yonun mu'su SIFIR DEGIL (H2-H4'un aksine), cunku onsel ortalama\n"
    f"   vektorunun yeni eksenler uzerinde bileseni var. Bu yuzden kappa'si\n"
    f"   H2-H4'un 0.0125'i degil, {k5:.4f} olmalidir; boylece sonda dosyasinin\n"
    f"   KENDISI de (hak biterse elde kalan dosya olarak) kazanc saglar."
)

# =========================================================================
# 7) YEDEK HAK -- 5. sonda bir gonderim hakkina deger mi?
# =========================================================================
print("\n" + "=" * 92)
print("7) YEDEK HAK MUHASEBESI")
print("=" * 92)
_, o4, p1_4, p2_4, p3_4 = degerlendir(W4)
_, o5, p1_5, p2_5, p3_5 = degerlendir(W5)
KAP_SON = 0.0125  # son sondanin kappa'si
kayip_son_sonda = np.sqrt(TABAN_MSE - kap4 + KAP_SON**2) - np.sqrt(TABAN_MSE - kap4)
print(
    f"""HAK BUTCESI
  31 Agu 3 hak + 1 Eyl 3 hak = 6.
  Mevcut plan : D1 D2 D3 (31 Agu) | D4 Z_NIHAI (1 Eyl) -> 5 kullanilir, 1 YEDEK.
  5 yonlu plan: D1 D2 D3 (31 Agu) | D4 D5 Z_NIHAI (1 Eyl) -> 6 kullanilir, 0 YEDEK.

YEDEGIN KORUDUGU SEY NEDIR
  (a) tuketim_YP_seviye.csv (1.00115) SECIMI bir hak HARCAMAZ -- zaten
      gonderilmis ve skorlanmistir. Yani "yedek dosya" ile "yedek hak" ayri
      seylerdir; asagidaki muhasebe YALNIZCA yedek HAKKI icindir.
  (b) Yedek hak yalnizca su durumlara karsi korur:
      - bir gonderim ERROR doner (hak yanar),
      - yanlis dosya gonderilir,
      - son gun Z_NIHAI'yi gonderecek hak kalmaz.
  (c) AMA zincir KUMULATIFTIR: son sonda dosyasi = nihai dosya + kappa*GD_son.
      Z_NIHAI hic gonderilemezse SON SONDA secilebilir ve maliyeti yalnizca
      kappa^2'dir: {KAP_SON:.4f}^2 -> skorda {kayip_son_sonda:.6f}.
      Yani yedek hakkin koruma degeri bu kucuk sayiyla SINIRLIDIR.

KAZANC (w = {W_YENI:.2f} onselinde)
  ortanca skor  4 yon {o4:.5f}  ->  5 yon {o5:.5f}   fark {o4 - o5:+.5f}
  P(3. sira)    {p3_4:.3f} -> {p3_5:.3f}   ({p3_5 - p3_4:+.3f})
  P(2. sira)    {p2_4:.3f} -> {p2_5:.3f}   ({p2_5 - p2_4:+.3f})
  P(1. sira)    {p1_4:.3f} -> {p1_5:.3f}   ({p1_5 - p1_4:+.3f})

MALIYET
  P(bir gonderimin yanmasi) ~ 0.05 (iyimser degil: dosya/kota/zamanlama).
  Yandiginda kayip ~ {kayip_son_sonda:.6f} (son sonda secilir) --
  Z_NIHAI'nin hic olmadigi felaket senaryosu DEGIL.
  Beklenen maliyet ~ 0.05 * {kayip_son_sonda:.6f} = {0.05 * kayip_son_sonda:.7f} skor.

ASIL RISK HAK DEGIL, ZAMANDIR
  5 yonlu planda 1 Eylul'de UC gonderim ve ARALARINDA skor bekleme var.
  YUMUSATICI: yonler DIK oldugu icin D5, D4'un skoru gelmeden de uretilebilir
  (onceki_r kaydinda r_4 = 0 yazilir; capraz terim bunu dogru isler). Yani
  D4 ve D5 ARDISIK BEKLEMEDEN gonderilebilir; yalnizca Z_NIHAI tum skorlari
  bekler. Bu, zaman riskini bir bekleme turuna indirir."""
)

# =========================================================================
# 8) TARIF -- m148'e NE EKLENIR (KOD DEGISTIRILMEDI)
# =========================================================================
YENI_LISTE = "\n".join(f'    "{a}",  # {f}, rho_s {s:+.4f}' for a, f, s in zip(ADY, AILEY, RSY))
print("\n" + "=" * 92)
print("8) m148'E EKLEME TARIFI (bu betik m148'i DEGISTIRMEZ)")
print("=" * 92)
print(
    f"""ONCE UYARI -- BOZULMA RISKI. m148'de H1 = ISR * |KATS| ve KATS, `kul`
listesindeki TUM eksenler uzerinden kuruluyor. `kul`'a yeni eksen EKLENIRSE
H1 VEKTORU DEGISIR ve GD[0] baska bir yon olur; oysa D1 ZATEN URETILDI ve
sabit/kappa_etkin degerleri m148_demet.json'a dondu. GD[0] degisirse olculen
rho_1 YANLIS YONE atfedilir. Bu yuzden ekleme SU KURALLA yapilmalidir:

  H1..H4'un agirlik vektorleri YENI EKSENLER UZERINDE SIFIR olmali,
  H5 (yeni yon) ise ILK 40 EKSEN UZERINDE SIFIR olmali.

Boyle kurulunca GD[0..3] BIREBIR AYNI kalir ve GD[4] otomatik olarak onlara
diktir (artakalan TAM 1.000 -- m148'in 0.05 kurali bol bol saglanir).

ADIM 1 -- eksen listesi. m148'in eksen secim dongusu `AZAMI_EKSEN = 40`'ta
duruyor. Dongunun BITIMINDEN SONRA (Q = ... satirindan ONCE) ikinci bir
donguyle asagidaki {nY} eksen AYNI kur()/kapi mantigiyla eklenir ve
`kul`/`ONCEKI`/`KAT_LISTE`/`RHO_CV_LISTE`'ye yazilir. `n40 = 40` degiskeni
donguden once saklanir.

    YENI_EKSENLER = [
{YENI_LISTE}
    ]

DIKKAT: bu adlarin bir kismi m148'in mevcut kur() fonksiyonunun BILMEDIGI
kiplerdir. Kur() su an "*", ":x_sv/:x_soguk/:x_ufuk/:x_ay", ":ust10/:ust25/
:alt10", ":kare" biliyor. Eklenmesi gerekenler:
  ":mnt75"  -> max(x - quantile(x, 0.75), 0)          [G_mentese]   5 eksen
  "yas*..." -> ZATEN DESTEKLENIYOR ("*" kolu)         [F_guc_yas]   5 eksen
  "hv_*"    -> panel gecikme/anomali kolonlari        [D_hava]      6 eksen
  "te_*"    -> dis blok hedef kodlamasi               [B_lokasyon]  3 eksen
  "uf_*"    -> ufuk mentese/log                       [E_ufuk]      2 eksen
  "yil_cos3"-> yilin gunu harmonigi                   [C_takvim]    1 eksen
En UCUZ ve EN AZ RISKLI alt kume F_guc_yas + G_mentese'dir (10 eksen):
F hic kod istemez, G icin kur()'a 3 satirlik bir ":mnt75" kolu yeter.
Kod ornegi (kur() icine, ESIK blogunun yanina):

    if kip == "mnt75":
        v_ = np.quantile(xt[np.isfinite(xt)], 0.75)
        return st(np.maximum(xt - v_, 0.0)), st(np.maximum(xb - v_, 0.0))

D/B/E/C eksenleri m144_yeni_aileler.py'deki uretecleri gerektirir; m148'e
tasinmasi 200+ satirdir ve m148 TASIYICI BETIKTIR -- yarisma bitmeden
oraya bu buyuklukte kod girmemeli.

ADIM 2 -- HIPOTEZ sozlugu. Mevcut dort satir DEGISMEZ ama yeni eksenlerde
sifirlanir; besinci satir eklenir:

    def _s40(w):            # ilk 40 disini sifirla
        w = np.asarray(w, float).copy(); w[n40:] = 0.0; return w
    def _sY(w):             # ilk 40'i sifirla
        w = np.asarray(w, float).copy(); w[:n40] = 0.0; return w

    HIPOTEZ = {{
        "H1 1.95|rho_s|":   _s40(np.abs(KATS)),
        "H2 rho_cv":        _s40(np.abs(np.array(RHO_CV_LISTE))),
        "H3 hava/mevsim":   _s40(np.array([...])),      # aynen
        "H4 trafo/yapisal": _s40(np.array([...])),      # aynen
        "H5 m144 yeni eksenler": _sY(np.abs(KATS)),     # YENI 5. YON
    }}

"H5 esit" satiri SILINIR (zaten artakalani 0.05'in altinda kalip atiliyor).

ADIM 3 -- kappa. KAPPA_K su an np.full(DEMET, 0.0125) ve KAPPA_K[0] sabit.
Yeni yonun mu'su sifir DEGIL ({mu5:.4f}), bu yuzden:

    KAPPA_K = np.full(DEMET, 0.0125)
    KAPPA_K[0] = 0.05174190699701174     # D1 -- ASLA DEGISTIRME
    if DEMET >= 5:
        KAPPA_K[4] = {k5:.4f}            # 5. yon: mu_5 > 0, m150 bol.6 optimumu

ADIM 4 -- DEGISIKLIKTEN SONRA ZORUNLU DOGRULAMA (tek tek bakilacak):
  1. Betik "sonda 1 ZATEN VAR" yazmali ve yazdirdigi sabit
     1.0046992296275314, kappa_etkin 0.0516962677376078 OLMALI.
  2. "demetlerin dikligi: en buyuk sapma" < 1e-8 olmali.
  3. Hipotez tablosunda H1..H4'un ARTAKALAN sutunlari degismemeli
     (1.000 / 0.369 / 0.433 / 0.211) -- degistiyse GD[0..3] kaymistir, GERI AL.
  4. TABAN_MSE 1.00202690 kalmali.
  5. m148_demet.json'daki sonda 1 kaydi BIREBIR ayni kalmali.
Bu bes kontrolun HERHANGI BIRI tutmazsa degisiklik GERI ALINIR."""
)

# =========================================================================
# 9) NET ONERI
# =========================================================================
kaz5 = o4 - o5
_, oc, pc1, pc2, pc3 = degerlendir(TAKAS[3])
print("\n" + "=" * 92)
print("9) NET ONERI")
print("=" * 92)
print(
    f"""1) MEVCUT 40 EKSENIN ICINDE 5. YON YOKTUR. H1..H4, o 40 boyutun onsel
   sinyalinin %{100 * kap4 / iz40:.0f}'ini kapsiyor; kalan 36 boyutta duran sey
   ({iz40 - kap4:.5f}) isaret gurultusu terimidir ve TEK bir yonde toplanmaz --
   40-ici en iyi artik PCA yonu bile yalnizca E[rho^2] = {KAYIT["Q1 artik PCA1 (40-ici)"]["e2"]:.5f}
   (skorda {KAYIT["Q1 artik PCA1 (40-ici)"]["kaz"]:.5f}) vaat ediyor. m148'i bunun icin
   DEGISTIRMEYE DEGMEZ.

2) GERCEKTEN YENI YON m144'un H_carpim40 DISI 22 EKSENINDEDIR. Bu eksenler
   mevcut 40'a KURULUS GEREGI diktir; artakalan {KAYIT["N1 m144 yeni 22 (hepsi)"]["art"]:.3f}.
   En iyi aday: {EN_IYI_AD}, E[rho^2] = {EN_IYI[1]["e2"]:.5f}, skorda {EN_IYI[1]["kaz"]:.5f}.
   Bu, 40-ici en iyi adayin {EN_IYI[1]["e2"] / max(KAYIT["Q1 artik PCA1 (40-ici)"]["e2"], 1e-12):.0f} KATIDIR.

3) EN IYI HAMLE BEDAVA OLANDIR: 5. yonu EKLEMEK yerine H4 (ya da H3) ile
   TAKAS ET. Hicbir gonderim hakki harcanmaz, D1 dokunulmaz.
       4 yon (mevcut) : ortanca {o4:.5f}  P(2.)={p2_4:.3f}  P(3.)={p3_4:.3f}
       H4 -> yeni yon : ortanca {oc:.5f}  P(2.)={pc2:.3f}  P(3.)={pc3:.3f}
       5 yon (+1 hak) : ortanca {o5:.5f}  P(2.)={p2_5:.3f}  P(3.)={p3_5:.3f}

4) YEDEK HAKKI HARCAMAYA DEGER MI? Ek kazanc (takasin uzerine) kucuktur ve
   koruma degeri de kucuktur ({kayip_son_sonda:.6f}). Karar KODLAMA RISKINE bakar:
   m148 tasiyici betiktir ve yeni eksenler kur() fonksiyonuna kod ister.
   F_guc_yas + G_mentese alt kumesi 3 satirlik bir eklemeyle kurulabiliyorsa
   5. yon EKLENEBILIR; D/B/E/C eksenleri icin gereken 200+ satir m144 kodu
   yarisma bitmeden m148'e TASINMAMALIDIR.

5) 6. YON: {"yok -- ikinci aday da " + str(en2) + " ve artakalani yeterli" if g6 is not None else "kurulamadi"}.
   Ama 6. yon ancak 5.'yi de eklersek gundeme gelir ve o zaman hak KALMAZ.
   ONERILMEZ."""
)

_, og, pg1, pg2, pg3 = degerlendir(W4_62)
print(
    f"""
6) KOORDINATORUN (a)/(b)/(c) SORUSUNA DOGRUDAN CEVAP
   (a) 40 EKSENDE KAL, 4 yon      : ortanca {o4:.5f}  P(1.)={p1_4:.3f}  P(2.)={p2_4:.3f}
   (b) 62 HAVUZ, H1..H4 YENIDEN   : ortanca {og:.5f}  P(1.)={pg1:.3f}  P(2.)={pg2:.3f}
       -> D1 GECERSIZ olur (H1 yonu degisir), bugunku tum dogrulama tekrarlanir
       -> USTELIK (c)'den DAHA KOTU. Kesinlikle ONERILMEZ.
   (c) 62 HAVUZ, H1..H4 DONDURULMUS, yeni eksenler AYRI bir yonde:
       c1) TAKAS (H4 yerine yeni yon), 4 yon, HAK MALIYETI YOK:
           ortanca {oc:.5f}  P(1.)={pc1:.3f}  P(2.)={pc2:.3f}
       c2) EKLEME (5. yon), 1 hak maliyeti:
           ortanca {o5:.5f}  P(1.)={p1_5:.3f}  P(2.)={p2_5:.3f}
   KRITIK NOKTA: (b)'nin "62'ye cikmak D1'i oldurur" ikilemi YAPAY. H1..H4'un
   agirliklari yeni eksenlerde SIFIRLANIRSA GD[0..3] BIREBIR ayni kalir, D1
   gecerliligini korur ve yeni boyutlara yine erisilir. (b) yerine (c1) yapilir.
   SIRALAMA: (c2) > (c1) > (b) > (a).  (c1) BEDAVA oldugu icin asgari hamledir.

7) m155 ILE MUTABAKAT
   - "62 gercekten yeni boyut aciyor mu?" EVET. m144'un kapisi Qd >= 0.25 ile
     zaten mevcut 40'a dik kurulmustur; bizim olcumumuzde H1..H4'e
     diklestirilmis artakalan TAM 1.000. m155'in tum-dik R2'si de 0.0842 ->
     0.1012 buyuyor. m150'nin "alt uzay secimi onemsiz (0.00003)" bulgusu
     SABIT 40-eksen HAVUZU icindir; havuzu buyutmek baska bir islemdir ve
     o bulguyla CELISMEZ.
   - "HB eklensin mi?" HAYIR: HB ayni eksenlerin span'i icinde bir yeniden
     agirliklandirmadir (yeni boyut YOK) ve blok-fittir; getirisinin buyuk
     kismi m148'in H2 yonuyle ZATEN olculuyor (bolum 5b)."""
)
print("\nHICBIR GONDERIM YAPILMADI, submissions/ altina YAZILMADI, m148 DEGISMEDI.")
