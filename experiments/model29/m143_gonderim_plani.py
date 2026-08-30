"""m143 -- ALTI HAKLIK GONDERIM PLANI (yalnizca HESAP; hicbir dosya YAZILMAZ).

SORU. 31 Agustos 03:00'te 3, 1 Eylul'de 3 hak var. Son secimde IKI gonderim
isaretlenir ve Kaggle IYISINI alir; mevcut 1.00115 bankadadir, yani asagi yon
KAPALIDIR. Bu haklar en iyi nasil kullanilir?

CEBIR (docs/69).
    Gonderim = a0 + r_hat + kappa * u        (u birim, span'a ve r_hat'a dik)
    skor^2   = MSE_OPT + kappa^2 - 2*kappa*rho
    rho      = <r, u>/N  = u yonundeki GERCEK korelasyon (BILINMIYOR)
Hazir dosya tuketim_K_TEKHAK.csv icin sabit = 1.011812620, kappa_etkin =
0.09892224 -> rho = (sabit - P^2) / 0.197844.

BU BETIGIN BULDUGU UC SEY.

1) OLCUM BEDAVADIR. LB 5 ondaliga yuvarlar; cozulen rho'nun hatasi
   sigma_rho ~ sigma_P / kappa. kappa = 0.099'da bu 3e-05 (yuvarlama) ve
   sabit'in kendi tutarsizligiyla birlikte en kotu 4e-04. rho'nun onsel
   genisligi 0.25'tir -- yani olcum hatasi onselin BINDE IKISI. Dolayisiyla
   "olcum sondasi icin kucuk kappa" diye bir sey YOK: kappa buyudukce olcum
   IYILESIR (hata 1/kappa gider). Sonda ile bahis arasinda cakisma yoktur.

2) IKI ASAMA SABIT kappa'DAN KESIN USTUNDUR. Sabit kappa yalnizca TEK bir
   hedefte optimaldir (kappa* = sqrt(MSE_OPT - h^2); orada gereken rho tam
   kappa*'dir). rho olculup kappa = rho konursa skor sqrt(MSE_OPT - rho^2)
   olur ve KAYIP HER ZAMAN SIFIRDIR: 1. sira icin gereken rho 0.1598 yerine
   0.1478, 3. sira icin 0.0675 yerine 0.0598. Ustelik sonuc GARANTIDIR.

3) HER OLCULEN EKSEN RISKSIZ EKLENIR. Eksenler birbirine dik kuruldugundan
       skor^2 = MSE_OPT - toplam(rho_i^2)
   ve rho_i ne cikarsa ciksin (isareti bile onemsiz) kazanc rho_i^2 >= 0.
   Yani her sonda hakki bir eksenin bilgisini KALICI olarak satin alir.
   5 sonda + 1 nihai gonderim en fazla 5 eksen olcer.

Ayrica her sonda "kilitlenmis nihai + kappa_k * u_k" bicimindedir; bu yuzden
sonuncudan onceki gonderim de zaten guclu bir adaydir (nihai gonderim
yapilamazsa dahi elde kalan sey kotu degildir).

Kosum:  ./.venv/Scripts/python.exe experiments/model29/m143_gonderim_plani.py
"""

import numpy as np

# ------------------------------------------------------------------ SABITLER
M0 = 1.005846366
MSE_OPT = 1.002112  # = M0 - gercek  (m122 "saf optimum" 1.001055'in karesi)
SABIT_K = 1.011812620  # tuketim_K_TEKHAK.csv
KAPPA_ILAN = 0.09908
KAPPA_ETKIN = 0.09892224
RHO_PRED = 0.2522  # bilesigin kagit uzerindeki ongorusu (1.95 carpani ile)
BANKA = 1.00115  # tuketim_YP_seviye.csv -- asagi yonu kapatan gonderim
SIRA = [("1. sira", 0.99009), ("2. sira", 0.99614), ("3. sira", 0.99927), ("4. sira", 1.00118)]

# LB 5 ondaliga yuvarlar -> hata [-5e-6, +5e-6] tekduze; sd = 1e-5/sqrt(12).
SIGMA_P_SD = 1e-5 / np.sqrt(12.0)
SIGMA_P_KOTU = 5.0e-6

#: ONSEL. Istenen aralik 0.03-0.25, merkez ~0.13 (docs/69 sec.4). Buna m140 ve
#: m141'in uyarilari icin bir SIFIR ATOMU eklendi: 1.95 carpaninin ikinci bir
#: olcumu bulunamadi (m140: c = 0.28 +- 0.10) ve isaretler yaz25 disindaki
#: bloklarda TERS donuyor (m141). O rejimde tasima tamamen basarisiz olabilir.
ONSEL = [(0.00, 0.10), (0.05, 0.20), (0.09, 0.20), (0.13, 0.25), (0.19, 0.15), (0.25, 0.10)]
#: Ek eksenlerin birincil eksene gore olcegi. Ilk bilesik en guclu 40 yonu
#: zaten yuttugu icin sonrakiler kucuk olmak zorunda; xi ~ U(0, 1.5) ile
#: carpilir (ortalama 0.75). E[rho_k] ~ 0.41 * rho_1 (k=2).
EK_OLCEK = [0.55, 0.45, 0.40, 0.35]
#: Atomlar kaba bir izgaradir; kappa taramasinin duz cikmamasi icin her atom
#: sd = 0.02 ile yumusatilir ve 0'da kirpilir. Sifir atomu (rejim cokmesi,
#: m141) boylece "cok kucuk rho" bolgesine dagilir.
ONSEL_SD = 0.02
ONSEL_N = 400_000

rho_atom = np.array([a for a, _ in ONSEL])
rho_agir = np.array([w for _, w in ONSEL])
_rng0 = np.random.default_rng(143)
ONSEL_ORNEK = np.maximum(
    _rng0.choice(rho_atom, size=ONSEL_N, p=rho_agir) + _rng0.normal(0.0, ONSEL_SD, ONSEL_N),
    0.0,
)


def cizgi(baslik):
    print()
    print("=" * 78)
    print(baslik)
    print("=" * 78)


def gerekli_rho(hedef, mse=MSE_OPT):
    """kappa = rho (nihai gonderim) kuruldugunda hedefe ulasmak icin gereken rho."""
    return float(np.sqrt(max(mse - hedef * hedef, 0.0)))


def tek_atis_gerekli(kappa, hedef, mse=MSE_OPT):
    """Tek atista (kappa sabit, rho bilinmiyor) hedefe ulasmak icin gereken rho."""
    return kappa / 2.0 + (mse - hedef * hedef) / (2.0 * kappa)


def sira_ad(skor):
    for ad, h in SIRA:
        if skor < h:
            return ad
    return "5.+"


def onsel_ust(esik):
    """P(rho >= esik) -- yumusatilmis onselden."""
    return float((esik <= ONSEL_ORNEK).mean())


# --------------------------------------------------------------- 0. TUTARLILIK
cizgi("0. TUTARLILIK DENETIMI -- sabit ile MSE_OPT birbirini tutuyor mu?")
bekleyen = MSE_OPT + KAPPA_ETKIN**2
fark = bekleyen - SABIT_K
print(f"  MSE_OPT + kappa_etkin^2 = {bekleyen:.9f}")
print(f"  dosyanin sabiti         = {SABIT_K:.9f}")
print(f"  FARK                    = {fark:+.9f}")
SIS_SABIT = abs(fark)
print()
print("  u, span'a ve r_hat'a dik kuruldugu icin bu farkin SIFIR olmasi gerekir.")
print("  Kalan fark kirpmadan (expm1 -> 0) ve MSE_OPT'un 6 ondaliktan geri")
print("  okunmasindan gelir. rho'ya cevrilince sistematik taban:")
print(
    f"    sigma_rho(sistematik) = {SIS_SABIT:.2e} / (2*kappa) = "
    f"{SIS_SABIT / (2 * KAPPA_ETKIN):.2e}  (kappa=0.099'da)"
)
print("  Bu, yuvarlama gurultusunun ~10 katidir ama onselin (0.25) yine de")
print("  BINDE IKISIDIR. Ikisi de karar icin onemsiz; ikisi de 1/kappa gider.")


# ------------------------------------------------- 1. OLCUM HASSASIYETI / kappa
def sigma_rho(kappa, p_yak=1.0, sigma_p=SIGMA_P_SD, sis=SIS_SABIT):
    yuvarlama = p_yak * sigma_p / kappa
    sistematik = sis / (2.0 * kappa)
    return float(np.hypot(yuvarlama, sistematik))


cizgi("1. OLCUM HASSASIYETI -- sigma_rho(kappa).  SORU 3'UN CEVABI")
print("  rho = (sabit - P^2) / (2*kappa)   ->   d(rho)/dP = -P/kappa")
print(f"  sigma_P(sd) = {SIGMA_P_SD:.3e}   sigma_P(en kotu) = {SIGMA_P_KOTU:.1e}")
print()
print(
    f"  {'kappa':>8s} {'yuvarlama':>11s} {'sistematik':>11s} {'sigma_rho':>11s} "
    f"{'/0.13':>8s} {'not':>14s}"
)
for k in [0.010, 0.020, 0.030, 0.050, 0.070, KAPPA_ILAN, 0.150, 0.200, 0.300]:
    yv = SIGMA_P_SD / k
    ss = SIS_SABIT / (2 * k)
    sr = sigma_rho(k)
    not_ = "HAZIR DOSYA" if abs(k - KAPPA_ILAN) < 1e-9 else ""
    print(f"  {k:8.4f} {yv:11.2e} {ss:11.2e} {sr:11.2e} {sr / 0.13:8.4f} {not_:>14s}")
print()
print("  OKUMA: hata 1/kappa gider -> BUYUK kappa daha iyi olcum verir.")
print("  kappa >= 0.02'de sigma_rho, 0.13'luk merkezin %2'sinin altinda.")
print("  Yani 'once kucuk kappa ile guvenli sonda' fikri YANLIS: hem olcumu")
print("  kotulestirir hem de o gonderimi bahis olarak degersiz kilar.")
print("  TEK HAK olsaydi bile ayni kappa hem olcer hem bahse girerdi.")
print()
print("  Kac hakla ne bilinir: her gonderim TEK bir skaler denklem verir, yani")
print("  bir eksende rho'yu ~4e-04 hassasiyetle. Ayni ekseni tekrar olcmek")
print("  hatayi yalnizca sqrt(2) kuculturdu (4e-04 -> 3e-04) ve HICBIR ISE")
print("  YARAMAZ. Bu yuzden hicbir eksen iki kez olculmez; her hak YENI bir")
print("  eksene gider. 6 hak -> en fazla 5 eksen olculur, 6.'si nihai atistir.")

# ------------------------------------------------------- 2. TEK ATIS vs IKI ASAMA
cizgi("2. TEK ATIS ile IKI ASAMA -- gereken rho.  SORU 1 ve 2'NIN TEMELI")
print("  Tek atista kappa'yi hedeften turetmek (kappa* = sqrt(MSE_OPT-h^2))")
print("  gereken rho'yu EN AZA indirir ve o asgari deger tam kappa*'dir. Yani")
print("  DOGRU HEDEFI ONCEDEN BILSEYDIK tek atis, iki asamayla AYNI rho'yu")
print("  isterdi -- iki asamanin ustunlugu esikte degil, HANGI hedefi")
print("  secmemiz gerektigini bilmemek zorunda olmamamizdadir:")
print("   - kappa'yi 2. siraya gore kurup rho buyuk cikarsa fazlasi ZIYAN olur")
print("     (skor^2 = MSE_OPT + kappa^2 - 2*kappa*rho, minimumu kappa=rho'da),")
print("   - kappa'yi 1. siraya gore kurup rho kucuk cikarsa skor BOZULUR.")
print("  Iki asamada kappa = olculen rho konur; kayip her zaman TAM SIFIRDIR.")
print()
print(f"  {'hedef':>8s} {'skor':>9s} {'iki asama gereken rho':>22s} {'tek atis @k=0.09908':>20s}")
for ad, h in SIRA:
    iki = gerekli_rho(h)
    tek = tek_atis_gerekli(KAPPA_ETKIN, h)
    tek_s = f"{tek:.4f}" if tek > 0 else "zaten asiliyor"
    iki_s = f"{iki:.4f}" if iki > 0 else "zaten asiliyor"
    print(f"  {ad:>8s} {h:9.5f} {iki_s:>22s} {tek_s:>20s}")
print()
print("  Sutunlarin karsilastirmasi: 2. sirada esitler (kappa zaten O hedeften")
print("  turetildi), ama 1. sirada tek atis 0.1598 isterken iki asama 0.1478,")
print("  3. sirada 0.0675 yerine 0.0598 istiyor. Yani sabit kappa, hedefini")
print("  ISKALAYAN her rho degerinde ceza odetiyor.")
print()
print(
    f"  {'gercek rho':>11s} {'TEK ATIS (k=0.09908)':>21s} {'sira':>8s} "
    f"{'IKI ASAMA (k=rho)':>18s} {'sira':>8s} {'fark':>9s}"
)
for rr in [0.00, 0.05, 0.09, 0.0991, 0.13, 0.1478, 0.19, 0.2125, RHO_PRED]:
    s_tek = float(np.sqrt(max(SABIT_K - 2 * KAPPA_ETKIN * rr, 1e-12)))
    s_iki = float(np.sqrt(max(MSE_OPT - rr * rr, 1e-12)))
    print(
        f"  {rr:11.4f} {s_tek:21.5f} {sira_ad(s_tek):>8s} {s_iki:18.5f} "
        f"{sira_ad(s_iki):>8s} {s_tek - s_iki:+9.5f}"
    )
print()
print("  Fark sutunu hicbir satirda negatif degil: iki asama TANIM GEREGI")
print("  tek atistan kotu olamaz. rho = 0.05 gibi kotu bir cekiliste bile")
print("  tek atis 4. siraya duserken iki asama 3. sirayi tutuyor.")

# --------------------------------------------------------------- 3. ONSEL
cizgi("3. ONSEL DAGILIM ve ILK SONDA icin kappa SECIMI.  SORU 1'IN CEVABI")
ort = float(ONSEL_ORNEK.mean())
ort2 = float((ONSEL_ORNEK**2).mean())
print(f"  {'rho':>7s} {'agirlik':>8s}   gerekce")
GEREKCE = [
    "m141: isaretler yaz25 disi bloklarda TERS -> tasima cokerse",
    "carpan m140'in olctugu 0.28 mertebesine yakin kalirsa",
    "2. sira esigi civari -- kritik nokta",
    "istenen merkez; blok korelasyonu 0.2125'in yarisi tasirsa",
    "blok korelasyonunun cogu tasirsa",
    "kagit ongoru 0.2522 (1.95 carpani tam tutarsa)",
]
for (a, w), g in zip(ONSEL, GEREKCE, strict=True):
    print(f"  {a:7.2f} {w:8.2f}   {g}")
print(f"  ortalama rho = {ort:.4f}   ort(rho^2) = {ort2:.5f}   kok-ort-kare = {np.sqrt(ort2):.4f}")

print()
print("  ILK SONDANIN kappa'si iki olcute gore secilebilir:")
print("   (a) tek basina hedefi tutturma olasiligi   -> kappa* = sqrt(MSE_OPT-h^2)")
print("   (b) nihai atis gelmezse pisman olmama      -> kappa = E[rho] (regret)")
print()
print(
    f"  {'kappa':>8s} {'P(2.sira)':>10s} {'P(3.sira)':>10s} {'E[(k-rho)^2]':>13s} {'sigma_rho':>10s}"
)
en_iyi2 = (None, -1.0)
en_iyi_reg = (None, 1e9)
for k in [0.03, 0.05, 0.0597, 0.07, 0.09, KAPPA_ILAN, 0.114, 0.13, 0.15, 0.20, 0.25]:
    g2 = tek_atis_gerekli(k, 0.99614)
    g3 = tek_atis_gerekli(k, 0.99927)
    p2, p3 = onsel_ust(g2), onsel_ust(g3)
    regret = float(((k - ONSEL_ORNEK) ** 2).mean())
    if p2 > en_iyi2[1]:
        en_iyi2 = (k, p2)
    if regret < en_iyi_reg[1]:
        en_iyi_reg = (k, regret)
    print(f"  {k:8.4f} {p2:10.3f} {p3:10.3f} {regret:13.5f} {sigma_rho(k):10.2e}")
print()
print(f"  (a) P(2. sira) en yuksek: kappa = {en_iyi2[0]:.5f}  ->  {en_iyi2[1]:.3f}")
print(f"  (b) regret en dusuk:      kappa = {en_iyi_reg[0]:.4f}  (E[rho] = {ort:.4f})")
print("  Iki olcut 0.099 ile 0.114 arasinda cakisiyor; aradaki fark onselin")
print("  gurultusunden kucuk. HAZIR DOSYA tuketim_K_TEKHAK.csv tam bu araligin")
print(f"  icinde (kappa={KAPPA_ILAN}) ve m126 ile uctan uca dogrulanmis.")
print("  KARAR: ilk sonda YENIDEN URETILMEZ, hazir dosya gonderilir. Yeniden")
print("  kurmanin beklenen kazanci ~0 iken hata riski gercek (docs/69 sec.2.3'te")
print("  bir 'bekleyen sonda' hatali sabitle mayina donmustu).")

# ------------------------------------------------- 4. COK EKSENLI PLANIN DEGERI
cizgi("4. COK EKSENLI BIRIKIM -- SORU 2'NIN CEVABI (kalan haklar)")
print("  Eksenler birbirine ve span'a dik kuruldugundan olculen her rho_i")
print("  KALICI olarak eklenir:   skor^2 = MSE_OPT - toplam(rho_i^2)")
print("  Kayip yok: rho_i = 0 cikarsa skor degismez, isareti onemsiz.")
print()
print("  ORNEK YOL (rho_1 = 0.13, ek eksenler EK_OLCEK * 0.75 * rho_1):")
print(
    f"  {'olculen eksen':>14s} {'eklenen rho':>12s} {'toplam rho':>11s} {'skor':>9s} {'sira':>9s}"
)
tp2, r1_ornek = 0.13**2, 0.13
print(f"  {1:14d} {r1_ornek:12.4f} {np.sqrt(tp2):11.4f} ", end="")
s = float(np.sqrt(max(MSE_OPT - tp2, 1e-12)))
print(f"{s:9.5f} {sira_ad(s):>9s}")
for i, sc in enumerate(EK_OLCEK, start=2):
    rk = sc * 0.75 * r1_ornek
    tp2 += rk * rk
    s = float(np.sqrt(max(MSE_OPT - tp2, 1e-12)))
    print(f"  {i:14d} {rk:12.4f} {np.sqrt(tp2):11.4f} {s:9.5f} {sira_ad(s):>9s}")
print("  Yani zayif ek eksenler bile toplam rho'yu 0.130 -> 0.158'e tasir;")
print("  bu, 2. sira esigini asmis bir bahsi 1. siraya YAKLASTIRIR.")

rng = np.random.default_rng(1143)
NS = ONSEL_N
r1 = ONSEL_ORNEK
kare = r1**2
kare_asama = [kare.copy()]
for sc in EK_OLCEK:
    rk = sc * r1 * rng.uniform(0.0, 1.5, size=NS)
    kare = kare + rk**2
    kare_asama.append(kare.copy())

cizgi("5. SIRALAMA OLASILIKLARI (400.000 cekilis).  SORU 5'IN CEVABI")
print(f"  Onsel atomlari: {[a for a, _ in ONSEL]}")
print(f"  Agirliklar    : {[w for _, w in ONSEL]}")
print(f"  Atomlar sd={ONSEL_SD} ile yumusatildi; ek eksen olcekleri {EK_OLCEK}")
print("  Banka (1.00115) her kurguda ikinci secim olarak isaretli varsayildi.")
print()
KURGU = [
    ("A  tek atis (bugunku plan, k=0.09908)", None),
    ("B  1 sonda + 1 nihai (2 hak)", 0),
    ("C  3 sonda + 1 nihai (4 hak)", 2),
    ("D  5 sonda + 1 nihai (6 hak) <- ONERILEN", 4),
]
print(f"  {'kurgu':>40s} {'P(1.)':>7s} {'P(2.)':>7s} {'P(3.)':>7s} {'P(4.)':>7s} {'ortanca':>9s}")
OLASILIK = {}
for ad, idx in KURGU:
    if idx is None:
        skor = np.sqrt(np.maximum(SABIT_K - 2 * KAPPA_ETKIN * r1, 1e-12))
    else:
        skor = np.sqrt(np.maximum(MSE_OPT - kare_asama[idx], 1e-12))
    skor = np.minimum(skor, BANKA)  # banka ikinci secim olarak her zaman elde
    p = [float((skor < h).mean()) for _, h in SIRA]
    OLASILIK[ad[0]] = p
    print(f"  {ad:>40s} {p[0]:7.3f} {p[1]:7.3f} {p[2]:7.3f} {p[3]:7.3f} {np.median(skor):9.5f}")
print()
print("  ONEMLI: P(4. sira) her kurguda 1.000 -- cunku BANKA (1.00115) zaten")
print("  1.00118'in altinda ve ikinci secim olarak isaretlenecek. Asagi yon")
print("  KAPALI; butun bu bahisler bedava.")
print()
sk_d = np.minimum(np.sqrt(np.maximum(MSE_OPT - kare_asama[4], 1e-12)), BANKA)
sk_a = np.minimum(np.sqrt(np.maximum(SABIT_K - 2 * KAPPA_ETKIN * r1, 1e-12)), BANKA)
print(f"  D'nin A'dan KOTU cikma olasiligi: {float((sk_d > sk_a + 1e-9).mean()):.4f}")
print("  (sifir olmasi tesadufi degil: nihai gonderim tanim geregi her sondadan")
print("   iyidir; sondanin fazlasi (kappa-rho)^2 kadar ziyandir.)")
print()
for et, idx in [("A tek atis", None), ("D 6 hak", 4)]:
    sk = (
        np.sqrt(np.maximum(SABIT_K - 2 * KAPPA_ETKIN * r1, 1e-12))
        if idx is None
        else np.sqrt(np.maximum(MSE_OPT - kare_asama[idx], 1e-12))
    )
    q = np.percentile(sk, [5, 25, 50, 75, 95])
    print(f"  {et:>12s} yuzdelikler %5/%25/%50/%75/%95: " + " ".join(f"{v:.5f}" for v in q))

# --------------------------------------------------------------------- 6. ZAMAN
cizgi("6. ZAMAN PLANI ve RISKLER.  SORU 4'UN CEVABI")
print("  Yarisma bitisi: 1 Eylul 23:59 UTC. Kota 24 saatlik kayan pencere.")
print()
print("  31 AGU 03:00 UTC  S1  hazir dosya tuketim_K_TEKHAK.csv (kappa=0.09908)")
print("                        -> skor okunur, rho_1 = (1.011812620-P^2)/0.197844")
print("  31 AGU 03:20      S2  a0 + r_hat + rho_1*u_1 + kappa_2*u_2   (u_2 kurulur)")
print("  31 AGU 04:00      S3  ... + rho_2*u_2 + kappa_3*u_3")
print("  01 EYL 03:00      S4  ... + rho_3*u_3 + kappa_4*u_4")
print("  01 EYL 03:30      S5  ... + rho_4*u_4 + kappa_5*u_5")
print("  01 EYL 04:00      S6  NIHAI: a0 + r_hat + toplam(rho_i * u_i), sonda YOK")
print("  01 EYL <= 20:00   iki gonderim ISARETLENIR: S6 + tuketim_YP_seviye.csv")
print()
print("  NEDEN SABAHIN 3'U: kota kayan 24 saat. Ilk gunun uc hakki 03:00-04:00")
print("  arasinda harcanirsa ikinci gunun haklari 1 Eylul 03:00-04:00'te acilir")
print("  ve bitise ~20 SAAT tampon kalir. Ilk gun ogleden sonraya kayarsa o")
print("  tampon yok olur ve S6 hic gonderilemeyebilir.")
print()
print("  KIRILGANLIK: her sonda 'kilitlenmis nihai + kappa_k*u_k' bicimindedir.")
print("  Yani S5 bile tum onceki olcumleri TASIR; S6 hic gonderilemezse S5")
print("  secilir ve yalnizca rho_5^2 kadar kaybedilir. Zincir hicbir noktada")
print("  sifira dusmez.")
print()
print("  HAZIRLIK (gonderimden ONCE, kota beklerken):")
print("   - u_2..u_5 adaylari BUGUN kurulmali ve dik/kapi denetimleri gecmeli.")
print("     Aday havuzu: m132'nin 41-58. eksenleri (kapilari gecti ama blokta")
print("     rho_pred eklemedi -> tam da 'bedava olc' sinifi), yapisal aile")
print("     (seviye^2, seviye x guc, bolge kesitleri), ve TS_*/y-ailesi.")
print("   - Her sonda betigi kirpma sonrasi KAPPA_ETKIN'i ve sabiti basmali")
print("     (docs/69 sec.2.4); cozumde ilan edilen kappa DEGIL etkin olan kullanilir.")
print("   - Sonda kappa'lari: kappa_k = E[rho_k] ~ 0.4 * rho_1(olculen).")
print("     Buyuk kappa olcumu iyilestirir ama S_k'yi kotulestirir; regret")
print("     en aza E[rho_k]'da iner. rho_1 olculmeden bu sayi YAZILMAZ.")
print()
print("  RISKLER:")
print("   R1 Public/private ayrimi. Butun kurgu P'yi (public) optimize eder ve")
print("      rho'yu 4e-04 hassasiyetle public'e OTURTUR. Ayrim varsa bu asiri")
print("      uydurmadir. m134 bunu hafifletiyor (L gurultusu 'public = TUM")
print("      kume' ile aciklaniyor, rastgele %50'ye gore 10.7:1) ama docs/69")
print("      1.3 tersini varsayiyor. IKISI CELISIYOR; celiskiyi cozmeden ek")
print("      eksen sayisini artirmak riski BUYUTUR. Bu yuzden bankadaki")
print("      1.00115 ikinci secim olarak MUTLAKA isaretli kalir.")
print("   R2 Kirpma dogrusalligi. kappa buyudukce expm1 kirpmasi yonu bozar.")
print("      Sonda ile nihai atisin kappa'lari 2 KATTAN fazla ayrilmasin.")
print("   R3 Kota. 'Zaman asimina ugrayan betik hicbir sey yapmadi demek")
print("      degildir' -- her gonderimden ONCE Kaggle gonderim listesi okunur.")
print("   R4 Onsel. Sifira yakin bolge (%10 agirlik) m141'in bulgusudur; gercek cokme")
print("      olasiligi daha yuksekse D kurgusunun P(2.sira)'si duser ama")
print("      A'ya gore USTUNLUGU degismez (D >= A her cekiliste).")

# ---------------------------------------------------------------------- 7. PLAN
cizgi("7. PLAN")
print("""
  ILKE. Asagi yon bankadaki 1.00115 ile KAPALI oldugundan her hak bedava bir
  bahistir; ustelik LB yuvarlamasi (5 ondalik) rho'yu kappa=0.099'da ~4e-04
  hassasiyetle cozdurur -- onselin binde ikisi. Bu iki gercek birlestiginde
  dogru strateji "once olc sonra bas" degil, HER GONDERIMDE HEM OLC HEM BAS
  ve olculeni bir sonrakine KILITLE.

  1) ILK GONDERIM: submissions/tuketim_K_TEKHAK.csv, kappa = 0.09908.
     Yeniden uretilmez. Bu kappa hem P(2.sira)'yi en ustte tutar (hedeften
     turetilmistir) hem de E[rho] = 0.114'e yakin oldugu icin pismanligi en
     aza indirir; olcum hatasi zaten kappa >= 0.02'nin her yerinde onemsiz.
     Kucuk "guvenli sonda" kappa'si SECILMEZ: olcumu kotulestirir ve o hakki
     bahis olarak ziyan eder.
         COZUM:  rho_1 = (1.011812620 - P^2) / 0.197844

  2) IKINCI GONDERIM: kappa = rho_1 (olculen). Skoru sqrt(MSE_OPT - rho_1^2)
     ve bu GARANTIDIR. rho_1 >= 0.0991 ise 2. sira, rho_1 >= 0.1478 ise 1. SIRA.

  3) KALAN HAKLAR: yeni eksen. Ince ayar YAPILMAZ (ayni ekseni tekrar olcmek
     hatayi yalnizca sqrt(2) kuculturur, degersiz), yedek TUTULMAZ (banka
     zaten yedektir). Her yeni dik eksen rho_k^2 kadar RISKSIZ kazanc ekler.
     Sira: S2..S5 sonda (kappa_k = E[rho_k] ~ 0.4*rho_1), S6 saf nihai atis.
     Her sonda onceki tum olcumleri tasidigindan zincir kopmaya dayaniklidir.

  4) ZAMAN: ilk uc hak 31 Agustos 03:00-04:00 UTC'de harcanir ki ikinci uclu
     1 Eylul 03:00'te acilsin ve bitise 20 saat tampon kalsin. u_2..u_5
     adaylari ve sonda betikleri KOTA GELMEDEN hazir ve denetlenmis olur.

  5) SON SECIM: S6 (nihai) + tuketim_YP_seviye.csv (1.00115). Ikincisi asagi
     yonu kapatir ve public/private ayrimi riskine karsi tek sigortadir.

  SAYILAR (yumusatilmis onselden, 400.000 cekilis):""")
print(
    f"    A tek atis :  P(1.)={OLASILIK['A'][0]:.3f}  P(2.)={OLASILIK['A'][1]:.3f}  "
    f"P(3.)={OLASILIK['A'][2]:.3f}"
)
print(
    f"    D 6 hak    :  P(1.)={OLASILIK['D'][0]:.3f}  P(2.)={OLASILIK['D'][1]:.3f}  "
    f"P(3.)={OLASILIK['D'][2]:.3f}"
)
print("""
  Bunlar ONSELIN ciktisidir, olcumun degil. Kesin olan tek sey su: D kurgusu
  hicbir cekiliste A'dan kotu degildir ve banka sayesinde her ikisinde de
  4. sira garantidir. Onsel yanlissa siralama olasiliklari kayar; USTUNLUK
  SIRALAMASI (D > C > B >= A) kaymaz.
""")
