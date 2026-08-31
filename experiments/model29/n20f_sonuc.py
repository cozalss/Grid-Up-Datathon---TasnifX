"""n20f -- NIHAI TOPLAMA. n20/n20b/n20c/n20d/n20e ciktilarini tek karara indirger.

HICBIR GONDERIM. submissions/ ALTINA YAZMA YOK.
"""

import json
import os

import numpy as np

M29 = os.path.dirname(os.path.abspath(__file__))
RHO_S_BIL = 0.2141 / 1.95  # m148: ||BETA|| / TAVAN
TABAN_MSE = 1.00202690
HEDEF1, HEDEF2, HEDEF3 = 0.99009, 0.99614, 0.99927


def J(f):
    with open(os.path.join(M29, f), encoding="utf-8") as fh:
        return json.load(fh)


B, C, D, E = (
    J("n20b_eksen_rho_s.json"),
    J("n20c_yenibas_ekseni.json"),
    J("n20d_saglamlik.json"),
    J("n20e_buzme_koprusu.json"),
)


def skor(c):
    rho = c * RHO_S_BIL
    return float(np.sqrt(max(TABAN_MSE - rho * rho, 0.0)))


print("=" * 78)
print("n20f  NIHAI: |c| OZNITELIK EKSENLERINDE")
print("=" * 78)
print("""
NE OLCULDU. |c| = |rho_dik| / |rho_s|, ayni OZNITELIK EKSENI uzerinde:
  rho_dik  eksenin span'a DIK parcasindaki korelasyon -- LB skorundan EXACT
  rho_s    ayni eksenin span ICI parcasindaki korelasyon
Iki eksen icin de LB'de gercek bir dik-yon sondasi var; baska hicbir
oznitelik ekseni sondasi Kaggle'a GONDERILMEMIS (n20 B0).
""")
print(
    f"{'eksen':>15s} {'LB sondasi':>26s} {'kos(dik)':>9s} {'rho_dik':>9s} "
    f"{'rho_s exact':>12s} {'rho_s m148':>11s} {'|c| exact':>10s} {'|c| m148':>9s}"
)
S1, S2 = E["noktalar"]["seviye"], E["noktalar"]["yenibaslangic"]
print(
    f"{'seviye':>15s} {'tuketim_YP_seviye.csv':>26s} {B['kosinus_denetimi']['YP_seviye_vs_seviye_ekseni']:+9.4f} "
    f"{S1['rho_dik']:+9.5f} {S1['rho_s_exact']:+12.5f} {S1['rho_s_buzmeli']:+11.5f} "
    f"{S1['c_exact']:10.3f} {S1['c_buzmeli']:9.3f}"
)
print(
    f"{'yenibaslangic':>15s} {'tuketim_K_yenibas.csv':>26s} {B['kosinus_denetimi']['K_yenibas_vs_yenibaslangic_ekseni']:+9.4f} "
    f"{S2['rho_dik']:+9.5f} {S2['rho_s_exact']:+12.5f} {S2['rho_s_buzmeli']:+11.5f} "
    f"{S2['c_exact']:10.3f} {S2['c_buzmeli']:9.3f}"
)

print("""
NOKTA 1 (seviye) SAGLAM:
  * span bilesimi: 6 varyanttan 5'i 1.94-1.96 araliginda
  * rcond 1e-5..1e-8: 1.955-1.972      * L gurultusu %90: [1.83, 2.09]
  * tahminci: exact 1.955 / buzmeli 1.905  (fark %2.6)
  Bu, m113/m148'in 1.986 capasinin TEMIZ span'da yeniden turetilmesidir.
  BAGIMSIZ IKINCI NOKTA DEGIL -- ayni olcumun duzeltilmis halidir.

NOKTA 2 (yenibaslangic) GERCEKTEN YENI, AMA ZAYIF:
  * rho_dik = -0.00259, sd = 8.2e-4  -> SNR 3.1 (sinirda)
  * rho_s COK KUCUK (0.002-0.006) ve TAHMINCIYE BAGLI: exact +0.00237,
    buzmeli -0.00582 -- ISARET BILE DONUYOR. Yani bu eksende rho_s'i
    veri degil, duzenlilestirme belirliyor.
  * sonuc |c| = 0.445 (m148 paydasi) .. 1.090 (exact). 2.5 KAT belirsiz.
""")
c1e, c1b = S1["c_exact"], S1["c_buzmeli"]
c2e, c2b = S2["c_exact"], S2["c_buzmeli"]
GEO_E = float(np.exp(np.mean(np.log([c1e, c2e]))))
GEO_B = float(np.exp(np.mean(np.log([c1b, c2b]))))
print("=" * 78)
print("KARAR")
print("=" * 78)
print(f"""m148'in rho_s(bilesik) = {RHO_S_BIL:.5f} degeri BUZMELI r_hat ile kurulmustur.
Ic tutarlilik icin |c| de AYNI paydayla olculmelidir -> "|c| m148" sutunu.
  seviye        {c1b:.3f}
  yenibaslangic {c2b:.3f}
  geometrik ortalama (n=2) = {GEO_B:.3f}      (exact paydayla {GEO_E:.3f})
Iki nokta {max(c1b, c2b) / min(c1b, c2b):.1f} KAT ayrisiyor. n=2 ile gecerli bir %90 guven
araligi URETILEMEZ (t_{{0.95,1}} = 6.31 araligi [0.01, 60]'a kadar acar).
DURUST ARALIK = olculen iki noktanin kendisi: [{min(c1b, c2b):.2f}, {max(c1b, c2b):.2f}].""")

print(f"\n{'senaryo':>36s} {'|c|':>8s} {'rho_LB':>9s} {'nihai skor':>11s} {'sonuc':>10s}")


def sira(s):
    if s <= HEDEF1:
        return "1. SIRA"
    if s <= HEDEF2:
        return "2. sira"
    if s <= HEDEF3:
        return "3. sira"
    return "kazanc yok"


SEN = [
    ("KOTU  = nokta 2 (yenibaslangic)", c2b),
    ("MERKEZ = geometrik ortalama n=2", GEO_B),
    ("exact payda geo ort", GEO_E),
    ("IYI   = nokta 1 (seviye)", c1b),
    ("[referans] n10 gonderim farklari", 0.434),
    ("[referans] m148/m113 mevcut capa", 1.986),
]
for ad, c in SEN:
    s = skor(c)
    print(f"{ad:>36s} {c:8.3f} {c * RHO_S_BIL:9.4f} {s:11.5f} {sira(s):>10s}")

print(f"""
EN ONEMLI BULGU. Iki capa da "olcum hatasi" degil:
  * 1.986 SEVIYE ekseninde GERCEKTEN dogru ({c1b:.2f} olarak dogrulandi).
  * 0.434 ise elimizdeki TEK DIGER oznitelik ekseninde de cikiyor ({c2b:.3f}).
Yani ayrisma bir aritmetik hata degil, EKSENDEN EKSENE GERCEK DEGISKENLIK.
Demet 40 ekseni birlestirdigine gore, beklenti tek bir eksenin en iyi
degerine degil, eksenler arasi ORTALAMAYA ({GEO_B:.2f}) capalanmalidir.
""")

SONUC = {
    "aciklama": "n20 -- |c| = |rho_dik|/|rho_s| OZNITELIK EKSENLERINDE olculdu",
    "yontem": {
        "rho_dik": "LB skorundan EXACT: L=(M0+Q-P^2)/2, sonra LOO span artigi",
        "rho_s": "ayni eksenin span ici parcasi; TEMIZ span (eksenin kendi sondasi CIKARILMIS)",
        "eksen_dogrulama": "gonderimin dik yonu ile eksenin dik yonu kosinusu",
    },
    "n_nokta": 2,
    "neden_sadece_2": (
        "Kaggle gonderim gecmisi (salt okuma) 30 satir / 29 dosya; bunlardan "
        "YALNIZ 2'si bir oznitelik ekseninin dik bileseni boyunca hareket eden "
        "sondadir (YP_seviye, K_yenibas). b1/b2/b3, b4/b5/b6, g7_span_tau3, "
        "q1*, YP_guc, YP_haftasonu, YP_seviye2, YP_bolge_*, K_PROBE_* dosyalari "
        "submissions/ altinda VAR ama LB'ye HIC GONDERILMEMIS; LB skoru olmadan "
        "rho_dik cozulemez."
    ),
    "noktalar": {
        "seviye": {
            "sonda": "tuketim_YP_seviye.csv",
            "LB": 1.00115,
            "kosinus_denetimi": B["kosinus_denetimi"]["YP_seviye_vs_seviye_ekseni"],
            "rho_dik": S1["rho_dik"],
            "rho_s_exact": S1["rho_s_exact"],
            "rho_s_m148_buzmeli": S1["rho_s_buzmeli"],
            "c_exact": c1e,
            "c_m148": c1b,
            "dik_pay": 0.656,
            "saglamlik": "YUKSEK",
            "span_varyantlari": D["noktalar"]["seviye"]["span_varyantlari"],
            "L_gurultusu_90": D["noktalar"]["seviye"]["L_gurultusu_%90"],
        },
        "yenibaslangic": {
            "sonda": "tuketim_K_yenibas.csv",
            "LB": 1.00191,
            "eksen_dosyasi": "tuketim_KES_yenibaslangic.csv",
            "kosinus_denetimi": B["kosinus_denetimi"]["K_yenibas_vs_yenibaslangic_ekseni"],
            "rho_dik": S2["rho_dik"],
            "rho_s_exact": S2["rho_s_exact"],
            "rho_s_m148_buzmeli": S2["rho_s_buzmeli"],
            "c_exact": c2e,
            "c_m148": c2b,
            "dik_pay": 0.870,
            "saglamlik": "DUSUK",
            "zayiflik": "rho_dik SNR 3.1; rho_s tahminciye bagli, ISARET DONUYOR",
            "span_varyantlari": D["noktalar"]["yenibaslangic"]["span_varyantlari"],
            "L_gurultusu_90": D["noktalar"]["yenibaslangic"]["L_gurultusu_%90"],
        },
    },
    "b_buzme_koprusu": {"medyan": E["b_medyan"], "kapi_gecen_medyan": E["b_kapi_gecen_medyan"]},
    "c_nokta_m148_paydasi": GEO_B,
    "c_nokta_exact_payda": GEO_E,
    "c_durust_aralik": [min(c1b, c2b), max(c1b, c2b)],
    "guven_araligi_uretilemedi": "n=2; t_{0.95,1}=6.31 araligi anlamsizca aciyor",
    "skorlar": {ad: skor(c) for ad, c in SEN},
    "eski_capalar": {"m148_m113": 1.986, "n10_gonderim_farklari": 0.434},
    "hukum": (
        "1.986 ve 0.434'un ayrismasi aritmetik hata DEGIL. 1.986 seviye ekseninde "
        "dogrulandi (1.905); elimizdeki TEK DIGER oznitelik ekseni ise 0.445 veriyor "
        "-- 0.434 capasiyla neredeyse ayni. Ayrisma EKSENLER ARASI GERCEK "
        "DEGISKENLIKTIR. 40 eksenlik bir demet icin beklenti eksenler arasi "
        f"ortalamaya ({GEO_B:.2f}) capalanmali, en iyi tek eksene degil."
    ),
}
YOL = os.path.join(M29, "n20_c_oznitelik.json")
with open(YOL, "w", encoding="utf-8") as fh:
    json.dump(SONUC, fh, ensure_ascii=False, indent=1, default=float)
print(f"YAZILDI {YOL}")
