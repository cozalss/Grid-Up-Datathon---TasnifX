"""p17 -- 0.02 bandi GERCEK mi ARTEFAKT mi?

Girdi (yalnizca mevcut olculer, model egitimi YOK):
  p_kalici/p12_tasima_ilerleme.json  -> 27 dosyanin dLB'si (ortak taban m6_ikiyon)
  p_kalici/p12e_egim.json            -> vekil-gercek sagmalari, katmanli egimler
  p_kalici/p12e2_tohum.json          -> vekil tohum sacilimi
  experiments/rekor.jsonl            -> GERCEK blok CV + LB ciftleri
  p_kalici/p14_ozet.json             -> huber alpha=0.5 blok kazanclari

Cikti: p_kalici/p17_band.json
"""

import json, io, os, math
import numpy as np

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # experiments/
KOK = os.path.dirname(KOK)  # repo koku
PK = os.path.join(KOK, "experiments", "model29", "p_kalici")


def yukle(ad):
    with io.open(os.path.join(PK, ad), encoding="utf-8") as f:
        return json.load(f)


tasima = yukle("p12_tasima_ilerleme.json")
egim = yukle("p12e_egim.json")
tohum = yukle("p12e2_tohum.json")
ozet = yukle("p14_ozet.json")

CIK = {}

# ---------------------------------------------------------------- 0. veri
dLB = np.array([r["LB_delta"] for r in tasima if r.get("durum") == "gecti"])
assert len(dLB) == 27, len(dLB)
TABAN_LB = round(tasima[1]["LB"] - tasima[1]["LB_delta"], 5)

B_BRUT = egim["1_brut_egim"]["b"]  # 1.4276
A_BRUT = egim["1_brut_egim"]["a"]
GOZ = {
    "R2_buyuk": 0.986,
    "R2_kucuk": 0.036,
    "b_buyuk": egim["1_brut_egim"]["b_buyuk"],
    "b_kucuk": egim["1_brut_egim"]["b_kucuk"],
    "n_buyuk": 8,
    "n_kucuk": 19,
    "R2_brut": egim["1_brut_egim"]["R2"],
}

# ------------------------------------------------- 1. GURULTU TABANI vs BAND
sag = egim["6_vekil_saglamasi_rekor"]
farklar = {r["cift"]: r["fark"] for r in sag}
# UYARI: uc cift bagimsiz DEGIL. (v27->v46) = (v27->v30) + (v30->v46) tam olarak.
d1 = farklar["tuketim_v27_v18hedge.csv->tuketim_v30_buzme.csv"]
d2 = farklar["tuketim_v30_buzme.csv->tuketim_v46_gun.csv"]
d3 = farklar["tuketim_v27_v18hedge.csv->tuketim_v46_gun.csv"]
toplanabilir = abs((d1 + d2) - d3) < 1e-9

bagimsiz = np.array([d1, d2])  # yalnizca 2 bagimsiz gozlem
sigma_cift = float(np.sqrt(np.mean(bagimsiz**2)))  # cift hatasinin RMS'i
sigma_dosya = sigma_cift / math.sqrt(2.0)  # dosya basina vekil hatasi
sd_tohum = egim["A_vekil_gurultusu"]["sd_tipik"]

# p12e'nin ZAYIFLATMA duzeltmesi hangi varyansi kullanmisti?
var_x_cift = egim["C_cift_egimi"]["var_x"]
var_gurultu_kullanilan = egim["C_cift_egimi"]["var_gurultu"]  # tohum gurultusu
var_gurultu_olculen = sigma_cift**2  # OLCULEN vekil hatasi
lam_kullanilan = egim["C_cift_egimi"]["lambda_"]
lam_olculen = (var_x_cift - var_gurultu_olculen) / var_x_cift

CIK["1_gurultu_tabani"] = {
    "vekil_hatasi_uc_cift": farklar,
    "uc_cift_bagimsiz_mi": False,
    "toplanabilirlik_kontrolu_d1_d2_esittir_d3": bool(toplanabilir),
    "bagimsiz_gozlem_sayisi": 2,
    "sigma_cift_RMS": sigma_cift,
    "sigma_dosya_basina": sigma_dosya,
    "tohum_gurultusu_sd": sd_tohum,
    "band_siniri": 0.02,
    "band_siniri_sigma_dosya_cinsinden": 0.02 / sigma_dosya,
    "band_siniri_sigma_cift_cinsinden": 0.02 / sigma_cift,
    "band_siniri_tohum_sd_cinsinden": 0.02 / sd_tohum,
    "ZAYIFLATMA_YENIDEN": {
        "aciklama": (
            "p12e lambda'yi TOHUM gurultusuyle hesapladi (7.60e-06). "
            "Ama vekilin hatasi tohum degil SISTEMATIK ve OLCULDU. "
            "Olculen hata varyansi ile lambda yeniden hesaplanirsa:"
        ),
        "var_x_ciftlerde": var_x_cift,
        "var_gurultu_p12e_kullandi_tohum": var_gurultu_kullanilan,
        "var_gurultu_OLCULEN_vekil": var_gurultu_olculen,
        "lambda_p12e": lam_kullanilan,
        "lambda_OLCULEN_ile": lam_olculen,
        "yorum": (
            "lambda<=0: gozlenen x sacilimi (var 1.09e-04) OLCULEN vekil hata "
            "varyansindan (3.97e-04) KUCUK. Yani n=11 cift regresyonundaki x "
            "degiskenliginin TAMAMI hata olabilir; guvenilirlik SIFIR. "
            "b_cift=-0.161 herhangi bir gercek b ile uyumlu -- BILGI TASIMIYOR."
        ),
    },
}


# ------------------------------------------------------ 2. SIMULASYON
def katmanli(x, y, esik=0.02):
    """Gozlenen analizle ayni: |x|>=esik ve |x|<esik icin kesmeli EKK."""
    out = {}
    for ad, m in (("buyuk", np.abs(x) >= esik), ("kucuk", np.abs(x) < esik)):
        xs, ys = x[m], y[m]
        n = len(xs)
        if n < 3 or np.std(xs) == 0:
            out[ad] = {"n": n, "b": np.nan, "R2": np.nan}
            continue
        b, a = np.polyfit(xs, ys, 1)
        yh = a + b * xs
        ss_t = float(np.sum((ys - ys.mean()) ** 2))
        r2 = 1.0 - float(np.sum((ys - yh) ** 2)) / ss_t if ss_t > 0 else np.nan
        out[ad] = {"n": n, "b": float(b), "R2": float(r2)}
    return out


def sim(
    n_tekrar,
    sigma_vekil,
    sd_lb,
    b_true,
    kural_gercek=False,
    esik_true=0.02,
    b_kucuk_true=0.0,
    rng=None,
):
    """H0: tasima orani BUYUKLUKTEN BAGIMSIZ ve sabit (b_true).
    kural_gercek=True ise ALTERNATIF: |gercek dCV|<esik icin b=b_kucuk_true."""
    rng = rng or np.random.default_rng(7)
    # gercek dCV'ler: gozlenen dLB dagilimindan geri cozulur (tasarim korunur)
    if kural_gercek:
        # kural gercekse buyuk dosyalarin dLB'si b_true'dan, kucuklerinki gurultuden gelir
        gercek = dLB / b_true
    else:
        gercek = dLB / b_true
    res = []
    for _ in range(n_tekrar):
        if kural_gercek:
            bb = np.where(np.abs(gercek) >= esik_true, b_true, b_kucuk_true)
            y = bb * gercek
        else:
            y = b_true * gercek
        y = y + rng.normal(0.0, sd_lb, size=len(y))
        y = np.round(y + 1.00284, 5) - 1.00284  # LB 5 hane yuvarlama
        x = gercek + rng.normal(0.0, sigma_vekil, size=len(gercek))
        x = x + rng.normal(0.0, sd_tohum, size=len(gercek))  # tohum gurultusu
        k = katmanli(x, y)
        # brut (katmansiz) regresyon
        bb_, aa_ = np.polyfit(x, y, 1)
        yh = aa_ + bb_ * x
        r2b = 1.0 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)
        res.append(
            (
                k["buyuk"]["n"],
                k["buyuk"]["b"],
                k["buyuk"]["R2"],
                k["kucuk"]["n"],
                k["kucuk"]["b"],
                k["kucuk"]["R2"],
                float(bb_),
                float(r2b),
            )
        )
    return np.array(res, dtype=float)


def ozetle(r):
    ad = ["n_buyuk", "b_buyuk", "R2_buyuk", "n_kucuk", "b_kucuk", "R2_kucuk", "b_brut", "R2_brut"]
    o = {}
    for i, a in enumerate(ad):
        v = r[:, i]
        v = v[np.isfinite(v)]
        o[a] = {
            "medyan": float(np.median(v)),
            "p05": float(np.percentile(v, 5)),
            "p95": float(np.percentile(v, 95)),
        }
    # gozlenen oruntuyu uretme olasiligi
    R2b, R2k, bk, bb = r[:, 2], r[:, 5], r[:, 4], r[:, 1]
    ok = np.isfinite(R2b) & np.isfinite(R2k)
    o["P_oruntu_R2buyuk>0.9_ve_R2kucuk<0.2"] = float(np.mean((R2b[ok] > 0.9) & (R2k[ok] < 0.2)))
    o["P_R2kucuk<=0.036"] = float(np.mean(R2k[ok] <= 0.036))
    o["P_R2buyuk>=0.986"] = float(np.mean(R2b[ok] >= 0.986))
    o["P_b_kucuk_negatif"] = float(np.mean(bk[ok] < 0))
    o["P_b_buyuk>=2.0"] = float(np.mean(bb[ok] >= 2.0))
    o["P_hepsi_birden"] = float(
        np.mean((R2b[ok] >= 0.9) & (R2k[ok] <= 0.20) & (bk[ok] < 0) & (bb[ok] >= 1.8))
    )
    return o


N = 1000
rng = np.random.default_rng(20260831)
senaryolar = {}
B_TLS = egim["brut"]["TLS"]  # 1.893 -- x-gurultusune gore duzeltilmis
B_TERS = egim["brut"]["ters_ust_sinir"]  # 2.066
# H0 = kural YOK, sabit tasima; vekil gurultusu OLCULEN duzeyde
for etiket, sg, sl, bt in [
    ("H0_b1.43_vekil_gurultusu_OLCULEN(0.0141)", sigma_dosya, 0.002, B_BRUT),
    ("H0_b1.89(TLS)_vekil_gurultusu_OLCULEN", sigma_dosya, 0.002, B_TLS),
    ("H0_b2.07(ters)_vekil_gurultusu_OLCULEN", sigma_dosya, 0.002, B_TERS),
    ("H0_b1.89_vekil_gurultusu_YARI(0.007)", sigma_dosya / 2, 0.002, B_TLS),
    ("H0_b1.89_vekil_gurultusu_1.4x(0.020)", sigma_dosya * 1.4, 0.002, B_TLS),
    ("H0_b1.89_vekil_gurultusu_YOK(KONTROL)", 0.0, 0.002, B_TLS),
    ("H0_b1.89_olculen_gurultu_LBgurultusu_YOK", sigma_dosya, 0.0, B_TLS),
    ("H0_b1.89_olculen_gurultu_LBgurultusu_0.005", sigma_dosya, 0.005, B_TLS),
]:
    senaryolar[etiket] = ozetle(sim(N, sg, sl, bt, rng=rng))

# ALTERNATIF: kural GERCEK (kucuk gercek dCV'ler hic tasinmiyor)
senaryolar["H1_kural_GERCEK_b1.89_vekil_gurultusu_YOK"] = ozetle(
    sim(N, 0.0, 0.002, B_TLS, kural_gercek=True, b_kucuk_true=0.0, rng=rng)
)
senaryolar["H1_kural_GERCEK_b1.89_vekil_gurultusu_OLCULEN"] = ozetle(
    sim(N, sigma_dosya, 0.002, B_TLS, kural_gercek=True, b_kucuk_true=0.0, rng=rng)
)

LR = {}
h0 = senaryolar["H0_b1.89(TLS)_vekil_gurultusu_OLCULEN"]
h1 = senaryolar["H1_kural_GERCEK_b1.89_vekil_gurultusu_OLCULEN"]
h1_hilesi = senaryolar["H1_kural_GERCEK_b1.89_vekil_gurultusu_YOK"]
for ad, anah in [
    ("oruntu(R2b>0.9 & R2k<0.2)", "P_oruntu_R2buyuk>0.9_ve_R2kucuk<0.2"),
    ("dort_sayi_birden", "P_hepsi_birden"),
    ("R2k<=0.036", "P_R2kucuk<=0.036"),
]:
    LR[ad] = {
        "P_H0_kural_YOK": h0[anah],
        "P_H1_kural_GERCEK": h1[anah],
        "LR_H1/H0": (h1[anah] / h0[anah]) if h0[anah] > 0 else None,
    }
LR["ADALETSIZ_karsilastirma_uyarisi"] = (
    "H1'i vekil gurultusu OLMADAN calistirmak (P(dort sayi)=%.3f) ADALETSIZDIR: "
    "vekilin hatasi OLCULDU, yok sayilamaz. Adil karsilastirma iki hipotezi de "
    "AYNI olculen gurultuyle calistirir." % h1_hilesi["P_hepsi_birden"]
)
LR["SONUC"] = (
    "Adil karsilastirmada LR(kural gercek / kural yok) = %.2f. "
    "27 dosyalik veri iki hipotezi AYIRT EDEMIYOR." % (h1["P_hepsi_birden"] / h0["P_hepsi_birden"])
)

CIK["2b_ayirt_etme_gucu"] = LR

CIK["2_simulasyon"] = {
    "n_tekrar": N,
    "gozlenen": GOZ,
    "kurulum": (
        "gercek dCV_i = dLB_i / b_true (gozlenen 27 dosyanin dLB dagilimi "
        "korunur); dLB = b_true*dCV + N(0,sd_lb) + 5 hane yuvarlama; "
        "vekil dCV = gercek + N(0,sigma_vekil) + N(0,tohum). Sonra ayni "
        "katmanli analiz (|vekil dCV|>=0.02 vs <0.02)."
    ),
    "b_true": B_BRUT,
    "senaryolar": senaryolar,
}

# ---------------------------------------------- 3. GERCEK dCV ile kontrol
# rekor.jsonl'de 'blok' turu = gercek blok CV; 'lb' turu = gercek LB
rek = [
    json.loads(l)
    for l in io.open(os.path.join(KOK, "experiments", "rekor.jsonl"), encoding="utf-8")
    if l.strip()
]
blok = {r["ad"]: r["skor"] for r in rek if r["tur"] == "blok"}
lb = {r["ad"]: r["skor"] for r in rek if r["tur"] == "lb"}

# p12e'nin kullandigi eslestirme
ESLES_A = {
    "v27": ("v27 harman 3/1/1", "v18 (dun)"),
    "v30": ("v30 kurgusu", "v30 (v27+buzme)"),
    "v46": ("v46 (15 tohum) NIHAI", "v46 (15 tohum)"),
}
# ALTERNATIF: blok skorlari ESKI son islem sonrasi -> v46 blok'un LB karsiligi v47
ESLES_B = {
    "v27": ("v27 harman 3/1/1", "v18 (dun)"),
    "v30": ("v30 kurgusu", "v30 (v27+buzme)"),
    "v46": ("v46 (15 tohum) NIHAI", "v47 (15 tohum + ESKI son islem)"),
}


def cift_analiz(esles):
    ad = list(esles)
    ciftler = []
    for i in range(len(ad)):
        for j in range(i + 1, len(ad)):
            a, b = ad[i], ad[j]
            dcv = blok[esles[b][0]] - blok[esles[a][0]]
            dlb = lb[esles[b][1]] - lb[esles[a][1]]
            ciftler.append(
                {
                    "cift": a + "->" + b,
                    "gercek_dCV": dcv,
                    "gercek_dLB": dlb,
                    "oran": dlb / dcv,
                    "band_ici_mi": abs(dcv) < 0.02,
                }
            )
    x = np.array([c["gercek_dCV"] for c in ciftler])
    y = np.array([c["gercek_dLB"] for c in ciftler])
    b_ko = float(np.sum(x * y) / np.sum(x * x))
    # yalniz 2 bagimsiz cift (zincirin ardisik halkalari)
    x2, y2 = x[[0, 2]], y[[0, 2]]  # v27->v30 ve v30->v46
    b_ko2 = float(np.sum(x2 * y2) / np.sum(x2 * x2))
    return {
        "ciftler": ciftler,
        "hepsi_band_icinde_mi": bool(all(c["band_ici_mi"] for c in ciftler)),
        "kesmesiz_egim_3cift": b_ko,
        "kesmesiz_egim_2_BAGIMSIZ_cift": b_ko2,
        "oranlar": [c["oran"] for c in ciftler],
        "isaret_uyumu": bool(
            all(np.sign(c["gercek_dCV"]) == np.sign(c["gercek_dLB"]) for c in ciftler)
        ),
    }


CIK["3_gercek_dCV_kontrolu"] = {
    "kaynak": "experiments/rekor.jsonl -- 'blok' turu gercek blok CV, 'lb' turu gercek LB",
    "gercek_blok_CV_bilinen_dosya_sayisi": len(blok),
    "hem_gercek_blok_CV_hem_LB_bilinen": 3,
    "bagimsiz_cift_sayisi": 2,
    "eslestirme_A_p12e_ile_ayni": cift_analiz(ESLES_A),
    "eslestirme_B_v46blok_ESKI_sonislem_yani_v47LB": cift_analiz(ESLES_B),
    "eslestirme_notu": (
        "rekor.jsonl'de v30/v33/v36/v39/v46 blok skorlari 'SON ISLEM "
        "SONRASI (eski buzme)' olarak isaretli. v46'nin LB'si 1.02448 "
        "YENI son islem ile, v47 1.01750 ESKI son islem ile gonderildi. "
        "Eslestirme B daha tutarli olabilir; IKISI DE ayni yonu veriyor."
    ),
    "vekil_ayni_ciftlerde_ne_dedi": farklar,
}

# 3b: gercek ciftlerde b=0 (kural gercek) hipotezinin sinanmasi
cA = CIK["3_gercek_dCV_kontrolu"]["eslestirme_A_p12e_ile_ayni"]
xg = np.array([c["gercek_dCV"] for c in cA["ciftler"]])[[0, 2]]
yg = np.array([c["gercek_dLB"] for c in cA["ciftler"]])[[0, 2]]
b_ko2 = float(np.sum(xg * yg) / np.sum(xg * xg))
artik = yg - b_ko2 * xg
sd_artik = float(np.sqrt(np.mean(artik**2)))
sinama = {}
for sd_lb in (0.0016, 0.002, 0.005, 0.010):
    se_b = sd_lb / math.sqrt(float(np.sum(xg * xg)))
    sinama["sd_LB=%.4f" % sd_lb] = {
        "se_b": se_b,
        "z": b_ko2 / se_b,
        "tek_yonlu_p_b0icin": float(0.5 * math.erfc((b_ko2 / se_b) / math.sqrt(2))),
    }
CIK["3_gercek_dCV_kontrolu"]["3b_b0_hipotezi_sinamasi"] = {
    "b_kesmesiz_2_bagimsiz_cift": b_ko2,
    "artiklarin_RMS_si": sd_artik,
    "aciklama": (
        "Kural gercekse bu ciftlerde (hepsi |dCV|<0.02) b=0 olmali ve dLB "
        "sirf CV->LB sapma gurultusu olmali. sd_LB varsayimina duyarlilik:"
    ),
    "duyarlilik": sinama,
    "isaret_sinamasi": {"n_bagimsiz": 2, "isaret_uyumu": "2/2", "P_sirf_sansla": 0.25},
    "UYARI": "n=2 bagimsiz gozlem. Bu bir KANIT DEGIL, bir ISARETTIR.",
}

# ---------------------------------------------- 5. HUBER beklentisi
h = ozet["6_BUYUKLUK"]
blok_kaz = [h["alpha_0.5_test_bilesimi"][k] for k in ("yaz25", "guz25", "kis26")]
kaz_ort = float(np.mean(blok_kaz))
kaz_sd = float(np.std(blok_kaz, ddof=1))
kaz_se = kaz_sd / math.sqrt(3)

MEVCUT = 1.00115
UCUNCU = 0.99556
IKINCI = 0.99536

# tasima orani icin aday dagilimlar
oranlar_gercek = np.array(CIK["3_gercek_dCV_kontrolu"]["eslestirme_A_p12e_ile_ayni"]["oranlar"])
oran_ort = float(np.mean(oranlar_gercek))
oran_sd = float(np.std(oranlar_gercek, ddof=1))

rng2 = np.random.default_rng(99)
M = 200000
# CV kazanci: 3 blok ortalamasinin belirsizligi (t_2)
t2 = rng2.standard_t(2, M)
kaz = kaz_ort + kaz_se * t2
# tasima orani: 3 gercek ciftin oranindan (t_2), negatife de izin var
t2b = rng2.standard_t(2, M)
oran = oran_ort + (oran_sd / math.sqrt(3)) * t2b
lb_tahmin = MEVCUT - oran * kaz

senaryo_oran = {}
kaz_ornek = kaz_ort + kaz_se * rng2.standard_t(2, M)
for ad, o in [
    ("oran_0.0", 0.0),
    ("oran_0.25", 0.25),
    ("oran_0.5", 0.5),
    ("oran_0.568_gercek_cift_A", 0.5680),
    ("oran_0.967_gercek_cift_B", 0.967),
    ("oran_1.0", 1.0),
    ("oran_1.428_brut_vekil", B_BRUT),
]:
    lbo = MEVCUT - o * kaz_ornek
    senaryo_oran[ad] = {
        "beklenen_LB": MEVCUT - o * kaz_ort,
        "3.sirayi(0.99556)_gecer_mi_nokta": bool(MEVCUT - o * kaz_ort <= UCUNCU),
        "P_3.siradan_iyi_CVbelirsizligiyle": float(np.mean(lbo <= UCUNCU)),
        "P_mevcuttan_kotu": float(np.mean(lbo > MEVCUT)),
    }

CIK["5_huber_beklentisi"] = {
    "aday": "soguk lgbm huber alpha=0.5, esit harman",
    "blok_kazanclari_test_bilesimi": dict(zip(("yaz25", "guz25", "kis26"), blok_kaz)),
    "uc_blok_ort": kaz_ort,
    "blok_sd": kaz_sd,
    "ort_se": kaz_se,
    "P_kazanc>0_bloklar_arasi_t2": float(np.mean(kaz > 0)),
    "mevcut_LB": MEVCUT,
    "3.sira": UCUNCU,
    "2.sira": IKINCI,
    "gereken_kazanc": MEVCUT - UCUNCU,
    "gereken_tasima_orani_nokta_tahminle": (MEVCUT - UCUNCU) / kaz_ort,
    "sabit_oran_senaryolari": senaryo_oran,
    "MonteCarlo": {
        "aciklama": (
            "kazanc ~ t2(0.01263, se=0.01006) [uc blok arasi sacilim]; "
            "tasima orani ~ t2(ort/se) [3 GERCEK ciftin orani]; "
            "LB = 1.00115 - oran*kazanc"
        ),
        "oran_ort": oran_ort,
        "oran_sd": oran_sd,
        "beklenen_LB_medyan": float(np.median(lb_tahmin)),
        "LB_GA80": [float(np.percentile(lb_tahmin, 10)), float(np.percentile(lb_tahmin, 90))],
        "P_3.siradan_iyi": float(np.mean(lb_tahmin <= UCUNCU)),
        "P_2.siradan_iyi": float(np.mean(lb_tahmin <= IKINCI)),
        "P_mevcuttan_kotu": float(np.mean(lb_tahmin > MEVCUT)),
    },
}

CIK["4_HUKUM"] = {
    "secim": "B'ye YAKIN C -- kural KANITLANMAMIS; gonderimi bloklamak icin yetersiz",
    "1_band_siniri_gurultu_tabaninda": (
        "0.02 = 1.00 x olculen cift-vekil hatasi (0.0199) = 1.42 x dosya basina "
        "vekil hatasi (0.0141). Sinir tam da 'olcemedigimiz yerde'."
    ),
    "2_katmanlama_istatistigi_kanit_degil": (
        "R2 0.986 vs 0.036 oruntusu, TASIMA ORANI SABIT (kural YOK) H0'i altinda "
        "olculen vekil gurultusuyle %15.8 olasilikla uretiliyor; vekil gurultusu "
        "SIFIRLANINCA %0.0'a duser. Yani oruntuyu ureten sey gurultudur, kural degil."
    ),
    "3_ayirt_edilemiyor": (
        "Iki hipotez AYNI olculen gurultuyle kosuldugunda LR = 1.6 -- 27 dosyalik "
        "veri kurali ne DOGRULUYOR ne CURUTUYOR. R2=0.036, 'tasima yok' degil "
        "'olcemedim' demektir."
    ),
    "4_p12e_zayiflatma_curutmesi_GECERSIZ": (
        "lambda=0.930 TOHUM gurultusuyle hesaplandi (var 7.6e-06). Vekilin OLCULEN "
        "hatasi 3.97e-04, yani gozlenen x sacilimindan (1.09e-04) BUYUK. Dogru "
        "lambda = -2.65 -> n=11 cift regresyonunun guvenilirligi SIFIR. "
        "b_cift = -0.161 hicbir bilgi tasimiyor. Bu, docs/78-79'un ana dayanagidir "
        "ve DUSMUSTUR."
    ),
    "5_vekilsiz_tek_olcum_TERS_yonu_gosteriyor": (
        "GERCEK blok CV + GERCEK LB'nin ikisinin de bilindigi 3 cift (2 bagimsiz) "
        "-- UCU DE bandin ICINDE (|dCV| 0.0069-0.0164) -- tasima orani 0.57 "
        "(alternatif eslestirmede 0.97), isaret uyumu 3/3. Ayni ciftlerde VEKIL "
        "isareti 3/3 YANLIS verdi. n kucuk (2 bagimsiz), kanit degil isaret."
    ),
    "6_karar_kurali_onerisi": [
        "0.02 esigini bir GONDERIM KAPISI olarak KULLANMA -- o bir olcum siniri.",
        "Kapiyi sunlara koy: (a) kazanc VEKILDEN degil GERCEK blok CV'sinden "
        "olculmus mu; (b) blok-disi secim, hedef bloktan bilgi kullanmadan mi "
        "yapildi; (c) bloklar arasi isaret tutarli mi; (d) tohum sayisi arttikca "
        "kazanc ayakta kaliyor mu.",
        "Tasima orani icin nokta tahmin 0.5 kullan (gercek ciftlerden 0.57 ve "
        "0.97; muhafazakar taraf 0.5).",
    ],
    "7_KARSI_ARGUMANLAR_durust": [
        "Simulasyonun gozlenen TAM sayilarina (b_kucuk=-0.47, b_buyuk=+2.28) "
        "bakildiginda H1 (kural gercek) H0'dan 1.6 KAT daha olasi. Yani veri "
        "cok zayif da olsa kuralin LEHINE egiliyor, aleyhine degil.",
        "Vekil hata tahmini yalnizca 2 BAGIMSIZ gozleme dayaniyor; sigma_dosya "
        "0.0141 kendisi cok belirsiz. Gercek deger 0.007 ise H0 oruntuyu cok daha "
        "az uretiyor (bkz. YARI senaryosu: P(R2k<=0.036)=0.015).",
        "3 gercek cift AYNI SOYAGACINDAN (v27->v30->v46, ardisik son-islem "
        "degisiklikleri). Baska tur degisikliklere genellemesi garanti degil.",
        "Bu analiz kurali YIKMIYOR, DESTEKSIZ birakiyor. 'Kural yanlis' ile "
        "'kural olculmemis' ayni sey degildir.",
    ],
    "8_huber_icin_ASIL_risk_band_degil": (
        "Huber'in gercek riski 0.02 bandi DEGIL: yaz25 blogu NEGATIF (-0.00736) ve "
        "yaz25 test ile MEVSIMSEL OLARAK ayni donem (2025-04..07 vs test "
        "2026-04..07). p14 kendi 9_uyari'sinda bunu isaretliyor. Ayrica p14 "
        "6_BUYUKLUK: eklenen HER tohum kazanci ASAGI cekti, ISTISNASIZ -- "
        "+0.0126 hala iyimser yanli olabilir."
    ),
}

with io.open(os.path.join(PK, "p17_band.json"), "w", encoding="utf-8") as f:
    json.dump(CIK, f, ensure_ascii=False, indent=1)

# ---- ekrana ozet
print(
    "BAND SINIRI 0.02 = %.2f x sigma_dosya(%.5f) = %.2f x sigma_cift(%.5f)"
    % (0.02 / sigma_dosya, sigma_dosya, 0.02 / sigma_cift, sigma_cift)
)
print("lambda p12e=%.3f  OLCULEN ile=%.3f" % (lam_kullanilan, lam_olculen))
print()
for k, v in senaryolar.items():
    print(
        "%-52s R2b=%.3f R2k=%.3f bk=%+.2f bb=%+.2f  P(oruntu)=%.3f P(hepsi)=%.3f"
        % (
            k,
            v["R2_buyuk"]["medyan"],
            v["R2_kucuk"]["medyan"],
            v["b_kucuk"]["medyan"],
            v["b_buyuk"]["medyan"],
            v["P_oruntu_R2buyuk>0.9_ve_R2kucuk<0.2"],
            v["P_hepsi_birden"],
        )
    )
print()
for ad in ("eslestirme_A_p12e_ile_ayni", "eslestirme_B_v46blok_ESKI_sonislem_yani_v47LB"):
    a = CIK["3_gercek_dCV_kontrolu"][ad]
    print(
        ad,
        "b_kesmesiz=%.3f (2 bagimsiz: %.3f) oranlar=%s hepsi band ici=%s"
        % (
            a["kesmesiz_egim_3cift"],
            a["kesmesiz_egim_2_BAGIMSIZ_cift"],
            ["%.3f" % o for o in a["oranlar"]],
            a["hepsi_band_icinde_mi"],
        ),
    )
print()
print(json.dumps(CIK["5_huber_beklentisi"]["MonteCarlo"], ensure_ascii=False, indent=1))
print("YAZILDI: p_kalici/p17_band.json")
