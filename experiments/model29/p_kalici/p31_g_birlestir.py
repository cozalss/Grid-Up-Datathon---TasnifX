"""p31_g -- butun p31 olcumlerini tek dosyada birlestir + HUKUM."""
import json
import os

import numpy as np

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
CIK = os.path.join(KOK, "experiments/model29/p_kalici")


def oku(ad):
    with open(os.path.join(CIK, ad), encoding="utf-8") as f:
        return json.load(f)


R = oku("p31_sulama.json")
for ad in ("p31_b_ara.json", "p31_c_ara.json", "p31_d_ara.json", "p31_f_ara.json"):
    R.update(oku(ad))
R["16_ADAY_DOSYALARI"] = oku("p31_e_ara.json")

LB0, HEDEF3, HEDEF10 = 1.00115, 0.00628, 0.00305
# dRMSLE ~= dMSE / (2*RMSLE); LB kazanci = tasima_orani * dRMSLE
gerek = {f"tasima_{t}": {"ilk3_icin_dMSE": round(2 * LB0 * HEDEF3 / t, 5),
                         "ilk10_icin_dMSE": round(2 * LB0 * HEDEF10 / t, 5)}
         for t in (0.5, 0.7, 1.0)}

d = R["12_DOGRULAMA_kis26_Oca_Mar"]
y = R["13_yaz25_AYNI_YIL_KAHIN"]
ob = R["14_ONYUKLEME_dogrulama_blogu"]

R["17_HUKUM"] = {
    "01_ANAHTAR": "47/47 ilce eslesti (p29: 18/47). Kapi ACILDI.",
    "02_SULAMA_IMZASI_47_ILCE_ILE_AYAKTA": {
        "yaz-kis farki std": R["03_SULAMA_IMZASI_47_ILCE"]["SICAK"]["std"],
        "kor_tarim_orani": R["03_SULAMA_IMZASI_47_ILCE"]["SICAK"]["kor_tarim_orani"],
        "p29_degeri": 0.541,
        "hukum": ("Isaret ve fizik AYAKTA (Odemis/Kinik/Sarigol/Golmarmara + ; "
                  "Guzelbahce/Foca/Bornova -). 47 ilcede r=+0.42, p29'un 18 ilceli "
                  "r=+0.54'unden dusuk ama ayni yonde ve anlamli."),
    },
    "03_ETKI_MODEL_ARTIGINDA_DA_VAR": {
        "ilce_x_ay_artik_std_yaz25": R["02_ILCE_x_AY_MODEL_ARTIGI"]["yaz25_SICAK"]["ilce_x_ay_std"],
        "kor_tarim_yaz25": R["02_ILCE_x_AY_MODEL_ARTIGI"]["yaz25_SICAK"]["kor_tarim_orani"],
        "kahin_tavani_dMSE_yaz25_TUM": R["09_KAHIN_TAVANI_blok_ICI"]["yaz25_TUM"]["kahin_ilce_x_ay"],
        "hukum": ("Odul GERCEK ve BUYUK: mukemmel bir ilce x ay duzeltmesi yaz25'te "
                  "0.0995 dMSE degerinde. Sorun buyukluk degil, TASINABILIRLIK."),
    },
    "04_YIL_UZERI_KARARLILIK": {
        "ilce_vektoru_kor": [k["korelasyon"] for k in R["10_YIL_UZERI_ILCE_VEKTORU"]["aylik"]],
        "ilce_vektoru_egim": [k["egim_2026_uzerine_2025"] for k in R["10_YIL_UZERI_ILCE_VEKTORU"]["aylik"]],
        "ortalama_egim": R["10_YIL_UZERI_ILCE_VEKTORU"]["ortalama_egim"],
        "p29_egimi": "0.25-0.39 (yanlis anahtar, 29 ilce)",
        "duzeltilmis": "0.34-0.67, ortalama 0.495 -- p29'dan IYI",
        "tarim_beta_yil_uzeri": R["11_YIL_UZERI_TARIM_EGIMI_beta"]["cift_2025_vs_2026"],
        "uyari": "Nisan-Temmuz beta'si tek yil -- DOGRULANAMAZ.",
    },
    "05_ASIL_KAPI_DUSTU": {
        "deney": ("kis26 Oca-Mar 2026 GERCEK CV tahminleri; capa YALNIZ 2025 "
                  "Oca-Mar'dan. Test kurgusunun birebir provasi."),
        "dogrudan_YIL_DISI_TUM": d["dogrudan|YIL_DISI_2025|TUM"],
        "parametrik_YIL_DISI_TUM": d["parametrik|YIL_DISI_2025|TUM"],
        "ayni_yil_KAHIN_TUM": d["dogrudan|AYNI_YIL_2026_kahin|TUM"],
        "onyukleme_dogrudan_l030_TUM": ob["dogrudan|TUM|0.3"],
        "onyukleme_dogrudan_l050_TUM": ob["dogrudan|TUM|0.5"],
        "onyukleme_parametrik_l030_TUM": ob["parametrik|TUM|0.3"],
        "HUKUM": ("YIL-DISI capa lambda>=0.3'te NEGATIF (dogrudan) ya da SIFIR "
                  "(parametrik). P(pozitif) en iyi durumda 0.79 ama buyukluk 3.5e-05 "
                  "-- gerekenin ~700 katindan kucuk."),
    },
    "06_UST_SINIR_HESABI": {
        "yaz25_ayni_yil_KAHIN_TUM_en_iyi": max(y["dogrudan|TUM"].values()),
        "aciklama": ("Bu deger HILELI: capa yaz25'in kendi yilindan. Test'te 2026 "
                     "Nis-Tem yok, boyle bir capa MUMKUN DEGIL."),
        "gercekci_indirim": ("kis26'da olculen yil-uzeri tasima orani G_dis/G_ic "
                             "NEGATIF. En comert varsayimla bile ilce vektoru "
                             "egimi 0.495 -> kazanc ~0.495^2 ile olceklenir: "
                             "0.0083 * 0.245 = 0.0020 dMSE."),
        "gereken_dMSE": gerek,
        "SONUC": ("Ilk 3 icin tasima 1.0'da bile 0.01257 dMSE gerekiyor. HILELI "
                  "ust sinir 0.0083, gercekci beklenti 0.0020. Ilk 10 icin "
                  "(0.00611 @ tasima 1.0) bile hileli ust sinir kil payi yetmiyor."),
    },
    "07_P_KAZANC_HESABI": {
        "olcut": "P(LB kazanci >= 0.00628)",
        "yontem": ("Onyukleme dagilimi (trafo kumeli, 500) dMSE uzerinde; "
                   "esik = 2*1.00115*0.00628/tasima."),
        "esik_dMSE": {"tasima_1.0": 0.01257, "tasima_0.7": 0.01796, "tasima_0.5": 0.02515},
        "en_iyi_varyantin_GA95_ust_ucu": max(
            ob[k]["GA95"][1] for k in ob),
        "P_ilk3": 0.0,
        "P_ilk10": 0.0,
        "not": ("Butun onyukleme dagilimlarinin 97.5 yuzdeligi 0.00066'nin altinda; "
                "esikler 0.0126-0.0252. Kesisim YOK -- P sifir."),
    },
    "08_NEDEN": {
        "ham_vekil_vs_model_artigi_kor": R["15_TESHIS_ham_vekil_vs_model_artigi"]["korelasyon"],
        "R2": R["15_TESHIS_ham_vekil_vs_model_artigi"]["R2"],
        "aciklama": ("Ham verideki ilce x ay sinyali ile modelin ARTIK yanliligi "
                     "yalniz kismen ortusuyor (R2=0.33) VE ayni-yil kahin capasi "
                     "kis26'da yalnizca 0.00035 dMSE aliyor (model-artigi kahini "
                     "0.066). Yani p29'un olctugu ham sulama imzasini model BUYUK "
                     "OLCUDE ZATEN GORUYOR (sicaklik + trafo mevsim oznitelikleri "
                     "uzerinden); geriye kalan ilce x ay artigi ise yil-uzeri "
                     "TASINMIYOR."),
    },
    "09_ADAYLAR": {
        "uretildi": sorted(R["16_ADAY_DOSYALARI"]["dosyalar"]),
        "dogrulama": "714688 satir, id sirasi test.csv ile birebir, NaN/negatif yok",
        "TAVSIYE": ("HICBIRI GONDERILMEMELI. Blok-disi + yil-disi olcumde "
                    "lambda>=0.5 NEGATIF, lambda=0.3 sifir. Dosyalar yalnizca "
                    "arastirma probu olarak duruyor."),
        "tek_temiz_katman": ("p08 olu trafo x0.50 (sicak satirlar, uc blokta da "
                             "pozitif, ezberden bagimsiz) -- bu p31'den BAGIMSIZ "
                             "ve hala gecerli."),
    },
    "10_DURUSTLUK": (
        "Sulama ekseni ELENDI. Bulunan sey gercek (imza fiziksel, odul buyuk) ama "
        "kullanilabilir degil: (a) Nisan-Temmuz capasi tek yil, dogrulanamaz; "
        "(b) dogrulanabilen tek prova (Oca-Mar) NEGATIF; (c) hileli ust sinir bile "
        "ilk 3 esiginin altinda. Bu, bugun kirilan yedinci 'dogrulanmis' bulgu "
        "olmasin diye kendi kapimi kendim koydum ve gecemedi."),
}
R["17_HUKUM"]["11_GEREKEN_dMSE_TABLOSU"] = gerek

with open(os.path.join(CIK, "p31_sulama.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print(json.dumps(R["17_HUKUM"], ensure_ascii=False, indent=1))
