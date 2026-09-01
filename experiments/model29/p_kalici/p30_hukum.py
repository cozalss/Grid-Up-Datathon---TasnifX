"""p30 hukum bolumlerini p30_ezber.json'a ekler. Olcum yok, yalnizca yazim."""
import json
import os

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
yol = os.path.join(KOK, "experiments/model29/p_kalici/p30_ezber.json")
R = json.load(open(yol, encoding="utf-8"))

R["07_havuzlanmis_TEMIZ"] = {
    "aciklama": "Uc blogun EZBERLENEMEZ (testle ayni kosul) soguk satirlari birlestirildi.",
    "n": 63407,
    "ham_satir_agirlikli_kazanc": -0.02208,
    "agr_satir_agirlikli_kazanc": -0.01746,
    "ham_blok_esit_ortalama": -0.00427,
    "agr_blok_esit_ortalama": 0.00578,
    "isaret_temiz_altkumelerde": "yaz25 +0.006 (GA95 sifiri kapsiyor), guz25 +0.004 "
    "(GA95 sifiri kapsiyor), kis26 -0.0227 (GA95 tamami negatif). 0/3 anlamli pozitif.",
}

R["08_LB_KANITI"] = {
    "dogrudan_cift_cat_tekil_vs_3_1_1": "YOK. Hicbir gonderim cifti yalnizca soguk harmani "
    "degistirmiyor.",
    "en_yakin_dogrudan_cift": {
        "v18 (soguk 3/1/1)": 1.03370,
        "v23 (soguk ESIT 1/1/1 + hafta gunu)": 1.04820,
        "yon": "cat agirligini AZALTMAK LB'de +0.0145 ZARARLI (karisik degisiklik)",
    },
    "docs80_ek_destek_iddiasi_YANLIS": {
        "iddia": "docs/80 §4: '3/1/1 ... o donemin BIRINCILIGI (LB 1.01750) onunla alindi'",
        "gercek": "experiments/rekor.jsonl: v46/v47 kaydi acikca 'cat-only soguk' diyor. "
        "BIRINCILIK 1.01750 CAT-TEKIL soguk harmanla alindi, 3/1/1 ile DEGIL. "
        "p21'in tek 'LB'de olculmus noktaya donus' gerekcesi COKTU.",
    },
    "rekor_jsonl_blok_CV_uretim_olcutu": {
        "v27 harman 3/1/1 soguk": 1.86509,
        "v32 cat-only soguk": 1.83606,
        "v30 (3/1/1 + buzme)": 1.83979,
        "v33 (cat-only + buzme)": 1.82250,
        "yorum": "URETIM olcutunde cat-tekil 3/1/1'den 0.029 IYI. p20'nin tersi.",
    },
    "gonderim_zaman_cizgisi": {
        "v30 (3/1/1, 3 tohum, sinir agi YOK)": 1.02639,
        "v47 (cat-tekil, 15 tohum, sinir agi VAR)": 1.01750,
        "not": "karisik degisiklik; ama 3/1/1 lehine LB kaniti YOK",
    },
}

R["09_DOKTRIN"] = {
    "kural": "Soguk rejim kararlari YALNIZ kis26 ile verilir (docs/36 §3, "
    "scripts/tuketim_model.py:830-836).",
    "dayanagi": "Ezber kanali. yaz25/guz25 soguk satirlarinin %97'si egitim katlarinda "
    "gorunen trafolara ait; testte %0. Kural mevsime degil OLCUM GECERLILIGINE dayaniyor.",
    "hala_gecerli_mi": "EVET -- p30/02 kuralin dayanagini bagimsiz olarak yeniden uretti "
    "(yaz25 %97.18, guz25 %97.70, kis26 %0.00, test %0.00).",
    "mevsim_analogu_ile_catisma": {
        "docs66_kural55": "olcut yaz25 (test'in mevsimsel ikizi)",
        "cozum": "CATISMA DEGIL, KAPSAM FARKI. Mevsim analogu SICAK satirlar icin gecerli "
        "(orada ezber kanali yok: sicak trafolarin gecmisi testte de var). SOGUK satirlarda "
        "yaz25 GECERSIZ bir olcum araci -- mevsimsel yakinlik ezber kirliligini kapatmaz. "
        "Ezber kirliligi bir YANLILIK; mevsim uyusmazligi bir VARYANS/kayma sorunu. "
        "Yanli bir olcumu daha ilgili bir populasyonda yapmak onu duzeltmez.",
        "kohort_sayilari (p14_guc)": {
            "agirlikli_etkin_trafo": {"yaz25": 34.3, "guz25": 156.5, "kis26": 192.7},
            "tv_teste": {"yaz25": 0.7227, "guz25": 0.531, "kis26": 0.5538},
            "test_hucre_kapsami": {"yaz25": 0.9727, "guz25": 0.9718, "kis26": 0.9997},
        },
        "sonuc": "kis26 hem tek TEMIZ blok, hem kohort kapsami en genis (%99.97), hem etkin "
        "trafo sayisi en yuksek (192.7 vs 34.3). yaz25'in tek ustunlugu takvim ayi -- ve "
        "onun agirlikli etkin trafo sayisi 34, yani zaten en gurultulu blok.",
    },
}

R["10_NET_HUKUM"] = {
    "SORU": "p21_harman311_olu50.csv yarin Hak 1 olarak gonderilmeli mi?",
    "CEVAP": "HAYIR. GONDERME.",
    "GEREKCE_SIRALI": [
        "1. KIRLILIK TESTI: p21'in kazanci EZBERLENEBILIR satirlarda yasiyor. yaz25 "
        "ezber +0.1164 / temiz +0.0058; guz25 ezber +0.0665 / temiz +0.0041; kis26 "
        "(%100 temiz) -0.0227. Testin TAMAMI temiz kosulda.",
        "2. Havuzlanmis temiz satirlarda (n=63.407, uc blok) kazanc NEGATIF (-0.022 ham / "
        "-0.017 agr). Uc temiz altkumenin HICBIRI anlamli pozitif degil (0/3); tek yuksek "
        "gucli temiz olcum (kis26, 1223 trafo) GA95'i tamamen negatif, P(+)=0.000.",
        "3. Kapi (c) isaret tutarliligi COKTU: docs/80'in '3/3 yapida pozitif' iddiasi "
        "ezber kirliligiyle uretilmis iki bloktan besleniyor. Kirliligi cikarinca 1/3 "
        "(ve o da negatif).",
        "4. p21'in tek LB dayanagi (docs/80 §4 'birincilik 3/1/1 ile alindi') OLGUSAL "
        "OLARAK YANLIS -- rekor.jsonl v46/v47 cat-tekil soguk diyor.",
        "5. Ayni yonde LB kaniti VAR ama TERS isaretli: v18 (3/1/1) 1.03370 -> v23 "
        "(cat agirligi azaltilmis) 1.04820.",
        "6. Doktrin (kis26) bu soru icin dogru mercidir ve p21'e HAYIR diyor. p28 zaten "
        "olcmustu: kis26'da her ikisi kendi optimum beta'sindayken 3/1/1 (1.83333) "
        "cat-tekilden (1.82519) KOTU.",
    ],
    "GUC_UYARISI": "yaz25/guz25 temiz altkumeleri kucuk (40 ve 60 trafo) ve gurultulu -- "
    "tek baslarina '3/1/1 kotu' demeye yetmezler. Hukmu tasiyan sey kis26'nin 1223 trafoluk "
    "TEMIZ ve NEGATIF olcumu ile temiz havuzun negatifligi.",
    "ALTERNATIF_HAK1": "Olculmus YP_seviye (1.00115) tabani KORUNSUN. p21 ve p28_beta50 "
    "ikisi de ayni kirli kaynaktan besleniyor (p28 zaten 'HAK 1 icin ONERILMEZ' demisti). "
    "Haklar soguk harmana degil, kirlilikten BAGIMSIZ katmanlara harcanmali: p08 olu trafo "
    "deltasi (sicak satirlar, uc blokta da pozitif, ezberden bagimsiz) ve dogrulanmis sicak "
    "taraf kazanclari (p15 cat tau=480).",
    "EGER_YINE_DE_GONDERILECEKSE": "Hak 1 olarak DEGIL. Once p08 gibi temiz bir katman "
    "olculur; p21 ancak elde bos hak kalirsa ve YEDEK secimde kalmak kosuluyla bir "
    "ARASTIRMA PROBU olarak gonderilebilir -- kazanc beklentisi negatiftir.",
}

with open(yol, "w", encoding="utf-8") as fh:
    json.dump(R, fh, indent=1, ensure_ascii=False)
print("yazildi:", yol)
