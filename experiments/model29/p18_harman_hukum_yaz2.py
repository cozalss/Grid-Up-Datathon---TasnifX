"""p18c-ek2: hukme p14'un kendi iddiasini, blok skorlarini ve bilesimi ekler."""

import json
import os

BURA = os.path.dirname(os.path.abspath(__file__))
YOL = os.path.join(BURA, "p_kalici", "p18_harman_hukmu.json")

with open(YOL, encoding="utf-8") as fh:
    R = json.load(fh)

R["3b_CELISKININ_YAZILI_KANITI"] = {
    "p_kalici/p14_ozet.json:harman": "ESIT (1/3 cat, 1/3 xgb, 1/3 lgbm) -- URETIM ile ayni, p06 agirligi KULLANILMADI",
    "p_kalici/p14_olc.json:harman": "ESIT (uretim ile ayni)",
    "scripts/tuketim_model.py:990-995": '"soguk": {"maske": 1.00, "cat": {"depth": 7}, "ek_koken": False, "agirlik": {"cat": 1.0}}',
    "hukum": "p14 'ESIT harman = URETIM' diye ACIKCA yaziyor. Kaynak kod ve gonderim dosyalarinin ampirik regresyonu bunun YANLIS oldugunu soyluyor. p14'un butun kazanc rakamlari bu yanlis varsayim uzerine kurulu.",
    "p11_b_lgbm_ayrica": "p11'in 'TABAN' sutunu (yaz25/1000 = 1.420952, kis26/1000 = 2.017536) ESIT harman DEGIL, YALNIZ-LGBM'dir (3 tohumlu lgbm-tekil: yaz25 1.40519, kis26 2.00527). docs/79 §6 tablosu bunu 'TABAN (uretim)' diye etiketlemis; o etiket de yanlis.",
}

R["3c_BLOK_SKORLARI_HAM"] = {
    "aciklama": "SON ISLEM ONCESI, 3 tohum (1000-1002) ortalamasi, log uzayinda kirpma (max(p,0)) ile. SOGUK = soguk_tahmin_*.npz, SICAK = aile_onbellek/*_uretim.npy (sinir_agi dahil).",
    "yaz25": {"soguk_URETIM_cat": 1.57454, "soguk_TEZGAH_esit": 1.43487, "soguk_lgbm": 1.40519,
              "sicak_URETIM_3_1_1_14": 0.79655, "sicak_TEZGAH_esit": 0.80304},
    "guz25": {"soguk_URETIM_cat": 1.69729, "soguk_TEZGAH_esit": 1.60782, "soguk_lgbm": 1.60749,
              "sicak_URETIM_3_1_1_14": 0.80548, "sicak_TEZGAH_esit": 0.80473},
    "kis26": {"soguk_URETIM_cat": 1.83864, "soguk_TEZGAH_esit": 1.90615, "soguk_lgbm": 2.00527,
              "sicak_URETIM_3_1_1_14": 0.74322, "sicak_TEZGAH_esit": 0.73831},
    "test_bilesimi_sqrt(0.2216*soguk+0.7784*sicak)": {
        "URETIM (cat soguk + 3/1/1/1.4 sicak)": {"yaz25": 1.02141, "guz25": 1.06930, "kis26": 1.08587, "ORT": 1.05886},
        "TEZGAH (esit + esit)": {"yaz25": 0.97888, "guz25": 1.03776, "kis26": 1.10881, "ORT": 1.04182},
        "URETIM sicak + ESIT soguk": {"yaz25": 0.97475, "guz25": 1.03821, "kis26": 1.11137, "ORT": 1.04144},
        "URETIM sicak + LGBM soguk": {"yaz25": 0.96512, "guz25": 1.03810, "kis26": 1.14937, "ORT": 1.05086},
    },
    "OLCEK": "Tezgah ile uretim arasindaki bilesim farki ORT 0.017. Yani CV tezgahi, gonderdigimizden ~0.017 DAHA IYI bir nesneyi olcuyor. Bu, tartisilan huber kazancindan (0.0126) BUYUK.",
}

R["3d_ACILAN_BUYUK_SORU"] = {
    "soru": "Soguk uzman CAT-TEKIL olmali mi?",
    "veri": "Soguk cat-tekil, yaz25'te esit harmandan 0.1397 KOTU, guz25'te 0.0895 KOTU, kis26'da 0.0675 IYI.",
    "karar_gecmisi": "tuketim_model.py'nin kendi yorumu: 'SOGUK HARMAN -> YALNIZ cat (2026-08-23, deney_soguk_taban.py). Eski 3/1/1 hukmu KIRLI bloklarin ortalamasindan geliyordu. kis26 tek basina bakildiginda ... olculdu'. Yani secim YALNIZ kis26'da yapildi.",
    "kalici_kural_ihlali": "docs (KALICI KURAL 10): 'kis26'da olculen seviye kazanci kesme-etiket mevsim bitisikliginden besleniyor ve TEST'in geometrisine tasinmiyor; boyle oneriler yaz25'te olculmelidir.' Soguk harman karari tam olarak bu kuralin yasakladigi bicimde verilmis.",
    "buyukluk": "Soguk tarafi esite dondurmek test bilesiminde ORT -0.0174 (yaz25 -0.0467, guz25 -0.0311, kis26 +0.0255). Isaret 2/3. Bu, simdiye kadar tartisilan her adaydan BUYUK bir eksen.",
    "UYARI_DURUST": [
        "Bu rakamlar SON ISLEM ONCESI. Uretim soguk buzmesi (beta) cat-tekil tahminlere gore kalibre edildi; buzme altinda siralama degisebilir. Kod yorumundaki kis26 tablosu buzme altinda da cat'i onde gosteriyor (1.82250 vs 3/1/1 1.83041) ama yaz25/guz25 buzme altinda HIC olculmedi.",
        "kis26 ters isaret veriyor ve isaret tutarliligi 2/3 -- docs/79 EK'teki (c) kapisini GECEMIYOR.",
        "Bu bir GONDERIM ONERISI DEGIL, olculmesi gereken bir eksendir. Olcum EGITIM GEREKTIRMIYOR (butun diziler onbellekte).",
    ],
}

R["5_NE_YAPILMALI"]["E_ONCELIK"] = [
    "1. p02_duzeltme.blok() ve turevlerinin harman kurulusunu URETIME hizala (soguk: yalniz cat; sicak: 3/1/1/1.4 + sinir_agi). EGITIM GEREKTIRMEZ.",
    "2. Hizalanmis tezgahta soguk harman eksenini son islem ALTINDA uc blokta olc. EGITIM GEREKTIRMEZ.",
    "3. Soguk kayip fonksiyonu sorusunu CAT icin yeniden sor (p11 yalniz lgbm'i taradi). Bu EGITIM GEREKTIRIR: 3 blok x 3-5 tohum x cat = ~25-40 dk bos makinede.",
    "4. Huber a=0.5 gonderim adayini ve p17'nin 5. bolumunu GEcERSIZ isaretle.",
]

with open(YOL, "w", encoding="utf-8") as fh:
    json.dump(R, fh, ensure_ascii=False, indent=1)
print("yazildi:", YOL)
