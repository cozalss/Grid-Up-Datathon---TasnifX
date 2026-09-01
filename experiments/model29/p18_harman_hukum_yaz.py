"""p18c-ek: p18_harman_hukmu.json'a HUKUM bolumunu ekler."""

import json
import os

BURA = os.path.dirname(os.path.abspath(__file__))
YOL = os.path.join(BURA, "p_kalici", "p18_harman_hukmu.json")

with open(YOL, encoding="utf-8") as fh:
    R = json.load(fh)

R["0_YONTEM_GECERLILIGI"] = {
    "iddia": "log1p(gonderim) uc aile kolonuna (cat,xgb,lgbm) regrese edilirse katsayi PAYLARI harman agirliklarini, katsayi TOPLAMI son islem buzmesini verir.",
    "KALIBRASYON_KANITI": {
        "tuketim_v27_v18hedge.csv": {
            "bilinen_gercek": "ESKI soguk harman 3/1/1 = (0.600, 0.200, 0.200); son islem YOK (scripts/son_islem.py docstring: 'v27 (son islem YOK) 1,03362')",
            "olculen_pay": [0.5968, 0.2383, 0.165],
            "olculen_toplam": 0.9965,
            "yorum": "Paylar 3/1/1'i, toplam 1.00 ise 'son islem yok'u DOGRU okudu.",
        },
        "tuketim_v30_buzme.csv": {
            "bilinen_gercek": "v30 = v27 + scripts/son_islem.py (beta=0.60)",
            "olculen_pay": [0.606, 0.2262, 0.1678],
            "olculen_toplam": 0.603,
            "yorum": "Paylar degismedi, toplam 0.603 = beta 0.60. Yontem hem harmani hem buzmeyi DOGRU okuyor.",
        },
    },
    "aile_kolonlari_ayirt_edilebilir_mi": "Evet. Aile korelasyonlari 0.83-0.93 (tam es-dogrusal degil); v27/v30 kalibrasyonu paylari +-0.04 icinde geri veriyor.",
}

R["1_KAYNAK_KOD"] = {
    "dosya": "scripts/tuketim_model.py",
    "REJIM_AYARLARI_tanimi_satir": 837,
    "SOGUK_bloku_satir": "1021-1026",
    "SOGUK_ifadesi": '"soguk": {"maske": 1.00, "cat": {"depth": 7}, "ek_koken": False, "agirlik": {"cat": 1.0}}',
    "SICAK_bloku_satir": "926-931",
    "SICAK_ifadesi": '"sicak": {"maske": 0.15, "cat": {"random_strength": 4.0, "l2_leaf_reg": 1.0, "depth": 6}, "ek_koken": True, "agirlik": {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0, "sinir_agi": 1.4}}',
    "kullanim": "tuketim_model.rejim_tahmini() satir 1221-1305: her rejim icin agirlik = ayar['agirlik']; cikti[maske] = sum(w*aile_tahmini(a,...)) / sum(w)",
    "degisiklik_commiti": "d04243f 'feat: soguk son islem yeniden kuruldu -- gun ekseni korunuyor, hedef kosullu' (2026-08-23 19:56). Soguk harman 3/1/1 -> YALNIZ cat bu commit'te sabitlendi.",
    "gerekce_yorumda": "REJIM_AYARLARI soguk yorumu: 'SOGUK HARMAN -> YALNIZ cat (2026-08-23, deney_soguk_taban.py)'; harman 1.84106 vs 3/1/1 1.86931 (kis26, beta=1.00).",
}

R["2_HUKUM"] = {
    "SOGUK_URETIM_HARMANI": "YALNIZ CAT ({'cat': 1.0}). Hem kaynak kod hem AMPIRIK regresyon soyluyor.",
    "ampirik_destek": {
        "tuketim_YP_seviye.csv (LB 1.00115, yedegimiz)": {
            "pay": [1.1146, -0.0992, -0.0155],
            "toplam": 0.8132,
        },
        "tuketim_m6_ikiyon.csv (LB 1.00284)": {"pay": [1.0761, -0.071, -0.0051], "toplam": 0.698},
        "tuketim_K_yenibas.csv": {"pay": [1.1147, -0.0988, -0.016], "toplam": 0.8162},
        "tuketim_v83_sicak_optimum.csv": {"pay": [1.001, -0.0028, 0.0018], "toplam": 0.6527},
        "tuketim_v80_optimum.csv": {"pay": [1.001, -0.0028, 0.0018], "toplam": 0.6527},
        "tuketim_v46_gun.csv": {"pay": [1.0101, -0.0156, 0.0055], "toplam": 0.5505},
        "tuketim_v90_temiz_sota.csv": {"pay": [0.9561, 0.0173, 0.0266], "toplam": 0.5239},
        "yorum": "2026-08-23 SONRASI uretilen HER dosyada cat payi ~1.0, xgb ve lgbm paylari ~0 (bazilari hafif NEGATIF). Oncekilerde (v27, v30) 0.60/0.24/0.17. Kirilma noktasi commit d04243f ile ortusuyor.",
    },
    "SICAK_URETIM_HARMANI": "{'cat': 3.0, 'xgb': 1.0, 'lgbm': 1.0, 'sinir_agi': 1.4} -> paylar cat %46.9, xgb %15.6, lgbm %15.6, sinir_agi %21.9. (Kaynak koddan; test tarafinda sicak aile dizisi diskte olmadigi icin ampirik dogrulanamadi. experiments/sicak_kaldirac/ortak.py bu agirligi DOGRU kullaniyor.)",
    "CV_TEZGAHININ_KULLANDIGI": "experiments/model29/p02_duzeltme.py:blok() -- SICAK icin aile_onbellek'ten cat/xgb/lgbm ESIT (sinir_agi YOK), SOGUK icin soguk_tahmin_*.npz'nin TUM anahtarlari ESIT (cat/xgb/lgbm). Ayni kurulus p01, p03-p14, p17'de tekrarlaniyor.",
}

R["3_GECERSIZ_OLAN_OLCUMLER"] = {
    "SOGUK_TARAFTA_TUMU": [
        "p11_b_lgbm.json -- soguk lgbm kayip fonksiyonu taramasi (huber a=1.0 / 2.0 / 0.5, l1, fair). TABAN sutunu ESIT harman.",
        "p14 huber a=0.5 olcumu (+0.01263 test bilesimi, docs/79 EK). Ayni ESIT harman uzerinde.",
        "p06 soguk harman agirligi (esit -> 0.05/0.35/0.60) ve p07'nin urettigi 12 aday CSV + 4 delta npy.",
        "p17_band.json'un 5_huber_beklentisi bolumu ve ona dayanan butun LB/olasilik hesaplari (P(3.sira)=0.52 dahil).",
    ],
    "NEDEN": "Uretim soguk uzmani lgbm'i HIC KULLANMIYOR. lgbm'in kayip fonksiyonunu degistirmek, iceriginde lgbm olmayan bir harmani DEGISTIRMEZ. Olculen kazanc uretimde birebir SIFIRDIR.",
    "p06_ozel_notu": "p06 deltasi A@(w_yeni - w_esit) idi; cat-tekil bir tabana eklenirse harmani (0.05,0.35,0.60)'a DEGIL, (1,0,0)+(w_yeni-w_esit) = (0.7167, 0.0167, 0.2667)'e goturur. Yani uretilmis 12 aday CSV'nin uyguladigi harman, p06'nin olctugu harman DEGILDIR.",
    "SICAK_TARAFTA_KISMEN_GECERSIZ": "p02/p15 sicak olcumleri cat/xgb/lgbm ESIT harman uzerinde; uretim 3/1/1 + sinir_agi 1.4. Bir sicak aile degisikliginin uretimdeki etkisi, olculen etkinin yaklasik (uretim_payi / 0.3333) kati: cat icin x1.41, xgb icin x0.47, lgbm icin x0.47. Isaret korunur, BUYUKLUK korunmaz. Ayrica harmanin %21.9'u (sinir_agi) hicbir olcumde YOK.",
    "GECERLI_KALANLAR": [
        "experiments/sicak_kaldirac/* -- ortak.py URETIM agirligini (3/1/1/1.4) kullaniyor, sinir_agi dahil.",
        "p08 olu trafo kurali -- harmandan bagimsiz, tahmin uzerinde carpan.",
        "p17'nin 1-4. bolumleri (band/gurultu tabani analizi) -- harmandan bagimsiz.",
    ],
}

R["4_TASIMA_BILMECESINE_ETKISI"] = {
    "hipotez": "'Kucuk CV kazanclari LB'ye tasinmiyor' bulgusunun bir parcasi, CV tezgahinin URETIM HATTINI OLCMEMESI olabilir.",
    "destek": "Tezgah soguk tarafta uretimde HIC KULLANILMAYAN iki aileyi (xgb, lgbm) 2/3 agirlikla tasiyor; sicak tarafta uretim harmaninin %21.9'unu (sinir_agi) hic tasimiyor ve cat'i %46.9 yerine %33.3 sayiyor.",
    "SINIR": "Bu bir ISARETTIR, kanit degil. docs/79 EK'teki 3 GERCEK cift (v27->v30->v46) blok CV'si de AYNI tezgahtan geliyor ve orada tasima orani 0.57-0.97 olculmustu -- yani tezgah tamamen kor degil. Ayrica v27/v30 ESKI (3/1/1) harman doneminden; tezgahin esit harmani o donemde uretime DAHA YAKINDI.",
    "YENI_SORU": "Tezgah-uretim uyusmazligi 2026-08-23'te (d04243f) DOGDU. O tarihten SONRAKI butun CV->LB karsilastirmalari bu sapmayi tasiyor. docs/79'daki 27 dosyalik tasima analizi iki donemi KARISTIRIYOR olabilir.",
}

R["5_NE_YAPILMALI"] = {
    "A_EN_HIZLI_DOGRU_OLCUM": "Mevcut onbellekler yeterli. soguk_tahmin_*.npz'de cat anahtarlari ZATEN var; p02_duzeltme.blok()'un soguk kolunu 'z tum anahtarlarin ortalamasi' yerine 'yalniz {tohum}_cat' yapmak, soguk tarafi URETIME hizalar. EGITIM GEREKTIRMEZ.",
    "B_SICAK_HIZALAMA": "aile_onbellek'te sinir_agi parcalari VAR (yaz25/guz25/kis26 x 1000-1002). Sicak kolu 3/1/1/1.4 agirligiyla kurmak da EGITIM GEREKTIRMEZ. experiments/sicak_kaldirac/ortak.py bunu zaten yapiyor -- oradan kopyalanabilir.",
    "C_SONRA": "Uretime hizalanmis tezgah kurulunca huber ve diger adaylar YENIDEN olculmelidir. Soguk tarafta anlamli soru artik 'lgbm huber mi' degil, 'CAT huber mi'dir -- ve p11_b_lgbm bu soruyu HIC sormadi.",
    "D_p18_DESTEGI": "p18_yeniden_egit.py aile bazinda uretiyor; harman asagi akista kuruluyor. Harman degisikligi p18'i etkilemez, YALNIZ p18_delta_vs_tam.py'nin harman kurulusu uretime hizalanmalidir.",
}

R["6_DURUSTLUK_NOTLARI"] = [
    "Ampirik regresyon SON ISLEM'i afin varsayiyor. Kova-merkezli varyant hucre sabitlerini temizliyor ama son islem tam olarak afin degilse paylar hafif kayabilir. Yine de cat 1.0 / xgb -0.10 / lgbm -0.02 gibi bir desen 'esit harman'la (0.33 her biri) UZLASMAZ.",
    "p06 dizisi tohum 1000-1002 ile uretildi; uretim gonderimleri baska tohumlar kullandi. Tohum gurultusu sd 0.00136 -- paylari 0.33'ten 1.00'a tasiyacak buyuklukte DEGIL.",
    "YP_seviye ve m6_ikiyon gonderim-uzayi cebiriyle kurulmus dosyalar; regresyon yine de net cat-tekil desen veriyor, cunku cebir soguk satirlarda kucuk bir duzeltme.",
    "SICAK harman ampirik olarak DOGRULANMADI (test tarafinda sicak aile dizisi yok). Hukum yalnizca kaynak koda dayaniyor.",
]

with open(YOL, "w", encoding="utf-8") as fh:
    json.dump(R, fh, ensure_ascii=False, indent=1)
print("yazildi:", YOL)
