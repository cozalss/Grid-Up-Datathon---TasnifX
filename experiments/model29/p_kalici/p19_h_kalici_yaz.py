"""p19-H: kalici kaydi (experiments/model29/p_kalici/p19_soguk_cat.json) guncelle."""

import json
import os
import shutil

SP = os.path.dirname(os.path.abspath(__file__))
PK = (r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
      r"/experiments/model29/p_kalici")
KAY = os.path.join(PK, "p19_soguk_cat.json")

eski = json.load(open(KAY, encoding="utf-8"))
B = json.load(open(os.path.join(SP, "p19_b_olc.json"), encoding="utf-8"))
E = json.load(open(os.path.join(SP, "p19_e_311_lgbm_huber.json"), encoding="utf-8"))
F = json.load(open(os.path.join(SP, "p19_f_311_bilesim.json"), encoding="utf-8"))
G = json.load(open(os.path.join(SP, "p19_g_blokdisi.json"), encoding="utf-8"))

BLOK = ("yaz25", "guz25", "kis26")


def kz(a, b):
    v = B["kazanclar"].get(a, {}).get(b)
    if not v:
        return None
    return dict(nt=v["n_tohum"], ham=v["ham"], agr=v["agr"], pg=v["pg"],
                bil=v["bilesim_agr"], oy_agr=v["onyukleme_agr"]["pozitif_oran"],
                oy_pg=v["onyukleme_pg"]["pozitif_oran"],
                oy_agr_ga95=v["onyukleme_agr"]["ga95"],
                tohum_bazli_agr=v["tohum_bazli_agr"])


adaylar = sorted(B["kazanclar"])
tablo = {a: {b: kz(a, b) for b in BLOK if kz(a, b)} for a in adaylar}

d = dict(eski)
d["durum"] = "KAMPANYA KAPANDI -- 2026-08-31 22:45, butun olcumler bitti"

d["01_ENVANTER"] = {
    "kosan_isler": "p19_s1_A/B/C (asama-1 kayiplar + ilk hp) ve p19_s2_D/E/F (hp + tohum tekrari) HEPSI BITTI",
    "TABAN_dogrulama": ("p19_a --aday TABAN uretim npz'si ile BIREBIR: yaz25 t1000, yaz25 t1001, "
                        "guz25 t1000, kis26 t1000 -> maxabs = 0.000e+00. TEZGAH GECERLI."),
    "TABAN_tamamlama": ("yaz25 t1002, guz25 t1001/1002, kis26 t1001/1002 icin TABAN yeniden "
                        "egitilmedi; uretim npz'sinin '{tohum}_cat' anahtarindan dogrudan yazildi "
                        "(birebir ozdeslik dort noktada kanitlandigi icin mesru)."),
    "olculen_aday_sayisi": len(adaylar),
    "coken_adaylar": ("yaz25'te hp_l2r10 / hp_rs4 / hp_lr03_it400 CatBoostError 'bad allocation', "
                      "hp_mdl50 MemoryError. AYNI adaylar guz25 ve kis26'da sorunsuz kostu -> "
                      "makine bellek darligi, aday kusuru degil. hp_mdl50 hicbir blokta yok."),
    "tau_izgarasi": ("OLCULMEDI. On-kayittaki TAU_YAPISAL_UYARI tau'nun soguk tarafta yaz25 icin "
                     "ETKISIZ, guz25 icin yarim etkili -- yani BLOK-DISI DOGRULANAMAZ oldugunu "
                     "zaten saptamisti. Hukum tasiyamayacak olcume zaman harcanmadi."),
}

d["02_TAM_TABLO_uretim_olcutu_cat_tekil"] = {
    "olcut": ("kohort-agirlikli soguk RMSLE kazanci, pozitif = TABAN'dan IYI. "
              "pg = PG(75,90] kovasi; bil = test bilesimi (0.2216 soguk)."),
    "taban_seviye": {b: B["bloklar"][b]["aday_seviye"]["TABAN"] for b in BLOK},
    "kazanclar": tablo,
    "npz_cat_tohum_gurultusu_agr_std": {
        b: B["npz_cat_tohum_gurultusu"][b]["agr_std"] for b in BLOK},
    "blok_disi_secim": {
        h: dict(secilen=v["secilen"],
                hedefte_agr=(v["hedefte_kazanc"] or {}).get("agr"),
                hedefte_bil=(v["hedefte_kazanc"] or {}).get("bilesim_agr"),
                hedefte_oy=((v["hedefte_kazanc"] or {}).get("onyukleme_agr") or {}).get("pozitif_oran"))
        for h, v in B["blok_disi_secim"].items()},
}

d["03_HUKUM_soguk_cat"] = {
    "SORU": "Soguk cat'ta URETIM tabanini (depth=7, RMSE) gecen bir yapilandirma VAR MI?",
    "CEVAP": ("KAYIP FONKSIYONU tarafinda YOK. HIPERPARAMETRE tarafinda tek sartli aday: depth=8. "
              "Ama gercek uretim harmani artik 3/1/1 oldugu icin etkisi pratikte SIFIR."),
    "kayip_fonksiyonlari_KAPANDI": {
        "olculenler": "huber delta 0.2/0.5/1.0/2.0/4.0, l1 (MAE), Quantile(0.5), MAPE",
        "desen": ("HEPSI yaz25'te belirgin NEGATIF (agr -0.016 .. -0.113). kis26'da pozitif "
                  "gorunuyorlar (tek tohum) ama isaret yaz25 ile CELISKILI."),
        "l1_ve_quantile05_ozdes": "CatBoost'ta MAE ve Quantile(alpha=0.5) BIREBIR ayni cikti verdi (beklenen).",
        "blok_disi_kanit": ("yaz25 hedefinde dis iki blok huber_a40'i seciyor -> yaz25'te -0.0155 agr, "
                            "onyukleme P(+)=0.356. Durust secim KAYBEDIYOR."),
        "yaz25_neden_onemli": ("Test donemi 2026-04..07; yaz25 = 2025-04..07 teste MEVSIMSEL olarak "
                               "denk gelen tek bloktur ve butun kayip fonksiyonlari tam orada kaybettiriyor."),
    },
    "hp_derin8_TEK_HAYATTA_KALAN": {
        "3_tohum_x_3_blok": tablo["hp_derin8"],
        "seviye": {b: dict(TABAN=B["bloklar"][b]["aday_seviye"]["TABAN"],
                           derin8=B["bloklar"][b]["aday_seviye"]["hp_derin8"]) for b in BLOK},
        "KAPILAR": {
            "a_agirlikli_pozitif": "yaz25 +0.0397 EVET | guz25 +0.0086 EVET | kis26 -0.0116 HAYIR",
            "b_PG_pozitif": "yaz25 +0.0534 EVET | guz25 +0.0191 EVET | kis26 -0.0119 HAYIR",
            "c_en_az_2_blok": "GECTI (2/3)",
            "d_tohum_1->3": ("yaz25 +0.0498 -> +0.0397 AYAKTA (3/3 tohum pozitif); "
                             "guz25 +0.0191 -> +0.0086 ZAYIFLADI (t1001 -0.0019, 2/3); "
                             "kis26 3/3 tohum NEGATIF"),
            "e_onyukleme": ("yaz25 P(+)=1.000 GA95 [+0.027,+0.065] GECTI; "
                            "guz25 P(+)=0.800 GA95 [-0.010,+0.026] GECMEDI; kis26 P(+)=0.044 TERS"),
        },
        "hukum": ("SARTLI KAZANC. a,b,c gecti; e yalniz yaz25'te gecti. LEHTE: tek net blok yaz25 -- "
                  "teste mevsimsel olarak denk gelen blok. ALEYHTE: kis26 uc tohumda da negatif; "
                  "guz25 kazanci (0.0086) npz cat tohum gurultusu std'sinin (0.0136) ALTINDA."),
        "buyukluk": {
            "test_bilesimi_kazanc": {b: tablo["hp_derin8"][b]["bil"] for b in BLOK},
            "uc_blok_ort": 0.00377,
            "cat_TEKIL_harmanda_LB_beklentisi_tasima_0.5": 0.0019,
            "gereken_LB_kazanci": 0.00559,
        },
    },
    "digerleri": {
        "hp_derin6": "yaz25 -0.0209 -- ters yon; derinlik ARTISI dogru yon",
        "hp_bt10": ("bagging_temperature=1.0 yaz25'te TABAN ile BIREBIR AYNI (agr farki 0.00000) -- "
                    "CatBoost varsayilani zaten 1.0, parametre ETKISIZ. Devir belgesindeki "
                    "'muhtemelen parametre etkisiz' tahmini DOGRULANDI."),
        "hp_l2r10": "guz25 +0.0192 / kis26 -0.0143 -- isaret celiskili, yaz25 bellek yuzunden olculemedi",
        "hp_rs4": "guz25 -0.0006 / kis26 -0.0100 -- YOK",
        "hp_lr03_it400": "guz25 +0.0034 / kis26 -0.0109 -- YOK",
    },
}

d["04_SORU5_311_harmaninda_lgbm_huber"] = {
    "SORU": ("p20/p21 adayi soguk harmani 3/1/1'e (cat 0.6 / xgb 0.2 / lgbm 0.2) donduruyor. "
             "lgbm uyesini HUBER yapmak bilesik soguk RMSLE'yi ne yapar?"),
    "yontem": ("EGITIM YOK. B = 0.6*cat + 0.2*xgb + 0.2*lgbm(L2) vs C1 = ayni ama lgbm HUBER. "
               "Girdiler data/interim/deney/soguk_tahmin_{blok}.npz ve scratchpad p11b_*.npy. "
               "Ortak tohum kumesi, kohort-agirlikli olcut, trafo-kumeli onyukleme (500)."),
    "denetim": ("p11b_{blok}_{tohum}_TABAN.npy npz'nin '{tohum}_lgbm' anahtariyla birebir mi diye "
                "assert edildi (esik 1e-5) -- uc blokta da GECTI, yani p11b tezgahi npz ile ayni nesneyi uretiyor."),
    "SONUC_alpha_1.0": E["toplama"]["huber"],
    "SONUC_alpha_0.5": E["toplama"]["huber_a05"],
    "blok_detay": {a: {b: dict(nt=v["n_tohum"], agr=v["agr"], pg=v["pg"], bil=v["bilesim_agr"],
                               oy=v["onyukleme_agr"]["pozitif_oran"],
                               tohum_bazli_agr=v["tohum_bazli_agr"])
                       for b, v in bb.items()} for a, bb in E["kazanclar"].items()},
    "CEVAP": ("KISMEN GERI GELIYOR, AMA YETERSIZ. 3/1/1'de lgbm payi 0.2 (esit harmanda 1/3 idi), "
              "bu yuzden p14'un kazanci ~0.6 katina iniyor: a=1.0 -> bilesim ort +0.0037 (2/3 blok), "
              "a=0.5 -> +0.0060 (2/3 blok). ISARET DESENI DEGISMEDI: yaz25 hala NEGATIF "
              "(a=1.0 -0.0039, a=0.5 -0.0052) ve yaz25 teste mevsimsel olarak denk gelen bloktur. "
              "p14'un olctugu 'her eklenen tohum kazanci asagi ceker' yanliligi da gecerli "
              "(guz25 n=2, kis26 n=1-2 tohum -- sayilar SISTEMATIK IYIMSER)."),
}

d["05_EK_KESIF_311_uye_degisimleri"] = {
    "etiket": "KESIF -- p19 on-kayit DISI, ayni agirlikli rig ile",
    "yapilar": "B = 3/1/1 std (p21 adayi) | C1 = lgbm->huber(a=1.0) | C2 = cat->depth8 | C3 = ikisi birden",
    "toplama": F["toplama"],
    "blok_detay": {k: {b: dict(agr=v["agr"], pg=v["pg"], bil=v["bilesim_agr"],
                               oy=v["onyukleme_agr"]["pozitif_oran"])
                       for b, v in bb.items()} for k, bb in F["kazanclar"].items()},
    "C3_UYARISI": ("C3'un 3/3 blokta pozitif gorunmesi DORT YAPI ARASINDAN TAM VERIYLE SECILDI. "
                   "Ustelik C3 kis26'da pozitif cikiyor cunku C1 (+0.0081) ile C2 (-0.0052) "
                   "birbirini DENGELIYOR -- ortak bir mekanizma degil, aritmetik tesadduf."),
    "DURUST_BLOK_DISI_SECIM": G["secim"],
    "durust_sonuc": G["durust_blokdisi_bilesim"],
    "HUKUM": ("Her hedef blok icin secim YALNIZ dis iki bloga bakilarak yapilinca bilesim kazanci "
              "ort +0.00215, 2/3 blok pozitif, yaz25 NEGATIF (-0.0039). Tasima orani 0.5 ile LB "
              "karsiligi ~0.001 -- gereken 0.00559'un besde biri. p21 adayinin USTUNE ONERILMEZ."),
}

d["06_NET_HUKUM"] = {
    "1": ("SOGUK CAT'TA URETIM TABANINI GECEN BIR KAYIP FONKSIYONU YOKTUR. huber (5 alfa), MAE, "
          "Quantile(0.5), MAPE olculdu; hepsi teste mevsimsel denk gelen yaz25 blogunda belirgin "
          "kaybettiriyor. YOL KAPANDI."),
    "2": ("Hiperparametre tarafinda depth=8 SARTLI bir kazanctir: cat-tekil olcutte yaz25 +0.0397 agr "
          "(3/3 tohum, onyukleme P(+)=1.000), guz25 +0.0086 (gurultu tabaninin altinda), "
          "kis26 -0.0116 (3/3 tohum negatif). Uc blok ortalama test bilesimi kazanci +0.0038."),
    "3": ("ANCAK URETIM ARTIK CAT-TEKIL DEGIL. p20/p21 adayi soguk harmani 3/1/1 yapiyor, cat payi 0.6. "
          "depth=8'in 3/1/1 ICINDE olculmus etkisi: yaz25 +0.0064, guz25 +0.0004, kis26 -0.0052, "
          "ORT +0.0005 -- PRATIKTE SIFIR."),
    "4": ("Soru 5: 3/1/1'de lgbm->huber GERI GELIYOR ama kucuk (ort +0.0037 a=1.0 / +0.0060 a=0.5, "
          "2/3 blok, yaz25 NEGATIF). Dort yapi uzerinde durust blok-disi secimle ort +0.00215."),
    "5": ("KAMPANYA KAPANIYOR. p19'dan gonderime girecek bir degisiklik CIKMADI. "
          "p20/p21 adayi (p20_harman_ESKI_3_1_1_V1_seviyesiz.csv) oldugu gibi kalir."),
    "tek_sart": ("Baska hicbir sey bulunamaz ve 3. gonderim hakki bosta kalirsa 3/1/1 + lgbm-huber(a=0.5) "
                 "yuksek varyansli bir ek bahis olarak DUSUNULEBILIR; ama yaz25 negatifligi teste "
                 "mevsimsel olarak en yakin bloktaki ters isarettir ve bu ONERILMEZ."),
}

d["07_URETILEN_DOSYALAR"] = {
    "scratchpad_json": ["p19_b_olc.json", "p19_c_harman.json", "p19_d_kirmizi.json",
                        "p19_e_311_lgbm_huber.json", "p19_f_311_bilesim.json", "p19_g_blokdisi.json"],
    "scratchpad_npy": "p19_{blok}_{tohum}_{aday}.npy -- soguk satirlarin log tahminleri",
    "p_kalici_betikler": ["p19_a_soguk_cat.py", "p19_b_olc.py", "p19_c_harman.py", "p19_d_kirmizi.py",
                          "p19_e_311_lgbm_huber.py", "p19_f_311_bilesim.py", "p19_g_blokdisi.py"],
    "YASAK_UYULDU": ("Test tahmini uretilmedi, Kaggle gonderimi yapilmadi, submissions/ altina "
                     "yazilmadi, commit yapilmadi."),
}

json.dump(d, open(KAY, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
for f in ("p19_e_311_lgbm_huber.py", "p19_f_311_bilesim.py", "p19_g_blokdisi.py"):
    shutil.copy(os.path.join(SP, f), os.path.join(PK, f))
print("yazildi:", KAY)
print("kopyalanan betik: 3")
