"""p06 NIHAI OZET -- soguk segment calismasinin butun olcumlerini tek dosyada topla."""

import json
import os

BURA = os.path.dirname(os.path.abspath(__file__))


def oku(ad):
    y = os.path.join(BURA, ad)
    if not os.path.exists(y):
        return None
    with open(y, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    T = oku("p06_soguk_tani.json")
    A = oku("p06_soguk_anatomi.json")
    Q = oku("p06_soguk_q.json")
    D = oku("p06_soguk_dogal.json")
    Rm = oku("p06_soguk_rampa.json")
    Y = oku("p06_soguk_yeniden.json")
    F = oku("p06_soguk_aile.json")
    F2 = oku("p06_soguk_aile2.json")
    W = oku("p06_soguk_agirlik.json")
    Af = oku("p06_soguk_afin.json")
    Te = oku("p06_soguk_test.json")

    R = {
        "gorev": "yaz25 soguk RMSLE 1,43592 -> <=1,4072 (toplam test-bilesimi kazanci >= 0,00579)",
        "taban": {
            "yaz25_soguk": 1.43592,
            "yaz25_sicak": 0.80319,
            "test_bilesimi": 0.97932,
            "LB": 1.00115,
        },
        "SONUC": {
            "bulunan_kazanc": "SOGUK harmanda AILE AGIRLIGI",
            "agirlik_cat_xgb_lgbm": [0.05, 0.35, 0.60],
            "secim_yeri": "guz25 soguk satirlari (tam izgara argmin) -- yaz25 hedefi KULLANILMADI",
            "yaz25_soguk": 1.41269,
            "yaz25_soguk_kazanc": 0.02323,
            "test_bilesimi": 0.9718,
            "test_bilesimi_kazanc": 0.00752,
            "esik_gecildi_mi": True,
            "beklenen_LB": 0.9935,
            "beklenen_LB_gerekce": "1,00115 x (0,97180 / 0,97932); yaz25 test-bilesimi ile LB "
            "arasindaki carpansal orani sabit varsayar",
            "onyukleme": {
                "ort": 0.02312,
                "std": 0.01086,
                "ci95": [0.00282, 0.04423],
                "pozitif_oran": 0.984,
            },
            "RISK": "kis26 soguk satirlari TERS yonu soyluyor (esit 1,90615 -> 1,94/1,97). "
            "Yani karar, test'in mevsimce yaz25+guz25'e kis26'dan daha yakin oldugu "
            "varsayimina dayanir (Kural 10 zaten bunu soyluyor). SICAK tarafta aile "
            "siralamasi UC BLOKTA DA ayni (xgb en iyi) -- yani bu bir blok artifakti "
            "degil, soguga OZGU bir kararsizlik.",
        },
        "olculen_fikirler": {
            "1_komsu_seviyesi": {
                "durum": "REDDEDILDI",
                "post_hoc_afin": T and T.get("oracle_afin"),
                "uretim_hattinda_yeniden_egitim": Y,
                "yorum": "Uretim cercevesinde g_ilce_log_ort / g_ilce_kova_ort / g_kova_log_ort / "
                "tanim_num ZATEN var; komsu capasi ek bilgi tasimiyor. Kahin afin "
                "sinavda capa katsayisi -0,027 (etkisiz). Ozellik olarak modele "
                "konunca soguk RMSLE 1,40520 -> 1,54703 (-0,142): capanin ozet "
                "penceresi bloklar arasinda 90/212/334 gun oldugu icin dagilim kayiyor.",
            },
            "2_rampa_egrisi": {
                "durum": "REDDEDILDI",
                "yorum": "kendi_gun = ufuk_gun - p_ilk_ofset kesitinde yanlilik SEKLI bloklar "
                "arasinda ISARET DEGISTIRIYOR (yaz25 1. gun -0,144; kis26 +0,345). "
                "Dis-blok ofseti uygulandiginda yaz25 soguk 1,43592 -> 1,43615/1,43813.",
                "olcum": Rm and {k: Rm[k] for k in ("duzeltme",) if k in Rm},
            },
            "3_toplu_giris": {
                "durum": "TEZGAHTA KARSILIGI YOK",
                "yorum": "Test'te 2026-05-11'de 1.326 trafo giriyor ve test soguk satirlarinin "
                "%93,2'si 61-100 gunluk pencereye sahip. yaz25'te en kalabalik giris "
                "gunu 2025-07-28 (177 trafo) ve blogun SONUNDA -- yani test'in baskin "
                "kohortunun yaz25'te dengi YOK. Bu, butun soguk olcumleri icin bir "
                "BILESIM UYARISIdir: 61-100 gun kovasinda ayri olculdu, ayni sonuc "
                "cikti (yaz25 esit 1,58661 -> 1,55826).",
            },
            "4_tahmin_penceresi_oznitelikleri": {
                "durum": "ZATEN URETIMDE",
                "yorum": "p_gun_sayisi, p_ilk_ofset, p_son_ofset, p_yayilma, p_doluluk, "
                "p_pencere_payi, ufuk_gun uretim cercevesinde MEVCUT (egitim.parquet). "
                "p02'nin w_gun/w_ilk/w_yayilma degiskenlerinin karsiligi bunlar.",
            },
            "5_daha_iyi_seviye_capasi": {
                "durum": "REDDEDILDI",
                "dogal_soguk_uzman": D and D.get("yaz25"),
                "afin_kalibrasyon": Af and Af.get("yaz25_uygulama"),
                "yorum": "Dogal (yapay maskelenmemis) soguk satirlarda egitilmis uzman yaz25 "
                "soguk 1,88648 -- uretimden cok kotu; dis-blokta secilen harman "
                "agirligi (w=0,35) yaz25'te 1,53354'e goturuyor. Soguga ozel afin "
                "kalibrasyonun EGIMI de bloklar arasi kararsiz (1,15 / 1,08 / 0,65).",
            },
            "6_soguga_ozel_sifir_siniflandirici": {
                "durum": "REDDEDILDI",
                "gerekce": "Soguk satirlarda sifirlar satirlarin %4'u ama KARE HATANIN %57,3'u "
                "(kahin: 1,43592 -> 0,93874). Uretimdeki P0 soguk tarafta anma=0 "
                "veriyor cunku t_* kolonlarina dayaniyor. Soguga OZEL bir q modeli "
                "(t_* haric 112 ozellik, guz25+kis26 soguk satirlarinda egitildi) "
                "AUC 0,58 (ic) / 0,61 (yaz25) -- ayirt edici degil. Kare hatanin "
                "%76,7'si en dusuk q kovasinda. Dis blokta secilen gamma = 0 "
                "(yani 'duzeltme yapma').",
                "olcum": Q
                and {k: Q.get(k) for k in ("auc_yaz25_soguk", "ic_secim", "duzeltme", "SECILEN")},
            },
        },
        "hata_anatomisi_soguk": A
        and {k: A.get(k) for k in ("sifir_kesiti", "tavanlar_sizintili", "trafo_sapmasi")},
        "aile_olcumleri": {"blok_bazli": F, "sicak_karsilastirmasi": F2, "agirlik_izgarasi": W},
        "test_ciktisi": Te,
        "betikler": [
            "p06_soguk_tani.py",
            "p06_soguk_anatomi.py",
            "p06_soguk_sifir.py",
            "p06_soguk_pencere.py",
            "p06_soguk_q.py",
            "p06_soguk_dogal.py",
            "p06_soguk_rampa.py",
            "p06_soguk_yeniden.py",
            "p06_soguk_aile.py",
            "p06_soguk_aile2.py",
            "p06_soguk_agirlik.py",
            "p06_soguk_afin.py",
            "p06_soguk_test.py",
            "p06_ozet.py",
        ],
        "sizinti_denetimi": [
            "yaz25 hedefi hicbir egitimde/ozellikte/kalibrasyonda kullanilmadi.",
            "Aile agirligi guz25 soguk satirlarinda tam izgara ile secildi (yaz25 hedefi yok).",
            "q modeli ve dogal-soguk uzmani guz25+kis26'da egitildi; trafo kumelerinin "
            "AYRIKLIGI assert ile denetlendi (yaz25-soguk bir trafonun Nis-Tem 2025'te satiri "
            "vardir, dolayisiyla guz25/kis26'da soguk olamaz).",
            "Agac sayisi / gamma / harman agirligi gibi tum ayarlar dis bloklarin kendi ic "
            "bolunmesinden (guz25 -> kis26) secildi.",
        ],
    }
    with open(os.path.join(BURA, "p06_soguk.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)
    print("yazildi p06_soguk.json")


if __name__ == "__main__":
    main()
