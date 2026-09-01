"""p27-08: butun p27 olcumlerini tek kalici JSON'da birlestir."""
import json
import os

CIK = os.path.dirname(os.path.abspath(__file__))
HEDEF = (r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX/"
         r"experiments/model29/p_kalici/p27_ufuk_anatomi.json")


def yukle(ad):
    with open(os.path.join(CIK, ad), encoding="utf-8") as f:
        return json.load(f)


j1, j2 = yukle("p27_01.json"), yukle("p27_02.json")
j3, j4 = yukle("p27_03.json"), yukle("p27_04.json")
j5, j6, j7 = yukle("p27_05.json"), yukle("p27_06.json"), yukle("p27_07.json")

OUT = {
    "00_KUNYE": {
        "tarih": "2026-09-01",
        "kapsam": "ufuk ekseni / seviye kaymasi + hata anatomisi + sifir cebi",
        "tezgah": "URETIM harmani (sicak cat3/xgb1/lgbm1/sinir_agi1.4, soguk cat-tekil, "
                  "son islem beta=0.60). p27_ortak.py",
        "kohort_agirligi": "test bilesimi soguk %22.16",
        "cv_lb_olcegi": 0.93907,
        "betikler": ["p27_ortak.py", "p27_01_taban.py", "p27_02_anatomi.py",
                     "p27_03_ufuk.py", "p27_04_capa.py", "p27_05_sifir.py",
                     "p27_06_sifir2.py", "p27_07_teshis.py"],
    },

    "01_HUKUM": {
        "ufuk_ekseni": "KAPALI",
        "gunluk_seviye_capasi": "KAPALI",
        "sifir_cebi": "KAPALI (ulasilabilir kismi zaten kullaniliyor)",
        "ozet": (
            "Kahin seviye tavani BUYUK (+0.032 LB) ama TAMAMI blok yapaylagindan "
            "geliyor. 27/27 durust blok-disi tahminci NEGATIF. Sifir cebinin kahin "
            "degeri, oznitelklerden ULASILAMAYAN alt kumede yogunlasmis; "
            "ulasilabilen kismi uretim modeli ZATEN cozmus."),
        "tek_cumle": (
            "UFUK SACILIMI buyutur (indirgenemez); YANLILIGI ise blogun egitiminde "
            "BULUNMAYAN takvim aylari yaratir -- ve testte o eksiklik YOK."),
    },

    "02_ufuk_egrisi": j3["01_ufuk_egrisi"],
    "03_kahin_seviye_tavani": j3["02_kahin_seviye_tavani"],
    "04_tavan_ozeti": j3["03_tavan_ozeti"],
    "05_ufuk_transfer_BASARISIZ": j3["04_ufuk_transfer"],
    "06_global_sabit_transfer_BASARISIZ": j3["05_global_sabit_transfer"],
    "07_takvim_kapsami": j3["00_kapsam"],
    "08_takvim_ayi_yanliligi": j3["06_takvim_ayi"],
    "09_kis26_dogal_deneyi": j3["07_kis26_dogal_deney"],

    "10_gunluk_capa": {
        "gun_yanliligi_korelasyon": j4["01_gun_yanliligi"],
        "blokdisi_sinav_BASARISIZ": j4["02_blokdisi_sinav"],
        "gun_kahin_ayrisim": j4["03_gun_kahin_ayrisim"],
        "test_capa_verisi": j4["04_test_capa"],
    },

    "11_izo_egri": j1["03_izo_egri"],
    "12_izo_tablo": j1["04_izo_tablo"],
    "13_cepler": j2["B_cepler"],
    "14_trafo_yogunlasmasi": j2["C_trafo_yogunlasmasi"],
    "15_guc_hucreleri": j2["E_guc_hucreleri"],

    "16_sifir": {
        "anatomi": j5["01_sifir_anatomisi"],
        "kahin_ayrisim": j5["02_kahin_ayrisim"],
        "blokdisi_auc": j6["01_auc"],
        "buzme_sonucu_BASARISIZ": j6["03_tablo"],
        "olu_trafo_kimligi": j7["01_olu_kimligi"],
        "esik_merdiveni": j7["02_esik_merdiveni"],
        "TESHIS": (
            "Blok sifirlarinin %68-84'u 4 ay boyunca HIC uretmeyen trafolardan. "
            "Trafo duzeyi blok-disi AUC 0.92-0.97 (satirda 0.96-0.99) -- yani "
            "belgelerdeki 'AUC 0.58-0.61' YALNIZ SOGUK alt kumesi icin gecerli. "
            "AMA: uretim modeli bu trafolara zaten ort. log 2.2-3.9 tahmin "
            "veriyor (canli trafolarda 6.5-6.8). Dogru yakalananlarin kahin "
            "degeri yalnizca +0.0005..+0.0052; kahin +0.11..+0.25'in tamami "
            "SINIFLANDIRICININ KACIRDIGI trafolarda -- onlar da ayni "
            "oznitelklerden gorunmuyor. Ulasilabilir artik kazanc YOK."),
    },
}

with open(HEDEF, "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
print("yazildi:", HEDEF, os.path.getsize(HEDEF), "bayt")
