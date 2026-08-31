"""p03: fark analizi nihai raporu."""
import json
import math
import os

B = os.path.dirname(os.path.abspath(__file__))


def oku(ad):
    p = os.path.join(B, ad)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


kesif = oku("p03_kesif.json")
taban = oku("p03_taban.json")
fikir = oku("p03_fikir.json")
dog = oku("p03_dogrula.json")
f2 = oku("p03_fikir2_kayit.json")
nih = oku("p03_nihai.json")
SOG = kesif["soguk"]["soguk_satir_orani"]

R = {}
R["ozet"] = {
    "soru": "LB farki (TasnifX 1.00115; 2. 0.99536; lider 0.98110) nereden geliyor",
    "gereken_kazanc": {"2_sira": 0.00579, "3_sira": 0.00559},
    "tezgah": (
        "SIZINTISIZ yaz25 geri-testi. EGITIM kesimi 2025-11-30 (gecmis "
        "2025-09-01..11-30 = 3 ay, hedef 2025-12-01..2026-03-31 = 4 ay). "
        "DEGERLENDIRME kesimi 2025-03-31 (gecmis 2025-01-01..03-31 = 3 ay, hedef "
        "2025-04-01..07-31 = yaz25, 4 ay). yaz25 hedefi ne etiket ne de ozellik "
        "olarak hicbir yerde kullanilmadi (egitim gecmisi 2025-09-01'de basliyor). "
        "idnum ozelligi ATILDI; degerlendirmede soguk olan trafolar egitimden "
        "cikarildi -> kimlik-ezberi kanali kapali."
    ),
    "betikler": [
        "p03_kesif.py", "p03_tezgah.py", "p03_taban.py", "p03_fikir.py",
        "p03_dogrula.py", "p03_fikir2.py", "p03_nihai.py", "p03_rapor.py",
    ],
}

R["1_dis_bilgi"] = {
    "kaggle_dosyalari": ["train.csv", "test.csv", "sample_submission.csv"],
    "public_kernel": "YOK (kaggle kernels list --competition -> Not found)",
    "yarisma_sayfasi": "404 -- in-class, oturum acmadan erisilemiyor",
    "web": (
        "Coderspace/GDZ/ADM duyurulari disinda acik bilgi yok; bu yarismanin "
        "notebooklari yarismadan SONRA paylasiliyor. Rakiplerin yaklasimi "
        "hakkinda ACIK BILGI YOK."
    ),
    "sonuc": "Disaridan ogrenilecek bir sey yok; fark veri/modelde aranmali.",
}

R["2_veri_yapisi"] = {
    "sutunlar": ["tanim", "guc", "tarih", "tuketim (hedef)", "lokasyon"],
    "kullanilmayan_yardimci_tablo": (
        "YOK. Ek trafo ustnitelik tablosu, musteri sayisi, hava, kesinti "
        "tablosu vb. yarismada VERILMIYOR."
    ),
    "hiyerarsi": (
        "lokasyon = il>bolge>ilce; 2 il, 20 bolge, 30 ilce, 47 essiz lokasyon. "
        "Testte YENI lokasyon YOK. Trafo basina lokasyon ve guc SABIT. "
        "TOPLAM-TUTARLILIGI KISITI YOK -- hicbir yerde il/ilce toplami "
        "verilmiyor, dolayisiyla hiyerarsi uzerinden ek KISIT cikmiyor; yalnizca "
        "grup ozelligi olarak degerli ve zaten kullaniliyor "
        "(m30_ozellik.grup_ozellik)."
    ),
    "guc": "41 essiz deger; testte 1 YENI deger (30930) -- tek basina onemsiz.",
    "panel": (
        "COK dengesiz. Egitimde trafolarin yalnizca %23,4'u 455 gunun hepsine "
        "sahip (medyan 170 gun). Testte %50,4'u 122 gunun hepsine sahip. Varlik "
        "deseni zaten ozellik (m30_ozellik.varlik_ozellik)."
    ),
    "EN_BUYUK_YAPISAL_GERCEK": (
        "TEST TRAFOLARININ 2024'U (7036'nin %28,8'i) EGITIMDE HIC GECMIYOR -> "
        "test SATIRLARININ %22,2'si tamamen SOGUK. yaz25 blogunda ayni oran "
        "yalnizca %7,5. Soguk satirlar geri-testte kare hatanin %30,4'unu "
        "olusturuyor; testte pay UC KAT daha buyuk."
    ),
    "sifirlar": (
        "Egitimde tuketim=0 orani %4,7 (yaz25'te %5,4). 298 trafo TAMAMEN sifir, "
        "254 trafo KISMEN sifir, 4792 trafonun hic sifiri yok. Aylik sifir orani "
        "mevsimlik: Ocak-Mayis ~%7,4, Temmuz ~%2,3."
    ),
}

R["3_rmsle_hata_ayristirmasi"] = {
    "kaynak": "p03_taban.json (yaz25, l2 taban, RMSLE 1,05327)",
    "tuketim_0_satirlar": {
        "satir_payi": 0.0542, "kare_hata_payi": 0.2692,
        "ortalama_sapma_log": 0.8588,
        "yorum": "satirlarin %5,4'u, hatanin %26,9'u; sistematik olarak FAZLA tahmin",
    },
    "tuketim_10_alti": {"satir_payi": 0.0645, "kare_hata_payi": 0.2983},
    "ust_kuyruk_1e5_ustu": {
        "satir_payi": 0.0027, "kare_hata_payi": 0.0487,
        "ortalama_sapma_log": -2.6686, "yorum": "ciddi EKSIK tahmin",
    },
    "soguk_vs_sicak": {
        "soguk_rmsle": 2.1196, "sicak_rmsle": 0.9137,
        "soguk_satir_payi_yaz25": 0.0750, "soguk_kare_hata_payi": 0.3039,
    },
}

F = []
if nih:
    A = nih["kosular"]["A_tek_l1"]
    Bk = nih["kosular"]["B_iki_huber"]
    C = nih["kosular"]["C_iki_huber_kom"]
    F.append({
        "sira": 1,
        "ad": "IKI ASAMALI AYRISTIRMA: P(tuketim>0) x E[log1p | tuketim>0]",
        "bir_cumle": (
            "Tek regresyon yerine bir ikili siniflandirici (sifir mi) ile YALNIZCA "
            "pozitif satirlarda egitilmis bir seviye regresyonunun CARPIMI; cunku "
            "RMSLE'nin optimal nokta tahmini kosullu ORTALAMA ve bu ortalama "
            "P(y>0)*E[log1p|y>0] olarak birebir ayrisiyor."
        ),
        "yaz25_taban_l1_tek_asamali": A["ort"]["rmsle"],
        "yaz25_rmsle": Bk["ort"]["rmsle"],
        "yaz25_kazanc": Bk["kazanc_vs_A"]["yaz25"],
        "test_bilesimine_agirlikli": {
            "aciklama": ("sqrt(0,778*RMSLE_sicak^2 + 0,222*RMSLE_soguk^2), testin "
                         "%22,2 soguk bilesimi"),
            "taban": A["ort"]["test_agirlikli"],
            "fikir": Bk["ort"]["test_agirlikli"],
            "kazanc": Bk["kazanc_vs_A"]["test_agirlikli"],
        },
        "kararlilik": {
            "tohumlar": [7, 17, 27],
            "taban_hepsi": [q["rmsle"] for q in A["tohumlar"]],
            "fikir_hepsi": [q["rmsle"] for q in Bk["tohumlar"]],
            "taban_std": A["std"]["rmsle"], "fikir_std": Bk["std"]["rmsle"],
        },
        "varyantlar_arasi": (f2["amac_iki_asamali"] if f2 else None),
        "tek_asamali_amac_karsilastirmasi": (f2["amac_tek_asamali"] if f2 else None),
        "iki_bilesen_de_sart": (
            "Yalnizca 'sifirsiz satirlarda egit' (carpansiz) 1,15708 -> l2 tabana "
            "gore -0,110 KAYBETTIRIYOR. Yalnizca '(1-P0) ile carp' (l2 taban "
            "uzerine) -0,0025 kaybettiriyor. Kazanc IKISININ CARPIMINDAN geliyor."
        ),
        "kazancin_kaynagi": (
            "Kazancin cogu sifir kovasindan DEGIL, UST kuyruktan: toplam kare "
            "hatanin (1e3,5e3] %4,2 + (5e3,1e5] %5,4 + >1e5 %1,8 kadari duzeliyor; "
            "sifir kovasindan yalnizca %3,5. Yani sifirlarin ORTAK regresyonu "
            "asagi cekmesi tum ust kuyrugu bozuyormus."
        ),
        "artik_duzeltmesi_tuzagi": (
            "HAYIR. Blok disinda ogrenilen bir SABIT tasinmiyor; model YAPISI "
            "degisiyor. Taban da fikir de ayni blok disi veriyle egitiliyor, "
            "karsilastirma adil."
        ),
        "uygulanabilirlik": (
            "m71/m30 hattina bir lgb ikili siniflandirici + pozitif alt kume "
            "regresyonu eklenerek kurulur. Egitim maliyeti ~2x; tam test tahmini "
            "dakikalar. Kapi denetimi etkilenmez -- carpim log-uzayinda negatif "
            "uretmez."
        ),
    })
    F.append({
        "sira": 2,
        "ad": "AMAC FONKSIYONU asama yapisiyla BIRLIKTE secilmeli",
        "bir_cumle": (
            "Tek asamali kurulumda log1p hedefi uzerinde l1 (0,99631) l2'yi "
            "(1,04672) 0,050 doverken huber (1,08008) en kotusu; iki asamali "
            "kurulumda sira TERSINE donuyor ve huber (0,96048) en iyisi oluyor."
        ),
        "yaz25_olculen": {
            "tek_asamali": (f2["amac_tek_asamali"] if f2 else None),
            "iki_asamali": (f2["amac_iki_asamali"] if f2 else None),
        },
        "yaz25_kazanc_iki_asamali_icinde_l2_den_huber_e": (
            (f2["amac_iki_asamali"]["l2"] - f2["amac_iki_asamali"]["huber2"])
            if f2 else None
        ),
        "yorum": (
            "Uretim hatti (m71) zaten huber+l1 harmani kullaniyor, yani tek "
            "asamalidaki dogru tarafta. Bu fikrin BAGIMSIZ kazanci kucuk; asil "
            "degeri, 1. fikri kurarken YANLIS amaci secmemek."
        ),
        "artik_duzeltmesi_tuzagi": "HAYIR -- egitim amaci degisiyor, sabit tasinmiyor.",
        "uygulanabilirlik": "Tek parametre; ek maliyet yok.",
    })
    F.append({
        "sira": 3,
        "ad": "AYNI ILCEDE idnum-KOMSU SEVIYESI (soguk trafo onceligi)",
        "bir_cumle": (
            "Her hedef trafo icin ayni ilcede idnum'a en yakin 8 komsunun gecmis "
            "ortalama log seviyesi ozellik olarak ekleniyor; gecmisi olmayan soguk "
            "trafolara (testte satirlarin %22,2'si) gercek bir on-bilgi veriyor."
        ),
        "yaz25_taban_l2": fikir["taban"],
        "yaz25_rmsle_l2_uzerine": fikir["f4_idnum_komsu"]["rmsle"],
        "yaz25_kazanc_l2_uzerine": fikir["f4_idnum_komsu"]["kazanc"],
        "soguk_satirlarda": {
            "once": fikir["f4_idnum_komsu"]["soguk_rmsle_once"],
            "sonra": fikir["f4_idnum_komsu"]["soguk_rmsle_sonra"],
        },
        "AMA_1_FIKIRLE_ORTUSUYOR": {
            "iki_asamali_huber_komsusuz": Bk["ort"]["rmsle"],
            "iki_asamali_huber_komsulu": C["ort"]["rmsle"],
            "ek_kazanc": Bk["ort"]["rmsle"] - C["ort"]["rmsle"],
            "ek_kazanc_test_agirlikli": (
                Bk["ort"]["test_agirlikli"] - C["ort"]["test_agirlikli"]
            ),
            "not": "Tek basina buyuk, 1. fikrin uzerine EK kazanci cok daha kucuk.",
        },
        "GERCEK_TESTE_TASINIR_MI": {
            "sonuc": "EVET",
            "kanit": dog["f4_kapsam"],
            "not": (
                "README'nin 'soguk cozulebilirlik TESTTE %0' uyarisi KIMLIK "
                "ezberine dair; komsu-SEVIYESI ozelligi farkli ve testte kapsami "
                "DAHA YUKSEK: yaz25 soguk satirlarin %66,0'inda, gercek testte "
                "soguk satirlarin %77,7'sinde hesaplanabiliyor."
            ),
        },
        "artik_duzeltmesi_tuzagi": "HAYIR -- yeni OZELLIK, blok disi sabit degil.",
        "uygulanabilirlik": (
            "p03_fikir_ortak.komsu_ozellik, ~15 sn; m30_ozellik.kur icine girer."
        ),
    })
R["4_UC_FIKIR"] = F

R["5_olculup_ELENEN"] = {
    "soguk_kalibrasyon_F2": fikir["f2_soguk_kalibrasyon"],
    "genel_kaydirma_F3": fikir["f3_genel_kaydirma"],
    "carpim_tek_basina_F1b": fikir["f1b_carpim"],
    "sifir_esigi_F1c": fikir["f1c_esik"],
    "not": (
        "F2 ve F3 tam olarak p01'in kapattigi ARTIK-DUZELTMESI yolu: blok "
        "disinda ogrenilen sabit kaydirma/buzme. Ikisi de yaz25'te KAYBETTIRDI "
        "(-0,0110 ve -0,0054). p01'in bulgusu bagimsiz olarak DOGRULANDI."
    ),
}

R["6_UYARI"] = (
    "Tezgahin MUTLAK seviyesi (taban ~1,00-1,05) LB seviyesinden (1,00115) "
    "yuksek; cunku gecmis 3 aya, egitim tek kesime kirpildi ve idnum atildi. "
    "Raporlanan KAZANCLAR farklardir; uretim hattinda ayni buyuklukte cikacaklari "
    "GARANTI DEGIL. Ancak 1. fikir bir kalibrasyon sabiti degil YAPISAL bir "
    "ayristirma oldugu icin, isaretinin blok degistirmesi p01'in gosterdigi "
    "kesit-yanliligi mekanizmasina tabi DEGIL."
)

json.dump(
    R, open(os.path.join(B, "p03_fark_analizi.json"), "w", encoding="utf-8"),
    indent=1, ensure_ascii=False,
)
print("yazildi")
