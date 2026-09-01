"""p05 ozet: tum p05/p03-uretim olcumlerini tek dosyada toplar ve teshisi yazar."""

import io
import json
import os

B = os.path.dirname(os.path.abspath(__file__))


def oku(ad):
    p = os.path.join(B, ad)
    return json.load(io.open(p, encoding="utf-8")) if os.path.exists(p) else None


ana = oku("p05_uretim_iki_asama.json")
p03u = oku("p03_uretim_iki_asama.json")
ikiz = oku("p05_ikiz_sinav.json")
tav = oku("p05_sifir_tavani.json")
tes = oku("p05_teshis.json")
ben = oku("p03_nihai.json")

T = ana["yaz25"]["uretim_tabani"]


def kaz(d):
    return {k: round(T[k] - d[k], 5) for k in T}


ozet = {
    "SONUC": "KAZANC YOK. Iki asamali ayristirma URETIM boru hattinda hicbir "
    "varyantta kazandirmadi; en iyi varyant bile test-bilesiminde "
    "-0.022 kaybettiriyor. GONDERILECEK DOSYA URETILMEDI.",
    "gereken_esik": {
        "gorevdeki_esik": 0.006,
        "olculen_en_iyi_varyant_kazanci_test_agirlikli": -0.02202,
    },
    "uretim_tabani_yaz25": T,
    "not_taban": "m148_demet_plani.py 95-113 ile ayni birlestirme; iki bagimsiz "
    "betikte (p03_uretim, p05_uretim) ayni sayi cikti: duz 0.86685, "
    "test-bilesimi 0.97961. LB'deki 1.00284 (m6_ikiyon, M0 tabani) "
    "ile tutarli -- ofset ~ +0.023.",
    "varyantlar_kazanc_test_agirlikli": {
        "(a) uretim x (1-P0)": -0.02351,
        "(b) siniflandirici x yalniz-pozitif huber regresyon": -0.14509,
        "(b) ayni, l2 amac (p03 olcumu)": -0.14971,
        "(c) (b) + ilce-ici idnum-komsu seviyesi": -0.13627,
        "(b2) harman w=0.25 uretim tabaniyla": -0.03919,
        "(d) delta-tasima (1-P0)*(pb + (ppos-pall))": -0.05045,
        "(d) delta-tasima yarim agirlik w=0.5": -0.02202,
        "(esik) P0>0.95 olan satirlari sifirla": -0.00616,
        "(esik) P0>0.99 olan satirlari sifirla (KAHIN esik secimi)": -0.00229,
    },
    "en_iyi_varyantin_kirilimi": {
        "ad": "(d) delta-tasima w=0.5",
        "sonuc": ana["yaz25"]["d_delta_carpim_w0.5"],
        "kazanc": kaz(ana["yaz25"]["d_delta_carpim_w0.5"]),
    },
    "TESHIS": {
        "1_ikiz_sinav_KARAR_VERICI": {
            "kurulus": "Ayni model sinifi (tek lgbm), ayni uretim oznitelikleri, ayni "
            "egitim bloklari: tek asamali vs iki asamali.",
            "tek_asamali_huber": ikiz["tek_asamali_huber"],
            "tek_asamali_l1": ikiz["tek_asamali_l1"],
            "iki_asamali_huber": ikiz["iki_asamali_huber"],
            "fikrin_ikiz_kazanci_vs_huber": ikiz["fikrin_ikiz_kazanci"],
            "fikrin_ikiz_kazanci_vs_l1": ikiz["fikrin_ikiz_kazanci_l1_tabana_gore"],
            "yorum": "Fikir TEK MODEL seviyesinde GERCEKTEN calisiyor: huber ikizine "
            "gore test-bilesiminde +0.0174, soguklarda +0.0735. Ama p03'un "
            "kendi tabani olan l1 ikizine gore -0.0151 KAYBETTIRIYOR. Daha "
            "onemlisi: en iyi TEK MODEL (l1, 1.10963 agirlikli) uretim "
            "demetinden (0.97961) 0.130 GERIDE. Fikrin tum ikiz kazanci "
            "(0.017) bu acigin sekizde biri. Yani fikir ZAYIF TABANI "
            "onaran bir yama; uretim hatti o hatayi zaten yapmiyor.",
        },
        "2_p03_tezgahi_ile_karsilastirma": {
            "tezgah_tek_asamali_l1": ben["kosular"]["A_tek_l1"]["ort"],
            "tezgah_iki_asamali_huber": ben["kosular"]["B_iki_huber"]["ort"],
            "uretim_hatti": T,
            "yorum": "Tezgahin iki-asamali SONUCU (sicak 0.8495, soguk 1.8627) hala "
            "uretim hattinin TABANINDAN kotu (sicak 0.8032, soguk 1.4359). "
            "Yani tezgahtaki +0.0224'luk kazanc, uretim hattinin cok "
            "asagisindaki bir noktadan yukari cikmaktir; uretim seviyesinde "
            "harcanacak bir hata birikimi kalmamis.",
        },
        "3_neden_carpan_kaybettiriyor": {
            "uretim_genel_sapma_yaz25": tes["bloklar"]["yaz25"]["genel_sapma"],
            "uretim_sifir_satirda_ortalama_tahmin": tes["bloklar"]["yaz25"][
                "sifir_satirda_ortalama_tahmin"
            ],
            "tezgah_sifir_satirda_ortalama_sapma": 0.8588,
            "yorum": "Uretim tahmini zaten E[log1p y] (sifirlar DAHIL) tahmin ediyor "
            "ve ortalama sapmasi -0.136, yani ZATEN ASAGI yanli. (1-P0) ile "
            "carpmak bu asagi yanliligi ~%5 daha buyutuyor: sifir olmayan "
            "%94.6 satirda kayip, sifirlarda kazanctan buyuk. Carpani "
            "yumusatmak da kurtarmiyor: (1-P0)^0.25 bile -0.0024.",
        },
        "4_sifir_ele_alisinin_TAVANI": {
            "kahin_gercek_sifirlari_sifirla": tav["kahin_sifirlari_sifirla"],
            "kahin_kazanc": tav["kahin_kazanc"],
            "siniflandirici_AUC": tav["P0_auc"],
            "en_iyi_KAHIN_esik_kazanci_agirlikli": max(
                r["kazanc_agirlikli"] for r in tav["esik_taramasi_KAHIN"]
            ),
            "yorum": "Sifir satirlari uretim hatasinin %31'i ve mukemmel bir sifir "
            "dedektoru test-bilesiminde 0.221 kazandirirdi. AMA elimizdeki "
            "siniflandirici AUC 0.988 / esik 0.99'da kesinlik 0.971 olmasina "
            "ragmen KAHIN esik secimiyle bile KAYBETTIRIYOR: log uzayinda "
            "yanlis sifirlanan bir gercek tuketici ~48 birim kare hata "
            "getirirken dogru sifirlanan bir satir ~4.3 birim kazandiriyor. "
            "Karli olmak icin kesinlik ~%99.5 uzeri gerekiyor. Bu, sifir "
            "yolunun kapali oldugunun sayisal ifadesi.",
        },
        "5_soguk_trafolar": {
            "uretim_soguk_rmsle": T["soguk"],
            "uretim_sicak_rmsle": T["sicak"],
            "soguk_pay_yaz25": ana["yaz25"]["soguk_pay"],
            "test_soguk_pay": 0.222,
            "en_iyi_varyantin_soguk_kazanci": kaz(ana["yaz25"]["d_delta_carpim_w0.5"])["soguk"],
            "P0_soguk_ortalama": p03u["P0"]["soguk_ortalama"],
            "P0_sicak_ortalama": p03u["P0"]["sicak_ortalama"],
            "gercek_sifir_orani_soguk": tav["sifir_orani_soguk"],
            "gercek_sifir_orani_sicak": tav["sifir_orani_sicak"],
            "yorum": "Soguklarda hicbir varyant kazandirmadi (hepsi negatif). "
            "Siniflandirici sogukta P0=0.0155 tahmin ediyor, gercek sifir "
            "orani 0.0400 -- gecmisi olmayan trafoda sifir olasiligi "
            "OGRENILEBILIR degil, model sistematik olarak dusuk tahmin "
            "ediyor. Ustelik carpani soguklarda uygulamak en cok orada "
            "kaybettiriyor (a: -0.054). Ilce-ici komsu seviyesi de "
            "soguklari duzeltmedi (c: -0.320).",
        },
    },
    "sizinti_kontrolu": ana["sizinti_kontrolu"],
    "betikler": [
        "p05_uretim_iki_asama.py",
        "p05_ikiz_sinav.py",
        "p05_sifir_tavani.py",
        "p05_teshis.py",
        "p03_uretim_iki_asama.py (kardes ajan, ayni taban)",
    ],
    "TAVSIYE": "Bu yolu KAPAT. Iki asamali ayristirma uretim hattina hicbir sey "
    "eklemiyor; kazanc tezgahin zayif tabanina ozgu. Elde 1.00115 "
    "bankada dururken bu fikirden uretilecek bir dosya LB'de "
    "yaklasik 1.005-1.15 arasi CIKARDI (test-bilesimi kaybi "
    "0.022-0.145 + ~0.023 ofset). GONDERILMEMELI.",
}

ana["OZET"] = ozet
json.dump(
    ana,
    io.open(os.path.join(B, "p05_uretim_iki_asama.json"), "w", encoding="utf-8"),
    indent=1,
    ensure_ascii=False,
)
print(json.dumps(ozet, indent=1, ensure_ascii=False))
