import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m10_ortak import JSON, json_yaz

json_yaz(
    "OZET",
    {
        "gorev": "SICAK (gecmisi olan) trafolarda 4 ay ileri SEVIYE kestirimi, satir duzeyinde RMSLE",
        "kesimler": [
            "2025-08-31",
            "2025-09-30",
            "2025-10-31",
            "2025-11-30",
            "(ek) 2025-03-31 takvim-ikizi",
            "(ek) 2025-12-31",
        ],
        "taban_son28_4kesim_ort": 0.8584,
        "nihai_4kesim_ort": 0.7996,
        "kazanc": 0.0588,
        "trafo_sabit_oracle_4kesim_ort": 0.4761,
        "takvim_ikizi_2025_03_31": {
            "son28": 0.8746,
            "nihai_mevsimsiz": 0.8601,
            "nihai_mevsim_lam1": 0.8051,
        },
        "eksen_hukumleri": {
            "1_pencere": "GERCEK. son-7-gun >> son-28. 4 kesim ort: son7 0.8303, son28 0.8584, son91 0.9270, tum 0.9496. Medyan HER ZAMAN ortalamadan kotu; trim ortalamaya esit/kotu. Ufuk arttikca siralamada degisim YOK (ay1..ay4 hepsinde son7 kazaniyor). Harman 0.75*son7+0.25*tum-gecmis ek 0.010-0.013.",
            "2_mevsimsellik": "EN BUYUK KALEM. Takvim-ayi profili (trafo-ici sapma) std 0.178. Mart-sonu seviyesine gore trafo-bazli kayma: Nis +0.01, May -0.02, Haz +0.30, Tem +0.63. Takvim ikizinde (kesim 2025-03-31 -> Nis-Tem 2025) global ay duzeltmesi -0.055 RMSLE, ilce x ay -0.070. Trafo-BAZLI mevsimsellik gurultu (Q1 2025 vs 2026 korelasyon 0.146) -> global/ilce profili kullan.",
            "3_trend": "GURULTU. son-90-gun egimi ekstrapolasyonu: en iyi lam 0-0.1, kazanc +-0.002; lam=0.5'te 0.03-0.10 KAYIP. Kullanma.",
            "4_susme_uyanma": "BUYUK. Sicak hata kutlesinin %13-34'u sifir-durumlu trafolardan. Gecmisi TAMAMEN sifir olan trafolarin %11.7'si uyaniyor; son-28-gun sifir olanlarin %33.9'u; son-7-gun sifir olanlarin %75.8'i. Optimal sabit (log1p uzayi) A=0.5-0.9, C=1.0-1.5, D=1.3-1.75. LOO kazanci ort +0.009 (6 kesimin 4'unde +, 2'sinde -).",
            "5_kisalik": "GURULTU. (guc,ilce) prior'una shrinkage: nsat>=4 icin optimal w=1.0 (yani hic shrink etme). Sadece nsat<4'te w=0.8 (7,706 satirda 0.034 kazanc = toplamda ihmal edilebilir). James-Stein YOK.",
            "6_gunluk_desen": "KUCUK AMA GERCEK. Global haftagunu etkisi ~0 (kazanc 0.0001). TRAFO-BAZLI haftagunu sapmasi x 0.5: 4/4 kesimde -0.003..-0.005 kazanc. Pazar -0.05, Persembe +0.03.",
        },
        "nihai_tarif": {
            "ly": "log1p(tuketim); tum kestirim log1p uzayinda, tahmin = expm1(p)",
            "ozellikler_trafo_bazli": {
                "k7": "kesimden onceki 7 gun ly ortalamasi",
                "ly_all": "tum gecmis ly ortalamasi",
                "s28": "trafonun KENDI son veri tarihinden geriye 28 gun ly ortalamasi",
                "maxt": "tum gecmiste max tuketim",
                "bosluk": "kesim - trafonun son veri tarihi (gun)",
                "smax7/smax28": "kendi son tarihinden geriye 7/28 gunde max tuketim",
                "dw": "trafo x haftagunu ortalama sapmasi (ly - trafo ly ortalamasi)",
            },
            "gruplar_sirayla": [
                "A: maxt<1 (tum gecmis sifir)",
                "B: bosluk>14",
                "C: smax28<1",
                "D: smax7<1",
                "E: digerleri",
            ],
            "tahmin": {
                "E": "0.75*k7 + 0.25*ly_all + 0.5*dw   (k7 yoksa ly_all, o da yoksa global ly ortalamasi)",
                "B": "0.70*s28 + 0.30*ly_all + 0.5*dw",
                "A": "sabit 0.6",
                "C": "sabit 1.1",
                "D": "sabit 1.3",
            },
            "mevsim_duzeltmesi_E_ve_B_icin": "p += lam * kayma[hedef_takvim_ayi], lam=0.75..1.0; A/C/D'ye UYGULAMA",
        },
        "mevsim_kaymalari_2026_nisan_temmuz": {
            "4": -0.1103,
            "5": -0.1424,
            "6": 0.2051,
            "7": 0.5289,
            "turetim": "kayma[M] = P[2025-M] + drift - 0.75*P[2026-03]; P = takvim-ayi profili (trafo-ici sapma ortalamasi, gecmisi>=120 gun ve max>0 trafolar), drift = ort(P[2026-m]-P[2025-m], m=1,2,3) = +0.1123",
            "dogrulama": "Ayni formul kesim 2025-03-31'de (-0.081,-0.113,+0.235,+0.559) ongordu; GERCEK artik ortalamalari (-0.032,-0.069,+0.252,+0.547) idi. Sekil dogru, ~+0.04 sabit fark (yil-ici buyume).",
            "uyari": "Yil-uzeri transfer OLCULEMEDI (veride tek yaz var). Kesitsel (yari-ornek) transfer olculdu: -0.055 (global) / -0.070 (ilce x ay) RMSLE.",
        },
        "test_bilesimi_2026_03_31": {
            "sicak_satir": 556319,
            "A_tum_sifir": 25566,
            "B_bayat": 35444,
            "C_son28_sifir": 3838,
            "D_son7_sifir": 1804,
            "E_normal": 489667,
        },
        "kritik_gozlem": "Sicak hata kutlesi asiri yogun: kesim 2025-10-31'de en kotu %1 satir kutlenin %67'si, en kotu 15 trafo (2,809 icinden) %43'u. Bunlarin cogu 'gecmisi tamamen sifir, hedefte uyanan' trafolar.",
    },
)
d = json.load(open(JSON, encoding="utf-8"))
print("JSON anahtarlari:", list(d.keys()))
print("dosya:", JSON, os.path.getsize(JSON), "bayt")
