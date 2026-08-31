"""p25-F: kirmizi takim hukmunu p25_kirmizi.json'a yaz."""

import json
import os

BURA = os.path.dirname(os.path.abspath(__file__))
yol = os.path.join(BURA, "p_kalici", "p25_kirmizi.json")
with open(yol, encoding="utf-8") as fh:
    R = json.load(fh)

R["HUKUM"] = {
    "hat1_afin_kopru": {
        "durum": "KIRILAMADI",
        "kanit": (
            "s bes bagimsiz yontemle tutarli: kova-3kolon 0.8132, trafo-ici-3kolon "
            "0.8097, trafolar-arasi-3kolon 0.8093, trafo-ici tek-cat 0.8004 "
            "(altorneklem GA95 [0.795, 0.805]). Kova basina egim 0.72-0.90 "
            "(cogunluk 0.77-0.80) -- afinlikten sapma +-%8, ama olcek duyarliligi "
            "dusuk: kazanc(m) egrisi m=0.7'de bile ORT +0.065 (tam olcegin %75'i), "
            "m yanlisligi kazanci yok etmiyor. dd'nin %83'u trafolar-ARASI ve "
            "trafolar-arasi s (0.78-0.81) ayni bantta. Buzme ort'u: delta log-"
            "ortalamasi tam 0 oldugu icin dongusel bagimlilik YOK."
        ),
    },
    "hat2_tohum_kumesi": {
        "durum": "KIRILAMADI",
        "kanit": (
            "yaz25/guz25'te 5 tohumla agr kazanc 0.1148/0.1366 vs 3 tohumla "
            "0.1058/0.1386 -- isaret ve buyukluk ayni. Capraz (taban 5 tohum, "
            "delta 3 tohum) 0.1062/0.1380 -- degismiyor. dd'nin tohum gurultusu "
            "satir basina sd 0.04-0.055; MSE maliyeti s^2*sd^2 ~ 0.0017 "
            "(kazancin %2'si). Uretim tohumlari (100+) farkli ama kopru "
            "regresyonu bunu s icinde emiyor."
        ),
    },
    "hat3_p06_dizisi": {
        "durum": "KIRILAMADI",
        "kanit": (
            "p06_soguk_test.py uretim soguk ayarlariyla birebir: maske 1.00, cat "
            "depth 7, ana-blok egitim (= ek_koken:False'un dar_egitim'i; "
            "deney.cerceveleri_kur ek koken ICERMIYOR), ayni oznitelik kurucusu "
            "(tm.*). b_cat=1.115 sapmasi TRAFOLAR-ARASI bir artefakt: trafo-ici "
            "paylar (1.042, -0.047, +0.005) ~ (1,0,0), yani dosya soguk satirlarda "
            "gercekten cat-tekil afin + trafo-duzeyi katmanlar. Trafo-ici R2 0.853; "
            "kalan %15 tohum farki + m112/m148 katmanlari, kopru bunlari emiyor."
        ),
    },
    "hat4_kohort_agirligi": {
        "durum": "KIRILAMADI",
        "kanit": (
            "+0.0867 kohort AGIRLIKLI (agr). Ham 0.1131, pg 0.1167 -- agr UCUNUN "
            "EN KUCUGU, yani agirlik kazanci sisirmiyor, kucultuyor. Dikkat: ham "
            "olcutte kis26 NEGATIF (-0.0256), '3/3' yalnizca agr/pg'de."
        ),
    },
    "hat5_merkezleme": {
        "durum": "KIRILDI",
        "kanit": (
            "V1 deltasi blokta AGIRLIKSIZ merkezlendi ama olcut TEST-AGIRLIKLI. "
            "Test-benzeri kohortlar blokta +0.025/+0.019/+0.053 kacak seviye aldi; "
            "gercek teste delta ortalamasi TEST dagiliminda 0 oldugu icin bu seviye "
            "TASINMAZ. Dogru taklit (agirlikli merkez): ORT +0.0867 -> +0.0712 "
            "(-%18) ve kis26 -0.0183'e DONUYOR -> isaret 3/3 degil 2/3. Kirpma "
            "merdiveninde kis26 her K'da negatif (-0.018..-0.027). Dosyanin "
            "KENDISI dogru (testte tek secenek genel merkez); yanlis olan CV "
            "BEKLENTISI."
        ),
        "duzeltilmis_beklenti": {
            "ORT_dMSE_agr": 0.07119,
            "test_dMSE": 0.015776,
            "dRMSLE": 0.00788,
            "LB_oran1.0": 0.99327,
            "LB_oran0.5": 0.99721,
            "K25_agirlikli_LB_oran0.5": 1.00063,
            "3_sira_icin_gereken_oran": 0.71,
            "not": (
                "docs/81'deki oran formulu payda 0.00964 -> 0.00788 olmali; "
                "3. sira esigi oran 0.58 degil 0.71."
            ),
        },
    },
    "hat6_olu_trafo": {
        "durum": "KIRILAMADI",
        "kanit": (
            "p21 = p20 uzerinde 15.407 satirda DOGRUDAN x0.5 (maxabs 1.3e-15), "
            "depodaki bayat p08_olu_delta_log_c050.npy KULLANILMAMIS (o D1_demet "
            "tabanindan, YP tabaninda 0.082'ye kadar sapardi). p08 satirlari ile "
            "soguk satirlarin kesisimi 0 -- etkilesim yapisal olarak sifir. "
            "Etkilenen tahminler zaten kucuk (medyan 0.22)."
        ),
    },
    "hat7_en_kotu_senaryo": {
        "beklenti_bandi_DUZELTILMIS": "0.9933 (oran 1.0, kirpmasiz) .. 1.0006 (oran 0.5, K25 agirlikli)",
        "onceki_iddia": "0.9915 .. 0.9998",
        "kis26_gibi_tasinirsa": (
            "agirlikli kis26 -0.018..-0.027 -> LB +0.0020..+0.0030 "
            "KOTULESIR (~1.0032'ye kadar). Test yaz-analogu oldugu "
            "icin dusuk olasilik; son secim sigortasi (YP_seviye "
            "1.00115) zarari sifirlar."
        ),
        "p08_katmani": "-0.0005..-0.0009, bagimsiz, riski yok",
    },
    "NET_HUKUM": (
        "p21_harman311_olu50.csv YARIN HAK 1 OLARAK GONDERILEBILIR -- dosya mekanik "
        "olarak dogru, kopru saglam, tohum/konfigurasyon tutarli, olu-trafo katmani "
        "temiz. AMA BEKLENTI DUSURULMELI: dogru merkezlemeyle CV kazanci +0.0712 "
        "(9/9 tohum kapisi ve yaz/guz gucu ayakta, kis26 isareti DONDU -> kapi (c) "
        "2/3), beklenen LB 0.9933 (oran 1.0) / 0.9972 (oran 0.5). 3. sirayi gecmek "
        "icin tasima orani >=0.71 gerekiyor (onceki iddia 0.58). Muhafazakar kosede "
        "(K25 + oran 0.5) kazanc ~sifir ama ZARAR da degil (1.0006). Asagi yonlu "
        "risk son secim sigortasiyla sinirli. DUZELTILMIS ADAY GEREKMIYOR: kusur "
        "dosyada degil OLCUMDE; testte genel merkezleme tek dogru secenek ve dosya "
        "zaten onu yapiyor."
    ),
}

with open(yol, "w", encoding="utf-8") as fh:
    json.dump(R, fh, ensure_ascii=False, indent=1)
print("yazildi:", yol)
print(json.dumps(R["HUKUM"]["NET_HUKUM"], ensure_ascii=False))
