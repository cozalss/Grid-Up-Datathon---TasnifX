"""Tum p0* olcumlerini TEK sonuc dosyasinda birlestirir: p01_hata_teshisi.json"""

import json
import os

import numpy as np
import pandas as pd
from p02_duzeltme import blok, skor

BURA = os.path.dirname(os.path.abspath(__file__))
yaz = blok("yaz25")
t0, t0w = skor(yaz, yaz.p.values)
tk = float((yaz.r**2).sum())


def pay(m):
    return round(float((yaz.loc[m, "r"] ** 2).sum() / tk), 4)


sifir = yaz.tuketim <= 0
kotu = sifir & (yaz.p > 4)
dusuk = yaz.p < 4
gecay = yaz.ay >= 6

KAYNAK = [
    dict(
        sira=1,
        ad="ONGORULEMEYEN SIFIR (kesinti/kopma)",
        tanim="gercek tuketim=0 ama tahmin log1p>4",
        n=int(kotu.sum()),
        satir_payi=round(float(kotu.mean()), 4),
        kare_pay=pay(kotu),
        ortalama_yanlilik=round(float(yaz.loc[kotu, "r"].mean()), 3),
        not_="tum sifir satirlar %5,4 satir / %31,2 kare hata; ama %83'u zaten "
        "p<1 tahmin ediliyor ve toplam kare hatanin yalnizca %0,47'sini tasiyor. "
        "Zarar bu 1.212 satirda yogunlasiyor ve sifir-siniflandirici (AUC 0,983) "
        "onlara q medyani 0,025 veriyor -- yani ONGORULEMEZ.",
        tavan_rmsle_kazanci=0.14775,
        olculmus_en_iyi_duzeltme=-0.00936,
    ),
    dict(
        sira=2,
        ad="MEVSIMSEL/UFUK KAYMASI (Haziran-Temmuz)",
        tanim="ay>=6 (ufkun son 2 ayi)",
        n=int(gecay.sum()),
        satir_payi=round(float(gecay.mean()), 4),
        kare_pay=pay(gecay),
        ortalama_yanlilik=round(float(yaz.loc[gecay, "r"].mean()), 3),
        saf_yanlilik_kare_payi=0.0713,
        aylik_yanlilik={"4": -0.045, "5": -0.052, "6": 0.199, "7": 0.399},
        not_="Temmuz yanliligi +0,399, Nisan -0,045. Model yaz yukunu SISTEMATIK "
        "olarak AZ tahmin ediyor. UYARI: geri-test modeli guz25+kis26 ile egitildi, "
        "yani HIC YAZ GORMEDI; uretim modeli yaz25'i goruyor. Bu kaynak geri-testte "
        "ABARTILI. guz25/kis26'da ufuk kaymasi ters isaretli -- dis-blok duzeltmesi "
        "tasinmiyor.",
        tavan_rmsle_kazanci=0.03333,
        olculmus_en_iyi_duzeltme=-0.00316,
    ),
    dict(
        sira=3,
        ad="TRAFO SEVIYE KAYMASI",
        tanim="trafo bazinda sabit artik ortalamasi",
        kare_pay=0.5566,
        kare_pay_sifirsiz=0.6824,
        yuzde50_kare_hata_trafo_sayisi=48,
        yuzde50_trafo_payi=0.0166,
        yuzde80_kare_hata_trafo_sayisi=310,
        kaliciligi_korelasyon=0.5058,
        not_="SISTEMATIK: ilk 30 gunun trafo kaymasi ile son 92 gunun kaymasi "
        "rho=0,51. Ama tahmin aninda kestirilemiyor: guz25/kis26 trafo kaymalarini "
        "buzerek tasimak yaz25'te 0,010-0,137 KAYBETTIRIYOR.",
        tavan_rmsle_kazanci=0.2896,
        olculmus_en_iyi_duzeltme=-0.00961,
    ),
]

OLCULEN = {}
for f in ("p02_duzeltme.json", "p04_sifir.json", "p07_seviye.json", "p08_dusuk_tahmin.json"):
    yol = os.path.join(BURA, f)
    if os.path.exists(yol):
        j = json.load(open(yol, encoding="utf-8"))
        for k, v in j.get("duzeltmeler", {}).items():
            OLCULEN[k] = dict(kazanc=v["kazanc"], kazanc_test_bilesimi=v.get("kazanc_test_bilesimi"))

R = dict(
    ozet=dict(
        blok="yaz25 (2025-04-01..07-31), test ufkunun mevsimsel ikizi",
        n=int(len(yaz)),
        n_trafo=int(yaz.tanim.nunique()),
        taban_rmsle=round(t0, 5),
        taban_rmsle_test_bilesimi=round(t0w, 5),
        hukum="Hata SISTEMATIK (kare hatanin %55,7'si trafo bazinda sabit kayma) "
        "ama TAHMIN ANINDA KESTIRILEMEZ. Denenen 20 duzeltmenin 19'u yaz25'te "
        "RMSLE'yi KOTULESTIRDI. Tek pozitif: dis-bloktan afin seviye kalibrasyonu "
        "+0,00127 (test bilesiminde +0,00062) -- 0,001'in ALTINDA, sisirilecek "
        "bir kazanc degil. yaz25'in kendi optimum buzme katsayisi beta=-0,03, "
        "LB'de olculen rho=-0,0304 ile birebir ayni: o yon ZATEN kullanilmis.",
    ),
    en_buyuk_uc_kaynak=KAYNAK,
    olculen_duzeltmeler=OLCULEN,
    ek_kesitler=dict(
        dusuk_tahmin_p_kucuk_4=dict(
            satir_payi=round(float(dusuk.mean()), 4),
            kare_pay=pay(dusuk),
            yanlilik=round(float(yaz.loc[dusuk, "r"].mean()), 3),
        ),
        soguk_trafolar=dict(satir_payi=0.075, kare_pay=0.2059, yanlilik=0.1334, rmsle=1.4359),
        yeni_trafo_yas_30gun_alti=dict(satir_payi=0.043, kare_pay=0.1448, yanlilik=0.0836),
    ),
    tavanlar_sizintili=dict(
        trafo_sabiti=0.2896, sifirlari_mukemmel_bil=0.14775, ufuk_sabiti=0.03333,
        ay_sabiti=0.03213, gun_sabiti=0.03471, kova_ofseti_p=0.01783, buzme_beta=0.00307,
    ),
    uyarilar=[
        "Geri-test modeli yaz25 blogunu HIC GORMEDI (blok_parcalari: hedef blok "
        "egitimden cikarilir). Uretim modeli test icin yaz25'i goruyor -> mevsimsel "
        "kayma gercek testte daha kucuk olmali.",
        "yaz25'te soguk trafo payi %7,5, gercek testte %22,2. Agirliksiz RMSLE "
        "karsilastirmalari icin test-bilesimi agirlikli sutun da verildi.",
        "yaz25'ten ONCE veri yok (ham train 2025-01-01 baslar), bu yuzden gercek "
        "testte kullanilabilecek 'onceki yil ayni mevsim' bilgisi geri-testte YOK.",
    ],
    betikler=["p01_hata_teshisi.py", "p02_duzeltme.py", "p03_bloklar.py", "p04_sifir.py",
              "p05_sifir_anatomi.py", "p06_sistematik.py", "p07_seviye.py", "p08_dusuk_tahmin.py"],
)
for f in ("p01_hata_teshisi.json",):
    pass
eski = os.path.join(BURA, "p01_hata_teshisi.json")
if os.path.exists(eski):
    R["ayrintili_ayristirma"] = json.load(open(eski, encoding="utf-8"))
json.dump(R, open(eski, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps({k: R[k] for k in ("ozet", "en_buyuk_uc_kaynak", "olculen_duzeltmeler")}, indent=1, ensure_ascii=False))
