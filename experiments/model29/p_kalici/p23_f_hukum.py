"""p23-F: ADIM 4-5 KARARI + NIHAI HUKUM -> p23_parti.json.

Adim 3 olcumu iddiayi BIZIM HATTA DOGRULAMADI (isaret ters ve kararsiz),
bu yuzden kayma adayi URETILMEDI. Gerekce ve muhasebe asagida.
"""

import json
import os

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
JSON_YOL = os.path.join(KOK, "experiments/model29/p_kalici/p23_parti.json")

with open(JSON_YOL, encoding="utf-8") as fh:
    R = json.load(fh)

R["adim4_kayma_adayi"] = {
    "karar": "ADAY URETILMEDI -- kazanc yok",
    "delta_zincir_p21_eksi_cat": -0.0667,
    "delta_zincir_aciklama": (
        "p21 zinciri parti-soguk satirlarini saf cat soguk uzmanina gore "
        "tarih-eslesmeli -0.067 EK asagi cekiyor (3/1/1 harmani: xgb/lgbm "
        "partiyi cat'ten cok fiyatliyor). Yani olculecek her kayma bunun "
        "USTUNE gelirdi; ama olculen kayma zaten negatif degil."
    ),
    "gerekce": [
        "P1 tarih-eslesmeli kopru-kontrol farki +0.209 (GA95 [-0.074, +0.453])"
        " -- iddianin TERSI isaret, sifiri kapsiyor",
        "P2 ayni olcum +0.046 (GA95 [-0.298, +0.344]) -- sifir; pencereye bagimli",
        "P2 capraz dogrulama B->A kaymayi -0.167 turetiyor ve A yarisinda "
        "dMSE -0.208 ile ANLAMLI ZARAR (GA95 [-0.306, -0.112]) -- yarimlar "
        "arasinda isaret donuyor, sabit parti-kaymasi diye bir sey yok, "
        "trafo-duzeyi heterojenlik var (p09 dersiyle ayni)",
        "dort-kapi kurali: (c) isaret tutarliligi 0/2 pencere, (d) GA95 "
        "sifiri kapsiyor -- aday kurulamaz",
        "kaba tabandaki -0.11 kaymayi p21'e uygulamak, bizim olcume gore "
        "ZARAR ederdi (bizim hatta isaret pozitif yonde)",
    ],
    "kiymik_notu": (
        "Negatif isaret yalniz 'kiymik' kopru alt-kohortunda gorunuyor "
        "(313 trafo, train'de sadece 2026-03-26/27 = 2 gunluk gecmis, 627 "
        "satir, fark -0.110). Bu dogum-gunu/kismi-olcum artefakti kalibi; "
        "kaba tabanin -0.11'lik kopru kaymasiyla birebir ortusmesi, o "
        "bulgunun bu artefakti olcmus olabilecegini dusunduruyor. Uzun "
        "gecmisli kopru (240 trafo, 5.888 satir) +0.243 ile TERS yonde."
    ),
}

R["adim5_durust_beklenti"] = {
    "beklenen_LB_etkisi": 0.0,
    "aciklama": (
        "Uygulanabilir kayma yok: iki pencerenin GA95'i de sifiri kapsiyor, "
        "isaret pencereler ve yarimlar arasinda donuyor. Soguk-parti satir "
        "payi %15.1 olsa da carpilacak kayma^2 terimi guvenilir bicimde "
        "sifirdan ayirt edilemiyor. Kaba tabandaki -0.0127 MSE beklentisi "
        "bizim hatta YOK."
    ),
    "senaryo_eger_kaba_kayma_uygulansaydi": {
        "kayma": -0.11,
        "not": (
            "bizim olcum (P1 +0.209) dogruysa, parti satirlarinda gercek "
            "artik pozitifken -0.11 kaydirmak satir ici MSE'yi "
            "2*(-0.11)*(+0.21) - 0.11^2 = -0.058 degistirir; test toplaminda "
            "0.151 * -0.058 = -0.0088 MSE = LB'de yaklasik +0.0044 KOTULESME"
        ),
    },
    "fiyatlanmislik_ozeti": (
        "Iddianin -0.38..-0.50'lik ham farki gercek olabilir ama uretim "
        "zaten fiyatliyor: p21 ayni kVA x ilce hucresinde parti-soguga "
        "-0.206 daha dusuk tahmin veriyor (cat ailesi -0.088, xgb -0.136, "
        "lgbm -0.173; zincir katmanlari ekstra -0.067). Kalan acik, kopru "
        "olcumune gore sifirdan ayirt edilemiyor."
    ),
}

R["hukum"] = (
    "Parti GERCEK (2.222 trafo 2026-05-11, 1.326 soguk / 896 kopru, sayilar "
    "iddiayla birebir) ama kaba tabandaki seviye-kaymasi kazanci BIZIM "
    "URETIM HATTINDA YOK. Uretim soguk uzmani + p21 zinciri parti "
    "dusukluqunun olculebilir kismini zaten fiyatliyor; kopru uzerinde "
    "kontrol-dusulmus artik kayma pozitif yonde, kararsiz ve GA95 sifiri "
    "kapsiyor. p23 aday CSV'si uretilmedi; p21_harman311_olu50.csv bu "
    "eksenden degismeden kalmali."
)

with open(JSON_YOL, "w", encoding="utf-8") as fh:
    json.dump(R, fh, indent=1, ensure_ascii=False)
print("yazildi:", JSON_YOL)
