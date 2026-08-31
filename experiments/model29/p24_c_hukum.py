"""p24-C: hukum ve kohort kararini p24_sicak_harman.json'a yazar."""

import json
import os

BURA = os.path.dirname(os.path.abspath(__file__))
YOL = os.path.join(BURA, "p_kalici", "p24_sicak_harman.json")

with open(YOL, encoding="utf-8") as fh:
    R = json.load(fh)

R["02_KOHORT_KARARI"] = {
    "karar": "AGR ana olcut, HAM da raporlandi",
    "gerekce": (
        "p24_kesif.json: kapsam 0.993-1.000, TV 0.14-0.16 -- soguktaki gibi "
        "bozuk DEGIL; ama pg(75,90] payi test %14.8 vs blok %1.4-3.5, tek "
        "eksende belirgin uyumsuzluk var. Agirliklarin ESS orani ~%24, "
        "w_max 73-94 -- agr olcutu gurultulu, bu yuzden iki olcut birden "
        "okundu; HUKUM ikisinde de ayni."
    ),
}

R["09_HUKUM"] = {
    "sonuc": "URETIM (3,1,1,1.4) AYAKTA KALDI -- soguktaki hatanin ikizi SICAKTA YOK",
    "madde": [
        "Hicbir 0-parametreli aday dort kapiyi gecmedi. Uc blok isareti 3/3 olan "
        "aday YOK; tohum 9/9 olan aday YOK; GA95'i sifiri dislayan aday YOK.",
        "Tek pozitif-ortalamali aday XGB_AGIR (1,3,1,1.4): agr +0.00217 "
        "(test +0.00169), ham +0.00017 (test +0.00013). Ama isaret 2/3 "
        "(kis26 agr NEGATIF -0.00743), tohum 4/9, GA95 [-0.00725,+0.01277] "
        "sifiri kapsiyor, P(+)=0.706. Kirpma merdiveninde K=25'te +0.0014'e "
        "eriyor. KANIT YETERSIZ -- gonderim hakki harcanacak guc yok.",
        "Yorumdaki 'sicakta xgb en iyi aile' iddiasi BU onbellekte TUTMUYOR: "
        "YALNIZ_XGB uretimden agr -0.00947 / ham -0.01323 KOTU (0/3 ham). O eski "
        "olcum ek_kokensiz ANA koldandi; aile_onbellegi.py:9-14 zaten 'aile "
        "siralamasi iki kolda TERS' diye uyariyordu. Soguktaki hatanin ikizi degil.",
        "sinir_agi'nin olculmemis 1.4 agirligi ZARARLI CIKMADI: SINIRSIZ (3,1,1,0) "
        "uretimden agr -0.00353 / ham -0.00219 KOTU; sinir_agi yaz25 (testin "
        "mevsimsel ikizi) ve kis26'da kazandiriyor, yalniz guz25'te zarar. Tek "
        "basina RMSLE'si kotu (guz25/kis26 ~0.915-0.917) ama harmanda cesitlilik "
        "katkisi gercek.",
        "YALNIZ_CAT da acik kotu (agr -0.02382, P(+)=0.056): sicakta cat-tekile "
        "indirgeme diye bir riziko da yok, harman gercekten calisiyor.",
    ],
    "test_adayi": "URETILMEDI -- uretim harmani en iyisi, degisiklik onerilmiyor",
    "cekince": (
        "onbellek 'yaklasik uretim' (docs/80 §8, maxabs 0.325); mutlak "
        "seviyeler yaklasik ama tum adaylar ayni onbellekten harmanlandigi "
        "icin karsilastirma gecerli"
    ),
}

with open(YOL, "w", encoding="utf-8") as fh:
    json.dump(R, fh, ensure_ascii=False, indent=1)
print("hukum yazildi:", YOL)
