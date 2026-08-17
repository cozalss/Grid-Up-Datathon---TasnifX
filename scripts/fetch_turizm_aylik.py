"""KTB AYLIK konaklama bultenlerinden il x ay geceleme/gelis/doluluk tablosu.

NEDEN BU BETIK
--------------
``fetch_turizm.py`` YILLIK ilce gecelemesini verir: "Bodrum ne kadar
turistik" sorusunun kararli cevabi. Ama sebekeyi zorlayan sey yilin HANGI
AYINDA kac kisi oldugudur (docs/10 bolum 5: Mugla'da yaz nufusu yerlesigin
2-5 kati). Yillik sayi bunu tasimaz; aylik il serisi tasir. Ikisi birlikte
"ilce yillik payi x il aylik profili" ile ilce-ay tahmini verir
(``gridup.features.tourism.district_monthly_estimate``).

KAYNAK
------
yigm.ktb.gov.tr > Konaklama Istatistikleri > Onceki Donemler > Aylik
Bultenler > {yil}. Her ay tek xlsx; "Il" sayfasi 81 il + TOPLAM satiri,
kolonlar: tesise gelis / geceleme / ortalama kalis / doluluk, her biri
yabanci-yerli-toplam. Yapi 2019-01'den 2026-06'ya kadar AYNI (olculdu:
7 farkli yil/ay ornegi, 85 x 13 hucre, ayni baslik dizilimi).

UC TUZAK (olculdu, 2026-08-17)
------------------------------
1. **Kapsam degisikligi -- IKI KIRILMA, ikisi de olculdu.** Ortuk yatak
   kapasitesi = geceleme / (doluluk x gun) her il-ay icin hesaplandi
   (doluluk KTB'nin kendi paydasidir; kapasite dogrudan cikar):
     * **2022-09**: Turkiye 1,04M -> 1,24M yatak; Mugla 82k -> 120k.
       Bulten BASLIGI ancak 2022-11'de "Turizm Isletme Belgeli" -> "Isletme
       ve Basit Belgeli" oldu; veri kapsami basliktan IKI AY ONCE degisti.
     * **2025-07**: Turkiye 1,46M -> 1,73M; Mugla 133k -> 219k; Fethiye
       yillik gecelemesi 2,3M -> 5,7M. Baslik degismedi. Doluluk sabit
       kaldi (%13 -> %15): tesis sayisi artti, turist degil (turizm amacli
       konut kiralama izinlerinin bakanlik belgesine gecisiyle uyumlu).
   Iki kolon tasir: ``kapsam`` = BASLIKTAN okunan etiket (belgeleme
   dogrulugu icin), ``kapsam_rejimi`` = OLCULEN kirilmalara gore 1/2/3
   (modelleme icin dogru olan). Yillar arasi kiyasta rejimden bagimsiz
   tek olcu ``doluluk``tur; ``geceleme`` yalnizca ayni rejim icinde
   kiyaslanabilir.
2. **Antalya Agustos dolulugu %100'u asar** (2019, 2022, 2024: %101-102;
   ilave yatak). KTB'nin kendi artefakti, hedef illerde yok; kirpilmaz.
3. **URL slug'lari tutarsiz.** "s-bat", "06", "yeni", "v2" gibi ekler var;
   ay adini slug'dan cikarmak guvenilmez. Bu yuzden yil-ay -> URL haritasi
   SABIT kodlanmis ve her dosyanin ayi/yili dosyanin KENDI ICINDEKI
   "Gelis-Geceleme Ay/Yil" sayfalarindan dogrulanir. Uyusmazlik = hata.

CAPRAZ DOGRULAMA (olculdu, 90 dosya)
------------------------------------
* Her dosyada 81 il toplami = TOPLAM satiri (0 sapma).
* Ulusal aylik seri sonraki bultenlerde HIC revize edilmemis (82 donem,
  maks fark %0,0) -- ilk yayin nihai.
* 12 ayin toplami = yillik bultenin il toplami, 2023-2025, HER ILDE %0,00.
  Aylik ve yillik seri ayni kaynaktan, tutarli.

YAYIN GECIKMESI
---------------
Slug'lardaki tarihlerden olculdu: ay M'nin bulteni M+2'nin 3-11'i arasi
cikar (2026 Ocak istisna: 6 Nisan). Yani bir tarih icin GUVENLE elde olan
en son ay M-3'tur; M-2 yalnizca ayin ortasindan sonra. Feature katmani
bunu ``lag_months >= 2`` zorunlulugu ve varsayilan 12 ile ele alir.

Ham dosyalar data/external/ham/ altina indirilir ve SAKLANIR: KTB sunucusu
hizli ardisik istekte baglantiyi kesiyor (fetch_turizm.py'de olculdu) --
betik dosya zaten varsa indirmez, cevrimdisi calisir.

Kullanim::

    python scripts/fetch_turizm_aylik.py
    python scripts/fetch_turizm_aylik.py --yillar 2024 2025 2026
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gridup.io_utils import (  # noqa: E402
    publish_bytes,
    publish_dataframe,
    validate_cached_file,
)
from gridup.turkish import join_key  # noqa: E402

EKLENTI = "https://yigm.ktb.gov.tr/Eklenti/"

#: (yil, ay) -> Eklenti yolu. Yil sayfalarindan (TR-232592 ... TR-448570)
#: 2026-08-17'de curl ile cikarildi. 2019 ve 2021 Ekim'de iki kopya vardi;
#: seri halinde yuklenen (yuksek id) tercih edildi, icerik dogrulamasi
#: yanlis secimi zaten reddeder.
BULTENLER: dict[tuple[int, int], str] = {
    (2019, 1): "90670,ocak-2019-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2019, 2): "90669,subat-2019-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2019, 3): "90671,mart-2019-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2019, 4): "90674,nisan-2019-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2019, 5): "90673,mayis-2019-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2019, 6): "90668,haziran-2019-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2019, 7): "90677,temmuz-2019-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2019, 8): "90667,agustos-2019-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2019, 9): "90678,eylul-2019-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2019, 10): "90672,ekim-2019-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2019, 11): "90675,kasim-2019-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2019, 12): "90676,aralik-2019-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2020, 1): "90698,ocak-2020-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2020, 2): "90696,subat-2020-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2020, 3): "90695,mart-2020-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2020, 4): "90702,nisan-2020-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2020, 5): "90693,mayis-2020-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2020, 6): "90703,haziran-2020-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2020, 7): "90701,temmuz-2020-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2020, 8): "90692,agustos-2020-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2020, 9): "90700,eylul-2020-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2020, 10): "90691,ekim-2020-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2020, 11): "90697,kasim-2020-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2020, 12): "90694,aralik-2020-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2021, 1): "96272,ocak-2021-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2021, 2): "96273,subat-2021-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2021, 3): "96269,mart-2021-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2021, 4): "96271,nisan-2021-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2021, 5): "96270,mayis-2021-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2021, 6): "96267,haziran-2021-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2021, 7): "96274,temmuz-2021-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2021, 8): "96263,agustos-2021-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2021, 9): "96266,eylul-2021-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2021, 10): "96265,ekim-2021-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2021, 11): "96268,kasim-2021-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2021, 12): "96264,aralik-2021-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2022, 1): "102183,ocak-2022-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2022, 2): "100834,subat-2022-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2022, 3): "97128,mart-2022-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2022, 4): "98631,nisan-2022-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2022, 5): "99473,konaklama-aylik-bulten-2022--mayis-turizm-isletme-belgelixlsx.xlsx?0",
    (2022, 6): "101911,haziran-2022-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2022, 7): "102586,temmuz-2022-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2022, 8): "104231,agustos-2022-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2022, 9): "105082,eylul-2022-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2022, 10): "106455,ekim-2022-isletme-belgeli-aylik-bultenxlsx.xlsx?0",
    (2022, 11): "110912,kasim-2022-isletme-ve-basit-belgeli-aylik-bultenxlsx.xlsx?0",
    (2022, 12): "110913,aralik-2022-isletme-belgeli-ve-basit-belgeli-aylik-bultenxlsx.xlsx?0",
    (2023, 1): "111791,ocak-2023-turizm-isletme-ve-basit-belgeli-aylik-bultenxlsx.xlsx?0",
    (2023, 2): "112667,subat-2023-turizm-isletme-ve-basit-belgeli-aylik-bultenxlsx.xlsx?0",
    (2023, 3): "113759,konaklama-aylik-bulten-2023---martxlsx.xlsx?0",
    (2023, 4): "115317,konaklama-aylik-bulten-2023---nisanxlsx.xlsx?0",
    (2023, 5): "116728,konaklama-aylik-bulten-2023---mayisxlsx.xlsx?0",
    (2023, 6): "117194,haziran-2023-turizm-isletme-ve-basit-belgeli-aylik-bultenxlsx.xlsx?0",
    (2023, 7): "117688,konaklama-aylik-bulten-2023---temmuzxlsx.xlsx?0",
    (2023, 8): "118149,konaklama-aylik-bulten-2023---agustosxlsx.xlsx?0",
    (2023, 9): "118727,eylul-2023-turizm-isletme-ve-basit-belgeli-aylik-bultenxlsx.xlsx?0",
    (2023, 10): "119380,konaklama-aylik-bulten-2023---ekimxlsx.xlsx?0",
    (2023, 11): "120155,konaklama-aylik-bulten-2023---kasim-yenixlsx.xlsx?0",
    (2023, 12): "120698,konaklama-aylik-bulten-2023---aralikxlsx.xlsx?0",
    (2024, 1): "122349,konaklama-aylik-bulten-2024--ocakxlsx.xlsx?0",
    (2024, 2): "122749,konaklama-aylik-bulten-2024--subatxlsx.xlsx?0",
    (2024, 3): "123300,konaklama-aylik-bulten-2024--martxlsx.xlsx?0",
    (2024, 4): "123817,konaklama-aylik-bulten-2024--nisanxlsx.xlsx?0",
    (2024, 5): "124384,konaklama-aylik-bulten-2024---mayisxlsx.xlsx?0",
    (2024, 6): "125161,konaklama-aylik-bulten-2024---haziranxlsx.xlsx?0",
    (2024, 7): "125522,konaklama-aylik-bulten-2024---temmuzxlsx.xlsx?0",
    (2024, 8): "126233,konaklama-aylik-bulten-2024---agustosxlsx.xlsx?0",
    (2024, 9): "128161,konaklama-aylik-bulten-2024---eylulxlsx.xlsx?0",
    (2024, 10): "128764,konaklama-aylik-bulten-2024---ekimxlsx.xlsx?0",
    (2024, 11): "129925,konaklama-aylik-bulten-2024---kasimxlsx.xlsx?0",
    (2024, 12): "130704,konaklama-aylik-bulten-2024---aralikxlsx.xlsx?0",
    (2025, 1): "131303,konaklama-aylik-bulten-2025---ocakxlsx.xlsx?0",
    (2025, 2): "132600,konaklama-aylik-bulten-2025-subatxlsx.xlsx?0",
    (2025, 3): "133136,konaklama-aylik-bulten-2025---mart-v2xlsx.xlsx?0",
    (2025, 4): "134366,konaklama-aylik-bulten-2025-nisan-03062025xlsx.xlsx?0",
    (2025, 5): "135379,konaklama-aylik-bulten-2025---mayis-07072025xlsx.xlsx?0",
    (2025, 6): "137030,konaklama-aylik-bulten-2025-haziran-05082025xlsx.xlsx?0",
    (2025, 7): "138038,konaklama-aylik-bulten-2025---temmuz--11092025xlsx.xlsx?0",
    (2025, 8): "138964,konaklama-aylik-bulten-2025---agustos-06102025xlsx.xlsx?0",
    (2025, 9): "139656,konaklama-aylik-b-lten-2025---eylul-05112025xlsx.xlsx?0",
    (2025, 10): "141233,konaklama-aylik-b-lten-2025---ekim-03122025xlsx.xlsx?0",
    (2025, 11): "142920,konaklama-aylik-b-lten-2025-kasim-07012026xlsx.xlsx?0",
    (2025, 12): "144509,konaklama-aylik-b-lten-2025---aralik-03022026xlsx.xlsx?0",
    (2026, 1): "146050,konaklama--ocak-bulteni-06042026xlsx.xlsx?0",
    (2026, 2): "146049,konaklama-aylik-b-lten-s-bat-07042026xlsx.xlsx?0",
    (2026, 3): "146557,konaklama-aylik-b-lten-2026---mart-07052026xlsx.xlsx?0",
    (2026, 4): "148039,konaklama-aylik-b-lten-2026---nisanxlsx.xlsx?0",
    (2026, 5): "148683,konaklama-mayis-bulteni-08072026xlsx.xlsx?0",
    (2026, 6): "150391,konaklama-aylik-b-lten-2026---06-2026-08-03xlsx.xlsx?0",
}

#: BELEDIYE (mahalli idare) belgeli tesislerin AYLIK bultenleri, ayni "Il"
#: sayfasi duzeni. Seri 2022-10'da BITER: 7334 sayili Kanun (28.07.2021)
#: belediye belgeli tesislere "basit konaklama turizm isletmesi belgesi"
#: zorunlulugu getirdi; belgelenenler Bakanlik serisine katildi (KTB
#: metaveri TR-201124: "2022 Kasim'dan itibaren Bakanlik Belgeli Konaklama
#: Istatistikleri olarak yayimlanmaktadir"). Bakanlik + belediye toplami,
#: 2022-09 kirilmasini KAPATAN "tum belgeli" serisidir. Yil sayfalari
#: TR-232595 (2019) .. TR-311439 (2022); slug'lardaki "hairan", "agusto"
#: yazim hatalari sitede gercekten boyle.
BELEDIYE_BULTENLER: dict[tuple[int, int], str] = {
    (2019, 1): "90681,ocak-2019-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2019, 2): "90684,subat-2019-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2019, 3): "90680,mart-2019-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2019, 4): "90690,nisan-2019-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2019, 5): "90685,mayis-2019-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2019, 6): "90689,haziran-2019-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2019, 7): "90682,temmuz-2019-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2019, 8): "90679,agustos-2019-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2019, 9): "90683,eylul-2019-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2019, 10): "90686,ekim-2019-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2019, 11): "90688,kasim-2019-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2019, 12): "90687,aralik-2019-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2020, 1): "90714,ocak-2020-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2020, 2): "90706,subat-2020-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2020, 3): "90712,mart-2020-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2020, 4): "90705,nisan-2020-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2020, 5): "90708,mayis-2020-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2020, 6): "90715,haziran-2020-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2020, 7): "90711,temmuz-2020-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2020, 8): "90704,agustos-2020-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2020, 9): "90709,eylul-2020-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2020, 10): "90713,ekim-2020-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2020, 11): "90710,kasim-2020-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2020, 12): "90707,aralik-2020-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2021, 1): "98265,ocak-2021-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2021, 2): "98266,subat-2021-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2021, 3): "98263,mart-2021-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2021, 4): "98268,nisan-2021-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2021, 5): "98264,mayis-2021-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2021, 6): "98260,hairan-2021-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2021, 7): "98267,temmuz-2021-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2021, 8): "98257,agusto-2021-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2021, 9): "98262,eylul-2021-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2021, 10): "98258,ekim-2021-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2021, 11): "98261,kasim-2021-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2021, 12): "98259,aralik-2021-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2022, 1): "93659,ocak-2022-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2022, 2): "100833,subat-2022-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2022, 3): "97127,mart-2022-belediye-belgeli-aylik-bultenxlsx.xlsx?0",
    (2022, 4): "111324,konaklama-aylik-bulten-2022---nisan-belediyexlsx.xlsx?0",
    (2022, 5): "111327,konaklama-aylik-bulten-2022---mayis-belediyexlsx.xlsx?0",
    (2022, 6): "111326,konaklama-aylik-bulten-2022---haziran-belediyexlsx.xlsx?0",
    (2022, 7): "111325,konaklama-aylik-bulten-2022---temmuz-belediyexlsx.xlsx?0",
    (2022, 8): "111321,konaklama-aylik-bulten-2022---agustos-belediyexlsx.xlsx?0",
    (2022, 9): "111322,konaklama-aylik-bulten-2022--eylul-belediyexlsx.xlsx?0",
    (2022, 10): "111323,konaklama-aylik-bulten-2022---ekim-belediyexlsx.xlsx?0",
}

#: Bultendeki ay adi (join_key bicimi) -> ay numarasi.
AY_ADLARI: dict[str, int] = {
    "ocak": 1, "subat": 2, "mart": 3, "nisan": 4, "mayis": 5, "haziran": 6,
    "temmuz": 7, "agustos": 8, "eylul": 9, "ekim": 10, "kasim": 11, "aralik": 12,
}  # fmt: skip

#: Sayfa adlari join_key bicimiyle esletirilir (dosyalarda "İl", "Geliş-Geceleme Ay").
SAYFA_IL = "il"
SAYFA_AY = "gelis-geceleme ay"
SAYFA_YIL = "gelis-geceleme yil"

#: "Il" sayfasi kolon dizilimi (0 = il adi). Her grup yabanci/yerli/toplam.
KOLON_GELIS = 1
KOLON_GECELEME = 4
KOLON_DOLULUK = 10
VERI_BASI = 3
BEKLENEN_IL_SAYISI = 81

HAM_DIZIN = Path("data/external/ham")
CIKTI_YOLU = Path("data/external/turizm_aylik_il.parquet")
#: Olculen kapsam kirilmalari (bkz. baslik). (yil, ay) bu donemden ITIBAREN
#: yeni rejim. Rejim 1: <=2022-08, 2: 2022-09..2025-06, 3: >=2025-07.
REJIM_KIRILMALARI: tuple[tuple[int, int], ...] = ((2022, 9), (2025, 7))

#: Belediye serisinden gelen kolonlar (2022-10'a kadar dolu, sonra NaN) ve
#: bakanlik + belediye toplami "tum belgeli" kolonlari (her donem dolu).
BELEDIYE_KOLONLARI = ["gelis_belediye", "geceleme_belediye", "doluluk_belediye"]
TUM_KOLONLARI = ["gelis_tum_belgeli", "geceleme_tum_belgeli", "doluluk_tum_belgeli"]
CIKTI_KOLONLARI = [
    "yil", "ay", "il", "il_key", "kapsam", "kapsam_rejimi",
    "gelis_yabanci", "gelis_yerli", "gelis",
    "geceleme_yabanci", "geceleme_yerli", "geceleme",
    "doluluk",
    *BELEDIYE_KOLONLARI, *TUM_KOLONLARI,
]  # fmt: skip

REQUEST_PAUSE_S = 6.0
TIMEOUT_S = 120
RETRIES = 4
MIN_BAYT = 50_000


def indir(yil: int, ay: int, eklenti: str, *, seri: str = "bakanlik") -> Path:
    """Bulteni ham dizine indirir; hash'i dogrulanmis dosya varsa dokunmaz."""
    url = EKLENTI + eklenti
    ek = "" if seri == "bakanlik" else f"{seri}_"
    hedef = HAM_DIZIN / f"ktb_konaklama_{ek}aylik_{yil}_{ay:02d}.xlsx"
    if hedef.exists():
        try:
            validate_cached_file(hedef, min_bytes=MIN_BAYT, source=url)
        except (OSError, ValueError) as hata:
            print(f"  {yil}-{ay:02d}: ham cache dogrulanamadi; yeniden indirilecek ({hata})")
        else:
            return hedef

    son_hata: Exception | None = None
    for deneme in range(1, RETRIES + 1):
        try:
            yanit = requests.get(
                url,
                timeout=TIMEOUT_S,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            yanit.raise_for_status()
            break
        except requests.RequestException as hata:
            son_hata = hata
            if deneme < RETRIES:
                time.sleep(REQUEST_PAUSE_S + 2**deneme)
    else:
        raise RuntimeError(
            f"{yil}-{ay:02d} bulteni {RETRIES} denemede inmedi. Son hata: {son_hata}"
        )

    if len(yanit.content) < MIN_BAYT:
        raise RuntimeError(
            f"{yil}-{ay:02d} bulteni supheli kucuk ({len(yanit.content)} bayt) -- "
            "muhtemelen hata sayfasi indi."
        )
    publish_bytes(yanit.content, hedef, source=url, min_bytes=MIN_BAYT)
    print(f"  {yil}-{ay:02d}: indirildi ({len(yanit.content):,} bayt)")
    time.sleep(REQUEST_PAUSE_S)
    return hedef


def _sayfa_bul(kitap: pd.ExcelFile, aranan: str) -> str:
    """Sayfa adini join_key esiyle bulur (Turkce I/İ farkina dayanikli)."""
    for ad in kitap.sheet_names:
        if join_key(ad).strip() == aranan:
            return ad
    raise ValueError(f"'{aranan}' sayfasi yok. Sayfalar: {kitap.sheet_names}")


def dosya_donemi(kitap: pd.ExcelFile) -> tuple[int, int]:
    """Bultenin (yil, ay) donemini ICERIKTEN okur.

    "Gelis-Geceleme Ay" sayfasinin ilk kolonundaki SON ay adi bultenin ayi,
    "Gelis-Geceleme Yil" sayfasindaki SON yil bultenin yilidir (olculdu:
    Temmuz bulteninde OCAK..TEMMUZ listelenir). URL slug'ina guvenilmez.
    """
    ay_sayfa = pd.read_excel(kitap, sheet_name=_sayfa_bul(kitap, SAYFA_AY), header=None)
    ay_adlari = [join_key(str(x)) for x in ay_sayfa[0].dropna()]
    aylar = [AY_ADLARI[a] for a in ay_adlari if a in AY_ADLARI]
    if not aylar:
        raise ValueError(f"Ay sayfasinda taninan ay adi yok: {ay_adlari}")

    yil_sayfa = pd.read_excel(kitap, sheet_name=_sayfa_bul(kitap, SAYFA_YIL), header=None)
    yillar = pd.to_numeric(yil_sayfa[0], errors="coerce").dropna()
    yillar = yillar[(yillar >= 2000) & (yillar <= 2100)]
    if yillar.empty:
        raise ValueError("Yil sayfasinda 2000-2100 arasi yil yok.")
    return int(yillar.iloc[-1]), aylar[-1]


def _kapsam(baslik: str) -> str:
    """Il sayfasi basligindan belge kapsamini etiketler.

    'turizm isletme belgeli' -> "isletme"; 'isletme ve basit belgeli' ->
    "isletme_basit". Ikisi de degilse hata: sessizce karistirmak, 2022
    Kasim seviye sicramasini gormezden gelmek olur.
    """
    anahtar = join_key(baslik)
    if "mahalli idare" in anahtar or "belediye" in anahtar:
        return "belediye"
    if "basit" in anahtar:
        return "isletme_basit"
    if "isletme belgeli" in anahtar:
        return "isletme"
    raise ValueError(f"Il sayfasi basligindan kapsam cikarilamadi: {baslik!r}")


def kapsam_rejimi(yil: int, ay: int) -> int:
    """(yil, ay) icin olculen kapsam rejimi (1'den baslar)."""
    return 1 + sum((yil, ay) >= kirilma for kirilma in REJIM_KIRILMALARI)


def il_tablosu(yol: Path, yil: int, ay: int, *, seri: str = "bakanlik") -> pd.DataFrame:
    """Bultenin "Il" sayfasini ortak semaya cevirir; donemi icerikten dogrular.

    ``seri="bakanlik"``: tam 81 il ve kapsam isletme/isletme_basit olmali.
    ``seri="belediye"``: kapsam "belediye" olmali; il sayisi 81'den AZ
    olabilir (olculdu: 2022-10'da Karaman yok -- belediye belgeli tesisi
    kalmamis). Eksik il "sifir tesis" demektir, birlestirmede 0 sayilir.

    Raises:
        ValueError: Donem uyusmazsa, baslik dizilimi degismisse, il sayisi
            beklenenden sapmissa veya kapsam etiketi seriyle celisiyorsa
            (yanlis URL'nin tek belirtisi budur).
    """
    with pd.ExcelFile(yol) as kitap:
        okunan = dosya_donemi(kitap)
        if okunan != (yil, ay):
            raise ValueError(f"{yol.name}: beklenen donem {yil}-{ay:02d}, icerik {okunan} diyor.")
        ham = pd.read_excel(kitap, sheet_name=_sayfa_bul(kitap, SAYFA_IL), header=None)

    baslik = " ".join(join_key(str(x)) for x in ham.iloc[1].tolist())
    for parca in ("gelis", "geceleme", "doluluk"):
        if parca not in baslik:
            raise ValueError(f"{yol.name}: baslikta '{parca}' yok -- yapi degismis. {baslik}")
    alt = [join_key(str(x)) for x in ham.iloc[2].tolist()]
    for grup_bas in (KOLON_GELIS, KOLON_GECELEME, KOLON_DOLULUK):
        if alt[grup_bas] != "yabanci" or alt[grup_bas + 2] != "toplam":
            raise ValueError(f"{yol.name}: kolon {grup_bas} yabanci/yerli/toplam degil: {alt}")

    veri = ham.iloc[VERI_BASI:].copy()
    veri = veri[veri[0].notna()]
    veri["il_key"] = veri[0].astype(str).map(join_key)
    veri = veri[~veri["il_key"].str.contains("toplam")]
    kapsam = _kapsam(str(ham.iloc[0, 0]))
    if seri == "bakanlik":
        if len(veri) != BEKLENEN_IL_SAYISI:
            raise ValueError(
                f"{yol.name}: {len(veri)} il satiri, {BEKLENEN_IL_SAYISI} bekleniyordu."
            )
        if kapsam == "belediye":
            raise ValueError(f"{yol.name}: bakanlik serisi bekleniyordu, baslik belediye diyor.")
    elif seri == "belediye":
        if not 1 <= len(veri) <= BEKLENEN_IL_SAYISI:
            raise ValueError(
                f"{yol.name}: {len(veri)} il satiri, 1..{BEKLENEN_IL_SAYISI} bekleniyordu."
            )
        if kapsam != "belediye":
            raise ValueError(f"{yol.name}: belediye serisi bekleniyordu, baslik {kapsam!r} diyor.")
    else:
        raise ValueError(f"Bilinmeyen seri: {seri!r}")

    def sayi(kolon: int) -> pd.Series:
        return pd.to_numeric(veri[kolon], errors="coerce")

    tablo = pd.DataFrame(
        {
            "yil": yil,
            "ay": ay,
            "il": veri[0].astype(str).str.strip(),
            "il_key": veri["il_key"],
            "kapsam": kapsam,
            "kapsam_rejimi": kapsam_rejimi(yil, ay),
            "gelis_yabanci": sayi(KOLON_GELIS),
            "gelis_yerli": sayi(KOLON_GELIS + 1),
            "gelis": sayi(KOLON_GELIS + 2),
            "geceleme_yabanci": sayi(KOLON_GECELEME),
            "geceleme_yerli": sayi(KOLON_GECELEME + 1),
            "geceleme": sayi(KOLON_GECELEME + 2),
            "doluluk": sayi(KOLON_DOLULUK + 2),
        }
    )
    if tablo["geceleme"].isna().any():
        eksik = tablo.loc[tablo["geceleme"].isna(), "il"].tolist()
        raise ValueError(f"{yol.name}: geceleme sayiya cevrilemedi: {eksik}")
    return tablo.reset_index(drop=True)


def _yatak_gun(geceleme: pd.Series, doluluk: pd.Series) -> pd.Series:
    """Ortuk yatak-gun = geceleme / (doluluk/100); doluluk 0/NaN ise NaN."""
    oran = doluluk / 100.0
    return (geceleme / oran.where(oran.gt(0))).where(oran.gt(0))


def tum_belgeli_birlestir(bakanlik: pd.DataFrame, belediye: pd.DataFrame | None) -> pd.DataFrame:
    """Bakanlik satirlarina belediye kolonlarini ve 'tum belgeli' toplamini ekler.

    * ``*_belediye``: belediye serisinin degeri; serinin bittigi 2022-10
      sonrasi NaN (yok degil, BAKANLIK serisine katildi).
    * ``*_tum_belgeli``: bakanlik + belediye (belediye NaN ise 0 sayilir).
      2022-11'den itibaren bakanlik degerine esittir. OLCULDU: 2022-09
      sicramasini kapatir (Turkiye ortuk yatak 1,50M -> 1,57M, oran 1,04;
      bakanlikta 1,19) AMA 2022-11'de TERS bir kirilma yaratir (1,44M ->
      1,03M): belediye belgeli tesislerin buyuk kismi basit belgeye
      GECMEMIS, istatistikten cikmis. Mugla Agustos: bakanlik 82k -> 125k
      yatak (2022 -> 2023, +%52), tum 154k -> 125k (-%19). Yani KUSURSUZ
      surekli bir seviye serisi YOKTUR; tum_belgeli 2019-2022 icin gercek
      turist yukune daha yakindir (Mugla'da belediye pansiyonlari +%40),
      2022-11 sonrasi bakanlikla aynidir. Yillar arasi seviye kiyasinda
      ``kapsam_rejimi`` ile birlikte kullanilmali; mevsim SEKLI icin
      doluluk tercih edilmeli. 2025-07 kirilmasini KAPATMAZ.
    * ``doluluk_tum_belgeli``: yatak-gun agirlikli birlesik doluluk. Yatak
      sayisi dogrudan yok; her serinin ortuk yatak-gunu = geceleme /
      (doluluk/100) ile geri cikarilir (bultenin kendi paydasi), sonra
      toplam geceleme / toplam yatak-gun alinir. Belediye yoksa bakanlik
      dolulugudur.

    Raises:
        ValueError: Belediye tablosunda bakanlikta olmayan (yil, ay, il) varsa
            -- eslesmeyen satir sessizce dusmemeli.
    """
    cikti = bakanlik.copy()
    if belediye is None or belediye.empty:
        for kolon in BELEDIYE_KOLONLARI:
            cikti[kolon] = float("nan")
    else:
        anahtar = ["yil", "ay", "il_key"]
        bel = belediye[[*anahtar, "gelis", "geceleme", "doluluk"]].rename(
            columns={
                "gelis": "gelis_belediye",
                "geceleme": "geceleme_belediye",
                "doluluk": "doluluk_belediye",
            }
        )
        eslesme = bel.merge(cikti[anahtar], on=anahtar, how="left", indicator=True)
        yetim = int((eslesme["_merge"] == "left_only").sum())
        if yetim:
            raise ValueError(f"Belediye serisinde bakanlikta olmayan {yetim} (yil, ay, il) var.")
        cikti = cikti.merge(bel, on=anahtar, how="left")

    bel_gelis = cikti["gelis_belediye"].fillna(0)
    bel_gec = cikti["geceleme_belediye"].fillna(0)
    cikti["gelis_tum_belgeli"] = cikti["gelis"] + bel_gelis
    cikti["geceleme_tum_belgeli"] = cikti["geceleme"] + bel_gec

    bak_yg = _yatak_gun(cikti["geceleme"], cikti["doluluk"])
    bel_yg = _yatak_gun(bel_gec, cikti["doluluk_belediye"].fillna(0)).fillna(0)
    toplam_yg = bak_yg + bel_yg
    birlesik_doluluk = 100.0 * cikti["geceleme_tum_belgeli"] / toplam_yg.where(toplam_yg.gt(0))
    cikti["doluluk_tum_belgeli"] = birlesik_doluluk.where(toplam_yg.gt(0), cikti["doluluk"])
    return cikti


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(CIKTI_YOLU), help="Cikti parquet yolu")
    parser.add_argument(
        "--yillar", nargs="*", type=int, default=None, help="Yalnizca bu yillar (vars. hepsi)"
    )
    args = parser.parse_args()

    donemler = sorted(BULTENLER)
    if args.yillar:
        donemler = [d for d in donemler if d[0] in set(args.yillar)]
    if not donemler:
        raise SystemExit("Secilen yillar icin bulten haritasinda kayit yok.")

    parcalar: list[pd.DataFrame] = []
    for yil, ay in donemler:
        yol = indir(yil, ay, BULTENLER[(yil, ay)])
        parcalar.append(il_tablosu(yol, yil, ay))
    bakanlik = pd.concat(parcalar, ignore_index=True)

    belediye_donemler = [d for d in sorted(BELEDIYE_BULTENLER) if d in set(donemler)]
    bel_parcalar: list[pd.DataFrame] = []
    for yil, ay in belediye_donemler:
        yol = indir(yil, ay, BELEDIYE_BULTENLER[(yil, ay)], seri="belediye")
        bel_parcalar.append(il_tablosu(yol, yil, ay, seri="belediye"))
    belediye = pd.concat(bel_parcalar, ignore_index=True) if bel_parcalar else None

    birlesik = tum_belgeli_birlestir(bakanlik, belediye)
    birlesik = birlesik[CIKTI_KOLONLARI].sort_values(["yil", "ay", "il_key"]).reset_index(drop=True)

    tekrar = birlesik.duplicated(subset=["yil", "ay", "il_key"])
    if tekrar.any():
        raise ValueError(f"Yinelenen (yil, ay, il) satirlari: {int(tekrar.sum())}")

    cikti = Path(args.out)
    publish_dataframe(
        birlesik,
        cikti,
        required_columns=CIKTI_KOLONLARI,
        min_rows=len(donemler) * BEKLENEN_IL_SAYISI,
        source="KTB monthly bulletins: "
        + f"{donemler[0][0]}-{donemler[0][1]:02d}..{donemler[-1][0]}-{donemler[-1][1]:02d}",
    )
    print(f"Yazildi: {cikti}")
    print(
        f"  {len(birlesik)} satir, {len(donemler)} donem, "
        f"rejimler: {sorted(birlesik['kapsam_rejimi'].unique())}"
    )
    if belediye_donemler:
        ilk, son = belediye_donemler[0], belediye_donemler[-1]
        print(
            f"  belediye serisi: {len(belediye_donemler)} donem ({ilk}..{son}); sonrasi bakanlikta"
        )
    print("  Kaynak: KTB Yatirim ve Isletmeler Gn.Md. aylik konaklama bultenleri.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
