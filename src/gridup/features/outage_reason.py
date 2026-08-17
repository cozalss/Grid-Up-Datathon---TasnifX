"""Ariza sebebi normalizasyonu: 919 serbest metinden kararli bir taksonomiye.

NEDEN BU MODUL VAR
------------------
EPIAS Seffaflik Platformu'ndan cekilen 30.285 gercek kesinti kaydinda
**919 FARKLI** sebep metni bulundu. Her dagitim sirketi kendi yazim bicimini
kullaniyor. Ayni fiziksel olay bes farkli sekilde yazilabiliyor::

    "Sigorta Atması"
    "SIGORTA ARIZASI"
    "Sigorta; NH Sigorta Atma/Değişim"
    "-Ekonomik Ömür - Sigorta Arıza"
    "Olumsuz Hava Sartlari - Sigorta Arıza"

Bu haliyle kategorik feature olarak kullanilamaz: 919 seviyeli bir kolon,
her seviyesi ortalama 33 ornek goren bir kodlama demektir -- saf gurultu.

Cozum: anahtar kelimeye dayali, oncelikli bir esleme. Sonuc kararli bir
tamsayi kod ve okunabilir bir aile adi.

TAKSONOMI NEDEN BU SEKILDE
--------------------------
Aileler fiziksel MEKANIZMAYA gore ayrildi, kelimeye gore degil. Cunku
feature'in isi modele nedeni ogretmektir:

  HAVA      -> hava durumu feature'lariyla etkilesir (ruzgar, yagis, yildirim)
  AGAC      -> mevsimsel (yapraklanma) + ruzgar + islak zemin
  ASIRI_YUK -> sicaklik ve talep feature'lariyla etkilesir
  UCUNCU_SAHIS -> insan faaliyeti; hafta ici/tatil takvimiyle etkilesir
  EKIPMAN   -> ekipman yasi ve bakim gecmisiyle etkilesir
  MANEVRA   -> PLANLI bir islem; plansiz ariza SAYILMAZ (bkz. asagi)
  ...

MANEVRA UYARISI
---------------
En sik gorulen kayit ``-SCADA - MANEVRA`` (1.496 kayit). Bu bir ARIZA DEGIL,
operatorun yaptigi bir sebeke manevrasidir. "Plansiz kesinti" listesinde
gorunmesi, veri setinin hedef tanimini dikkatle okumak gerektigini gosterir:
yarisma hedefi bunlari sayiyor mu, saymiyor mu? ``is_fault`` bayragi bu
ayrimi tasir.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from ..turkish import join_key

__all__ = [
    "REASON_FAMILIES",
    "REASON_CODES",
    "classify_reason",
    "add_reason_features",
    "reason_family_report",
]

# Aile -> (kod, arizan mi, anahtar kelimeler)
# Sira ONEMLI: yukaridakiler once eslenir. Cok etiketli metinlerde
# ("Olumsuz Hava Sartlari - Sigorta Ariza") daha ACIKLAYICI olan kazanir:
# hava kosulu, ekipmanin hangi parcasinin bozuldugundan daha bilgilendiricidir.
REASON_FAMILIES: tuple[tuple[str, int, bool, tuple[str, ...]], ...] = (
    # --- Dis etkenler: hava feature'lariyla etkilesecek olanlar once ---
    (
        "HAVA",
        1,
        True,
        (
            "olumsuz hava",
            "hava sart",
            "elverissiz hava",
            "hava kosul",
            "firtina",
            "yildirim",
            "kar ",
            "buzlanma",
            "sel ",
            "dolu ",
            "ruzgar",
        ),
    ),
    ("AGAC", 2, True, ("agac", "dal ", "bitki", "orman")),
    ("HAYVAN", 3, True, ("kus ", "hayvan", "kedi", "leylek", "yilan")),
    (
        "UCUNCU_SAHIS",
        4,
        True,
        (
            "3.sahis",
            "3. sahis",
            "ucuncu sahis",
            "kazi",
            "arac carpma",
            "is makinesi",
            "vinc",
            "hasar",
            "yabanci cisim",
        ),
    ),
    # UST SEBEKE: TEIAS iletim seviyesi kesintisi. Dagitim arizasindan FARKLI
    # bir mekanizmadir -- tum bolgeyi ayni anda vurur ve dagitim sirketinin
    # kontrolunde degildir. Ayri tutulmasi, modelin "bolgesel es zamanli
    # kesinti" desenini ogrenmesini saglar.
    ("UST_SEBEKE", 17, True, ("teias", "iletim", "ust sebeke", "enterkonnekte")),
    # --- Ariza OLMAYANLAR: bunlar HAVA'dan sonra ama ekipmandan ONCE gelmeli,
    #     cunku "Borctan Kesme - Hattan kesme" metninde "hat" gecer ve yanlislikla
    #     KABLO ailesine duserdi. Idari kesme bir ariza DEGILDIR. ---
    ("IDARI_KESME", 22, False, ("borctan kesme", "yikimdan kesme", "kacak", "usulsuz", "tahsilat")),
    (
        "EMNIYET",
        23,
        False,
        ("emniyet", "can ve mal", "guvenlik", "tehlike onleyici", "tedbir amacli"),
    ),
    (
        "MANEVRA",
        20,
        False,
        (
            "manevra",
            "scada",
            "sebeke calis",
            "planli",
            "bakim",
            "yatirim",
            "yeni baglanti",
            "revizyon",
            "kolon islem",
            "deplase",
            "dahili onarim",
            "veri duzeltme",
            "kesinti talep",
        ),
    ),
    ("TALEP", 21, False, ("abone talep", "musteri talep", "resmi kurum")),
    # --- Sebeke ici mekanizmalar ---
    (
        "ASIRI_YUK",
        5,
        True,
        ("asiri yuk", "yuklenme", "overload", "asiri akim", "termik ac", "asiri akim toprak"),
    ),
    (
        "KABLO",
        6,
        True,
        (
            "kablo",
            "iletken",
            "hat kop",
            "kop",
            "havai hat",
            "jumper",
            "rekortmen",
            "rekortman",
            "notr ariza",
        ),
    ),
    ("TRAFO", 7, True, ("trafo", "transformator", "guc trafo")),
    ("SIGORTA", 8, True, ("sigorta", "nh sigorta")),
    (
        "KESICI",
        9,
        True,
        (
            "kesici",
            "ayirici",
            "rekloser",
            "fider ac",
            "tms",
            "salter",
            "kontaktor",
            "sekonder koruma",
        ),
    ),
    ("IZOLATOR", 10, True, ("izolator", "mesnet", "atlama", "flashover")),
    ("KLEMENS", 11, True, ("klemens", "oksit", "ek yeri", "bag")),
    ("SAYAC", 12, True, ("sayac", "olcu", "pano")),
    ("DIREK", 14, True, ("direk", "trvers", "traves", "konsol")),
    ("KUTU", 15, True, ("dagitim kutusu", "kofre", "saha dagitim", "buat")),
    ("EKONOMIK_OMUR", 13, True, ("ekonomik omur", "yaslanma", "eskime", "korozyon", "asinma")),
    # --- Genel sebeke arizasi: spesifik ekipman belirtilmemis ---
    (
        "SEBEKE",
        16,
        True,
        ("og ariza", "ag ariza", "og-ariza", "ag-ariza", "sebeke ariza", "aydinlatma"),
    ),
    # --- Siniflandirilamayanlar ---
    ("GECICI", 30, True, ("gecici", "transient", "otomatik tekrar")),
    (
        "BILINMEYEN",
        99,
        True,
        ("bilinmeyen", "belirsiz", "tespit edilemedi", "girilmemis", "girilmemistir"),
    ),
)

REASON_CODES: dict[str, int] = {name: code for name, code, _, _ in REASON_FAMILIES}
REASON_CODES["DIGER"] = 0

_IS_FAULT: dict[int, bool] = {code: fault for _, code, fault, _ in REASON_FAMILIES}
_IS_FAULT[0] = True  # siniflandirilamayan -> ihtiyatla ariza say

_CODE_TO_NAME: dict[int, str] = {code: name for name, code in REASON_CODES.items()}

# Noktalama ve fazla bosluk temizligi -- anahtar kelime aramasindan once.
_PUNCTUATION = re.compile(r"[;,/\\\-_()\[\]]+")


def _normalize_text(value: object) -> str:
    """Sebep metnini anahtar kelime aramasina hazirlar.

    ``join_key`` Turkce buyuk/kucuk harf tuzagini cozer (``İ`` -> ``i``,
    ``I`` -> ``ı``) ve aksani duzler. Bu sart: metinler ``SIGORTA ARIZASI``,
    ``Sigorta Atması`` ve ``sigorta arıza`` bicimlerinde karisik geliyor.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = join_key(str(value))
    text = _PUNCTUATION.sub(" ", text)
    return f" {' '.join(text.split())} "


def classify_reason(reason: object) -> tuple[int, str, bool]:
    """Tek bir sebep metnini siniflandirir.

    Returns:
        ``(kod, aile_adi, ariza_mi)``.

    >>> classify_reason("Olumsuz Hava Sartlari - Sigorta Arıza")[1]
    'HAVA'
    >>> classify_reason("-SCADA - MANEVRA")[2]
    False
    >>> classify_reason("KUŞ ÇARPMASI")[1]
    'HAYVAN'
    """
    text = _normalize_text(reason)
    if not text.strip():
        return 0, "DIGER", True

    for name, code, is_fault, keywords in REASON_FAMILIES:
        if any(keyword in text for keyword in keywords):
            return code, name, is_fault

    return 0, "DIGER", True


def add_reason_features(
    frame: pd.DataFrame, reason_column: str, *, prefix: str = "sebep"
) -> pd.DataFrame:
    """Serbest metin sebep kolonundan feature uretir. YENI frame dondurur.

    Uretilen kolonlar:
        ``{prefix}_kod``     kararli tamsayi aile kimligi
        ``{prefix}_aile``    okunabilir aile adi (kategorik)
        ``{prefix}_ariza``   gercek ariza mi (manevra/planli is DEGIL)
        ``{prefix}_hava``    hava kaynakli mi -- hava feature'lariyla etkilesim icin

    ``{prefix}_hava`` ayri bir kolon cunku modelin ogrenmesi gereken en guclu
    etkilesim budur: ruzgar yuksekken HAVA kaynakli arizalar artar, EKIPMAN
    kaynakli olanlar artmaz.
    """
    if reason_column not in frame.columns:
        raise KeyError(f"Kolon '{reason_column}' frame icinde yok.")

    # Benzersiz metin sayisi satir sayisindan cok kucuk (919 vs 30.285);
    # her satiri ayri siniflandirmak israf.
    unique = frame[reason_column].astype(object).drop_duplicates()
    lookup = {value: classify_reason(value) for value in unique}

    codes = frame[reason_column].astype(object).map(lambda v: lookup[v][0])
    families = frame[reason_column].astype(object).map(lambda v: lookup[v][1])
    faults = frame[reason_column].astype(object).map(lambda v: lookup[v][2])

    weather_code = REASON_CODES["HAVA"]
    return frame.assign(
        **{
            f"{prefix}_kod": codes.astype("int16"),
            f"{prefix}_aile": families.astype("category"),
            f"{prefix}_ariza": faults.astype("int8"),
            f"{prefix}_hava": (codes == weather_code).astype("int8"),
        }
    )


def reason_family_report(reasons: Iterable[object]) -> pd.DataFrame:
    """Sebep metinlerinin ailelere dagilimini raporlar.

    Veri geldiginde ILK calistirilacaklardan: ``DIGER`` orani yuksekse
    taksonomi bu veri setine uymuyor demektir ve anahtar kelimeler
    genisletilmelidir.
    """
    series = pd.Series(list(reasons), dtype=object)
    classified = series.map(classify_reason)

    frame = pd.DataFrame(
        {
            "aile": classified.map(lambda item: item[1]),
            "ariza": classified.map(lambda item: item[2]),
        }
    )
    report = (
        frame.groupby("aile", observed=True)
        .agg(kayit=("aile", "size"), ariza_mi=("ariza", "first"))
        .sort_values("kayit", ascending=False)
    )
    report["yuzde"] = (report["kayit"] / len(series) * 100).round(2)
    return report.reset_index()
