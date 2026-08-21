"""Yarisma uygunlugu: hangi verinin MODELE GIRDI olabilecegini kod duzeyinde tutar.

NEDEN BU MODUL VAR
------------------
Coderspace'in GDZ'22 Case-1 yarismasi -- bizimkiyle ayni problem, gunluk
kesinti tahmini -- kural sayfasinda soyle diyor::

    "Ariza SONUC verileri internette halka acik olarak erisilebilmektedir.
     Notebook degerlendirme surecinde, bu verilerin KULLANILMADIGI ve
     MODELDE GIRDI OLARAK YER ALMADIGI konusu detayli olarak incelenecektir."

Ayni ailenin 2023 yarismasi ise dis veriyi ACIKCA tesvik ediyor ("ekiplerin
kullandiklari dis veriler ... degerlendirilecektir"). Yani kural dis veriyi
degil, HEDEFIN GECMISINI yasakliyor -- cunku o, sizintinin ta kendisidir.

Elimizdeki ``data/external/epias/*`` kesinti kayitlari tam olarak o sinifta.
Veri degerlidir ve atilmaz; yeri bellidir:

    PROVA ZEMINI  (izinli)  hatti gercek veriyle sinamak, olcum kalibrasyonu
    MODEL GIRDISI (YASAK)   feature olarak panele baglanmak

Manifestte prosa bir uyari zaten vardi. Bir JSON dosyasindaki cumle hicbir
kodu durdurmaz -- bu modul onu MAKINE OKUNUR bir kapiya cevirir.

TASARIM: kapi KOLON ADINA degil KAYNAK DOSYAYA bakar
-----------------------------------------------------
Yarismanin KENDI hedefinden turetilen lag feature'lari (``kesinti_adedi_lag7``
gibi) mesrudur -- o veri yarismanin bize verdigi veridir. Kolon adiyla filtre
kurmak bunlari da vururdu. Koken, tek guvenilir ayirt edicidir.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

__all__ = [
    "MANIFEST_YOLU",
    "TARANAN_DIZINLER",
    "kaynak_ihlallerini_tara",
    "model_girdisi_yasak_yollar",
    "yasakli_aileleri_dogrula",
]

#: Depo kokune gore veri kaynagi manifesti.
MANIFEST_YOLU = "data/sources.yml"

#: Statik taramanin bakacagi dizinler: MODELLEME kutuphanesi.
#:
#: ``scripts/`` bilerek DISARIDA. Prova ve olcum betiklerinin kesinti verisini
#: okumasi yalnizca mesru degil, GEREKLIDIR -- ``dusmanca_prova.py`` gercek
#: 96 ilce x 4,6 yil veriden hasim bir yarisma dosyasi uretir. Yasak olan,
#: verinin MODELE GIRDI olmasidir; gelistirme sirasinda onunla test etmek
#: degil. Bu ayrimi yanlis kurmak ya kapiyi ise yaramaz kilar (her seyi
#: yasaklar) ya da delik birakir (hicbir seyi yasaklamaz).
TARANAN_DIZINLER = ("src/gridup",)


def _kok(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    return Path(__file__).resolve().parents[2]


def model_girdisi_yasak_yollar(root: str | Path | None = None) -> dict[str, str]:
    """Manifestte ``model_girdisi: false`` isaretli artifact'lari dondurur.

    Returns:
        ``gorece yol -> gerekce``. Gerekce, kuralin kendisini alintilamalidir;
        aksi halde alti ay sonra kimse neden yasak oldugunu bilmez ve isaret
        "gereksiz" diye silinir.
    """
    manifest = _kok(root) / MANIFEST_YOLU
    veri = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        kayit["path"]: str(kayit.get("model_girdisi_gerekce", ""))
        for kayit in veri.get("artifacts", [])
        if kayit.get("model_girdisi") is False
    }


def kaynak_ihlallerini_tara(root: str | Path | None = None) -> list[tuple[str, int, str]]:
    """Modelleme kutuphanesinde yasakli yola yapilan referanslari bulur.

    Metin taramasi bilerek secildi: ithal grafigi takip etmek, dinamik yol
    kurulumunu (``kok / "data" / dosya_adi``) kaciririr. Yol dizgesinin
    kendisini aramak kaba ama kacamaz.

    Returns:
        ``(gorece dosya, satir no, yasakli yol)`` listesi. Bos liste = temiz.
    """
    kok = _kok(root)
    yasak = model_girdisi_yasak_yollar(kok)
    if not yasak:
        return []

    # Yol parcalarini ayri ayri ara: kod yolu "data/external/epias/x.parquet"
    # diye tek dizgede tutmayabilir, ``kok / "data/external/epias" / ad`` diye
    # parcalayabilir. Dosya ADI en dayanikli imzadir.
    imzalar = {Path(yol).name: yol for yol in yasak}

    ihlaller: list[tuple[str, int, str]] = []
    for dizin in TARANAN_DIZINLER:
        for dosya in sorted((kok / dizin).rglob("*.py")):
            metin = dosya.read_text(encoding="utf-8")
            if not any(imza in metin for imza in imzalar):
                continue
            for numara, satir in enumerate(metin.splitlines(), start=1):
                for imza, yol in imzalar.items():
                    if imza in satir:
                        ihlaller.append((dosya.relative_to(kok).as_posix(), numara, yol))
    return ihlaller


def yasakli_aileleri_dogrula(
    aile_yollari: Mapping[str, str], *, root: str | Path | None = None
) -> None:
    """``attach_external`` aile->yol haritasini calisma aninda denetler.

    Statik tarama kutuphanenin bugunku halini korur; bu kapi ise CAGRI
    ANINDA, haritanin nasil kuruldugundan bagimsiz olarak calisir.

    Args:
        aile_yollari: ``aile adi -> okunan artifact yolu``.
        root: Depo koku (test icin).

    Raises:
        ValueError: Herhangi bir aile yasakli bir artifact okuyorsa.
    """
    yasak = model_girdisi_yasak_yollar(root)
    if not yasak:
        return
    imzalar = {Path(yol).name for yol in yasak}

    ihlal = {aile: yol for aile, yol in aile_yollari.items() if Path(str(yol)).name in imzalar}
    if not ihlal:
        return

    satirlar = "\n".join(f"    {aile}  ->  {yol}" for aile, yol in ihlal.items())
    raise ValueError(
        "YARISMA UYGUNLUK IHLALI -- egitim baslatilmadi.\n"
        f"{satirlar}\n\n"
        "Bu dosyalar yarisma HEDEFININ gecmisidir. Coderspace'in ayni problemi\n"
        "kullanan GDZ'22 Case-1 yarismasi acikca soyle diyor: 'Notebook\n"
        "degerlendirme surecinde, bu verilerin kullanilmadigi ve modelde girdi\n"
        "olarak yer almadigi konusu detayli olarak incelenecektir.'\n\n"
        "Veri ATILMAZ; yeri farklidir:\n"
        "  IZINLI  -- prova zemini: hatti gercek veriyle sinamak, olcum\n"
        "             kalibrasyonu, dusmanca prova (scripts/dusmanca_prova.py)\n"
        "  YASAK   -- feature olarak panele baglanmak\n\n"
        "Yarisma kurallarinda harici veri izni ACIKCA dogrulandiysa ve kesinti\n"
        "gecmisi izinliyse, manifestteki 'model_girdisi' isaretini kaldir --\n"
        "ama once kurali oku, sonra kaldir."
    )
