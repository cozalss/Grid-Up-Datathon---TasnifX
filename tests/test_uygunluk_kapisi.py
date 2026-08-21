"""ELEME KAPISI: yarisma hedefinin gecmisi MODELE GIRDI OLAMAZ.

NEDEN BU TEST DOSYASI (2026-08-21, kural arastirmasi)
-----------------------------------------------------
Coderspace'in GDZ'22 Case-1 yarismasi -- bizimkiyle ayni problem, gunluk
kesinti tahmini -- kural sayfasinda sunu yaziyor::

    "Ariza SONUC verileri internette halka acik olarak erisilebilmektedir.
     Notebook degerlendirme surecinde, bu verilerin KULLANILMADIGI ve
     MODELDE GIRDI OLARAK YER ALMADIGI konusu detayli olarak incelenecektir."

Elimizde tam olarak o veri var: ``data/external/epias/kesinti_plansiz.parquet``
(405.819 gercek GDZ+ADM plansiz kesinti kaydi, 2022-2026) ve ondan turetilen
ilce x gun panelleri.

Bu veri DEGERLIDIR ve atilmaz -- ama yeri bellidir:

    PROVA ZEMINI  (izinli)  hatti gercek 96 ilce x 4,6 yil veriyle sinamak,
                            olcum aletini kalibre etmek, dusmanca prova
    MODEL GIRDISI (YASAK)   feature olarak panele baglanmak

Ayrim onemli: gelistirme sirasinda kamuya acik veriyle test etmek "modelde
girdi olarak yer almak" degildir. Ayni ayrimi 2023 GDZ yarismasi da yapiyor
ve dis veriyi (hava, arazi ortusu, altyapi) ACIKCA tesvik ediyor.

TASARIM: prosa uyari yetmez, KOD durdurmali
--------------------------------------------
Manifestte zaten "MODELE GIRMEZ" diye bir not vardi. Bir JSON dosyasindaki
cumle hicbir kodu durdurmaz. Bu yuzden:

  1. Manifest artifact'i ``model_girdisi: false`` ile ISARETLER (makine okur).
  2. Bu testler modelleme kutuphanesinin o dosyalara HIC dokunmadigini
     STATIK olarak dogrular.
  3. ``attach_external`` aile->yol haritasi CALISMA ANINDA denetlenir.

Yanlis pozitif riski dusunuldu: yarismanin KENDI hedefinden turetilen lag
feature'lari (or. ``kesinti_adedi_lag7``) mesrudur ve engellenmemelidir.
Bu yuzden kapi KOLON ADINA degil, KAYNAK DOSYAYA bakar -- koken tek
guvenilir ayirt edicidir.
"""

from __future__ import annotations

import pytest

from gridup.uygunluk import (
    kaynak_ihlallerini_tara,
    model_girdisi_yasak_yollar,
    yasakli_aileleri_dogrula,
)


def test_manifest_kesinti_verisini_yasakli_isaretler() -> None:
    """Uc kesinti artifact'i da makine-okunur bicimde yasakli olmali."""
    yasak = model_girdisi_yasak_yollar()

    assert "data/external/epias/kesinti_plansiz.parquet" in yasak
    assert any("panel_ilce_gun" in yol for yol in yasak)


def test_yasakli_yollar_gerekce_tasir() -> None:
    """Her yasak, sebebini yaninda tasimali -- yoksa alti ay sonra silinir."""
    yasak = model_girdisi_yasak_yollar()

    for yol, gerekce in yasak.items():
        assert gerekce.strip(), f"{yol}: gerekce bos"
        assert len(gerekce) > 40, f"{yol}: gerekce cok kisa, kural alintisi yok"


def test_modelleme_kutuphanesi_yasakli_dosyalara_dokunmuyor() -> None:
    """ELEME KAPISI. Bu test kirmizi ise notebook degerlendirmesinde eleniriz."""
    ihlaller = kaynak_ihlallerini_tara()

    assert ihlaller == [], (
        "Modelleme kutuphanesi yasakli kesinti verisine referans veriyor:\n"
        + "\n".join(f"  {dosya}:{satir} -> {yol}" for dosya, satir, yol in ihlaller)
    )


def test_temiz_aile_haritasi_gecer() -> None:
    aileler = {
        "hava": "data/external/hava_gunluk.parquet",
        "arazi_ortusu": "data/external/arazi_ortusu_ilce.parquet",
    }

    yasakli_aileleri_dogrula(aileler)  # istisna atmamali


def test_yasakli_aile_calisma_aninda_durdurulur() -> None:
    """Biri paneli aile olarak baglarsa egitim BASLAMADAN durur."""
    aileler = {
        "hava": "data/external/hava_gunluk.parquet",
        "gecmis_kesinti": "data/external/epias/panel_ilce_gun.parquet",
    }

    with pytest.raises(ValueError, match="gecmis_kesinti"):
        yasakli_aileleri_dogrula(aileler)


def test_hata_mesaji_kurali_alintiar() -> None:
    """Hata okunup anlasilmali; sadece 'yasak' demek yeterli degil."""
    with pytest.raises(ValueError) as hata:
        yasakli_aileleri_dogrula({"x": "data/external/epias/kesinti_plansiz.parquet"})

    metin = str(hata.value)
    assert "notebook" in metin.lower()
    assert "prova" in metin.lower()  # nerede KULLANILABILECEGI de yazmali
