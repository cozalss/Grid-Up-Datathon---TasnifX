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


def test_manifest_yokken_ithal_cokmez(tmp_path) -> None:
    """Kutuphane, gelistirme deposu OLMADAN da ithal edilebilmeli.

    OLCULDU 2026-08-22: CI'in "kurulmus wheel'i dumanla" adimi coktu.
    ``import gridup`` bu kapiyi ithal aninda kosuyor; kapi manifesti
    ``__file__``in iki ust dizininde ariyor. Depoda orasi kok, kurulmus
    pakette ``site-packages/../..`` -- orada ``data/`` yok::

        FileNotFoundError: .../lib/python3.11/data/sources.yml

    Yani paket, kendi deposu olmadan ITHAL EDILEMIYORDU. Bir kutuphanenin
    ithal edilmesi, gelistirme agacinin varligina bagli olamaz.

    Atlamak korunmayi kaldirmaz: manifestin olmadigi yerde ``data/`` dizini
    de yoktur, dolayisiyla yasakli aile o dosyayi zaten OKUYAMAZ.
    """
    assert model_girdisi_yasak_yollar(root=tmp_path) == {}
    # Yasakli bir aile verilse bile CAGRI PATLAMAMALI -- orada okunacak
    # dosya yok; hata artifact'in kendisinde cikar.
    yasakli_aileleri_dogrula({"kotu": "data/external/epias/kesinti_plansiz.parquet"}, root=tmp_path)


def test_bozuk_manifest_hala_patlar(tmp_path) -> None:
    """Tolerans DAR: yalnizca dosyanin YOKLUGU. Bozuk manifest patlamali.

    "Dosya yok" ile "dosya bozuk" ayri seylerdir. Birincisi kutuphane
    olarak kurulmus olmak demek; ikincisi manifestin bozulmus olmasi --
    ve onu sessiz gecirmek, kapinin varlik sebebini ortadan kaldirir.
    """
    import json

    veri = tmp_path / "data"
    veri.mkdir()
    (veri / "sources.yml").write_text("{bozuk json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        model_girdisi_yasak_yollar(root=tmp_path)


def test_manifest_varken_kapi_aynen_calisir() -> None:
    """Duzeltmenin kapiyi ZAYIFLATMADIGININ kaniti.

    Yukaridaki iki test toleransi olcer; bu test toleransin depoda
    HICBIR SEYI degistirmedigini olcer. Ucu birlikte olmazsa "kapi
    atlandi" ile "kapi calisiyor" ayirt edilemez.
    """
    yasak = model_girdisi_yasak_yollar()
    assert yasak, "Depoda yasakli yol goruNMUYOR -- kapi fiilen kapali."

    with pytest.raises(ValueError, match="UYGUNLUK IHLALI"):
        yasakli_aileleri_dogrula({"kotu": next(iter(yasak))})
