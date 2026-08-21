"""Hedef kolonu METIN geldiyse ERKEN yakala -- 7/7'de degil.

NEDEN BU TEST DOSYASI (2026-08-21, dusmanca prova olctu)
--------------------------------------------------------
Hasim yarisma dosyasi ondalik VIRGULLU yazildi (Turkce Excel varsayilani).
``sniff_dialect_shared`` dosyaya bakip ``ondalik=','`` dedi -- dogru karar.
Ama ayni dosyadaki hedef kolon NOKTA ondalikliydi::

    kesinti_adedi = "5.0"   ->  decimal=',' ile sayiya cevrilemez  ->  dtype str

Hicbir hata cikmadi. Betik yedi asamanin ALTISINI kosturdu -- profil, CV,
feature, baseline, 5 tohumlu yeniden egitim -- ve ancak SUBMISSION adiminda,
tum egitim maliyeti odendikten sonra coktu::

    TypeError: operation 'mod' not supported for dtype 'str'

Yarisma gununde bunun bedeli submission'siz gecen bir saattir.

TASARIM: iki ondalik yorumu da denenir ama SESSIZ SECIM YOKTUR. Hangisinin
kullanildigi ve kac degerin kurtarildigi raporlanir; ikisi de olmuyorsa
istisna ERKEN atilir.
"""

from __future__ import annotations

import pandas as pd
import pytest

from gridup.validation import hedefi_sayisallastir


def test_zaten_sayisal_hedef_dokunulmaz() -> None:
    # Arrange
    seri = pd.Series([0, 3, 5, 2])

    # Act
    sonuc = hedefi_sayisallastir(seri, ad="kesinti")

    # Assert
    assert sonuc.deger.tolist() == [0, 3, 5, 2]
    assert sonuc.donusum == "gerek yok"


def test_nokta_ondalikli_metin_kurtarilir() -> None:
    """Olculen ariza: ondalik-virgullu dosyada nokta-ondalikli hedef kolon."""
    seri = pd.Series(["5.0", "0.0", "12.0"])

    sonuc = hedefi_sayisallastir(seri, ad="kesinti_adedi")

    assert sonuc.deger.tolist() == [5.0, 0.0, 12.0]
    assert sonuc.donusum == "nokta"


def test_virgul_ondalikli_metin_kurtarilir() -> None:
    seri = pd.Series(["5,5", "0,0", "12,25"])

    sonuc = hedefi_sayisallastir(seri, ad="sure")

    assert sonuc.deger.tolist() == [5.5, 0.0, 12.25]
    assert sonuc.donusum == "virgul"


def test_bos_degerler_korunur() -> None:
    """Eksik deger NaN kalir; 0 ile DOLDURULMAZ (0 gercek bir kesinti sayisidir)."""
    seri = pd.Series(["5.0", "", None, "1.0"])

    sonuc = hedefi_sayisallastir(seri, ad="kesinti")

    assert sonuc.deger.isna().tolist() == [False, True, True, False]


def test_daha_cok_deger_kurtaran_yorum_secilir() -> None:
    """Iki yorum da kismen calisiyorsa daha COK degeri cozen kazanir."""
    # 'virgul' yorumu 3/4'unu cozer, 'nokta' yorumu 1/4'unu
    seri = pd.Series(["1,5", "2,5", "3,5", "4.5"])

    sonuc = hedefi_sayisallastir(seri, ad="karisik")

    assert sonuc.donusum == "virgul"
    assert sonuc.kurtarilan == 3


def test_cozulemeyen_hedef_erken_patlar() -> None:
    """Metin hedef sayiya donmuyorsa 7/7'ye kadar beklenmez."""
    seri = pd.Series(["az", "cok", "orta"])

    with pytest.raises(ValueError, match="sayisallastirilamadi"):
        hedefi_sayisallastir(seri, ad="seviye")


def test_kismi_kurtarma_esigin_altinda_patlar() -> None:
    """Yarisindan azi cozuluyorsa bu bir bicim sorunu degil, YANLIS KOLONDUR."""
    seri = pd.Series(["1.0", "elma", "armut", "kiraz", "erik"])

    with pytest.raises(ValueError, match="sayisallastirilamadi"):
        hedefi_sayisallastir(seri, ad="karisik")


def test_rapor_satiri_sayilari_tasir() -> None:
    """Log'a dusen satir tek basina okunabilir olmali."""
    sonuc = hedefi_sayisallastir(pd.Series(["5.0", "1.0"]), ad="kesinti")

    metin = str(sonuc)
    assert "kesinti" in metin
    assert "nokta" in metin
