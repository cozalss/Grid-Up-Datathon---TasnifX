"""Aile katkisina HUKUM: gurultuyu gecmeyen etki "faydali" sayilmaz.

NEDEN BU TEST DOSYASI
---------------------
2026-08-21 olcumu: ayni ayna verisinde ayni ablasyon iki kez kosuldu ve
YEDI ailenin BESINDE isaret degisti.

    konvektif  +4,12 -> -0,41      gunes  -1,16 -> +0,85
    epias      +3,13 -> -1,51      hava   -0,58 -> +0,23
    hava_saatlik +2,65 -> -0,33

Sebep: modelin tohum gurultusu ~1,24 MAE, aile etkileri ise +-2 MAE. Etki
gurultuyle AYNI MERTEBEDE oldugu icin kucuk etkilerin isareti yazi-turadir.
Buna ragmen betik tek kosunun ham deltasini SIRALAMA olarak raporluyordu --
yani yanlis guven uretiyordu.

Duzeltme: cok tohumlu, ESLESTIRILMIS fark. Ayni tohum ve ayni fold'larda
"aile var" ile "aile yok" karsilastirilir, boylece ortak gurultu sadelesir.
Sonra tohumlar arasi yayilim bakilir ve hukum verilir:

  FAYDALI  -- etki hem pratik esigi hem istatistiksel belirsizligi asiyor
  ZARARLI  -- ayni sey ters yonde
  KARARSIZ -- ayirt edilemiyor; "bilmiyorum" demek dogru cevaptir

"Bilmiyorum" demek bu depoda bir basarisizlik degil; yarisma gunu gurultuye
inanip 2 gun yanlis yone kosmayi engelleyen sey tam olarak budur.
"""

from __future__ import annotations

import pytest

from gridup.ablation import aile_hukmu


def test_buyuk_ve_tutarli_etki_faydali() -> None:
    # Arrange: lag ailesi gibi -- buyuk, her tohumda ayni yon
    deltalar = [8.5, 8.1, 8.9, 8.3, 8.6]

    # Act
    hukum = aile_hukmu(deltalar, gurultu=1.24)

    # Assert
    assert hukum.karar == "FAYDALI"
    assert hukum.ortalama == pytest.approx(8.48, abs=0.01)


def test_buyuk_ve_tutarli_negatif_zararli() -> None:
    # Arrange: tatil ailesi gibi
    deltalar = [-7.8, -8.2, -7.1, -8.9, -7.5]

    # Act / Assert
    assert aile_hukmu(deltalar, gurultu=1.24).karar == "ZARARLI"


def test_isaret_donen_etki_kararsiz() -> None:
    """Bu, duzeltmenin varlik sebebi: CAPE'in gercek olcum deseni."""
    # Arrange: iki kosuda +4,12 ve -0,41 gormustuk; benzer bir yayilim
    deltalar = [4.12, -0.41, 1.2, -2.0, 0.8]

    # Act
    hukum = aile_hukmu(deltalar, gurultu=1.24)

    # Assert: ortalama pozitif ama yayilim onu yutuyor
    assert hukum.ortalama > 0
    assert hukum.karar == "KARARSIZ"


def test_kucuk_ama_tutarli_etki_pratik_esikte_kalir() -> None:
    """Istatistiksel olarak tutarli ama PRATIK olarak onemsiz etki.

    Yayilim cok kucuk oldugu icin "sifirdan farkli" denebilir; ama etki
    tohum gurultusunun altindaysa gonderimde bir sey degistirmez. Iki esik
    de ayri ayri aranir -- yalnizca istatistige bakmak, olcum hassaslastikca
    her seyi "anlamli" ilan etmeye goturur.
    """
    # Arrange: ortalama 0,3 MAE, yayilim neredeyse yok
    deltalar = [0.30, 0.31, 0.29, 0.30, 0.30]

    # Act
    hukum = aile_hukmu(deltalar, gurultu=1.24)

    # Assert
    assert hukum.karar == "KARARSIZ"
    assert "pratik" in hukum.gerekce.lower()


def test_tek_olcum_asla_hukum_veremez() -> None:
    """Tek tohumla yayilim olculemez -- eski davranisin ta kendisi."""
    # Act
    hukum = aile_hukmu([4.12], gurultu=1.24)

    # Assert
    assert hukum.karar == "KARARSIZ"
    assert "tek" in hukum.gerekce.lower()


def test_bos_liste_hata() -> None:
    with pytest.raises(ValueError, match="en az bir"):
        aile_hukmu([], gurultu=1.24)


def test_negatif_gurultu_hata() -> None:
    with pytest.raises(ValueError, match="gurultu"):
        aile_hukmu([1.0, 2.0], gurultu=-1.0)


def test_hukum_metni_sayilari_tasir() -> None:
    """Rapor satiri tek basina okunabilir olmali (log'a dusunce anlasilsin)."""
    # Act
    metin = str(aile_hukmu([8.5, 8.1, 8.9], gurultu=1.24))

    # Assert
    assert "FAYDALI" in metin
    assert "8.5" in metin or "8.50" in metin
