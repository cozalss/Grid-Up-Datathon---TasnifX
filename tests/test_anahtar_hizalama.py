"""Panel ilce anahtarini REFERANSA hizalama: sessiz kismi eslesmeyi bitirir.

NEDEN BU TEST DOSYASI (2026-08-21, dusmanca prova yakaladi)
-----------------------------------------------------------
``attach_external`` %0 eslesmede ValueError atiyor, %50'nin altinda uyariyor.
Arada bir kor bant var ve gercek veri tam oraya dusuyor:

    96 ilcenin 91'i esliyor  ->  %94,8  ->  ne hata, ne uyari.

Eslesmeyen 5 ilce EPIAS'in kendi yazimindan geliyor::

    'BOZKURT / DENIZLI'   niteleyici eki   (ayni ad Kastamonu'da da var)
    'KALE / DENIZLI'      niteleyici eki
    'KOPRUBASI / MANISA'  niteleyici eki
    'YENIPAZAR / AYDIN'   niteleyici eki
    'AYDIN MERKEZ'        2012 oncesi ad   (bugun 'efeler')

Bu 5 ilce icin 219 dis kolonun HEPSI NaN olur ve model bunu "bilgi yok" diye
degil "bu ilcede orman yok, altyapi yok, turizm yok" diye okur.

TASARIM KARARI: yeniden yazim REFERANSA KOSULLUDUR
--------------------------------------------------
Niteleyici atmayi kosulsuz uygulamak yeni bir hata sinifi acardi: adinda
tire veya parantez bulunan mesru bir ilce sessizce baska bir ilceye
baglanabilirdi. Bu yuzden her kurtarma adayi ancak REFERANS KUMESINE
dusuyorsa kabul edilir; dusmuyorsa ham anahtar korunur ve
``BULUNAMADI`` olarak RAPORLANIR. Uydurma yok, sessiz duzeltme yok.
"""

from __future__ import annotations

import pytest

from gridup.turkish import hizala_ilce_anahtarlari

REFERANS = ("bozkurt", "kale", "koprubasi", "yenipazar", "efeler", "cigli", "konak")


def test_zaten_dogru_anahtar_dogrudan_gecer() -> None:
    # Arrange / Act
    sonuc = hizala_ilce_anahtarlari(["cigli"], referans=REFERANS)

    # Assert
    assert sonuc["cigli"].anahtar == "cigli"
    assert sonuc["cigli"].yontem == "dogrudan"


def test_turkce_buyuk_harf_dogrudan_cozulur() -> None:
    """``ÇİĞLİ`` -> ``cigli``: i-tuzagi join_key katmaninda coz."""
    sonuc = hizala_ilce_anahtarlari(["ÇİĞLİ"], referans=REFERANS)

    assert sonuc["ÇİĞLİ"].anahtar == "cigli"
    assert sonuc["ÇİĞLİ"].yontem == "dogrudan"


def test_niteleyici_eki_atilir() -> None:
    """Gercek EPIAS yazimi: ayni ad birden fazla ilde oldugu icin niteleniyor."""
    sonuc = hizala_ilce_anahtarlari(["BOZKURT / DENIZLI"], referans=REFERANS)

    assert sonuc["BOZKURT / DENIZLI"].anahtar == "bozkurt"
    assert sonuc["BOZKURT / DENIZLI"].yontem == "niteleyici"


def test_bilesik_anahtardan_ilce_alinir() -> None:
    """2024 GDZ bicimi: ``izmir-cigli`` -- ilce SAGDA."""
    sonuc = hizala_ilce_anahtarlari(["izmir-cigli"], referans=REFERANS)

    assert sonuc["izmir-cigli"].anahtar == "cigli"
    assert sonuc["izmir-cigli"].yontem == "bilesik"


def test_takma_ad_son_care_olarak_uygulanir() -> None:
    """2012 buyuksehir yasasi: 'Aydin Merkez' bugun 'Efeler'."""
    sonuc = hizala_ilce_anahtarlari(
        ["AYDIN MERKEZ"], referans=REFERANS, takma_adlar={"aydin merkez": "efeler"}
    )

    assert sonuc["AYDIN MERKEZ"].anahtar == "efeler"
    assert sonuc["AYDIN MERKEZ"].yontem == "takma_ad"


def test_referansta_olmayan_ad_uydurulmaz() -> None:
    """Kurtarilamayan anahtar SESSIZCE degistirilmez, BULUNAMADI olarak isaretlenir."""
    sonuc = hizala_ilce_anahtarlari(["Vestel Sehri"], referans=REFERANS)

    assert sonuc["Vestel Sehri"].yontem == "BULUNAMADI"
    assert sonuc["Vestel Sehri"].anahtar == "vestel sehri"


def test_benzeyen_ad_yanlis_ilceye_baglanmaz() -> None:
    """Kritik guvence: kurtarma bir adi BASKA bir ilceye baglayamaz.

    ``kale-i sultaniye`` referansta olan ``kale`` ile ayni harflerle
    basliyor. Kosulsuz "tireden once kes" kurali bunu sessizce Denizli'nin
    Kale ilcesine baglardi -- her gunu yanlis ilcenin havasiyla etiketlemek
    demektir bu. Kurtarma zinciri hicbir adayi referansta bulamayinca adi
    DEGISTIRMEZ; ``BULUNAMADI`` der ve karari cagirana birakir.

    Ayni cagride gercek niteleyici (``KALE / DENIZLI``) cozulmeye devam
    etmeli: guvenlik, isleyen kurtarmayi bozmamali.
    """
    sonuc = hizala_ilce_anahtarlari(["kale-i sultaniye", "KALE / DENIZLI"], referans=REFERANS)

    assert sonuc["kale-i sultaniye"].yontem == "BULUNAMADI"
    assert sonuc["kale-i sultaniye"].anahtar != "kale"
    assert sonuc["KALE / DENIZLI"].anahtar == "kale"
    assert sonuc["KALE / DENIZLI"].yontem == "niteleyici"


def test_bos_referans_hata() -> None:
    with pytest.raises(ValueError, match="referans"):
        hizala_ilce_anahtarlari(["cigli"], referans=[])


def test_tekrarlanan_ham_ad_tek_kayit_uretir() -> None:
    sonuc = hizala_ilce_anahtarlari(["cigli", "cigli", "ÇİĞLİ"], referans=REFERANS)

    assert set(sonuc) == {"cigli", "ÇİĞLİ"}


def test_gercek_epias_besligi_tamamen_kurtarilir() -> None:
    """Olculen 5 arizanin HEPSI tek cagriyla cozulmeli."""
    # Arrange: dusmanca provada BULUNAMAYAN tam liste
    ham = [
        "BOZKURT / DENIZLI",
        "KALE / DENIZLI",
        "KÖPRÜBAŞI / MANISA",
        "YENIPAZAR / AYDIN",
        "AYDIN MERKEZ",
    ]

    # Act
    sonuc = hizala_ilce_anahtarlari(ham, referans=REFERANS, takma_adlar={"aydin merkez": "efeler"})

    # Assert
    assert [sonuc[a].anahtar for a in ham] == [
        "bozkurt",
        "kale",
        "koprubasi",
        "yenipazar",
        "efeler",
    ]
    assert all(sonuc[a].yontem != "BULUNAMADI" for a in ham)
