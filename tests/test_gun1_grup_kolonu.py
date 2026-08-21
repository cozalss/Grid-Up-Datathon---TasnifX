"""Grup kolonu SEZILDIYSE benimsenmeli -- yoksa 219 dis kolon sessizce kaybolur.

NEDEN BU TEST DOSYASI (2026-08-21, dusmanca prova olctu)
--------------------------------------------------------
``day_one.py`` grup kolonunu ``suggest_scheme`` ile seziyor ve ekrana
yaziyordu ("Grup kolonu: il"), ama sonucu ``args.group_column``a GERI
YAZMIYORDU. Iki yer bu degiskene bakiyor::

    if time_column and args.group_column:      -> panel kurulumu
    if ... and args.group_column and ...:      -> attach_external

Yani ``--group`` bayragi elle verilmediginde her ikisi de SESSIZCE atlaniyordu.
Hata cikmiyor, uyari cikmiyor; kosu "basarili" bitiyor ve gecerli bir
submission uretiyor -- yalnizca hava, arazi ortusu, altyapi, turizm, deprem,
yangin ailelerinin HICBIRI baglanmamis oluyor. Olculdu: 27 feature (beklenen
~230), MAE 3.07.

Yarisma gunu bunun bedeli, sebebi gorunmeyen kalici bir skor kaybidir.

IKINCI SORUN: HANGI aday secilecek?
------------------------------------
Sezici birden fazla aday buldugunda ilkini aliyordu ve gercek veride bu ``il``
oldu (5 deger) -- oysa dis tablolarin tamami ``ilce`` anahtarli (96 deger).
Dogru aday, DIS TABLOLARIN ANAHTARINA EN COK ESLESEN adaydir; bu bir tercih
degil, olculebilir bir sey. ``grup_adayini_sec`` bunu olcer.
"""

from __future__ import annotations

import pandas as pd
import pytest

from gridup.turkish import grup_adayini_sec

REFERANS = ("cigli", "konak", "bornova", "efeler", "bodrum")


def test_ilce_adayi_il_adayina_tercih_edilir() -> None:
    """Olculen ariza: sezici 'il' dedi, dis tablolar 'ilce' anahtarli."""
    # Arrange
    frame = pd.DataFrame(
        {
            "il": ["IZMIR", "IZMIR", "AYDIN", "MUGLA"],
            "ilce": ["ÇİĞLİ", "KONAK", "EFELER", "BODRUM"],
        }
    )

    # Act
    secim = grup_adayini_sec(frame, adaylar=["il", "ilce"], referans=REFERANS)

    # Assert
    assert secim.kolon == "ilce"
    assert secim.eslesme_orani == pytest.approx(1.0)


def test_aday_sirasi_sonucu_degistirmez() -> None:
    """Secim OLCUME dayanir, listedeki siraya degil."""
    frame = pd.DataFrame(
        {
            "ilce": ["ÇİĞLİ", "KONAK"],
            "il": ["IZMIR", "IZMIR"],
        }
    )

    ileri = grup_adayini_sec(frame, adaylar=["il", "ilce"], referans=REFERANS)
    geri = grup_adayini_sec(frame, adaylar=["ilce", "il"], referans=REFERANS)

    assert ileri.kolon == geri.kolon == "ilce"


def test_hicbiri_eslesmezse_ilk_aday_dondurulur() -> None:
    """Referansa hic eslesme yoksa sezicinin sirasina SAYGI duyulur.

    Eslesme yoksa bu bir ilce paneli olmayabilir (baska bir varlik tipi).
    Uydurma bir secim yapmak yerine sezicinin karari korunur ve oran 0
    raporlanir -- cagiran taraf bunu gorup karar verir.
    """
    frame = pd.DataFrame({"magaza": ["A", "B"], "bolge": ["X", "Y"]})

    secim = grup_adayini_sec(frame, adaylar=["magaza", "bolge"], referans=REFERANS)

    assert secim.kolon == "magaza"
    assert secim.eslesme_orani == 0.0


def test_niteleyicili_adlar_da_sayilir() -> None:
    """Kurtarilabilir ad eslesme sayilir -- hizalama zaten onu cozecek."""
    frame = pd.DataFrame({"ilce": ["BOZKURT / DENIZLI", "ÇİĞLİ"]})

    secim = grup_adayini_sec(frame, adaylar=["ilce"], referans=(*REFERANS, "bozkurt"))

    assert secim.eslesme_orani == pytest.approx(1.0)


def test_bos_aday_listesi_none_dondurur() -> None:
    secim = grup_adayini_sec(pd.DataFrame({"a": [1]}), adaylar=[], referans=REFERANS)

    assert secim.kolon is None
    assert secim.eslesme_orani == 0.0


def test_framede_olmayan_aday_atlanir() -> None:
    frame = pd.DataFrame({"ilce": ["ÇİĞLİ"]})

    secim = grup_adayini_sec(frame, adaylar=["yok_boyle", "ilce"], referans=REFERANS)

    assert secim.kolon == "ilce"


def test_rapor_metni_olculen_orani_tasir() -> None:
    frame = pd.DataFrame({"ilce": ["ÇİĞLİ", "KONAK"], "il": ["IZMIR", "IZMIR"]})

    metin = str(grup_adayini_sec(frame, adaylar=["il", "ilce"], referans=REFERANS))

    assert "ilce" in metin
    assert "%" in metin
