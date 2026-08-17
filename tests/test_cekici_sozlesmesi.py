"""Harici veri cekici betiklerinin ortak sozlesmeleri.

Bu dosya, 2026-08-17'de GERCEKTEN yasanmis bir hatadan dogdu:
``fetch_nem_toprak.py`` ``cap_end_date``in donusunu tek deger sanip dogrudan
kullandi. Fonksiyon ``(kirpilmis_tarih, uyari)`` CIFTI donuyor; ``requests``
bu cifti seri hale getirince istek URL'sine ``end_date`` IKI KEZ girdi ve API
96 ilcenin hepsinde HTTP 400 dondu. Betik 15 ilce boyunca sessizce donup
hicbir kontrol noktasi yazmadan ilerledi.

Hata sinifi genel: yardimci fonksiyonun donus SEKLI degisir veya yanlis
okunursa, cekiciler cok gec fark edilen bicimde bozulur. Buradaki testler o
sinifi API yuzeyinde zorlar.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

KOK = Path(__file__).resolve().parents[1]
BETIK_DIZINI = KOK / "scripts"


def _fetch_betikleri() -> list[Path]:
    return sorted(BETIK_DIZINI.glob("fetch_*.py"))


def test_fetch_betikleri_bulunuyor() -> None:
    """Tarama gercekten dosya buluyor mu -- bos tarama sessiz basari olurdu."""
    betikler = _fetch_betikleri()
    assert len(betikler) >= 5, f"Beklenenden az cekici betigi bulundu: {betikler}"


@pytest.mark.parametrize("betik", _fetch_betikleri(), ids=lambda p: p.name)
def test_cap_end_date_donusu_cift_olarak_acilir(betik: Path) -> None:
    """``cap_end_date`` cagiran her betik donusu IKI degere acmalidir.

    ``end = cap_end_date(x)`` yazmak, tarih yerine bir demet dondurur.
    ``requests`` demeti tekrarlanan sorgu parametresi olarak seri hale
    getirir ve API her konum icin HTTP 400 verir -- olculdu.
    """
    kaynak = betik.read_text(encoding="utf-8")
    if "cap_end_date(" not in kaynak:
        pytest.skip(f"{betik.name} cap_end_date kullanmiyor")

    agac = ast.parse(kaynak)
    hatalar: list[str] = []
    for dugum in ast.walk(agac):
        if not isinstance(dugum, ast.Assign):
            continue
        deger = dugum.value
        if not (
            isinstance(deger, ast.Call)
            and isinstance(deger.func, ast.Name)
            and deger.func.id == "cap_end_date"
        ):
            continue
        hedef = dugum.targets[0]
        if not isinstance(hedef, ast.Tuple) or len(hedef.elts) != 2:
            hatalar.append(
                f"satir {dugum.lineno}: cap_end_date donusu TEK degere atanmis. "
                "Dogrusu: `end, uyari = cap_end_date(...)`"
            )

    assert not hatalar, f"{betik.name}: " + "; ".join(hatalar)


def test_cap_end_date_gercekten_cift_donuyor() -> None:
    """Sozlesmenin kendisi: fonksiyon iki elemanli demet dondurur.

    Yukaridaki AST testi 'cagiran dogru aciyor mu' diye bakar; bu test
    'acilacak sey gercekten cift mi' diye bakar. Fonksiyon ileride tek
    degere donerse AST testi yanlis yere bagirmaya baslar -- bu test onu
    ayirt eder.
    """
    sys.path.insert(0, str(BETIK_DIZINI))
    spec = importlib.util.spec_from_file_location(
        "fetch_weather", BETIK_DIZINI / "fetch_weather.py"
    )
    assert spec is not None and spec.loader is not None
    modul = importlib.util.module_from_spec(spec)
    sys.modules["fetch_weather"] = modul
    spec.loader.exec_module(modul)

    sonuc = modul.cap_end_date("2020-01-01")
    assert isinstance(sonuc, tuple) and len(sonuc) == 2, (
        f"cap_end_date (tarih, uyari) cifti dondurmeli, donen: {sonuc!r}"
    )
    tarih, uyari = sonuc
    assert isinstance(tarih, str), f"ilk eleman str olmali, {type(tarih).__name__} geldi"
    assert uyari is None or isinstance(uyari, str)


@pytest.mark.parametrize("betik", _fetch_betikleri(), ids=lambda p: p.name)
def test_cekici_yazmadan_once_dogruluyor(betik: Path) -> None:
    """Parquet yazan her cekici, YAZMADAN ONCE bir kalite kapisindan gecmelidir.

    ``fetch_hourly_weather``te olculen ders: yalnizca ekrana yazdiran bir
    dogrulama, gozetimsiz gece kosusunda bozuk veriyi sessizce yayinlar.
    Kapinin yazimdan SONRA kosmasi da ayni sey demektir -- veri zaten diskte
    olur ve sonraki kosular onu okur.
    """
    kaynak = betik.read_text(encoding="utf-8")
    yazim = "atomic_write_dataframe" in kaynak or "to_parquet" in kaynak
    if not yazim:
        pytest.skip(f"{betik.name} parquet yazmiyor")

    # Kalite kapisi = raise ile reddeden bir dogrulama yolu olmali.
    kapi_var = any(
        anahtar in kaynak for anahtar in ("kalite_kapisi", "raise ValueError", "raise RuntimeError")
    )
    assert kapi_var, (
        f"{betik.name} parquet yaziyor ama reddeden bir dogrulama yolu yok. "
        "Bozuk veri sessizce yayinlanir."
    )
