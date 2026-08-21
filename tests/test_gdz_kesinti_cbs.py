from __future__ import annotations

import importlib.util
from pathlib import Path

import requests


def _modul():
    yol = Path(__file__).resolve().parents[1] / "scripts" / "fetch_gdz_kesinti_cbs.py"
    spec = importlib.util.spec_from_file_location("fetch_gdz_kesinti_cbs", yol)
    assert spec is not None and spec.loader is not None
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


def test_wkt_nokta_lon_lat_sirasini_guvenli_cevirir():
    modul = _modul()
    assert modul._wkt_nokta("POINT(27.1914 38.4764)") == (38.4764, 27.1914)
    assert modul._wkt_nokta(None) == (None, None)


def test_satir_trafo_eslesmesi_uydurmaz():
    modul = _modul()
    satir = modul._satira_cevir(
        {
            "Kesinti_ID": "3797307",
            "Sehir": "\u0130ZM\u0130R",
            "Ilce": "BORNOVA",
            "CBS_Koordinat": "POINT(27.1914 38.4764)",
            "Mahalle": ["Atat\u00fcrk"],
            "Sokak": ["900. Sk.", "905. Sk."],
            "Musteri_Koordinat": ["POINT(27.19 38.47)"],
        },
        cekilme_zamani="2026-08-21T00:00:00+00:00",
    )

    assert satir["cbs_lat"] == 38.4764
    assert satir["cbs_lon"] == 27.1914
    assert satir["sokak_sayisi"] == 2
    assert satir["trafo_kodu"] is not None
    assert satir["trafo_eslesme_durumu"].startswith("eslesmedi")


def test_json_utf8_turkce_karakterleri_korur():
    modul = _modul()
    yanit = requests.Response()
    yanit._content = '{"data":["İZMİR","Ağ çalışması"]}'.encode()
    yanit.encoding = "ISO-8859-1"  # Sunucunun eksik charset durumunu taklit eder.

    assert modul._json_utf8(yanit)["data"] == ["İZMİR", "Ağ çalışması"]
