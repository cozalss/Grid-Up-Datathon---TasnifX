"""FEATURE AILESI ABLASYONU: ``scripts/ablation_gercek.py`` sozlesme testleri.

Betigin tam kosusu gercek veri ister ve ~40 sn surer -- burada KOSULMAZ.
Bu testler hafif sozlesmeyi kilitler:

  1. Betik derlenebilir (veri gununde SyntaxError ile karsilasmayalim).
  2. ``experiments/ablasyon_gercek.json`` gecerli ve zorunlu alanlari tasiyor.
  3. Ic tutarlilik: delta = mae_ailesiz - tam_mae, siralama delta'ya gore
     azalan, panel satir = ilce x gun (tam izgara).

OLCULEN (2026-08-15 kosusu, bu dosyanin dogruladigi JSON):
  tam_mae=313.6376  sifir_baseline=366.9741  fold_std=94.305
  panel 22.184 = 47 x 472, 7 aile / 76 kolon, en buyuk delta lag=+22.3431
"""

from __future__ import annotations

import json
import py_compile
from pathlib import Path

import pytest

KOK = Path(__file__).resolve().parents[1]
BETIK = KOK / "scripts" / "ablation_gercek.py"
SONUC = KOK / "experiments" / "ablasyon_gercek.json"

#: Gorev tanimindaki yedi aile -- eksigi/fazlasi sozlesme ihlalidir.
AILELER = {"takvim", "tatil", "hava", "gunes", "lag", "komsu", "frekans"}

ZORUNLU_ALANLAR = {
    "tam_mae",
    "sifir_baseline",
    "fold_std",
    "aileler",
    "siralama",
    "gain_top15",
    "panel",
}


@pytest.fixture(scope="module")
def sonuc() -> dict:
    assert SONUC.exists(), f"{SONUC} yok -- once betigi kos: python scripts/ablation_gercek.py"
    return json.loads(SONUC.read_text(encoding="utf-8"))


def test_betik_derlenebilir():
    """Veri gunu betigi acilmadan SyntaxError yakalansin."""
    py_compile.compile(str(BETIK), doraise=True)


def test_zorunlu_alanlar_tam(sonuc):
    assert set(sonuc) >= ZORUNLU_ALANLAR, f"eksik alan: {ZORUNLU_ALANLAR - set(sonuc)}"
    for alan in ("tam_mae", "sifir_baseline", "fold_std"):
        assert isinstance(sonuc[alan], (int, float)), f"{alan} sayi olmali"
        assert sonuc[alan] >= 0


def test_yedi_ailenin_hepsi_olculmus(sonuc):
    assert set(sonuc["aileler"]) == AILELER
    for ad, bilgi in sonuc["aileler"].items():
        assert isinstance(bilgi["mae_ailesiz"], (int, float)), ad
        assert bilgi["mae_ailesiz"] > 0, ad
        assert isinstance(bilgi["delta"], (int, float)), ad
        assert isinstance(bilgi["kolon_sayisi"], int), ad
        assert bilgi["kolon_sayisi"] >= 1, f"{ad} ailesi bos -- kurulum bozuk"


def test_delta_tanimi_tutarli(sonuc):
    """delta = mae_ailesiz - tam_mae; JSON kendi icinde celismemeli.

    Yuvarlama 4 basamak oldugu icin tolerans 1e-3 (iki yuvarlama farki).
    """
    for ad, bilgi in sonuc["aileler"].items():
        beklenen = bilgi["mae_ailesiz"] - sonuc["tam_mae"]
        assert abs(bilgi["delta"] - beklenen) < 1e-3, (
            f"{ad}: delta={bilgi['delta']} ama mae_ailesiz - tam_mae={beklenen:.4f}"
        )


def test_siralama_deltaya_gore_azalan(sonuc):
    """Siralama = veri gununun oncelik listesi -- en cok katki veren onde."""
    assert set(sonuc["siralama"]) == AILELER
    assert len(sonuc["siralama"]) == len(AILELER), "tekrarli aile adi var"
    deltalar = [sonuc["aileler"][ad]["delta"] for ad in sonuc["siralama"]]
    assert deltalar == sorted(deltalar, reverse=True), (
        f"siralama delta'ya gore azalan degil: {deltalar}"
    )


def test_gain_top15_bicimi(sonuc):
    tablo = sonuc["gain_top15"]
    assert 1 <= len(tablo) <= 15
    for giris in tablo:
        assert len(giris) == 2, f"[kolon, onem] bekleniyor: {giris}"
        kolon, onem = giris
        assert isinstance(kolon, str) and kolon
        assert isinstance(onem, (int, float)) and onem >= 0
    onemler = [onem for _, onem in tablo]
    assert onemler == sorted(onemler, reverse=True), "gain azalan sirali olmali"


def test_panel_tam_izgara(sonuc):
    """build_panel tam izgara kurar: satir = ilce x gun olmali.

    OLCULDU: 22.184 = 47 x 472. Esitlik bozulursa panel kurulumunda satir
    kaybi/tekrari var demektir -- fold'lar sessizce kayar.
    """
    panel = sonuc["panel"]
    for alan in ("satir", "ilce", "gun"):
        assert isinstance(panel[alan], int) and panel[alan] > 0
    assert panel["satir"] == panel["ilce"] * panel["gun"]


def test_model_sifir_baselinei_geciyor(sonuc):
    """Tam model hep-sifir tahmininden iyi olmali; degilse hat bozuk.

    OLCULDU: 313.6376 < 366.9741 (%14.5 daha az sapma).
    """
    assert sonuc["tam_mae"] < sonuc["sifir_baseline"]
