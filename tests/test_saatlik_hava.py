"""Saatlik hava turev tablosunun (hava_saatlik_turev.parquet) kontratlari.

Bu testler GERCEK indirilmis veri uzerinde calisir (data/external). Veri
yoksa atlanir -- CI'da / taze klonda veri olmayabilir ama yerelde varsa
sema, kapsam ve fiziksel makullik dogrulanmis olur. Ayni koruma kalibi
tests/test_weather_spatial.py'de de var.

Tablonun ureticisi: scripts/fetch_hourly_weather.py
Kolon sozlesmesi orada FINAL_COLUMNS olarak tanimlidir; buradaki liste
onun bagimsiz bir kopyasidir -- betikten import etmiyoruz ki betikteki
bir gerileme testi de sessizce pesinden suruklemesin.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SAATLIK_PATH = ROOT / "data" / "external" / "hava_saatlik_turev.parquet"
REFERANS_PATH = ROOT / "data" / "reference" / "ilceler_gdz_adm.parquet"

BEKLENEN_KOLONLAR = [
    "ilce_key",
    "tarih",
    "basinc_min",
    "basinc_ort",
    "ruzgar_15ms_saat",
    "ruzgar_20ms_saat",
    "hamle_20ms_saat",
    "yon_std",
    "yon_degisim",
]


@pytest.fixture(scope="module")
def saatlik() -> pd.DataFrame:
    if not SAATLIK_PATH.exists():
        pytest.skip(
            "Saatlik hava turevi yok: scripts/fetch_hourly_weather.py calistir "
            "(taze klonda beklenen durum)"
        )
    return pd.read_parquet(SAATLIK_PATH)


class TestSema:
    def test_kolonlar_birebir_sozlesme(self, saatlik: pd.DataFrame) -> None:
        """Fazla kolon da eksik kolon kadar hatadir -- sema birebir eslesmelidir."""
        assert list(saatlik.columns) == BEKLENEN_KOLONLAR

    def test_tekrar_eden_satir_yok(self, saatlik: pd.DataFrame) -> None:
        assert saatlik.duplicated(subset=["ilce_key", "tarih"]).sum() == 0


class TestKapsam:
    def test_tum_ilceler_mevcut(self, saatlik: pd.DataFrame) -> None:
        if not REFERANS_PATH.exists():
            pytest.skip("Referans ilce tablosu yok: scripts/fetch_districts.py calistir")
        referans = set(pd.read_parquet(REFERANS_PATH)["ilce_key"])
        eksik = referans - set(saatlik["ilce_key"])
        assert not eksik, f"Referanstaki {len(eksik)} ilce hava verisinde yok: {sorted(eksik)}"

    def test_tarih_araligi_2020_2026_kapsar(self, saatlik: pd.DataFrame) -> None:
        assert saatlik["tarih"].min() <= pd.Timestamp("2020-01-31")
        assert saatlik["tarih"].max() >= pd.Timestamp("2026-08-01")


class TestMakullik:
    def test_basinc_fiziksel_aralikta(self, saatlik: pd.DataFrame) -> None:
        """surface_pressure ISTASYON seviyesidir, deniz seviyesi degil: en
        yuksek rakimli ilce Cameli (~1400 m) ~850 hPa civarinda oturur ve
        derin bir alcakta altina sarkar (olculen taban: 848.8, 2026-02-13).
        Alt sinir 800: rakim + firtina payi birakir ama birim karismasini
        yine yakalar (Pa ~1e5, kPa ~100, inHg ~30, mmHg tipik 730-780)."""
        for kolon in ["basinc_min", "basinc_ort"]:
            degerler = saatlik[kolon].dropna()
            assert degerler.between(800.0, 1100.0).all(), f"{kolon} aralik disi"

    def test_basinc_min_ortalamayi_asamaz(self, saatlik: pd.DataFrame) -> None:
        ortak = saatlik.dropna(subset=["basinc_min", "basinc_ort"])
        assert (ortak["basinc_min"] <= ortak["basinc_ort"] + 1e-3).all()

    def test_saat_kolonlari_0_ile_24_arasinda(self, saatlik: pd.DataFrame) -> None:
        """Gunde 24 saat vardir; Turkiye kalici UTC+3 (yaz saati yok), 25
        saatlik gun de olamaz."""
        for kolon in ["ruzgar_15ms_saat", "ruzgar_20ms_saat", "hamle_20ms_saat"]:
            degerler = saatlik[kolon].dropna()
            assert degerler.between(0, 24).all(), f"{kolon} aralik disi"

    def test_yon_degisim_0_ile_180_arasinda(self, saatlik: pd.DataFrame) -> None:
        """Iki yon arasindaki dairesel fark en fazla 180 derecedir; ustu,
        modulo hesabinin yanlis oldugu anlamina gelir."""
        degerler = saatlik["yon_degisim"].dropna()
        assert degerler.between(0.0, 180.0).all()

    def test_yon_std_negatif_degil(self, saatlik: pd.DataFrame) -> None:
        assert (saatlik["yon_std"].dropna() >= 0.0).all()
