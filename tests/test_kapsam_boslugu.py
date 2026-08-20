"""ZAMAN kapsami ortusmeyen aile COKERTMEZ, ATLANIR.

NEDEN BU TEST DOSYASI
---------------------
2026-08-21, gercek veri provasi (``real_data_rehearsal.py``) yakaladi:
ayna verisi 2021-2022, ``turizm_geceleme`` tablosu YALNIZCA 2023-2025.
``year_lag=1`` ile panelin istedigi yillar 2020-2021 -- hic ortusmuyor.
Sonuc: ``add_annual_district_attribute`` %0 eslesme gorup ValueError firlatti
ve ``attach_external`` TUM hatti durdurdu.

Kapi iki FARKLI arizayi ayni kefeye koyuyordu:

  1. ANAHTAR BOZUK -- ilce adlari uyusmuyor (join_key/split_il_ilce sorunu).
     Bu gercekten durdurulmali: sessiz NaN sutunu uretmek en kotusudur.
  2. ZAMAN KAPSAMI ORTUSMUYOR -- anahtarlar gayet dogru, kaynak sadece baska
     bir donemi kapsiyor. Burada durmak YANLIS: eksik olan bir aile, bozuk
     bir anahtar degil.

Yarisma gunu ikincisi cok muhtemeldir (yarismanin donemi bizim tablolarimizin
donemiyle ayni olmak zorunda degil) ve tek aile yuzunden gun-1 hattinin komple
durmasi 30 dakikaya mal olurdu. Bu testler ayrimi sabitler.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.features.external import KapsamBoslugu, attach_external


def _panel(yil: int) -> pd.DataFrame:
    gunler = pd.date_range(f"{yil}-03-01", periods=10, freq="D")
    ilceler = ["konak", "bozdogan"]
    frame = pd.DataFrame([{"ilce_key": ilce, "gun": gun} for ilce in ilceler for gun in gunler])
    frame["hedef"] = np.arange(len(frame), dtype="float64")
    return frame


def _kaynaklari_yaz(kok, *, yillar: list[int], ilceler: list[str]) -> None:
    dizin = kok / "data" / "external"
    dizin.mkdir(parents=True, exist_ok=True)
    (kok / "data" / "reference").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "il_key": ["izmir"] * len(ilceler),
            "ilce_key": ilceler,
            "nufus": [100_000] * len(ilceler),
        }
    ).to_parquet(kok / "data" / "reference" / "ilceler_gdz_adm.parquet")
    pd.DataFrame(
        [
            {
                "yil": yil,
                "il_key": "izmir",
                "ilce_key": ilce,
                "geceleme": 1000.0 + yil,
                "tesise_gelis": 500.0 + yil,
            }
            for yil in yillar
            for ilce in ilceler
        ]
    ).to_parquet(dizin / "turizm_geceleme.parquet")


def test_zaman_ortusmezse_aile_atlanir_hat_durmaz(tmp_path) -> None:
    """Kaynak BASKA bir donemi kapsiyorsa: skip + rapor, cokme YOK."""
    # Arrange: panel 2021, kaynak 2023-2025 -> year_lag=1 ile hic ortusme yok
    panel = _panel(2021)
    _kaynaklari_yaz(tmp_path, yillar=[2023, 2024, 2025], ilceler=["konak", "bozdogan"])

    # Act
    sonuc = attach_external(
        panel,
        key_column="ilce_key",
        time_column="gun",
        horizon=7,
        families=("turizm_yillik",),
        root=tmp_path,
    )

    # Assert: hat AYAKTA, aile raporlanmis sekilde atlanmis
    assert len(sonuc.frame) == len(panel)
    assert "turizm_yillik" in sonuc.skipped
    assert "kapsam" in sonuc.skipped["turizm_yillik"].lower()
    assert "turizm_yillik" not in sonuc.families


def test_zaman_ortusuyorsa_normal_baglanir(tmp_path) -> None:
    """Kontrol grubu: ortusme VARSA aile normal baglanmali."""
    # Arrange: panel 2024, kaynak 2023-2025 -> year_lag=1 ile 2023 ortusur
    panel = _panel(2024)
    _kaynaklari_yaz(tmp_path, yillar=[2023, 2024, 2025], ilceler=["konak", "bozdogan"])

    # Act
    sonuc = attach_external(
        panel,
        key_column="ilce_key",
        time_column="gun",
        horizon=7,
        families=("turizm_yillik",),
        root=tmp_path,
    )

    # Assert
    assert "turizm_yillik" not in sonuc.skipped
    assert sonuc.families["turizm_yillik"]


def test_anahtar_bozuksa_hala_hata_verir(tmp_path) -> None:
    """Ayrim korunuyor mu: yil ortusuyor ama ILCE adlari uyusmuyorsa DURMALI.

    Bu testin varlik sebebi, kapsam muafiyetinin anahtar hatalarini da
    yutmasini engellemektir -- sessiz NaN sutunu en kotu sonuctur.
    """
    # Arrange: yillar ortusuyor (2023), ama kaynak baska ilceleri tasiyor
    panel = _panel(2024)
    _kaynaklari_yaz(tmp_path, yillar=[2023, 2024], ilceler=["yok1", "yok2"])

    # Act / Assert
    with pytest.raises(ValueError, match="HIC eslesmedi"):
        attach_external(
            panel,
            key_column="ilce_key",
            time_column="gun",
            horizon=7,
            families=("turizm_yillik",),
            root=tmp_path,
        )


def test_kapsam_boslugu_valueerror_alt_sinifi() -> None:
    """``KapsamBoslugu`` ValueError'dan turemeli -- eski yakalamalar kirilmasin."""
    assert issubclass(KapsamBoslugu, ValueError)
