"""EPIAS kesinti panelinin sozlesmesi (P1-6, 2026-08-18 denetimi).

Uc tuzak sabitlenir:
  1. KAPSAMA: EPIAS arsivi delik desik (1690 gunun 406'sinda hic kayit yok).
     Bu gunler "kesinti olmadi" DEGIL "yayimlanmadi"dir; sahte sifir olarak
     panele girerse sifir orani %54,5'ten %65,4'e siser.
  2. MERKEZ ADLARI: "Aydin Merkez" -> efeler KURTARILIR (tek merkez ilce);
     Denizli/Manisa merkez adlari 2012'de IKIYE bolundugu icin BELIRSIZDIR
     ve uydurma bolme yerine DISARIDA birakilir.
  3. SATIR SAYISI: panel her zaman referans ilce x gun izgarasidir; olay
     kaydinin satir sayisiyla karistirilamaz.

Testler sentetik veriyle kosar -- gercek parquet CI'da yoktur.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

KOK = Path(__file__).resolve().parents[1]


def _betik():
    yol = KOK / "scripts" / "epias_panel.py"
    spec = importlib.util.spec_from_file_location("epias_panel", yol)
    modul = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(modul)
    return modul


def _referans() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "il_key": ["aydin", "aydin", "denizli", "manisa"],
            "ilce_key": ["efeler", "kusadasi", "merkezefendi", "sehzadeler"],
            "sirket": ["ADM", "ADM", "ADM", "GDZ"],
        }
    )


def _kayit(il: str, ilce: str, gun: str, dakika: int = 60, abone: int = 100) -> dict:
    bas = pd.Timestamp(f"{gun} 08:00", tz="Europe/Istanbul")
    return {
        "province": il,
        "district": ilce,
        "date": bas.isoformat(),
        "startTime": bas.isoformat(),
        "endTime": (bas + pd.Timedelta(minutes=dakika)).isoformat(),
        "effectedSubscribers": abone,
    }


def test_kapsanmayan_gun_sahte_sifir_uretmiyor() -> None:
    m = _betik()
    ham = pd.DataFrame(
        [
            _kayit("Aydın", "Kuşadası", "2026-01-01"),
            _kayit("Aydın", "Kuşadası", "2026-01-03"),  # 01-02 EPIAS'ta YOK
        ]
    )
    panel, rapor = m.panel_kur(ham, _referans())
    assert rapor["toplam_gun"] == 3 and rapor["kapsanan_gun"] == 2 and rapor["bos_gun"] == 1
    bos = panel[panel["gun"] == pd.Timestamp("2026-01-02")]
    assert len(bos) == 4 and (bos["kapsanan_gun"] == 0).all()
    # Kapsanan gunlerin sifir orani, tum izgaradan DUSUK olmali (sahte sifir yok)
    assert rapor["sifir_orani_kapsanan"] < rapor["sifir_orani_hepsi"]


def test_aydin_merkez_efelere_kurtariliyor() -> None:
    m = _betik()
    ham = pd.DataFrame(
        [
            _kayit("Aydın", "Aydın Merkez", "2026-01-01", dakika=30),
            _kayit("Aydın", "AYDIN", "2026-01-01", dakika=90),
            _kayit("Aydın", "Kuşadası", "2026-01-01"),
        ]
    )
    panel, rapor = m.panel_kur(ham, _referans())
    assert rapor["merkez_kurtarilan_kayit"] == 2
    efeler = panel[(panel["ilce_key"] == "efeler") & (panel["gun"] == pd.Timestamp("2026-01-01"))]
    assert len(efeler) == 1
    assert efeler["kesinti_adet"].iloc[0] == 2
    assert efeler["kesinti_dk"].iloc[0] == pytest.approx(120.0)


def test_belirsiz_merkezler_uydurulmuyor() -> None:
    """Denizli/Manisa merkezi IKI ilceye bolundu; kayit hangisine ait belli degil."""
    m = _betik()
    ham = pd.DataFrame(
        [
            _kayit("Denizli", "DENİZLİ", "2026-01-01"),
            _kayit("Manisa", "Manisa", "2026-01-01"),
            _kayit("Denizli", "Merkezefendi", "2026-01-01"),
        ]
    )
    panel, rapor = m.panel_kur(ham, _referans())
    assert rapor["belirsiz_merkez_kayit"] == 2
    merkez = panel[
        (panel["ilce_key"] == "merkezefendi") & (panel["gun"] == pd.Timestamp("2026-01-01"))
    ]
    # Yalnizca ACIKCA Merkezefendi olan kayit sayilir; belirsiz olan EKLENMEZ
    assert merkez["kesinti_adet"].iloc[0] == 1


def test_panel_referans_izgarasi_ve_kapsanmayan_ilce_raporu() -> None:
    m = _betik()
    ham = pd.DataFrame([_kayit("Aydın", "Kuşadası", "2026-01-01")])
    panel, rapor = m.panel_kur(ham, _referans())
    # 4 ilce x 1 gun
    assert len(panel) == 4
    assert set(panel["ilce_key"]) == set(_referans()["ilce_key"])
    assert list(panel.columns) == m.CIKTI_KOLONLARI
    # EPIAS'ta hic kaydi olmayan ilceler ACIKCA raporlanir (hep-sifir tuzagi)
    assert set(rapor["epias_kaydi_olmayan_ilce"]) == {"efeler", "merkezefendi", "sehzadeler"}


def test_negatif_sureli_kayit_atiliyor() -> None:
    m = _betik()
    bozuk = _kayit("Aydın", "Kuşadası", "2026-01-01")
    bozuk["endTime"] = pd.Timestamp("2026-01-01 07:00", tz="Europe/Istanbul").isoformat()
    ham = pd.DataFrame([bozuk, _kayit("Aydın", "Kuşadası", "2026-01-01")])
    _, rapor = m.panel_kur(ham, _referans())
    assert rapor["ham_kayit"] == 1
