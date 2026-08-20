"""``arazi_ortusu`` ve ``osm_altyapi`` ailelerinin orkestrator sozlesmesi.

NEDEN BU TEST DOSYASI
---------------------
docs/18 bolum A: bu iki aile, kesinti tahmini literaturunun EN BUYUK belgelenmis
ablasyon kazanciydi ve bizde SIFIRDI. Ikisi de ILCE BASINA TEK SATIR, zaman
boyutsuz statik tablolardir -- bu yuzden ozel bir risk tasirlar:

  * Zamansiz oldugu icin ufuk/ambargo kapisi onlari denetlemez; join yanlissa
    hicbir sizinti alarmi calmaz, yalnizca NaN dolu bir kolon olusur.
  * Her ilce icin ayni deger tekrar ettiginden, YANLIS ilceye baglanmis bir
    satir sessizce "makul" gorunur.

Bu yuzden sozlesme testle sabitlenir: satir sayisi ARTMAZ (tekillik), eslesme
tam olur, kolonlar aileye dogru atanir ve bozuk anahtarda SESSIZ NaN yerine
HATA yukselir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.features.external import EXTERNAL_FAMILIES, attach_external


@pytest.fixture
def panel() -> pd.DataFrame:
    gunler = pd.date_range("2024-01-01", periods=4, freq="D")
    ilceler = ["konak", "bozdogan", "kavaklidere"]
    satirlar = [{"ilce_key": ilce, "gun": gun} for ilce in ilceler for gun in gunler]
    frame = pd.DataFrame(satirlar)
    frame["hedef"] = np.arange(len(frame), dtype="float64")
    return frame


#: Ilce -> (il, agac, yerlesim, tarim). Konumsal listeler yerine ISIMLE
#: eslenir: fixture'in ilce siralamasi degisince degerler sessizce kaymasin.
ORTU_DEGERLERI: dict[str, tuple[str, float, float, float]] = {
    "konak": ("izmir", 0.10, 0.54, 0.20),
    "bozdogan": ("aydin", 0.45, 0.03, 0.40),
    "kavaklidere": ("mugla", 0.83, 0.01, 0.10),
}
#: Ilce -> (il, direk, hat_km, direk_yogunlugu).
ALTYAPI_DEGERLERI: dict[str, tuple[str, float, float, float]] = {
    "konak": ("izmir", 120.0, 45.0, 1.2),
    "bozdogan": ("aydin", 8.0, 12.0, 0.08),
    "kavaklidere": ("mugla", 3.0, 6.0, 0.03),
}


def _ortu_tablosu(ilceler: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "il_key": ORTU_DEGERLERI[ilce][0],
                "ilce_key": ilce,
                "agac_orani": ORTU_DEGERLERI[ilce][1],
                "yerlesim_orani": ORTU_DEGERLERI[ilce][2],
                "tarim_orani": ORTU_DEGERLERI[ilce][3],
                "ortu_piksel": 10000,
            }
            for ilce in ilceler
        ]
    )


def _altyapi_tablosu(ilceler: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "il_key": ALTYAPI_DEGERLERI[ilce][0],
                "ilce_key": ilce,
                "osm_direk": ALTYAPI_DEGERLERI[ilce][1],
                "osm_toplam_hat_km": ALTYAPI_DEGERLERI[ilce][2],
                "osm_direk_yogunlugu": ALTYAPI_DEGERLERI[ilce][3],
            }
            for ilce in ilceler
        ]
    )


@pytest.fixture
def kok(tmp_path, panel):
    """Ailelerin bekledigi dizin duzenini kurar."""
    dizin = tmp_path / "data" / "external"
    dizin.mkdir(parents=True)
    ilceler = sorted(panel["ilce_key"].unique())
    _ortu_tablosu(ilceler).to_parquet(dizin / "arazi_ortusu_ilce.parquet")
    _altyapi_tablosu(ilceler).to_parquet(dizin / "osm_altyapi_ilce.parquet")
    return tmp_path


def test_aileler_kanonik_listede() -> None:
    """Aile adlari EXTERNAL_FAMILIES'e girmeli; yoksa attach_external reddeder."""
    assert "arazi_ortusu" in EXTERNAL_FAMILIES
    assert "osm_altyapi" in EXTERNAL_FAMILIES


def test_arazi_ortusu_satir_sayisini_degistirmez(panel, kok) -> None:
    # Arrange / Act
    sonuc = attach_external(
        panel,
        key_column="ilce_key",
        time_column="gun",
        horizon=1,
        families=("arazi_ortusu",),
        root=kok,
    )

    # Assert: statik tabloda ilce basina TEK satir olmali; coklama olursa
    # panel satir sayisi artar ve hedef sessizce coklanir.
    assert len(sonuc.frame) == len(panel)


def test_arazi_ortusu_degerleri_dogru_ilceye_gider(panel, kok) -> None:
    # Act
    sonuc = attach_external(
        panel,
        key_column="ilce_key",
        time_column="gun",
        horizon=1,
        families=("arazi_ortusu",),
        root=kok,
    )

    # Assert
    frame = sonuc.frame
    assert frame.loc[frame["ilce_key"] == "kavaklidere", "agac_orani"].eq(0.83).all()
    assert frame.loc[frame["ilce_key"] == "konak", "yerlesim_orani"].eq(0.54).all()
    assert frame["agac_orani"].notna().all()


def test_osm_altyapi_kolonlari_aileye_atanir(panel, kok) -> None:
    # Act
    sonuc = attach_external(
        panel,
        key_column="ilce_key",
        time_column="gun",
        horizon=1,
        families=("osm_altyapi",),
        root=kok,
    )

    # Assert
    kolonlar = sonuc.families["osm_altyapi"]
    assert "osm_direk" in kolonlar
    assert "osm_toplam_hat_km" in kolonlar
    # Anahtar kolonlari feature olarak sizmamali.
    assert "il_key" not in kolonlar
    assert "ilce_key" not in kolonlar


def test_ikisi_birlikte_baglanabilir(panel, kok) -> None:
    # Act
    sonuc = attach_external(
        panel,
        key_column="ilce_key",
        time_column="gun",
        horizon=1,
        families=("arazi_ortusu", "osm_altyapi"),
        root=kok,
    )

    # Assert
    assert len(sonuc.frame) == len(panel)
    assert set(sonuc.families) == {"arazi_ortusu", "osm_altyapi"}


def test_bozuk_anahtar_sessiz_nan_yerine_hata(panel, kok) -> None:
    """Hicbir ilce eslesmezse SESSIZ NaN degil HATA olmali.

    Statik tabloda bu ozellikle onemli: zaman boyutu olmadigi icin ufuk
    kapisi devreye girmez, yani yanlis anahtari yakalayacak baska bir kapi
    YOKTUR.
    """
    # Arrange: tabloyu taninmayan anahtarlarla yeniden yaz
    bozuk = _ortu_tablosu(sorted(ORTU_DEGERLERI))
    bozuk["ilce_key"] = ["yok1", "yok2", "yok3"]
    bozuk.to_parquet(kok / "data" / "external" / "arazi_ortusu_ilce.parquet")

    # Act / Assert
    with pytest.raises(ValueError, match="HIC eslesmedi"):
        attach_external(
            panel,
            key_column="ilce_key",
            time_column="gun",
            horizon=1,
            families=("arazi_ortusu",),
            root=kok,
        )


def test_dosya_yoksa_atlanir_ve_raporlanir(panel, tmp_path) -> None:
    # Arrange: bos dizin
    (tmp_path / "data" / "external").mkdir(parents=True)

    # Act
    sonuc = attach_external(
        panel,
        key_column="ilce_key",
        time_column="gun",
        horizon=1,
        families=("arazi_ortusu", "osm_altyapi"),
        root=tmp_path,
    )

    # Assert: eksik dosya sessizce yok sayilmaz, raporlanir
    assert set(sonuc.skipped) == {"arazi_ortusu", "osm_altyapi"}
    assert len(sonuc.frame) == len(panel)
