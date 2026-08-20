"""``median_blend`` sozlesmesi -- MAE metriginde DOGRU toplayici.

NEDEN BU TEST DOSYASI
---------------------
2026-08-20 arastirmasi (docs/18 bolum B2): harmani agirlikli ORTALAMA ile
kurmustuk, oysa MAE'yi minimize eden tahmin MEDYANDIR. Yani harmani reddeden
yuvalanmis kontrolumuz, metrikle celisen bir adayi reddetmisti. Bu testler
medyan toplayicinin sozlesmesini sabitler:

  1. agirliksiz cagri ``np.median`` ile BIREBIR ayni,
  2. esit agirlikli cagri da ``np.median`` ile ayni (cift uye dahil),
  3. MAE'de medyan, aykiri uye varken ortalamadan IYI,
  4. dogrulama hatalari sessiz gecmiyor.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridup.ensemble import median_blend


def test_agirliksiz_numpy_medyani_ile_ayni() -> None:
    # Arrange
    tahminler = {
        "a": np.array([1.0, 10.0, 5.0]),
        "b": np.array([2.0, 20.0, 5.0]),
        "c": np.array([3.0, 30.0, 100.0]),
    }

    # Act
    harman = median_blend(tahminler)

    # Assert
    beklenen = np.median(np.column_stack(list(tahminler.values())), axis=1)
    np.testing.assert_array_equal(harman, beklenen)


@pytest.mark.parametrize("uye_sayisi", [2, 3, 4, 5, 6, 7, 8])
def test_esit_agirlik_numpy_medyanina_indirgenir(uye_sayisi: int) -> None:
    """Esit agirlik verildiginde sonuc ``np.median`` ile ayni olmali.

    CIFT uye sayisi kritik: standart "alt agirlikli medyan" tanimi burada
    iki orta degerin KUCUGUNU secer ve np.median'dan sapar. Sozlesmemiz
    interpolasyonlu tanim -- bu test sapmanin geri gelmesini engeller.
    """
    # Arrange
    rng = np.random.default_rng(0)
    tahminler = {f"m{i}": rng.normal(size=50) for i in range(uye_sayisi)}
    agirliklar = {ad: 1.0 / uye_sayisi for ad in tahminler}

    # Act
    harman = median_blend(tahminler, agirliklar)

    # Assert
    beklenen = np.median(np.column_stack(list(tahminler.values())), axis=1)
    np.testing.assert_allclose(harman, beklenen, atol=1e-12)


def test_agirlik_medyani_kaydiriyor() -> None:
    # Arrange: b'ye ezici agirlik verilirse medyan b'ye oturmali
    tahminler = {
        "a": np.array([0.0, 0.0]),
        "b": np.array([5.0, 5.0]),
        "c": np.array([9.0, 9.0]),
    }

    # Act
    harman = median_blend(tahminler, {"a": 0.05, "b": 0.90, "c": 0.05})

    # Assert
    np.testing.assert_allclose(harman, [5.0, 5.0])


def test_mae_de_medyan_aykiri_uyeye_ortalamadan_dayanikli() -> None:
    """Bir uye cuvallarsa medyan ayakta kalir, ortalama surüklenir.

    Harmanin varlik sebebi tam olarak budur; MAE'de bu fark toplayici
    seciminden dogar, uye kalitesinden degil.
    """
    # Arrange
    rng = np.random.default_rng(42)
    gercek = rng.uniform(0, 100, size=500)
    iyi_1 = gercek + rng.normal(0, 5, size=500)
    iyi_2 = gercek + rng.normal(0, 5, size=500)
    cuvallayan = gercek + rng.normal(200, 5, size=500)  # sistematik sapma
    tahminler = {"iyi_1": iyi_1, "iyi_2": iyi_2, "cuvallayan": cuvallayan}

    # Act
    medyan = median_blend(tahminler)
    ortalama = np.column_stack(list(tahminler.values())).mean(axis=1)

    # Assert
    mae_medyan = float(np.abs(gercek - medyan).mean())
    mae_ortalama = float(np.abs(gercek - ortalama).mean())
    assert mae_medyan < mae_ortalama


def test_tek_uye_kendisini_dondurur() -> None:
    # Arrange
    tahminler = {"tek": np.array([3.0, -1.0, 7.5])}

    # Act / Assert
    np.testing.assert_array_equal(median_blend(tahminler), tahminler["tek"])


def test_bos_sozluk_hata() -> None:
    with pytest.raises(ValueError, match="Bos tahmin"):
        median_blend({})


def test_uzunluk_uyusmazligi_hata() -> None:
    with pytest.raises(ValueError, match="ayni uzunlukta"):
        median_blend({"a": np.array([1.0, 2.0]), "b": np.array([1.0])})


def test_eksik_agirlik_hata() -> None:
    with pytest.raises(ValueError, match="Eksik agirlik"):
        median_blend({"a": np.array([1.0]), "b": np.array([2.0])}, {"a": 1.0})


def test_fazla_agirlik_hata() -> None:
    with pytest.raises(ValueError, match="fazla agirlik"):
        median_blend(
            {"a": np.array([1.0])},
            {"a": 1.0, "hayalet": 1.0},
        )


def test_negatif_agirlik_hata() -> None:
    with pytest.raises(ValueError, match="Negatif agirlik"):
        median_blend(
            {"a": np.array([1.0]), "b": np.array([2.0])},
            {"a": -1.0, "b": 2.0},
        )


def test_sifir_agirlik_toplami_hata() -> None:
    with pytest.raises(ValueError, match="Agirlik toplami sifir"):
        median_blend(
            {"a": np.array([1.0]), "b": np.array([2.0])},
            {"a": 0.0, "b": 0.0},
        )


def test_sifir_agirlikli_uye_sonuca_girmez() -> None:
    """Agirligi 0 olan uye, medyani kaydirmamali."""
    # Arrange
    tahminler = {
        "a": np.array([1.0, 1.0]),
        "b": np.array([2.0, 2.0]),
        "yok_say": np.array([1000.0, 1000.0]),
    }

    # Act
    harman = median_blend(tahminler, {"a": 0.5, "b": 0.5, "yok_say": 0.0})

    # Assert: a ve b'nin (esit agirlikli) medyani = 1.5
    np.testing.assert_allclose(harman, [1.5, 1.5])
