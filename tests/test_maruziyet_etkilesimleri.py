"""Maruziyet etkilesimleri: ruzgar TEK BASINA degil, AGACLA birlikte keser.

NEDEN BU TEST DOSYASI (2026-08-21 arastirmasi)
----------------------------------------------
Dagitim sebekesi kesinti literaturunde en tekrarlanan bulgu, etkinin
CARPIMSAL oldugudur -- toplamsal degil:

  * NHESS 2023 (10.5194/nhess-23-1665-2023): AYNI ruzgar hizinda kesinti
    olasiligi yaprakli mevsimde 3-4x, islak toprakta 2-3x, ikisi birlikteyken
    4-5x. Yalnizca-ruzgar modeli buyuk kesintileri 2-5x EKSIK tahmin ediyor.
  * NHESS 2021 (10.5194/nhess-21-607-2021): 7 gunluk ONCUL yagis, ruzgardan
    sonra 2. en onemli degisken (onem 0,33 / 1,00).
  * UConn/Eversource OPM (WO2018013148A1): esik-ustu maruziyet SURESI, anlik
    degerden ustun; medyan APE %130 -> %59.

Fiziksel mekanizma tek cumleyle: ruzgar hattin kendisini nadiren koparir;
AGACI devirir, agac hatta duser. Dolayisiyla agac ortusu bir CARPANDIR.
Yaprakli agac daha cok ruzgar tutar; islak toprak kokun tutunmasini zayiflatir.

Depoda ESIK-USTU SAATLER ve KANTILLER zaten vardi (hava_saatlik_turev:
ruzgar_8ms_saat ... hamle_25ms_saat, ruzgar_q90). ESA WorldCover agac ortusu
orani da vardi. EKSIK OLAN ikisinin CARPIMIYDI -- GBDT etkilesimleri
ogrenebilir ama sinirli derinlikte ve seyrek bolgede zorlanir; fizik biliniyorsa
onu acikca vermek ucuzdur.

TASARIM: eksik kolonda COKMEZ, ATLAR ve RAPORLAR
------------------------------------------------
Yarisma verisinin semasini bilmiyoruz. Bir etkilesimin girdisi yoksa dogru
davranis hattı durdurmak degil, o etkilesimi atlayip hangisinin neden
atlandigini SOYLEMEKTIR. Sessizce atlamak da kabul degil -- rapor doner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.features.maruziyet import add_maruziyet_etkilesimleri, yaprak_mevsimi_orani


def _panel(gun_sayisi: int = 40) -> pd.DataFrame:
    gunler = pd.date_range("2024-06-01", periods=gun_sayisi, freq="D")
    rng = np.random.default_rng(0)
    parcalar = []
    for ilce, agac in (("ormanlik", 0.80), ("sehir", 0.05)):
        parcalar.append(
            pd.DataFrame(
                {
                    "ilce_key": ilce,
                    "tarih": gunler,
                    "agac_orani": agac,
                    "ruzgar_max": rng.uniform(2, 18, gun_sayisi),
                    "hamle_20ms_saat": rng.integers(0, 6, gun_sayisi).astype(float),
                    "yagis_toplam": rng.uniform(0, 12, gun_sayisi),
                    "toprak_nem_ort": rng.uniform(0.1, 0.45, gun_sayisi),
                }
            )
        )
    return pd.concat(parcalar, ignore_index=True)


def test_agac_carpani_uretilir() -> None:
    # Arrange
    panel = _panel()

    # Act
    sonuc = add_maruziyet_etkilesimleri(panel, time_column="tarih", key_column="ilce_key")

    # Assert
    assert "maruziyet_hamle_agac" in sonuc.frame.columns
    assert "maruziyet_hamle_agac" in sonuc.uretilen


def test_ormanlik_ilce_sehirden_yuksek_maruziyet_alir() -> None:
    """Fizik kontrolu: ayni ruzgarda agacli ilce daha yuksek carpan almali."""
    # Arrange: iki ilcenin hamle saatleri ayni dagilimdan, agac ortusu farkli
    panel = _panel()
    panel.loc[:, "hamle_20ms_saat"] = 4.0

    # Act
    sonuc = add_maruziyet_etkilesimleri(panel, time_column="tarih", key_column="ilce_key")
    ortalama = sonuc.frame.groupby("ilce_key")["maruziyet_hamle_agac"].mean()

    # Assert
    assert ortalama["ormanlik"] > ortalama["sehir"]


def test_oncul_yagis_gecmise_bakar() -> None:
    """Oncul islaklik GECMIS yagistir; ayni gunun yagisi ONCUL DEGILDIR."""
    # Arrange: tek ilce, yalnizca 5. gunde yagis var
    gunler = pd.date_range("2024-06-01", periods=10, freq="D")
    panel = pd.DataFrame(
        {
            "ilce_key": "a",
            "tarih": gunler,
            "agac_orani": 0.5,
            "yagis_toplam": [0, 0, 0, 0, 100.0, 0, 0, 0, 0, 0],
            "ruzgar_max": 5.0,
        }
    )

    # Act
    sonuc = add_maruziyet_etkilesimleri(panel, time_column="tarih", key_column="ilce_key")
    oncul = sonuc.frame.set_index("tarih")["oncul_yagis_7g"]

    # Assert: yagis gununde oncul HENUZ 0; ertesi gun 100'u gormeli
    assert oncul.iloc[4] == pytest.approx(0.0)
    assert oncul.iloc[5] == pytest.approx(100.0)


def test_yaprak_mevsimi_yazin_yuksek_kisin_dusuk() -> None:
    """Ege'de yaprakli mevsim ~nisan-kasim; ocakta yaprak yok."""
    temmuz = yaprak_mevsimi_orani(pd.Series([pd.Timestamp("2024-07-15")]))
    ocak = yaprak_mevsimi_orani(pd.Series([pd.Timestamp("2024-01-15")]))

    assert float(temmuz.iloc[0]) > 0.9
    assert float(ocak.iloc[0]) < 0.1


def test_yaprak_mevsimi_sinirlarda_kalir() -> None:
    yil = pd.Series(pd.date_range("2024-01-01", periods=366, freq="D"))
    oran = yaprak_mevsimi_orani(yil)

    assert float(oran.min()) >= 0.0
    assert float(oran.max()) <= 1.0


def test_eksik_kolonda_cokmez_atlar_ve_raporlar() -> None:
    """Yarisma semasini bilmiyoruz: eksik girdi hatti DURDURMAMALI."""
    # Arrange: agac ortusu YOK
    panel = _panel().drop(columns=["agac_orani"])
    oncesi = list(panel.columns)

    # Act
    sonuc = add_maruziyet_etkilesimleri(panel, time_column="tarih", key_column="ilce_key")

    # Assert
    assert "maruziyet_hamle_agac" not in sonuc.frame.columns
    assert any("agac" in neden for neden in sonuc.atlanan.values())
    assert list(panel.columns) == oncesi  # girdi mutasyona ugramadi


def test_satir_sayisi_ve_sirasi_korunur() -> None:
    panel = _panel()

    sonuc = add_maruziyet_etkilesimleri(panel, time_column="tarih", key_column="ilce_key")

    assert len(sonuc.frame) == len(panel)
    pd.testing.assert_series_equal(sonuc.frame["ilce_key"], panel["ilce_key"])


def test_hedef_kolonu_asla_kullanilmaz() -> None:
    """Sizinti guvencesi: uretilen hicbir kolon hedefe dokunmaz.

    Panele bir hedef kolonu konur ve etkilesimlerin onunla korelasyonu
    OLCULUR degil -- kolonun ADI uretimde hic gecmemeli. Bu, tasarim
    geregi boyle: fonksiyon hedef adini parametre olarak dahi ALMAZ.
    """
    panel = _panel()
    panel["kesinti_adedi"] = np.arange(len(panel), dtype=float)

    sonuc = add_maruziyet_etkilesimleri(panel, time_column="tarih", key_column="ilce_key")

    assert all("kesinti" not in ad for ad in sonuc.uretilen)


def test_rapor_metni_okunabilir() -> None:
    sonuc = add_maruziyet_etkilesimleri(_panel(), time_column="tarih", key_column="ilce_key")

    metin = sonuc.ozet()
    assert "maruziyet" in metin
    assert str(len(sonuc.uretilen)) in metin


def _termal_panel() -> pd.DataFrame:
    """Sicak dalgasi olan tek ilce: 3 sicak, 1 serin, 4 sicak gun."""
    return pd.DataFrame(
        {
            "ilce_key": "a",
            "tarih": pd.date_range("2024-07-01", periods=8, freq="D"),
            "sicaklik_max": [35.0, 36.0, 34.0, 28.0, 33.0, 37.0, 38.0, 39.0],
            "yerlesim_orani": 0.6,
            "sogutma_derece_gun": 8.0,
        }
    )


def test_sicak_sureklilik_blok_icinde_sayar() -> None:
    """Trafo ilk sicak gunde degil, SUREKLI yukte arizalanir -- sayac bunu olcer."""
    # Act
    sonuc = add_maruziyet_etkilesimleri(_termal_panel(), time_column="tarih", key_column="ilce_key")

    # Assert: 3 sicak -> serin gun sifirlar -> 4 sicak
    assert sonuc.frame["sicak_sureklilik"].tolist() == [1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 4.0]


def test_sicak_sureklilik_ilceler_arasi_sizmaz() -> None:
    """Bir ilcenin sicak dalgasi digerinin sayacini baslatmamali."""
    # Arrange: b ilcesi hep serin
    a = _termal_panel()
    b = a.assign(ilce_key="b", sicaklik_max=20.0)
    panel = pd.concat([a, b], ignore_index=True)

    # Act
    sonuc = add_maruziyet_etkilesimleri(panel, time_column="tarih", key_column="ilce_key")
    b_sayac = sonuc.frame.loc[sonuc.frame.ilce_key == "b", "sicak_sureklilik"]

    # Assert
    assert (b_sayac == 0.0).all()


def test_kentsel_etkilesimler_uretilir() -> None:
    """OLCULEN bulgu (rho +0,155 yerlesim vs -0,058 agac) kentsel kolu gerektirdi."""
    sonuc = add_maruziyet_etkilesimleri(_termal_panel(), time_column="tarih", key_column="ilce_key")

    assert "maruziyet_sicak_yerlesim" in sonuc.uretilen
    assert "maruziyet_sicak_sureklilik_yerlesim" in sonuc.uretilen
