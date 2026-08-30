from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODEL29 = Path(__file__).resolve().parents[1] / "experiments" / "model29"
sys.path.insert(0, str(MODEL29))

from m112_kalibre import (  # noqa: E402
    cikti_adini_dogrula,
    dosya_adaylarini_ekle,
    gonderim_olcumlerini_ekle,
    gonderim_olcumu,
    hedef996_yonleri,
    idye_hizala,
    onsele_dayali_duzeltme,
    skoru_dogrula,
)


def test_gonderim_olcumu_leaderboard_skorunu_birebir_yeniden_kurar() -> None:
    taban = np.array([1.0, 1.4, 2.1, 0.3], dtype=np.float64)
    tahmin = np.array([1.2, 1.1, 2.4, 0.4], dtype=np.float64)
    m0 = 1.37
    skor = 1.12345

    yon, ic_carpim = gonderim_olcumu(taban, tahmin, skor, m0=m0)

    yeniden_kurulan_mse = m0 + np.mean(yon * yon) - 2.0 * ic_carpim
    assert np.sqrt(yeniden_kurulan_mse) == pytest.approx(skor, abs=1e-12)


def test_gonderim_olcumu_satir_sayisi_uyusmazligini_reddeder() -> None:
    with pytest.raises(ValueError, match="satir sayisi"):
        gonderim_olcumu(
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            1.0,
            m0=1.0,
        )


def test_kayitli_gonderimler_soyut_yon_yerine_dosyanin_kendisini_kullanir() -> None:
    taban = np.array([0.2, 0.4, 0.8], dtype=np.float64)
    gonderilen = np.array([0.5, 0.1, 0.9], dtype=np.float64)
    yonler: list[np.ndarray] = []
    ic_carpimlar: list[float] = []
    olcumler = [{"aday": "seviye", "dosya": "probe.csv", "skor": 0.91}]

    gonderim_olcumlerini_ekle(
        taban,
        yonler,
        ic_carpimlar,
        olcumler,
        okuyucu=lambda _: gonderilen,
        m0=1.2,
    )

    assert len(yonler) == 1
    assert yonler[0] == pytest.approx(gonderilen - taban)
    yeniden_kurulan = 1.2 + np.mean(yonler[0] ** 2) - 2.0 * ic_carpimlar[0]
    assert np.sqrt(yeniden_kurulan) == pytest.approx(0.91, abs=1e-12)


def test_onsel_duzeltmesi_yeni_yonleri_sirayla_diklestirir() -> None:
    bilinen = np.array([[1.0], [-1.0], [1.0], [-1.0]])
    gram = bilinen.T @ bilinen / len(bilinen)
    adaylar = {
        "ay": np.array([1.0, 1.0, -1.0, -1.0]),
        "haftasonu": np.array([1.0, -1.0, -1.0, 1.0]),
    }

    duzeltme, bilgi = onsele_dayali_duzeltme(
        adaylar,
        bilinen,
        gram,
        [("ay", 0.03), ("haftasonu", -0.02)],
        len(bilinen),
    )

    assert [satir["aday"] for satir in bilgi] == ["ay", "haftasonu"]
    assert np.mean(duzeltme * bilinen[:, 0]) == pytest.approx(0.0, abs=1e-12)
    assert np.mean(duzeltme**2) == pytest.approx(0.03**2 + 0.02**2, abs=1e-12)


def test_hedef996_yonleri_merkezli_birim_ve_sonlu_uretilir() -> None:
    ozellikler = pd.DataFrame(
        {
            "tarih": pd.date_range("2026-04-01", periods=8, freq="15D"),
            "t_hg_genligi": [1.0, 2.0, np.nan, 4.0, 7.0, 8.0, 10.0, 12.0],
            "yas": [2.0, 3.0, 5.0, 8.0, 13.0, 21.0, np.inf, 55.0],
            "t_log_ort": [3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0],
            "gunes_radyasyon": [0.0, 2.0, 5.0, 9.0, 14.0, 20.0, 27.0, 35.0],
            "sicaklik_ort": [4.0, 4.5, 6.0, 8.0, 11.0, 15.0, 20.0, 26.0],
            "ulusal_gunluk": [100, 105, 111, 120, 132, 147, 165, 186],
        }
    )
    a0 = np.linspace(8.0, 9.4, len(ozellikler))
    a0.setflags(write=False)

    yonler = hedef996_yonleri(ozellikler, a0)

    assert tuple(yonler) == (
        "h",
        "t_hg_genligi",
        "sv_yas",
        "h_t_log_ort",
        "gunes_radyasyon",
        "h_sicaklik_ort",
        "ulusal_gunluk",
    )
    for yon in yonler.values():
        assert np.isfinite(yon).all()
        assert np.mean(yon) == pytest.approx(0.0, abs=1e-12)
        assert np.mean(yon**2) == pytest.approx(1.0, abs=1e-12)


def test_dosya_adaylari_taban_farki_olarak_eklenir() -> None:
    taban = np.array([1.0, 1.5, 2.0])
    adaylar: dict[str, np.ndarray] = {"mevcut": np.ones(3)}

    dosya_adaylarini_ekle(
        taban,
        adaylar,
        {"y46": "y46.csv", "p42": "p42.csv"},
        okuyucu=lambda ad: {
            "y46.csv": np.array([1.2, 1.4, 2.3]),
            "p42.csv": np.array([0.9, 1.8, 2.2]),
        }[ad],
    )

    assert adaylar["y46"] == pytest.approx([0.2, -0.1, 0.3])
    assert adaylar["p42"] == pytest.approx([-0.1, 0.3, 0.2])


def test_dosya_adayi_satir_sayisi_uyusmazligini_reddeder() -> None:
    with pytest.raises(ValueError, match="satir sayisi"):
        dosya_adaylarini_ekle(
            np.array([1.0, 2.0]),
            {},
            {"bozuk": "bozuk.csv"},
            okuyucu=lambda _: np.array([1.0]),
        )


@pytest.mark.parametrize("skor", [None, -1.0, 0.0, np.nan, np.inf, 3.0])
def test_gecersiz_lb_skoru_durum_kaydina_giremez(skor: float | None) -> None:
    with pytest.raises(ValueError, match="LB skoru"):
        skoru_dogrula(skor)


def test_gecerli_lb_skoru_kabul_edilir() -> None:
    assert skoru_dogrula(0.996) == pytest.approx(0.996)


@pytest.mark.parametrize(
    "ad",
    [
        "../disari.csv",
        "alt/dosya.csv",
        r"alt\dosya.csv",
        "tuketim_m6_ikiyon.csv",
        "tuketim_y46_amnezik_kirpik.csv",
        "sonuc.txt",
    ],
)
def test_cikti_adi_kaynaklari_ve_yol_gecisini_korur(ad: str) -> None:
    with pytest.raises(ValueError, match="cikti"):
        cikti_adini_dogrula(ad)


def test_yeni_csv_cikti_adi_kabul_edilir() -> None:
    assert cikti_adini_dogrula("tuketim_K_yeni.csv") == "tuketim_K_yeni.csv"


def test_durumdaki_olculmus_dosya_da_kaynak_olarak_korunur() -> None:
    with pytest.raises(ValueError, match="cikti"):
        cikti_adini_dogrula("olculmus.csv", ek_korunan={"olculmus.csv"})


def test_tam_boy_gonderim_idye_gore_test_sirasina_hizalanir() -> None:
    gonderim = pd.DataFrame({"id": [30, 10, 20], "tuketim": [3.0, 1.0, 2.0]})

    hizali = idye_hizala(gonderim, np.array([10, 20, 30]))

    assert hizali.id.tolist() == [10, 20, 30]
    assert hizali.tuketim.tolist() == [1.0, 2.0, 3.0]


def test_tam_boy_gonderimde_eksik_id_reddedilir() -> None:
    gonderim = pd.DataFrame({"id": [10, 20, 40], "tuketim": [1.0, 2.0, 4.0]})

    with pytest.raises(ValueError, match="id kumesi"):
        idye_hizala(gonderim, np.array([10, 20, 30]))


def test_kisa_teshis_dosyasi_exact_span_tarafindan_atlanmak_uzere_kalir() -> None:
    gonderim = pd.DataFrame({"id": [10], "tuketim": [1.0]})

    hizali = idye_hizala(gonderim, np.array([10, 20, 30]))

    assert hizali.equals(gonderim)
