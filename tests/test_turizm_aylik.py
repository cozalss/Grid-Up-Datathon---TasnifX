"""KTB AYLIK turizm cekicisi ve aylik feature katmani testleri.

Iki soru:
  1. ``fetch_turizm_aylik.il_tablosu`` bultenin donemini ICERIKTEN dogruluyor
     ve kapsam degisikligini (2022 Kasim) etiketliyor mu?
  2. ``add_monthly_attribute`` yayimlanmamis ayi panele sokabiliyor mu?
     (lag < 2 reddedilmeli; lag 12 gecen yilin ayni ayini vermeli.)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridup.features.tourism import (
    MIN_LAG_MONTHS,
    add_monthly_attribute,
    district_monthly_estimate,
)

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))


def _betik():
    yol = KOK / "scripts" / "fetch_turizm_aylik.py"
    spec = importlib.util.spec_from_file_location("fetch_turizm_aylik", yol)
    modul = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(modul)
    return modul


ILLER = ["Adana", "Aydın", "Muğla", "İzmir"]


def _sahte_bulten(
    yol: Path,
    *,
    yil: int,
    ay: int,
    kapsam_basit: bool,
    il_sayisi: int = 81,
    belediye: bool = False,
    carpan: int = 1,
) -> None:
    """KTB aylik bultenin sayfa yapisini birebir taklit eden xlsx yazar."""
    ay_adlari = [
        "OCAK", "ŞUBAT", "MART", "NİSAN", "MAYIS", "HAZİRAN",
        "TEMMUZ", "AĞUSTOS", "EYLÜL", "EKİM", "KASIM", "ARALIK",
    ]  # fmt: skip
    if belediye:
        baslik = "MAHALLİ İDARELERCE BELGELENDİRİLEN KONAKLAMA TESİSLERİNDE ..."
    elif kapsam_basit:
        baslik = "İŞLETME VE BASİT BELGELİ KONAKLAMA TESİSLERİNDE ..."
    else:
        baslik = "TURİZM İŞLETME BELGELİ KONAKLAMA TESİSLERİNDE ..."
    yil_sayfa = pd.DataFrame([[baslik, None, None], ["YILLAR", "TESİSE GELİŞ", "GECELEME"]]
                             + [[str(y), 1, 2] for y in range(yil - 3, yil + 1)])  # fmt: skip
    ay_sayfa = pd.DataFrame([[baslik, None, None], ["AYLAR", "TESİSE GELİŞ", "GECELEME"]]
                            + [[ay_adlari[i], 1, 2] for i in range(ay)])  # fmt: skip

    iller = ILLER + [f"Il{i}" for i in range(il_sayisi - len(ILLER))]
    satirlar = [
        [baslik] + [None] * 12,
        [
            "İLLER",
            "TESİSE GELİŞ SAYISI",
            None,
            None,
            "GECELEME",
            None,
            None,
            "ORTALAMA KALIŞ",
            None,
            None,
            "DOLULUK ORANI (%)",
            None,
            None,
        ],  # fmt: skip
        [None] + ["YABANCI", "YERLİ", "TOPLAM"] * 4,
    ]
    for sira, il in enumerate(iller, start=1):
        c = carpan
        satirlar.append([il, sira * c, sira * 2 * c, sira * 3 * c, sira * 10 * c, sira * 20 * c,
                         sira * 30 * c, 1.5, 1.5, 1.5, 10.0, 20.0, 30.0])  # fmt: skip
    satirlar.append(["TOPLAM"] + [999] * 12)
    il_sayfa = pd.DataFrame(satirlar)

    with pd.ExcelWriter(yol, engine="openpyxl") as yazici:
        icindekiler = pd.DataFrame([["İçindekiler"]])
        icindekiler.to_excel(yazici, sheet_name="İçindekiler", header=False, index=False)
        yil_sayfa.to_excel(yazici, sheet_name="Geliş-Geceleme Yıl", header=False, index=False)
        ay_sayfa.to_excel(yazici, sheet_name="Geliş-Geceleme Ay", header=False, index=False)
        il_sayfa.to_excel(yazici, sheet_name="İl", header=False, index=False)


# --------------------------------------------------------------------------
# Cekici: il_tablosu
# --------------------------------------------------------------------------


def test_bulten_haritasi_2019_2026_bosluksuz() -> None:
    """Yil-ay haritasi 2019-01'den son doneme kadar ATLAMASIZ olmali.

    Eksik bir ay, lag-12 join'de o ay icin sessiz NaN uretir.
    """
    m = _betik()
    donemler = sorted(m.BULTENLER)
    beklenen = []
    yil, ay = donemler[0]
    while (yil, ay) <= donemler[-1]:
        beklenen.append((yil, ay))
        ay += 1
        if ay > 12:
            yil, ay = yil + 1, 1
    assert donemler == beklenen, "Haritada atlanan ay var."
    assert donemler[0] == (2019, 1)
    assert len(set(m.BULTENLER.values())) == len(m.BULTENLER), "Ayni URL iki doneme atanmis."


def test_il_tablosu_donemi_icerikten_dogrular(tmp_path: Path) -> None:
    """Dosya adi ne derse desin, icerik farkli bir ay soyluyorsa reddedilmeli."""
    m = _betik()
    yol = tmp_path / "x.xlsx"
    _sahte_bulten(yol, yil=2025, ay=7, kapsam_basit=True)
    with pytest.raises(ValueError, match="beklenen donem 2025-06"):
        m.il_tablosu(yol, 2025, 6)
    with pytest.raises(ValueError, match="beklenen donem 2024-07"):
        m.il_tablosu(yol, 2024, 7)


def test_il_tablosu_sema_ve_kapsam(tmp_path: Path) -> None:
    """81 il, TOPLAM disarida, kapsam etiketi basliktan, degerler dogru kolondan."""
    m = _betik()
    eski = tmp_path / "eski.xlsx"
    yeni = tmp_path / "yeni.xlsx"
    _sahte_bulten(eski, yil=2022, ay=10, kapsam_basit=False)
    _sahte_bulten(yeni, yil=2022, ay=11, kapsam_basit=True)

    t_eski = m.il_tablosu(eski, 2022, 10)
    t_yeni = m.il_tablosu(yeni, 2022, 11)
    temel = [k for k in m.CIKTI_KOLONLARI if k not in (*m.BELEDIYE_KOLONLARI, *m.TUM_KOLONLARI)]
    assert list(t_eski.columns) == temel
    assert len(t_eski) == 81 and "toplam" not in set(t_eski["il_key"])
    assert set(t_eski["kapsam"]) == {"isletme"}
    assert set(t_yeni["kapsam"]) == {"isletme_basit"}

    mugla = t_eski[t_eski["il_key"] == "mugla"].iloc[0]
    sira = ILLER.index("Muğla") + 1
    assert mugla["gelis"] == sira * 3
    assert mugla["geceleme_yabanci"] == sira * 10
    assert mugla["geceleme"] == sira * 30
    assert mugla["doluluk"] == 30.0
    # Turkce anahtar: "İzmir" -> "izmir", "Aydın" -> "aydin"
    assert {"izmir", "aydin"} <= set(t_eski["il_key"])


def test_il_tablosu_eksik_il_reddedilir(tmp_path: Path) -> None:
    """80 il = yapi degismis; sessizce kabul edilmez."""
    m = _betik()
    yol = tmp_path / "eksik.xlsx"
    _sahte_bulten(yol, yil=2024, ay=3, kapsam_basit=True, il_sayisi=80)
    with pytest.raises(ValueError, match="80 il satiri"):
        m.il_tablosu(yol, 2024, 3)


def test_kapsam_rejimi_olculen_kirilmalar() -> None:
    """Rejim, basliga degil OLCULEN yatak sicramasina gore: 2022-09 ve 2025-07."""
    m = _betik()
    assert m.REJIM_KIRILMALARI == ((2022, 9), (2025, 7))
    assert m.kapsam_rejimi(2019, 1) == 1
    assert m.kapsam_rejimi(2022, 8) == 1
    assert m.kapsam_rejimi(2022, 9) == 2
    assert m.kapsam_rejimi(2025, 6) == 2
    assert m.kapsam_rejimi(2025, 7) == 3
    assert m.kapsam_rejimi(2026, 6) == 3


def test_il_tablosu_kapsam_rejimi_kolonu(tmp_path: Path) -> None:
    """Baslik 2022-10'da hala 'isletme' der ama rejim 2'dir -- ikisi de tasinmali."""
    m = _betik()
    yol = tmp_path / "x.xlsx"
    _sahte_bulten(yol, yil=2022, ay=10, kapsam_basit=False)
    t = m.il_tablosu(yol, 2022, 10)
    assert set(t["kapsam"]) == {"isletme"}
    assert set(t["kapsam_rejimi"]) == {2}


def test_belediye_haritasi_2019_01_2022_10_bosluksuz() -> None:
    """Belediye serisi 2019-01..2022-10; sonrasi bakanlik serisine katildi."""
    m = _betik()
    donemler = sorted(m.BELEDIYE_BULTENLER)
    assert donemler[0] == (2019, 1) and donemler[-1] == (2022, 10)
    assert len(donemler) == 46
    assert len(set(m.BELEDIYE_BULTENLER.values())) == 46
    assert not set(m.BELEDIYE_BULTENLER.values()) & set(m.BULTENLER.values())


def test_il_tablosu_seri_kapsam_capraz_dogrular(tmp_path: Path) -> None:
    """Belediye dosyasi bakanlik olarak (veya tersi) okunursa reddedilmeli."""
    m = _betik()
    bel = tmp_path / "bel.xlsx"
    bak = tmp_path / "bak.xlsx"
    _sahte_bulten(bel, yil=2021, ay=6, kapsam_basit=False, belediye=True, il_sayisi=79)
    _sahte_bulten(bak, yil=2021, ay=6, kapsam_basit=False)

    t = m.il_tablosu(bel, 2021, 6, seri="belediye")
    assert len(t) == 79 and set(t["kapsam"]) == {"belediye"}
    with pytest.raises(ValueError, match="belediye serisi bekleniyordu"):
        m.il_tablosu(bak, 2021, 6, seri="belediye")
    with pytest.raises(ValueError, match="il satiri"):
        m.il_tablosu(bel, 2021, 6)  # bakanlik olarak: 79 il -> red
    _sahte_bulten(bel, yil=2021, ay=6, kapsam_basit=False, belediye=True, il_sayisi=81)
    with pytest.raises(ValueError, match="baslik belediye diyor"):
        m.il_tablosu(bel, 2021, 6)
    with pytest.raises(ValueError, match="Bilinmeyen seri"):
        m.il_tablosu(bel, 2021, 6, seri="x")


def test_tum_belgeli_birlestir_toplam_ve_doluluk() -> None:
    """tum = bakanlik + belediye; birlesik doluluk yatak-gun agirlikli; belediye yoksa bakanlik."""
    m = _betik()
    bak = pd.DataFrame(
        {"yil": [2022, 2022, 2023], "ay": [8, 8, 8], "il_key": ["mugla", "izmir", "mugla"],
         "gelis": [100.0, 50.0, 300.0], "geceleme": [800.0, 200.0, 2400.0],
         "doluluk": [80.0, 40.0, 60.0]}
    )  # fmt: skip
    # Mugla 2022-08: belediye 200 geceleme, %20 doluluk -> yatak-gun 1000; bakanlik yg 1000
    bel = pd.DataFrame(
        {"yil": [2022], "ay": [8], "il_key": ["mugla"], "gelis": [40.0],
         "geceleme": [200.0], "doluluk": [20.0]}
    )  # fmt: skip
    kopya = bak.copy()
    out = m.tum_belgeli_birlestir(bak, bel).set_index(["yil", "ay", "il_key"])
    pd.testing.assert_frame_equal(bak, kopya)
    mug22 = out.loc[(2022, 8, "mugla")]
    assert mug22["geceleme_tum_belgeli"] == 1000.0 and mug22["gelis_tum_belgeli"] == 140.0
    assert np.isclose(mug22["doluluk_tum_belgeli"], 100.0 * 1000.0 / 2000.0)  # %50
    izm = out.loc[(2022, 8, "izmir")]  # belediye satiri yok -> 0 sayilir
    assert izm["geceleme_tum_belgeli"] == 200.0 and np.isnan(izm["geceleme_belediye"])
    assert izm["doluluk_tum_belgeli"] == 40.0
    mug23 = out.loc[(2023, 8, "mugla")]  # seri bitmis -> bakanlik degeri
    assert mug23["geceleme_tum_belgeli"] == 2400.0 and mug23["doluluk_tum_belgeli"] == 60.0
    # belediye tamamen yoksa da kolonlar var ve tum = bakanlik
    out2 = m.tum_belgeli_birlestir(bak, None)
    assert (out2["geceleme_tum_belgeli"] == bak["geceleme"]).all()
    assert out2["geceleme_belediye"].isna().all()


def test_tum_belgeli_birlestir_yetim_belediye_satiri_hata() -> None:
    m = _betik()
    bak = pd.DataFrame({"yil": [2022], "ay": [8], "il_key": ["mugla"], "gelis": [1.0],
                        "geceleme": [1.0], "doluluk": [50.0]})  # fmt: skip
    bel = pd.DataFrame({"yil": [2022], "ay": [8], "il_key": ["van"], "gelis": [1.0],
                        "geceleme": [1.0], "doluluk": [50.0]})  # fmt: skip
    with pytest.raises(ValueError, match="bakanlikta olmayan"):
        m.tum_belgeli_birlestir(bak, bel)


# --------------------------------------------------------------------------
# Cekici (yillik): Alsancak -> Konak katlama
# --------------------------------------------------------------------------


def _yillik_betik():
    yol = KOK / "scripts" / "fetch_turizm.py"
    spec = importlib.util.spec_from_file_location("fetch_turizm", yol)
    modul = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(modul)
    return modul


def test_alsancak_konaga_katlanir() -> None:
    """Alsancak satiri Konak'a EKLENIR; Izmir toplami degismez; ad 'Konak' kalir."""
    m = _yillik_betik()
    tablo = pd.DataFrame(
        {
            "yil": [2024, 2024, 2024],
            "il": ["İzmir"] * 3,
            "ilce": ["Alsancak", "Konak", "Çeşme"],
            "il_key": ["izmir"] * 3,
            "ilce_key": ["alsancak", "konak", "cesme"],
            "tesise_gelis": [10.0, 100.0, 50.0],
            "geceleme": [20.0, 200.0, 80.0],
        }
    )
    kopya = tablo.copy()
    sonuc = m._ilceleri_katla(tablo)
    pd.testing.assert_frame_equal(tablo, kopya)  # girdi degismez
    assert "alsancak" not in set(sonuc["ilce_key"])
    konak = sonuc[sonuc["ilce_key"] == "konak"].iloc[0]
    assert konak["geceleme"] == 220.0 and konak["tesise_gelis"] == 110.0
    assert konak["ilce"] == "Konak"
    assert sonuc["geceleme"].sum() == tablo["geceleme"].sum()
    assert list(sonuc.columns) == list(tablo.columns)


def test_katlama_haritasi_bos_eslesmede_dokunmaz() -> None:
    m = _yillik_betik()
    tablo = pd.DataFrame(
        {"yil": [2024], "il": ["Muğla"], "ilce": ["Bodrum"], "il_key": ["mugla"],
         "ilce_key": ["bodrum"], "tesise_gelis": [1.0], "geceleme": [2.0]}
    )  # fmt: skip
    pd.testing.assert_frame_equal(m._ilceleri_katla(tablo), tablo)


# --------------------------------------------------------------------------
# Feature: add_monthly_attribute
# --------------------------------------------------------------------------


def _aylik(yillar=(2025, 2026), iller=("mugla", "denizli")) -> pd.DataFrame:
    satirlar = []
    for il in iller:
        for yil in yillar:
            for ay in range(1, 13):
                deger = 1000.0 * ay if il == "mugla" else 100.0
                satirlar.append({"il_key": il, "yil": yil, "ay": ay, "geceleme": deger + yil})
    return pd.DataFrame(satirlar)


def _panel() -> pd.DataFrame:
    gunler = pd.date_range("2026-01-01", "2026-12-31", freq="MS") + pd.Timedelta(days=14)
    return pd.DataFrame(
        [(il, g) for il in ("mugla", "denizli") for g in gunler], columns=["il_key", "tarih"]
    )


@pytest.mark.parametrize("lag", [0, 1, -3])
def test_yayin_gecikmesi_altindaki_lag_reddedilir(lag: int) -> None:
    """lag 0/1 = henuz yayimlanmamis ay; sessiz kabul sizintidir."""
    with pytest.raises(ValueError, match="yayin gecikmesinin altinda"):
        add_monthly_attribute(
            _panel(), _aylik(), key_column="il_key", time_column="tarih",
            value_columns=["geceleme"], lag_months=lag,
        )  # fmt: skip
    assert MIN_LAG_MONTHS == 2


def test_lag_12_gecen_yilin_ayni_ayini_verir() -> None:
    """2026-07 satiri 2025-07 degerini gormeli; 2026 degerlerinin hicbiri sizmamali."""
    sonuc = add_monthly_attribute(
        _panel(), _aylik(), key_column="il_key", time_column="tarih",
        value_columns=["geceleme"], lag_months=12, prefix="turizm",
    )  # fmt: skip
    mugla_temmuz = sonuc[(sonuc["il_key"] == "mugla") & (sonuc["tarih"].dt.month == 7)]
    assert mugla_temmuz["turizm_geceleme"].tolist() == [7000.0 + 2025]
    # 2026 kaynakli deger (…+2026) hicbir satirda olmamali
    assert not (sonuc["turizm_geceleme"] % 10 == 6).any(), "SIZINTI: 2026 verisi kullanilmis."
    assert sonuc["turizm_geceleme"].notna().all()


def test_lag_2_iki_ay_geriyi_verir() -> None:
    """lag 2: 2026-03 satiri 2026-01 degerini gormeli."""
    sonuc = add_monthly_attribute(
        _panel(), _aylik(), key_column="il_key", time_column="tarih",
        value_columns=["geceleme"], lag_months=2, add_year_share=False,
    )  # fmt: skip
    mart = sonuc[(sonuc["il_key"] == "mugla") & (sonuc["tarih"].dt.month == 3)]
    assert mart["aylik_geceleme"].tolist() == [1000.0 + 2026]
    ocak = sonuc[(sonuc["il_key"] == "mugla") & (sonuc["tarih"].dt.month == 1)]
    assert ocak["aylik_geceleme"].tolist() == [11000.0 + 2025]  # 2025-11


def test_yil_payi_kapsamdan_bagimsiz_ve_bire_toplanir() -> None:
    """Ay payi kaynak yilin 12 ayina bolunur; il icinde 12 ay toplami 1'dir."""
    sonuc = add_monthly_attribute(
        _panel(), _aylik(), key_column="il_key", time_column="tarih",
        value_columns=["geceleme"], lag_months=12,
    )  # fmt: skip
    for il in ("mugla", "denizli"):
        paylar = sonuc.loc[sonuc["il_key"] == il, "aylik_geceleme_yil_payi"]
        assert np.isclose(paylar.sum(), 1.0), f"{il}: paylar {paylar.sum()} topluyor."
    # Denizli duz profil: her ay ~1/12; Mugla Temmuz payi Ocak'in ~7 kati
    denizli = sonuc.loc[sonuc["il_key"] == "denizli", "aylik_geceleme_yil_payi"]
    assert np.allclose(denizli, 1 / 12, atol=1e-3)


def test_yil_payi_eksik_yilda_nan() -> None:
    """Kaynak yilin 12 ayi yoksa pay HESAPLANMAZ (yanlis payda ile oran uretilmez)."""
    aylik = _aylik(yillar=(2025,))
    aylik = aylik[~((aylik["yil"] == 2025) & (aylik["ay"] == 12))]  # Aralik eksik
    sonuc = add_monthly_attribute(
        _panel(), aylik, key_column="il_key", time_column="tarih",
        value_columns=["geceleme"], lag_months=12,
    )  # fmt: skip
    assert sonuc["aylik_geceleme_yil_payi"].isna().all()
    # Ham deger yine de baglanmali (Aralik disinda)
    assert sonuc["aylik_geceleme"].notna().sum() == 2 * 11


def test_add_monthly_attribute_girdiyi_degistirmez() -> None:
    panel, aylik = _panel(), _aylik()
    p_kopya, a_kopya = panel.copy(), aylik.copy()
    add_monthly_attribute(
        panel, aylik, key_column="il_key", time_column="tarih", value_columns=["geceleme"]
    )
    pd.testing.assert_frame_equal(panel, p_kopya)
    pd.testing.assert_frame_equal(aylik, a_kopya)


# --------------------------------------------------------------------------
# Feature: district_monthly_estimate
# --------------------------------------------------------------------------


def test_ilce_ay_tahmini_pay_carpi_il_aylik() -> None:
    """Bodrum il gecelemesinin %60'iysa, her ayda il aylik x 0.6 almali."""
    yillik = pd.DataFrame(
        {
            "ilce_key": ["bodrum", "menteşe", "pamukkale"],
            "il_key": ["mugla", "mugla", "denizli"],
            "yil": [2025, 2025, 2025],
            "geceleme": [600.0, 400.0, 50.0],
        }
    )
    aylik = _aylik(yillar=(2025, 2026))
    tahmin = district_monthly_estimate(yillik, aylik)

    assert set(tahmin["yil"]) == {2025}, "Yalnizca ortak yil uretilmeli."
    bodrum = tahmin[tahmin["ilce_key"] == "bodrum"].set_index("ay")
    assert len(bodrum) == 12
    assert np.isclose(bodrum.loc[7, "ilce_il_payi"], 0.6)
    assert np.isclose(bodrum.loc[7, "geceleme_tahmini"], 0.6 * (7000.0 + 2025))
    pamukkale = tahmin[tahmin["ilce_key"] == "pamukkale"]
    assert np.allclose(pamukkale["ilce_il_payi"], 1.0)
    assert list(tahmin.columns) == [
        "ilce_key", "il_key", "yil", "ay", "geceleme_tahmini", "ilce_il_payi",
    ]  # fmt: skip


def test_ilce_ay_tahmini_ortak_yil_yoksa_hata() -> None:
    yillik = pd.DataFrame(
        {"ilce_key": ["bodrum"], "il_key": ["mugla"], "yil": [2019], "geceleme": [1.0]}
    )
    with pytest.raises(ValueError, match="ortak yil yok"):
        district_monthly_estimate(yillik, _aylik(yillar=(2025,)))


def test_ilce_ay_tahmini_lag12_ile_panele_baglanir() -> None:
    """Tahmin tablosu ``add_monthly_attribute`` ile ilce anahtarindan baglanabilmeli."""
    yillik = pd.DataFrame(
        {"ilce_key": ["bodrum", "menteşe"], "il_key": ["mugla", "mugla"],
         "yil": [2025, 2025], "geceleme": [600.0, 400.0]}
    )  # fmt: skip
    tahmin = district_monthly_estimate(yillik, _aylik(yillar=(2025,)))
    panel = pd.DataFrame(
        {"ilce_key": ["bodrum", "bodrum"], "tarih": pd.to_datetime(["2026-07-15", "2026-01-03"])}
    )
    sonuc = add_monthly_attribute(
        panel, tahmin, key_column="ilce_key", time_column="tarih",
        value_columns=["geceleme_tahmini"], lag_months=12, prefix="turizm",
    )  # fmt: skip
    assert np.isclose(sonuc.loc[0, "turizm_geceleme_tahmini"], 0.6 * (7000.0 + 2025))
    assert np.isclose(sonuc.loc[1, "turizm_geceleme_tahmini"], 0.6 * (1000.0 + 2025))
    paylar = sonuc["turizm_geceleme_tahmini_yil_payi"]
    assert np.isclose(paylar.iloc[0] / paylar.iloc[1], (7000.0 + 2025) / (1000.0 + 2025))


def test_ilce_ay_tahmini_referans_ilceleri_sifirlar() -> None:
    """Yillik tabloda olmayan referans ilce NaN degil 0 alir; yil bazinda."""
    yillik = pd.DataFrame(
        {"ilce_key": ["bodrum", "bayindir"], "il_key": ["mugla", "izmir"],
         "yil": [2025, 2024], "geceleme": [600.0, 10.0]}
    )  # fmt: skip
    ref = pd.DataFrame(
        {"il_key": ["mugla", "mugla", "izmir", "izmir", "van"],
         "ilce_key": ["bodrum", "kavaklidere", "bayindir", "konak", "ercis"]}
    )  # fmt: skip
    aylik = _aylik(yillar=(2024, 2025), iller=("mugla", "izmir"))
    tahmin = district_monthly_estimate(yillik, aylik, districts=ref)

    kav = tahmin[tahmin["ilce_key"] == "kavaklidere"]
    assert set(kav["yil"]) == {2024, 2025} and len(kav) == 24
    assert (kav["ilce_il_payi"] == 0).all() and (kav["geceleme_tahmini"] == 0).all()
    # Bayindir 2024'te listeli (pay 1.0), 2025'te degil (0)
    bay = tahmin[tahmin["ilce_key"] == "bayindir"].groupby("yil")["ilce_il_payi"].first()
    assert bay[2024] == 1.0 and bay[2025] == 0.0
    # Ili aylik tabloda olmayan referans ilce (Van/Ercis) uretilmez -- gercekten bilinmiyor
    assert "ercis" not in set(tahmin["ilce_key"])
    assert tahmin["geceleme_tahmini"].notna().all()


def test_ilce_ay_tahmini_districts_kolon_eksikse_hata() -> None:
    yillik = pd.DataFrame(
        {"ilce_key": ["bodrum"], "il_key": ["mugla"], "yil": [2025], "geceleme": [1.0]}
    )
    with pytest.raises(KeyError, match="districts icinde"):
        district_monthly_estimate(
            yillik, _aylik(yillar=(2025,)), districts=pd.DataFrame({"x": [1]})
        )
