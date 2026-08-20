"""Kapsam deseni kapisinin KENDISININ sozlesmeleri.

``scripts/kapsam_deseni.py``, bir feature'in egitim ve test blogunda ayni
dolulukta olup olmadigini olcer. Bir kapinin en tehlikeli hali GECIRDIGI
seyi sessizce gecirmesidir; buradaki testler kapinin gercekten kapi
oldugunu, kasitli bozulmus girdilerle kanitlar.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

KOK = Path(__file__).resolve().parents[1]


def _modul():
    yol = KOK / "scripts" / "kapsam_deseni.py"
    spec = importlib.util.spec_from_file_location("kapsam_deseni", yol)
    assert spec is not None and spec.loader is not None
    modul = importlib.util.module_from_spec(spec)
    sys.modules["kapsam_deseni"] = modul
    spec.loader.exec_module(modul)
    return modul


def _panel(gun: int = 100) -> pd.DataFrame:
    tarihler = pd.date_range("2024-01-01", periods=gun, freq="D")
    return pd.DataFrame({"tarih": list(tarihler) * 2, "ilce_key": ["a"] * gun + ["b"] * gun})


def test_testte_kaybolan_kolon_yakalaniyor() -> None:
    """Egitimde dolu / testte bos kolon HATA olmali.

    Bu desen EKSIK feature'dan daha kotudur: model kolona guvenmeyi ogrenir,
    sonra tam teslim aninda o bilgi yoktur. Gercek ornek (2026-08-20):
    ``ARCHIVE_LAG_DAYS`` yanlis oldugu icin saatlik hava turevleri son 30
    gunde %70, tum panelde %99.6 doluydu.
    """
    modul = _modul()
    frame = _panel()
    sinir = frame["tarih"].max() - pd.Timedelta(days=29)
    # Degeri DEGISKEN tutmak sart: sabit bir kolon ayrica "bilgi tasimayan"
    # bulgusu uretir ve testin neyi olctugu bulanir.
    frame["kaybolan"] = range(len(frame))
    frame["kaybolan"] = frame["kaybolan"].astype("float64")
    frame.loc[frame["tarih"] >= sinir, "kaybolan"] = None
    frame["saglam"] = 1.0
    frame.loc[frame.index % 7 == 0, "saglam"] = 2.0  # degisken ama hep dolu

    olcum = modul.desen_olc(frame, ["kaybolan", "saglam"], test_gun=30)
    hata, uyari = modul.rapor(olcum, {}, modul.VARSAYILAN_ESIK)

    assert hata == 1, f"Testte kaybolan kolon HATA sayilmadi (hata={hata})"
    assert uyari == 0


def test_yalnizca_son_donemde_var_olan_kolon_uyari() -> None:
    """Egitimde bos / testte dolu kolon UYARI olmali.

    Model bu kolonu egitimde neredeyse hic gormez ama testte ona split
    acabilir. Gercek ornek: turizm feature'lari panelin 2020-2023
    bolumunde %0, 2024-2026'da %100 doluydu.
    """
    modul = _modul()
    frame = _panel()
    sinir = frame["tarih"].max() - pd.Timedelta(days=29)
    frame["yeni"] = None
    frame.loc[frame["tarih"] >= sinir, "yeni"] = 1.0
    frame.loc[frame["tarih"] == frame["tarih"].max(), "yeni"] = 2.0

    olcum = modul.desen_olc(frame, ["yeni"], test_gun=30)
    hata, uyari = modul.rapor(olcum, {}, modul.VARSAYILAN_ESIK)

    assert uyari == 1, f"Yalnizca son donemde var olan kolon UYARI sayilmadi ({uyari})"
    assert hata == 0, "Bu desen HATA degil UYARI olmali -- model onu gormezden gelebilir"


def test_bilgi_tasimayan_kolon_yakalaniyor() -> None:
    """Tek degerli kolon HATA olmali.

    Gercek ornek (2026-08-20): ``ruzgar_20ms_saat`` 2.326.080 saatte bir kez
    bile tetiklenmedi -- surekli ruzgar Ege'de 10 m yukseklikte 20 m/s'ye
    ULASMIYOR (olculen max 18.5). Kolon her satirda 0 yaziyordu.
    """
    modul = _modul()
    frame = _panel()
    frame["olu"] = 0.0
    frame["bos"] = None

    olcum = modul.desen_olc(frame, ["olu", "bos"], test_gun=30)
    hata, _ = modul.rapor(olcum, {}, modul.VARSAYILAN_ESIK)

    assert hata == 2, f"Bilgi tasimayan iki kolonun ikisi de yakalanmadi (hata={hata})"


def test_saglam_kolon_bosuna_bagirmiyor() -> None:
    """Her zaman HATA doneni bir kapi degildir -- temiz girdi TEMIZ gecmeli."""
    modul = _modul()
    frame = _panel()
    frame["saglam"] = range(len(frame))

    olcum = modul.desen_olc(frame, ["saglam"], test_gun=30)
    hata, uyari = modul.rapor(olcum, {}, modul.VARSAYILAN_ESIK)

    assert (hata, uyari) == (0, 0), "Kusursuz kolon icin bulgu uretildi -- kapi asiri hassas"


def test_ileri_blok_boluyor_rastgele_degil() -> None:
    """Test blogu panelin SON gunleri olmali.

    Rastgele bolme bu deseni GIZLER: rastgele bir test kumesi egitimle ayni
    tarih dagilimini tasir ve "son gunlerde kayboluyor" hatasi hic gorunmez.
    Bu test, bolmenin gercekten tarihe gore yapildigini zorlar.
    """
    modul = _modul()
    frame = _panel(gun=100)
    # Yalnizca SON 10 gunde dolu bir kolon: ileri blok bolmede test %100,
    # egitim %0 gorunmeli.
    sinir = frame["tarih"].max() - pd.Timedelta(days=9)
    frame["son_on_gun"] = None
    frame.loc[frame["tarih"] >= sinir, "son_on_gun"] = 1.0

    olcum = modul.desen_olc(frame, ["son_on_gun"], test_gun=10).set_index("kolon")
    assert olcum.loc["son_on_gun", "test_dolu"] == pytest.approx(1.0)
    assert olcum.loc["son_on_gun", "egitim_dolu"] == pytest.approx(0.0)


def test_gercek_veride_kapsam_deseni_temiz() -> None:
    """Depodaki gercek veri kendi kapsam sozlesmesini SAGLAMALI.

    Veri dosyalari yoksa atlanir (CI checkout'unda data/ olmayabilir), ama
    varsa hatasiz gecmek ZORUNDADIR.
    """
    modul = _modul()
    if not (KOK / "data" / "external" / "hava_gunluk.parquet").is_file():
        pytest.skip("harici veri bu ortamda yok")

    import warnings

    from gridup.features.external import attach_external

    son = modul.panel_sonu()
    panel = modul.panel_kur(son)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sonuc = attach_external(
            panel, key_column="ilce_key", time_column="tarih", horizon=modul.UFUK, root=KOK
        )
    kolonlar = [k for k in sonuc.frame.columns if k not in ("ilce_key", "tarih")]
    olcum = modul.desen_olc(sonuc.frame, kolonlar, test_gun=modul.VARSAYILAN_TEST_GUN)

    fark = olcum["test_dolu"] - olcum["egitim_dolu"]
    kaybolan = olcum.loc[fark < -modul.VARSAYILAN_ESIK, "kolon"].tolist()
    olu = olcum.loc[olcum["tekil_deger"] <= 1, "kolon"].tolist()

    assert not kaybolan, (
        f"Su kolonlar test blogunda kayboluyor: {kaybolan}. "
        "Kaynak cekicisini panelin sonuna kadar tekrar calistir."
    )
    assert not olu, f"Su kolonlar hic bilgi tasimiyor: {olu}."
