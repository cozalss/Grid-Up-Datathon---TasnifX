"""Trafo ozetleri FOLD-GUVENLI mi -- yani hedef sizmiyor mu.

NEDEN BU TEST DOSYASI
---------------------
``gridup.features.trafo`` hedefi okuyan tek feature modulu. Olculdu (gercek
train.csv, 1.226.237 satir): trafo seviyesi tek basina ``log1p(tuketim)``
varyansinin %90,1'ini acikliyor. Bu, modele verilebilecek en guclu sinyal --
ve ayni nedenle en tehlikelisi. Bir satirin ozeti kendi hedef degerini
iceriyorsa CV mukemmele yakin cikar, leaderboard cokar.

Sizinti burada BIR ISTISNA FIRLATMAZ. Sessizce daha iyi bir skor uretir.
Bu yuzden test, "patliyor mu" degil "ozet gercekten yalnizca ``uydur``
cercevesinden mi geliyor" sorusunu soruyor: uygulama cercevesindeki hedef
degerleri BOZULUYOR ve ciktinin degismemesi bekleniyor. Degisirse, hedef
oradan okunuyor demektir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.features.trafo import (
    grup_seviyeleri_ekle,
    guc_kovasi,
    trafo_ozetleri_cikar,
    trafo_ozetleri_uygula,
)


def _panel(*, trafolar: int = 6, gun: int = 40, tohum: int = 0, olcek: float = 1.0) -> pd.DataFrame:
    """Trafo x gun paneli. Her trafonun KENDI seviyesi var."""
    rng = np.random.default_rng(tohum)
    satirlar = []
    for i in range(trafolar):
        seviye = 100.0 * (i + 1)
        for g in range(gun):
            satirlar.append(
                {
                    "tanim": f"T{i:02d}",
                    "tarih": pd.Timestamp("2025-01-01") + pd.Timedelta(days=g),
                    "guc": 250.0 * (i + 1),
                    "tuketim": max(0.0, olcek * seviye * (1.0 + 0.1 * rng.normal())),
                    "sogutma_derece_gun": float(g % 11),
                    "isitma_derece_gun": float((g * 3) % 7),
                    "ilce_key": "bornova" if i % 2 == 0 else "kiraz",
                }
            )
    return pd.DataFrame(satirlar)


def _bol(panel: pd.DataFrame, bolme: str = "2025-01-25") -> tuple[pd.DataFrame, pd.DataFrame]:
    return panel[panel["tarih"] < bolme].copy(), panel[panel["tarih"] >= bolme].copy()


# ------------------------------------------------------------ SIZINTI


def test_uygulama_hedefi_bozulunca_ozetler_degismez() -> None:
    """Sizintinin tek gecerli kaniti budur.

    Uygulama cercevesindeki hedef degerleri 1000 katina cikariliyor. Ozetler
    yalnizca ``uydur``dan geliyorsa cikti BIREBIR ayni kalmalidir.
    """
    # Arrange
    uydur, uygula = _bol(_panel())
    ozetler = trafo_ozetleri_cikar(uydur)
    temiz = trafo_ozetleri_uygula(uygula, ozetler)

    bozuk_cerceve = uygula.copy()
    bozuk_cerceve["tuketim"] = bozuk_cerceve["tuketim"] * 1000.0

    # Act
    bozuk = trafo_ozetleri_uygula(bozuk_cerceve, ozetler)

    # Assert
    ozet_kolonlari = [k for k in temiz.columns if k.startswith(("t_", "soguk"))]
    assert ozet_kolonlari, "hic ozet kolonu uretilmemis -- test bir sey olcmuyor"
    pd.testing.assert_frame_equal(
        temiz[ozet_kolonlari].reset_index(drop=True),
        bozuk[ozet_kolonlari].reset_index(drop=True),
    )


def test_uydur_hedefi_bozulunca_ozetler_degisir() -> None:
    """Onceki testin karsi kutbu: fonksiyon gercekten hedefi okuyor mu.

    Bu olmadan onceki test, hicbir sey hesaplamayan bos bir fonksiyonda da
    gecerdi. Ikisi birlikte 'yalnizca ve tam olarak uydur'dan' der.
    """
    uydur, uygula = _bol(_panel())
    temiz = trafo_ozetleri_uygula(uygula, trafo_ozetleri_cikar(uydur))

    bozuk_uydur = uydur.copy()
    bozuk_uydur["tuketim"] = bozuk_uydur["tuketim"] * 1000.0
    bozuk = trafo_ozetleri_uygula(uygula, trafo_ozetleri_cikar(bozuk_uydur))

    assert not np.allclose(
        temiz["t_log_ort"].to_numpy(), bozuk["t_log_ort"].to_numpy(), equal_nan=True
    )


def test_grup_seviyeleri_de_uygulamadan_okumaz() -> None:
    uydur, uygula = _bol(_panel())
    temiz = grup_seviyeleri_ekle(uygula, uydur)

    bozuk_cerceve = uygula.copy()
    bozuk_cerceve["tuketim"] = bozuk_cerceve["tuketim"] * 1000.0
    bozuk = grup_seviyeleri_ekle(bozuk_cerceve, uydur)

    kolonlar = ["g_kova_log_ort", "g_ilce_log_ort", "g_ilce_kova_ort", "g_ilce_kova_n"]
    pd.testing.assert_frame_equal(
        temiz[kolonlar].reset_index(drop=True), bozuk[kolonlar].reset_index(drop=True)
    )


# ------------------------------------------------------------ SOGUK BASLANGIC


def test_soguk_trafo_ozetleri_nan_kalir_sifir_degil() -> None:
    """0 yazmak 'tuketimi sifir' iddiasidir; NaN 'bilmiyorum' der.

    Test satirlarinin %22,16'si bu durumda (olculdu). Sessizce 0 yazan bir
    uygulama, o satirlarin tamamini sistematik olarak asagi ceker ve RMSLE
    log olcekte oldugu icin ceza buyur.
    """
    uydur, _ = _bol(_panel())
    yeni = _panel(trafolar=2, tohum=9)
    yeni["tanim"] = "YENI_" + yeni["tanim"]

    sonuc = trafo_ozetleri_uygula(yeni, trafo_ozetleri_cikar(uydur))

    assert sonuc["t_log_ort"].isna().all()
    assert sonuc["t_log_son14"].isna().all()
    assert sonuc["t_trend"].isna().all()
    assert (sonuc["soguk_mu"] == 1).all()


def test_bilinen_trafo_soguk_isaretlenmez() -> None:
    uydur, uygula = _bol(_panel())
    sonuc = trafo_ozetleri_uygula(uygula, trafo_ozetleri_cikar(uydur))
    assert (sonuc["soguk_mu"] == 0).all()
    assert sonuc["t_log_ort"].notna().all()


# ------------------------------------------------------------ DEGERLER


def test_seviye_dogru_hesaplanir() -> None:
    """El ile hesaplanabilir bir ornek -- fonksiyon gercekten ne yapiyor."""
    uydur = pd.DataFrame(
        {
            "tanim": ["A", "A", "A", "B"],
            "tarih": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-01"]),
            "guc": [100.0, 100.0, 100.0, 200.0],
            "tuketim": [0.0, 99.0, 999.0, 500.0],
        }
    )
    ozetler = trafo_ozetleri_cikar(uydur)
    beklenen = float(np.mean([np.log1p(0.0), np.log1p(99.0), np.log1p(999.0)]))
    assert ozetler.seviye.loc["A", "t_log_ort"] == pytest.approx(beklenen)
    assert ozetler.seviye.loc["A", "t_sifir_orani"] == pytest.approx(1 / 3)
    # yuk faktoru = ort(tuketim) / (guc * 24)
    assert ozetler.seviye.loc["B", "t_yuk_faktoru"] == pytest.approx(500.0 / (200.0 * 24.0))


def test_haftagunu_sapmasi_sifir_toplamlidir() -> None:
    """Sapma seviyeye gore tanimli; dengeli bir panelde toplami ~0 olmali.

    Bu, sapmanin yanlislikla MUTLAK ortalamaya donmedigini gosterir --
    mutlak ortalama tutulsaydi ``t_log_ort`` ile esdogrusal olurdu ve
    modele hicbir yeni bilgi tasimazdi.
    """
    uydur, _ = _bol(_panel(gun=70), bolme="2025-03-01")
    ozetler = trafo_ozetleri_cikar(uydur)
    for _tanim, parca in ozetler.haftagunu.groupby("tanim"):
        assert abs(float(parca["t_hg_sapma"].mean())) < 0.05


def test_isil_egim_sabit_sicaklikta_nan() -> None:
    """Uc farkli sicaklik degeri yoksa egim TANIMSIZDIR.

    0 dondurmek 'bu trafo sicakliga tepki vermiyor' iddiasidir ve olculmemis
    bir iddiadir.
    """
    uydur = pd.DataFrame(
        {
            "tanim": ["A"] * 5,
            "tarih": pd.date_range("2025-01-01", periods=5),
            "guc": [100.0] * 5,
            "tuketim": [10.0, 20.0, 30.0, 40.0, 50.0],
            "sogutma_derece_gun": [7.0] * 5,
        }
    )
    ozetler = trafo_ozetleri_cikar(uydur)
    assert np.isnan(ozetler.isil_egim.loc["A", "t_egim_sogutma_derece_gun"])


def test_isil_egim_gercek_tepkiyi_yakalar() -> None:
    n = 30
    sicak = np.linspace(0.0, 15.0, n)
    uydur = pd.DataFrame(
        {
            "tanim": ["A"] * n,
            "tarih": pd.date_range("2025-01-01", periods=n),
            "guc": [100.0] * n,
            # log1p(tuketim) sicakliga DOGRUSAL bagli olsun -> egim pozitif
            "tuketim": np.expm1(2.0 + 0.3 * sicak),
            "sogutma_derece_gun": sicak,
        }
    )
    ozetler = trafo_ozetleri_cikar(uydur)
    assert ozetler.isil_egim.loc["A", "t_egim_sogutma_derece_gun"] == pytest.approx(0.3, abs=0.01)


# ------------------------------------------------------------ KOVA


def test_guc_kovasi_sirali_ve_monoton() -> None:
    guc = pd.Series([40.0, 100.0, 250.0, 630.0, 1600.0, 35900.0])
    kova = guc_kovasi(guc)
    assert list(kova) == sorted(kova)
    assert kova.iloc[0] < kova.iloc[-1]


def test_ilce_kova_hucre_sayisi_uretilir() -> None:
    """Seyrek hucre ortalamasi gurultudur; model bunu ancak N'i GORURSE
    indirime tabi tutabilir."""
    uydur, uygula = _bol(_panel())
    sonuc = grup_seviyeleri_ekle(uygula, uydur)
    assert "g_ilce_kova_n" in sonuc.columns
    assert (sonuc["g_ilce_kova_n"] > 0).any()
    assert sonuc["g_ilce_kova_n"].notna().all()


# ------------------------------------------------------------ KAPILAR


def test_bos_uydur_cercevesi_reddedilir() -> None:
    with pytest.raises(ValueError, match="bos"):
        trafo_ozetleri_cikar(pd.DataFrame(columns=["tanim", "tarih", "tuketim", "guc"]))


def test_eksik_kolon_reddedilir() -> None:
    with pytest.raises(KeyError, match="kolon"):
        trafo_ozetleri_cikar(pd.DataFrame({"tanim": ["A"], "tarih": [pd.Timestamp("2025-01-01")]}))


def test_satir_sayisi_korunur() -> None:
    """Merge'ler cogaltma yapmamali -- ``validate='many_to_one'`` bunu
    zorluyor ama testi olmadan bir gun biri onu kaldirir."""
    uydur, uygula = _bol(_panel())
    sonuc = trafo_ozetleri_uygula(uygula, trafo_ozetleri_cikar(uydur))
    assert len(sonuc) == len(uygula)
    assert len(grup_seviyeleri_ekle(uygula, uydur)) == len(uygula)


def test_soguk_satirlarda_profil_de_bostur() -> None:
    """Soguk trafo yalnizca SEVIYE'yi degil PROFIL'i de kaybetmeli.

    Sessiz dogrulama yanliligi buradaydi: ``profil_kaynak`` hedef blogun
    disindaki her seyi icerdigi icin, bir blokta soguk olan trafo blogun
    SONRASINDAN profil kazaniyordu. Gercek test'te soguk trafonun hicbir
    yerde kaydi yok; dogrulamanin da oyle davranmasi sart.
    """
    # Arrange: uydur penceresinde OLMAYAN, ama profil kaynaginda OLAN trafo
    uydur, uygula = _bol(_panel())
    yeni = _panel(trafolar=2, tohum=5)
    yeni["tanim"] = "SONRADAN_" + yeni["tanim"]
    profil_kaynak = pd.concat([uydur, yeni], ignore_index=True)

    # Act
    ozetler = trafo_ozetleri_cikar(uydur, profil_kaynak=profil_kaynak)
    sonuc = trafo_ozetleri_uygula(yeni, ozetler)

    # Assert: profil kaynagi onlari GORDU ama tasinmadi
    assert (sonuc["soguk_mu"] == 1).all()
    assert sonuc["t_hg_sapma"].isna().all()
    assert sonuc["t_ay_sapma"].isna().all()


def test_sicak_satirlarda_profil_tasinir() -> None:
    """Onceki testin karsi kutbu: silme kurali fazla genis olmamali."""
    uydur, uygula = _bol(_panel())
    sonuc = trafo_ozetleri_uygula(uygula, trafo_ozetleri_cikar(uydur))
    assert (sonuc["soguk_mu"] == 0).all()
    assert sonuc["t_hg_sapma"].notna().any()
