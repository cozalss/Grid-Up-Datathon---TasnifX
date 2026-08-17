"""Komsu ilce sinyalinin sizinti yuzeyleri -- olculmus regresyon testleri.

Bu dosya UC somut kusuru kilitler. Her testin docstring'i, duzeltmeden ONCE ve
SONRA GERCEKTEN olculmus sayilari tasir; sayi tutmuyorsa test degil kod
degismistir.

Panel ureteci ``_panel`` bilincli olarak gunluk ORTAK bir bolgesel sok
kullanir (firtina tum ilceleri ayni gun vurur). Komsu sinyalinin degeri de,
sizinti riski de tam olarak bu ortak soktan gelir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.features.spatial import (
    add_neighbour_feature_mean,
    add_neighbour_target_lag,
    nearest_neighbours,
)

N_ILCE = 20
N_GUN = 60
ANAHTAR = "ilce_key"
ZAMAN = "tarih"
HEDEF = "hedef"


def _koordinatlar() -> pd.DataFrame:
    """20 ilce, Ege bolgesi olceginde (hepsi 120 km komsuluk yaricapinda)."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            ANAHTAR: [f"ILCE_{i:02d}" for i in range(N_ILCE)],
            "lat": 38.0 + rng.uniform(-0.6, 0.6, N_ILCE),
            "lon": 27.5 + rng.uniform(-0.6, 0.6, N_ILCE),
        }
    )


def _panel(satir_gun: int, tohum: int = 7) -> pd.DataFrame:
    """Ilce basina gunde ``satir_gun`` satir. 1 = temiz panel, >1 = tekrarli."""
    rng = np.random.default_rng(tohum)
    gunler = pd.date_range("2024-01-01", periods=N_GUN, freq="D")
    sok = rng.gamma(2.0, 3.0, N_GUN)
    kayit = []
    for i in range(N_ILCE):
        for gun_no, gun in enumerate(gunler):
            taban = sok[gun_no] * (1.0 + 0.1 * i)
            for _ in range(satir_gun):
                kayit.append(
                    {
                        ANAHTAR: f"ILCE_{i:02d}",
                        ZAMAN: gun,
                        HEDEF: float(taban + rng.normal(0, 1.0)),
                    }
                )
    return pd.DataFrame(kayit)


@pytest.fixture
def komsuluk() -> pd.DataFrame:
    return nearest_neighbours(_koordinatlar(), key_column=ANAHTAR, k=3, max_distance_km=120)


@pytest.fixture
def gunluk_panel() -> pd.DataFrame:
    """Ilce-gun bazinda TEK satir -- fonksiyonlarin bekledigi sekil."""
    return _panel(satir_gun=1)


# ---------------------------------------------------------------------------
# 1) SATIR bazli shift, tekrarli (ilce, gun) panelinde ayni gunu sizdiriyordu
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("satir_gun", "horizon", "beklenen_tekrar"),
    [(40, 30, 46800), (60, 7, 70800)],
)
def test_tekrarli_ilce_gun_paneli_reddediliyor(
    komsuluk, satir_gun: int, horizon: int, beklenen_tekrar: int
):
    """Gunde birden cok satir varsa shift(horizon) ayni gune dusuyordu.

    OLCULDU (20 ilce x 60 gun, feature ile satirin KENDI ayni gunku hedefi
    arasindaki korelasyon):
      40 satir/gun, horizon=30 -> ayni-gun orani %25.0, corr=+0.1682, ISTISNA YOK
      60 satir/gun, horizon= 7 -> ayni-gun orani %88.3, corr=+0.8098, ISTISNA YOK
    SONRA: her ikisi de ValueError -- korelasyon hic uretilemiyor.
    """
    frame = _panel(satir_gun=satir_gun)
    assert int(frame.duplicated([ANAHTAR, ZAMAN]).sum()) == beklenen_tekrar

    with pytest.raises(ValueError, match="SATIR kaydirir, GUN degil"):
        add_neighbour_target_lag(
            frame,
            komsuluk,
            key_column=ANAHTAR,
            time_column=ZAMAN,
            target_column=HEDEF,
            horizon=horizon,
            statistics=("mean",),
        )


def test_tekrar_hatasi_hangi_anahtarin_tekrarladigini_soyluyor(komsuluk):
    """Sessiz duzeltme yok: hata mesaji sayiyi ve ornek anahtari vermeli."""
    frame = _panel(satir_gun=2)

    with pytest.raises(ValueError) as bilgi:
        add_neighbour_target_lag(
            frame,
            komsuluk,
            key_column=ANAHTAR,
            time_column=ZAMAN,
            target_column=HEDEF,
            horizon=7,
        )

    mesaj = str(bilgi.value)
    assert "1200 satirda tekrarliyor" in mesaj  # 20 ilce x 60 gun x (2-1)
    assert "ILCE_00" in mesaj
    assert "groupby" in mesaj  # dogru kullanimi da soyluyor


def test_gunluk_toplanmis_panel_hala_calisiyor(komsuluk, gunluk_panel):
    """YANLIS POZITIF KORUMASI: temiz panelde davranis DEGISMEDI.

    OLCULDU (ayni ham veri once ilce-gun bazinda toplanarak, horizon=30):
      corr(feature, BUGUNKU hedef) = -0.1771  (ONCE ve SONRA ayni)
    Yani duzeltme dogru kullanimi kirmiyor; yalnizca tekrarli girdiyi kesiyor.
    """
    sonuc = add_neighbour_target_lag(
        gunluk_panel,
        komsuluk,
        key_column=ANAHTAR,
        time_column=ZAMAN,
        target_column=HEDEF,
        horizon=30,
        statistics=("mean",),
    )

    assert len(sonuc) == len(gunluk_panel)
    assert f"komsu_{HEDEF}_ufuk30_mean" in sonuc.columns


def test_komsunun_gecmisi_kullaniliyor_bugunu_degil(komsuluk):
    """Kaydirmanin YONU: 2. gunun feature'i komsunun 1. gunku degeri olmali.

    Elle kurulmus iki ilce: ILCE_00 hep 0.0, ILCE_01 = [7, 8, 9, 10].
    horizon=1 -> ILCE_00'in 2. gun feature'i 7.0 (komsunun DUNU), 8.0 degil.
    """
    tarihler = pd.date_range("2024-01-01", periods=4, freq="D")
    frame = pd.concat(
        [
            pd.DataFrame({ANAHTAR: "ILCE_00", ZAMAN: tarihler, HEDEF: [0.0] * 4}),
            pd.DataFrame({ANAHTAR: "ILCE_01", ZAMAN: tarihler, HEDEF: [7.0, 8.0, 9.0, 10.0]}),
        ],
        ignore_index=True,
    )
    yakin = pd.DataFrame([{ANAHTAR: "ILCE_00", "komsu": "ILCE_01", "mesafe_km": 10.0}])

    sonuc = add_neighbour_target_lag(
        frame,
        yakin,
        key_column=ANAHTAR,
        time_column=ZAMAN,
        target_column=HEDEF,
        horizon=1,
        statistics=("max",),
    )

    ilce = sonuc[sonuc[ANAHTAR] == "ILCE_00"].sort_values(ZAMAN)
    assert pd.isna(ilce[f"komsu_{HEDEF}_ufuk1_max"].iloc[0])
    assert ilce[f"komsu_{HEDEF}_ufuk1_max"].iloc[1] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# 2) add_neighbour_feature_mean hedef korumasi
# ---------------------------------------------------------------------------


def test_komsu_ortalamasi_hedefi_reddediyor(komsuluk, gunluk_panel):
    """Hedef value_columns'a girerse komsunun AYNI GUNKU hedefi feature olurdu.

    OLCULDU: ISTISNA YOK, corr(komsu_hedef_mean, satirin AYNI GUNKU hedefi)
    = 0.813099. Ikinci savunma hatti da tutmuyordu: leakage_report korelasyon
    esigi 0.95 ve 0.813099 onun ALTINDA -- rapor "0 kritik" diyordu.
    SONRA: ValueError.
    """
    with pytest.raises(ValueError, match="value_columns icinde"):
        add_neighbour_feature_mean(
            gunluk_panel,
            komsuluk,
            key_column=ANAHTAR,
            time_column=ZAMAN,
            value_columns=[HEDEF],
            target_column=HEDEF,
        )


def test_target_column_verilmezse_calismiyor(komsuluk, gunluk_panel):
    """Nobetci opt-in OLAMAZ: varsayilan None olsaydi koruma hic calismazdi.

    ONCE: target_column parametresi YOKTU, cagri sessizce geciyordu (corr
    0.813099). SONRA: TypeError.
    """
    with pytest.raises(TypeError, match="target_column ACIKCA verilmelidir"):
        add_neighbour_feature_mean(
            gunluk_panel,
            komsuluk,
            key_column=ANAHTAR,
            time_column=ZAMAN,
            value_columns=[HEDEF],
        )


def test_masum_hava_kolonu_engellenmiyor(komsuluk, gunluk_panel):
    """YANLIS POZITIF KORUMASI: hedef DISI kolon hedef bildirilse de gecmeli.

    OLCULDU: 1200 satirlik panelde komsu_sicaklik_mean uretiliyor, hedef
    korumasi ateslenmiyor. Komsunun bugunku havasi tahmin aninda bilinir --
    kaydirma gerekmez ve sizinti degildir.
    """
    hava = gunluk_panel.assign(sicaklik=np.linspace(10, 30, len(gunluk_panel)))

    sonuc = add_neighbour_feature_mean(
        hava,
        komsuluk,
        key_column=ANAHTAR,
        time_column=ZAMAN,
        value_columns=["sicaklik"],
        target_column=HEDEF,
        statistics=("mean",),
    )

    assert "komsu_sicaklik_mean" in sonuc.columns
    assert len(sonuc) == len(hava) == 1200
    assert sonuc["komsu_sicaklik_mean"].notna().all()


def test_hedef_yokken_none_bilincli_karar_olarak_kabul_ediliyor(komsuluk, gunluk_panel):
    """Test frame'inde hedef olmayabilir -- ``None`` acik bir karardir, gecer."""
    hava = gunluk_panel.drop(columns=[HEDEF]).assign(ruzgar=1.0)

    sonuc = add_neighbour_feature_mean(
        hava,
        komsuluk,
        key_column=ANAHTAR,
        time_column=ZAMAN,
        value_columns=["ruzgar"],
        target_column=None,
        statistics=("mean",),
    )

    assert "komsu_ruzgar_mean" in sonuc.columns


# ---------------------------------------------------------------------------
# 3) Komsu 'sum' istatistigi: "bilgi yok" != "kesinti yok"
# ---------------------------------------------------------------------------


def test_komsu_toplami_bilgi_yokken_nan_kaliyor(komsuluk, gunluk_panel):
    """Serinin ilk gununde kaydirilmis komsu degeri YOKTUR -- toplam da NaN olmali.

    OLCULDU (ilk gun, 20 ilce, horizon=1):
      ONCE : mean NaN orani=1.00, sum NaN orani=0.00  (tum sum degerleri 0.0)
      SONRA: mean NaN orani=1.00, sum NaN orani=1.00
    pandas sum varsayilani min_count=0 oldugu icin tum-NaN grup 0.0 doner;
    LightGBM bunu 'gercek sifir' dali olarak isler ve her varligin ilk
    ``horizon`` gununde sistematik hata olusur.
    """
    sonuc = add_neighbour_target_lag(
        gunluk_panel,
        komsuluk,
        key_column=ANAHTAR,
        time_column=ZAMAN,
        target_column=HEDEF,
        horizon=1,
        statistics=("mean", "sum"),
    )

    ilk_gun = sonuc[sonuc[ZAMAN] == sonuc[ZAMAN].min()]
    ortalama_nan = ilk_gun[f"komsu_{HEDEF}_ufuk1_mean"].isna().mean()
    toplam_nan = ilk_gun[f"komsu_{HEDEF}_ufuk1_sum"].isna().mean()

    assert ortalama_nan == 1.0
    assert toplam_nan == 1.0, "sum, 'bilgi yok' halini 0.0'a duduruyor"


def test_komsuda_gercek_sifir_varken_toplam_sifir_kaliyor():
    """YANLIS POZITIF KORUMASI: olculmus 0.0 ile 'olcum yok' ayri kalmali.

    Komsunun DUNKU kesintisi gercekten 0.0 ise toplam da 0.0 olmali (NaN
    degil) -- aksi halde duzeltme bu sefer ters yonde bilgi kaybettirirdi.
    """
    frame = pd.DataFrame(
        {
            ANAHTAR: ["ILCE_00", "ILCE_00", "ILCE_01", "ILCE_01"],
            ZAMAN: pd.to_datetime(["2024-01-01", "2024-01-02"] * 2),
            HEDEF: [5.0, 5.0, 0.0, 0.0],
        }
    )
    yakin = pd.DataFrame([{ANAHTAR: "ILCE_00", "komsu": "ILCE_01", "mesafe_km": 10.0}])

    sonuc = add_neighbour_target_lag(
        frame,
        yakin,
        key_column=ANAHTAR,
        time_column=ZAMAN,
        target_column=HEDEF,
        horizon=1,
        statistics=("sum",),
    )

    ikinci_gun = sonuc[(sonuc[ANAHTAR] == "ILCE_00") & (sonuc[ZAMAN] == pd.Timestamp("2024-01-02"))]
    assert ikinci_gun[f"komsu_{HEDEF}_ufuk1_sum"].iloc[0] == 0.0


def test_kismi_nan_grupta_toplam_mevcut_degerleri_topluyor():
    """Bir komsu NaN, digeri dolu -> toplam dolu olani vermeli (min_count=1)."""
    tarihler = pd.to_datetime(["2024-01-01", "2024-01-02"])
    frame = pd.concat(
        [
            pd.DataFrame({ANAHTAR: "ILCE_00", ZAMAN: tarihler, HEDEF: [1.0, 1.0]}),
            pd.DataFrame({ANAHTAR: "ILCE_01", ZAMAN: tarihler, HEDEF: [4.0, 9.0]}),
            pd.DataFrame({ANAHTAR: "ILCE_02", ZAMAN: tarihler, HEDEF: [np.nan, 3.0]}),
        ],
        ignore_index=True,
    )
    yakin = pd.DataFrame(
        [
            {ANAHTAR: "ILCE_00", "komsu": "ILCE_01", "mesafe_km": 10.0},
            {ANAHTAR: "ILCE_00", "komsu": "ILCE_02", "mesafe_km": 20.0},
        ]
    )

    sonuc = add_neighbour_target_lag(
        frame,
        yakin,
        key_column=ANAHTAR,
        time_column=ZAMAN,
        target_column=HEDEF,
        horizon=1,
        statistics=("sum",),
    )

    ikinci_gun = sonuc[(sonuc[ANAHTAR] == "ILCE_00") & (sonuc[ZAMAN] == pd.Timestamp("2024-01-02"))]
    # ILCE_01'in dunu 4.0, ILCE_02'nin dunu NaN -> toplam 4.0 (NaN yutulur)
    assert ikinci_gun[f"komsu_{HEDEF}_ufuk1_sum"].iloc[0] == pytest.approx(4.0)
