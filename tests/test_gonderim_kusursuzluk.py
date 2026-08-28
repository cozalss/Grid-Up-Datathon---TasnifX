"""Gonderim dosyalarinin SIFIR HATA ve KUSURSUZLUK test paketi.

YEREL KAPI TESTIDIR. `submissions/*.csv` ve `data/raw/*` depoda tutulmaz
(gitignore -- her biri ~28 MB), bu yuzden CI'da bu testler ATLANIR. Amaclari
gonderim oncesi yerel dogrulama yapmaktir; CI'da yesil gorunmeleri bir sey
kanitlamaz, dosya orada zaten yoktur.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

KOK = Path(__file__).resolve().parents[1]
GONDERIM = KOK / "submissions"
DATA = KOK / "data" / "raw"
BEKLENEN_SATIR = 714688
BEKLENEN_TRAFO = 7036

#: Kapidan gecmesi gereken aday dosyalar. Yoksa o parametre atlanir.
ADAYLAR = (
    "tuketim_v91_grupb_kaldirma.csv",
    "tuketim_v90_temiz_sota.csv",
    "tuketim_v89_genis_taban.csv",
    "tuketim_v88_olu_taban.csv",
    "tuketim_v87_tabanli.csv",
    "tuketim_v87_olu_izole.csv",
    "tuketim_v85_gram_rank2.csv",
    "tuketim_sota_v1.csv",
)

veri_gerekir = pytest.mark.skipif(
    not (DATA / "sample_submission.csv").exists() or not (DATA / "test.csv").exists(),
    reason="data/raw depoda tutulmuyor (gitignore); yerel kapi testi",
)


def _oku(dosya_adi: str) -> pd.DataFrame:
    """Aday dosyayi oku; yoksa testi atla."""
    yol = GONDERIM / dosya_adi
    if not yol.exists():
        pytest.skip(f"{dosya_adi} yerelde yok")
    return pd.read_csv(yol)


@pytest.fixture(scope="module")
def ss_df():
    return pd.read_csv(DATA / "sample_submission.csv")


@pytest.fixture(scope="module")
def test_meta():
    return pd.read_csv(DATA / "test.csv")


@veri_gerekir
@pytest.mark.parametrize("dosya_adi", ADAYLAR)
def test_gonderim_satir_ve_id_sirasi(dosya_adi, ss_df):
    df = _oku(dosya_adi)
    assert len(df) == BEKLENEN_SATIR, f"Satir sayisi hatali: {len(df)} != {BEKLENEN_SATIR}"
    assert list(df.columns) == ["id", "tuketim"], f"Kolon adlari hatali: {list(df.columns)}"
    assert (df["id"].values == ss_df["id"].values).all(), (
        "ID siralamasi sample_submission ile birebir eslesmiyor!"
    )
    assert df["id"].duplicated().sum() == 0, "Tekrarlanan (duplicate) ID bulundu!"


@pytest.mark.parametrize("dosya_adi", ADAYLAR)
def test_gonderim_sayisal_gecerlilik(dosya_adi):
    df = _oku(dosya_adi)
    v = df["tuketim"].to_numpy(dtype="float64")

    assert not np.isnan(v).any(), "NaN deger tespit edildi!"
    assert not np.isinf(v).any(), "Sonsuz (inf) deger tespit edildi!"
    assert (v >= 0).all(), "Negatif tuketim degeri tespit edildi!"
    assert v.max() < 500000.0, f"Asiri buyuk fizik disi tuketim degeri: {v.max()}"


@veri_gerekir
@pytest.mark.parametrize("dosya_adi", ADAYLAR)
def test_gonderim_zaman_serisi_butunlugu(dosya_adi, test_meta):
    df = _oku(dosya_adi)

    merged = test_meta[["id", "tanim", "tarih", "guc"]].merge(df, on="id")
    assert len(merged) == BEKLENEN_SATIR

    # Test.csv ile birebir ayni trafo ve gun dagilimi
    beklenen_gun_sayilari = test_meta.groupby("tanim")["tarih"].count()
    gercek_gun_sayilari = merged.groupby("tanim")["tarih"].count()
    assert (gercek_gun_sayilari == beklenen_gun_sayilari).all(), (
        "Test.csv ile trafo gun sayilari uyusmuyor!"
    )
    assert len(gercek_gun_sayilari) == BEKLENEN_TRAFO, (
        f"Trafo sayisi hatali: {len(gercek_gun_sayilari)}"
    )


@pytest.mark.parametrize("dosya_adi", ADAYLAR)
def test_gonderim_log_dagilim_olcegi(dosya_adi):
    df = _oku(dosya_adi)
    lg = np.log1p(df["tuketim"].to_numpy())

    # Hedef dagilimin ortalamasi 6.20 - 6.80 bandinda olmali
    ort = lg.mean()
    std = lg.std()
    assert 6.20 <= ort <= 6.80, f"Log1p ortalamasi supheli olcekte: {ort:.4f}"
    assert 1.60 <= std <= 2.10, f"Log1p standart sapmasi supheli: {std:.4f}"
