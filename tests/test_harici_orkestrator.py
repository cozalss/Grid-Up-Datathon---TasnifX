"""``attach_external`` sozlesmesi -- 12 harici kaynagin TEK kapisi (P1-5).

2026-08-18 denetimi: on bir kaynagin sekizi yalnizca kutuphane fonksiyonuydu,
hicbir pipeline cagirmiyordu. Bu testler orkestratorun sozlesmesini sabitler:
satir sirasi/sayisi korunur, aile secimi calisir, eksik dosya SESSIZ NaN degil
raporlanan atlama uretir, %0 eslesme HATA verir, kolon cakismasi reddedilir.

Testler gercek parquet'lere BAGIMLI DEGILDIR: sentetik kaynaklar tmp_path
altina yazilir, boylece CI'da (veri yokken) da kosarlar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridup.features.external import EXTERNAL_FAMILIES, attach_external


def _panel(n_ilce: int = 3, n_gun: int = 40) -> pd.DataFrame:
    gunler = pd.date_range("2026-01-01", periods=n_gun, freq="D")
    ilceler = [f"ilce{i}" for i in range(n_ilce)]
    return pd.DataFrame(
        [(ilce, gun) for ilce in ilceler for gun in gunler], columns=["ilce_key", "gun"]
    )


def _kaynaklari_yaz(kok: Path, panel: pd.DataFrame) -> None:
    """Sentetik harici kaynaklar: gunluk tablolar, nokta olay, turizm, izsu, epias."""
    (kok / "data/external/epias").mkdir(parents=True, exist_ok=True)
    (kok / "data/reference").mkdir(parents=True, exist_ok=True)
    ilceler = sorted(panel["ilce_key"].unique())
    gunler = pd.DatetimeIndex(sorted(panel["gun"].unique()))
    izgara = pd.DataFrame([(i, g) for i in ilceler for g in gunler], columns=["ilce_key", "tarih"])

    rng = np.random.default_rng(0)
    hava = izgara.assign(
        konum="x", konum_key="x", il_key="il", sicaklik_ort=rng.normal(20, 5, len(izgara))
    )
    hava.to_parquet(kok / "data/external/hava_gunluk.parquet", index=False)
    izgara.assign(basinc_min=rng.normal(1000, 5, len(izgara))).to_parquet(
        kok / "data/external/hava_saatlik_turev.parquet", index=False
    )
    izgara.assign(pm10_ort=rng.gamma(2, 5, len(izgara))).to_parquet(
        kok / "data/external/hava_kalitesi_gunluk.parquet", index=False
    )
    izgara.assign(cape_max=rng.gamma(2, 50, len(izgara))).to_parquet(
        kok / "data/external/konvektif_gunluk.parquet", index=False
    )
    izgara.assign(nem_ort=rng.uniform(30, 90, len(izgara))).to_parquet(
        kok / "data/external/nem_toprak_gunluk.parquet", index=False
    )
    izgara.assign(
        anahtar=izgara["ilce_key"].map(lambda k: f"il|{k}"),
        gun_uzunlugu_saat=rng.uniform(9, 15, len(izgara)),
    ).drop(columns=["ilce_key"]).to_parquet(kok / "data/external/gunes_gunluk.parquet", index=False)

    referans = pd.DataFrame(
        {
            "ilce_key": ilceler,
            "il_key": ["il"] * len(ilceler),
            "lat": [38.4 + 0.1 * i for i in range(len(ilceler))],
            "lon": [27.1 + 0.1 * i for i in range(len(ilceler))],
            "nufus": [100_000 + 1000 * i for i in range(len(ilceler))],
        }
    )
    referans.to_parquet(kok / "data/reference/ilceler_gdz_adm.parquet", index=False)

    pd.DataFrame(
        {"tarih": gunler[:10], "lat": 38.4, "lon": 27.1, "frp": 50.0, "guven": "h", "aygit": "x"}
    ).to_parquet(kok / "data/external/yanginlar.parquet", index=False)
    pd.DataFrame(
        {"tarih": gunler[:5], "lat": 38.4, "lon": 27.1, "buyukluk": 4.5, "derinlik_km": 10.0}
    ).to_parquet(kok / "data/external/depremler.parquet", index=False)

    pd.DataFrame(
        {
            "yil": [2025] * len(ilceler),
            "il_key": ["il"] * len(ilceler),
            "ilce_key": ilceler,
            "geceleme": [1000.0 * (i + 1) for i in range(len(ilceler))],
            "tesise_gelis": [500.0 * (i + 1) for i in range(len(ilceler))],
        }
    ).to_parquet(kok / "data/external/turizm_geceleme.parquet", index=False)
    pd.DataFrame(
        {
            "il_key": ["il"] * 12,
            "yil": [2025] * 12,
            "ay": list(range(1, 13)),
            "geceleme": [1000.0 * a for a in range(1, 13)],
        }
    ).to_parquet(kok / "data/external/turizm_aylik_il.parquet", index=False)
    pd.DataFrame(
        {
            "ilce_key": [i for i in ilceler for _ in range(12)],
            "ay": list(range(1, 13)) * len(ilceler),
            "su_ay_endeksi": rng.uniform(0.8, 1.3, 12 * len(ilceler)),
        }
    ).to_parquet(kok / "data/external/izsu_su_profili.parquet", index=False)
    saatler = pd.date_range(gunler.min(), gunler.max() + pd.Timedelta(days=1), freq="h")
    pd.DataFrame(
        {"zaman": saatler, "consumption": rng.normal(30000, 3000, len(saatler))}
    ).to_parquet(kok / "data/external/epias/tuketim_saatlik.parquet", index=False)


def test_tum_aileler_baglaniyor_ve_satir_sirasi_korunuyor(tmp_path: Path) -> None:
    panel = _panel()
    _kaynaklari_yaz(tmp_path, panel)
    sonuc = attach_external(
        panel, key_column="ilce_key", time_column="gun", horizon=7, root=tmp_path
    )
    assert set(sonuc.families) == set(EXTERNAL_FAMILIES), sonuc.skipped
    assert not sonuc.skipped
    assert len(sonuc.frame) == len(panel)
    pd.testing.assert_frame_equal(sonuc.frame[["ilce_key", "gun"]], panel[["ilce_key", "gun"]])
    assert len(sonuc.feature_columns) > 20
    # Gunluk tablolar tam eslesmeli
    for aile in ("hava", "hava_kalitesi", "konvektif", "nem_toprak", "gunes"):
        assert sonuc.match_rates[aile] == pytest.approx(1.0)
    assert "attach_external" in sonuc.summary()


def test_aile_secimi_yalnizca_istenen_kolonlari_uretir(tmp_path: Path) -> None:
    panel = _panel()
    _kaynaklari_yaz(tmp_path, panel)
    sonuc = attach_external(
        panel,
        key_column="ilce_key",
        time_column="gun",
        horizon=7,
        families=["hava", "izsu"],
        root=tmp_path,
    )
    assert set(sonuc.families) == {"hava", "izsu"}
    assert "cape_max" not in sonuc.frame.columns
    with pytest.raises(ValueError, match="Bilinmeyen aile"):
        attach_external(
            panel, key_column="ilce_key", time_column="gun", horizon=7,
            families=["yok"], root=tmp_path,
        )  # fmt: skip


def test_eksik_kaynak_sessiz_nan_degil_raporlanan_atlama(tmp_path: Path) -> None:
    panel = _panel()
    _kaynaklari_yaz(tmp_path, panel)
    (tmp_path / "data/external/konvektif_gunluk.parquet").unlink()
    sonuc = attach_external(
        panel, key_column="ilce_key", time_column="gun", horizon=7, root=tmp_path
    )
    assert "konvektif" in sonuc.skipped
    assert "kaynak dosya yok" in sonuc.skipped["konvektif"]
    assert "konvektif" not in sonuc.families
    assert "ATLANDI" in sonuc.summary()


def test_sifir_eslesme_hata_verir(tmp_path: Path) -> None:
    """Anahtar bicimi bozuksa join SESSIZ NaN degil, hata uretmeli."""
    panel = _panel()
    _kaynaklari_yaz(tmp_path, panel)
    bozuk = panel.assign(ilce_key=panel["ilce_key"] + "_BOZUK")
    with pytest.raises(ValueError, match="HIC eslesmedi"):
        attach_external(
            bozuk, key_column="ilce_key", time_column="gun", horizon=7,
            families=["hava"], root=tmp_path,
        )  # fmt: skip


def test_kolon_cakismasi_reddedilir_ve_ufuk_dogrulanir(tmp_path: Path) -> None:
    panel = _panel().assign(sicaklik_ort=1.0)
    _kaynaklari_yaz(tmp_path, panel)
    with pytest.raises(ValueError, match="panelde zaten var"):
        attach_external(
            panel, key_column="ilce_key", time_column="gun", horizon=7,
            families=["hava"], root=tmp_path,
        )  # fmt: skip
    with pytest.raises(ValueError, match="horizon"):
        attach_external(
            _panel(), key_column="ilce_key", time_column="gun", horizon=0, root=tmp_path
        )
    with pytest.raises(KeyError, match="panel icinde"):
        attach_external(_panel(), key_column="yok", time_column="gun", horizon=7, root=tmp_path)


def test_gecmise_donuk_kaynaklar_ufuk_kadar_kaydirilmis(tmp_path: Path) -> None:
    """Nokta olay ve ulusal seri kolonlari ufuk duvarinin ARKASINDA olmali."""
    panel = _panel(n_gun=60)
    _kaynaklari_yaz(tmp_path, panel)
    sonuc = attach_external(
        panel,
        key_column="ilce_key",
        time_column="gun",
        horizon=10,
        families=["yangin", "epias"],
        root=tmp_path,
    )
    # Ham (kaydirilmamis) gunluk yogunluk kolonu ciktida BIRAKILMAZ
    ham_adaylar = [k for k in sonuc.families["yangin"] if "shift" not in k and "kayan" not in k]
    assert all("gun" not in k or "shift" in k or "kayan" in k for k in ham_adaylar)
    # Ilk horizon gunu bilgi tasiyamaz: en az bir kolon bas taraflarda NaN
    ilk_gunler = sonuc.frame["gun"] <= sonuc.frame["gun"].min() + pd.Timedelta(days=5)
    epias_kolon = sonuc.families["epias"][0]
    assert sonuc.frame.loc[ilk_gunler, epias_kolon].isna().any()
