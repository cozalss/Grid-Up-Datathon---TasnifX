"""DERIN KAZI 3. DALGA TAKTIKLERI (docs/10 bolum 6 "Simdi").

Yedi olculebilir aktarim, her biri kaynakli:

  * kararlilik-cezali harman secimi (Home Credit 2024 + M5 1.si)
  * kuvvet ortalamasi harmani (ASHRAE 1.-2.)
  * exposure-offset: init_score/base_margin (sigorta pratigi)
  * Hawkes-esinli ustel-bozunumlu recency (alan-disi sayim bilimi)
  * son olaydan gecen gun (Sivas tezi #3 feature'i)
  * mesafe-agirlikli komsu + genis komsu istatistikleri (KDD Cup 2018)
  * monotonik kisit yardimcisi (arXiv 2512.17945, maliyet ~%0-2.9)

Stil referansi: tests/test_kazanan_taktikleri.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.ensemble import hill_climb_weights, power_mean_blend, tune_power_mean
from gridup.features.spatial import (
    MIN_NEIGHBOUR_DISTANCE_KM,
    add_neighbour_feature_mean,
    add_neighbour_target_lag,
)
from gridup.features.temporal import add_days_since_event_features, add_event_decay_features
from gridup.models import cross_validate, monotone_constraints_for
from gridup.validation import purged_time_series_split

# --------------------------------------------------------------------------
# Kararlilik-cezali harman secimi (Home Credit 2024 + M5 1.si)
# --------------------------------------------------------------------------


def _oynak_kararli_oof() -> tuple[dict[str, np.ndarray], np.ndarray, list[np.ndarray]]:
    """Hedef 0; 'oynak' ort. 0.8 ama fold'lar arasi sapmali, 'kararli' sabit 1.0.

    oynak : fold1 mae 0.0, fold2 mae 1.6 -> ortalama 0.8, std 0.8
    kararli: fold1 mae 1.0, fold2 mae 1.0 -> ortalama 1.0, std 0.0
    Cezasiz objektifte oynak, cezali (>=2) objektifte kararli kazanmali.
    """
    y = np.zeros(200)
    oof = {
        "oynak": np.concatenate([np.zeros(100), np.full(100, 1.6)]),
        "kararli": np.ones(200),
    }
    dilimler = [np.arange(100), np.arange(100, 200)]
    return oof, y, dilimler


def test_ceza_sifir_eski_davranisla_bit_esit():
    """GERIYE UYUM: penalty=0 iken fold_slices verilse bile eski yol calisir."""
    rng = np.random.default_rng(7)
    y = rng.gamma(2.0, 10.0, 300)
    oof = {
        "a": y + rng.normal(0, 3, 300),
        "b": y + rng.normal(0, 5, 300),
        "c": y * 0.9 + rng.normal(0, 4, 300),
    }
    dilimler = [np.arange(150), np.arange(150, 300)]

    eski = hill_climb_weights(oof, y, metric="mae", verbose=False)
    yeni = hill_climb_weights(
        oof, y, metric="mae", stability_penalty=0.0, fold_slices=dilimler, verbose=False
    )
    assert eski == yeni  # bit-esit: ayni kod yolu, ayni float'lar


def test_ceza_kararli_uyeye_agirlik_kaydiriyor():
    """Ceza acilinca objektif ortalama + ceza*std olur ve kararli uye kazanir."""
    oof, y, dilimler = _oynak_kararli_oof()

    cezasiz = hill_climb_weights(oof, y, metric="mae", verbose=False)
    cezali = hill_climb_weights(
        oof, y, metric="mae", stability_penalty=2.0, fold_slices=dilimler, verbose=False
    )
    # Cezasiz: oynak genel MAE 0.8 < 1.0 -> tum agirlik oynakta.
    assert cezasiz["oynak"] == pytest.approx(1.0)
    # Cezali: oynak objektifi 0.8 + 2*0.8 = 2.4 > 1.0 -> tum agirlik kararlida.
    assert cezali["kararli"] == pytest.approx(1.0)
    assert cezali["oynak"] == pytest.approx(0.0)


def test_ceza_buyuk_iyi_metrikte_isaret_donuyor():
    """greater_is_better metrikte objektif ortalama - ceza*std olmali (smoke)."""
    rng = np.random.default_rng(3)
    y = rng.normal(0, 1, 200)
    oof = {"a": y + rng.normal(0, 0.1, 200), "b": y + rng.normal(0, 0.3, 200)}
    dilimler = [np.arange(100), np.arange(100, 200)]

    agirlik = hill_climb_weights(
        oof, y, metric="r2", stability_penalty=1.0, fold_slices=dilimler, verbose=False
    )
    assert sum(agirlik.values()) == pytest.approx(1.0)
    assert all(deger >= 0 for deger in agirlik.values())


def test_ceza_parametre_hatalari():
    oof, y, dilimler = _oynak_kararli_oof()
    with pytest.raises(ValueError, match="stability_penalty"):
        hill_climb_weights(oof, y, metric="mae", stability_penalty=-0.5, verbose=False)
    with pytest.raises(ValueError, match="fold_slices ZORUNLU"):
        hill_climb_weights(oof, y, metric="mae", stability_penalty=0.5, verbose=False)
    with pytest.raises(ValueError, match="en az 2 dilim"):
        hill_climb_weights(
            oof,
            y,
            metric="mae",
            stability_penalty=0.5,
            fold_slices=[np.arange(100)],
            verbose=False,
        )
    with pytest.raises(ValueError, match="bos"):
        hill_climb_weights(
            oof,
            y,
            metric="mae",
            stability_penalty=0.5,
            fold_slices=[np.arange(100), np.array([])],
            verbose=False,
        )
    with pytest.raises(ValueError, match="disina tasiyor"):
        hill_climb_weights(
            oof,
            y,
            metric="mae",
            stability_penalty=0.5,
            fold_slices=[np.arange(100), np.array([500])],
            verbose=False,
        )


# --------------------------------------------------------------------------
# Kuvvet ortalamasi harmani (ASHRAE 1.-2.)
# --------------------------------------------------------------------------

_KUVVET_TAHMIN = {"a": np.array([1.0, 4.0]), "b": np.array([9.0, 16.0])}
_KUVVET_AGIRLIK = {"a": 0.5, "b": 0.5}


def test_kuvvet_p1_agirlikli_aritmetige_bit_esit():
    """p=1 dogrusal harmanla BIT-ESIT olmali -- negatif tahminde bile."""
    tahminler = {"a": np.array([-2.0, 3.0, 7.5]), "b": np.array([4.0, -1.0, 2.5])}
    agirlik = {"a": 0.3, "b": 0.7}
    beklenen = 0.3 * tahminler["a"] + 0.7 * tahminler["b"]
    np.testing.assert_array_equal(power_mean_blend(tahminler, agirlik, p=1.0), beklenen)


def test_kuvvet_p0_geometrik_ortalama():
    """p=0 log-uzayda geometrik: sqrt(1*9)=3, sqrt(4*16)=8."""
    np.testing.assert_allclose(power_mean_blend(_KUVVET_TAHMIN, _KUVVET_AGIRLIK, p=0.0), [3.0, 8.0])


def test_kuvvet_p2_el_hesabi():
    """p=2: sqrt(0.5*1 + 0.5*81) = sqrt(41)."""
    np.testing.assert_allclose(
        power_mean_blend(_KUVVET_TAHMIN, _KUVVET_AGIRLIK, p=2.0),
        [np.sqrt(41.0), np.sqrt(0.5 * 16.0 + 0.5 * 256.0)],
    )


def test_kuvvet_negatif_tahmin_kirpiliyor():
    """p != 1 icin kesirli us negatifte tanimsiz -> 0'a kirpilir."""
    tahminler = {"a": np.array([-5.0]), "b": np.array([4.0])}
    sonuc = power_mean_blend(tahminler, _KUVVET_AGIRLIK, p=2.0)
    assert sonuc[0] == pytest.approx(np.sqrt(0.5 * 16.0))


def test_kuvvet_p0_sifir_uyede_patlamiyor():
    tahminler = {"a": np.array([0.0]), "b": np.array([4.0])}
    sonuc = power_mean_blend(tahminler, _KUVVET_AGIRLIK, p=0.0)
    assert np.isfinite(sonuc).all()
    assert sonuc[0] == pytest.approx(0.0, abs=1e-3)


def test_kuvvet_agirlik_hatalari():
    with pytest.raises(ValueError, match="Bos"):
        power_mean_blend({}, {})
    with pytest.raises(ValueError, match="ayni adlari"):
        power_mean_blend(_KUVVET_TAHMIN, {"a": 1.0})
    with pytest.raises(ValueError, match="ayni adlari"):
        power_mean_blend(_KUVVET_TAHMIN, {**_KUVVET_AGIRLIK, "hayalet": 0.1})
    with pytest.raises(ValueError, match="Negatif"):
        power_mean_blend(_KUVVET_TAHMIN, {"a": -0.5, "b": 1.5})
    with pytest.raises(ValueError, match="toplami sifir"):
        power_mean_blend(_KUVVET_TAHMIN, {"a": 0.0, "b": 0.0})
    with pytest.raises(ValueError, match="uzunlukta"):
        power_mean_blend({"a": np.ones(3), "b": np.ones(4)}, _KUVVET_AGIRLIK)


def test_tune_kuvvet_dogru_p_yi_buluyor():
    """Hedef tam p=2 harmaniysa tarama p=2'yi sifir hatayla bulmali."""
    y = power_mean_blend(_KUVVET_TAHMIN, _KUVVET_AGIRLIK, p=2.0)
    en_iyi, skor, tablo = tune_power_mean(
        _KUVVET_TAHMIN, y, weights=_KUVVET_AGIRLIK, p_grid=(0.5, 1.0, 2.0)
    )
    assert en_iyi == pytest.approx(2.0)
    assert skor == pytest.approx(0.0, abs=1e-12)
    assert len(tablo) == 3


def test_tune_kuvvet_p1_daima_izgarada():
    """p=1 izgarada olmasa bile eklenir: 'dogrusal harman' hep denenir."""
    y = np.array([5.0, 10.0])
    _, _, tablo = tune_power_mean(_KUVVET_TAHMIN, y, weights=_KUVVET_AGIRLIK, p_grid=(0.5, 2.0))
    assert 1.0 in set(np.round(tablo["p"], 6))
    assert len(tablo) == 3


def test_tune_kuvvet_kapsam_maskesine_uyuyor():
    """Kapsanmayan (dolgu) satirlar taramaya girmemeli -- purged CV geregi."""
    tahminler = {
        "a": np.array([1.0, 4.0, 1.0, 4.0]),
        "b": np.array([9.0, 16.0, 9.0, 16.0]),
    }
    y = power_mean_blend(tahminler, _KUVVET_AGIRLIK, p=2.0)
    y_bozuk = y.copy()
    y_bozuk[:2] = 999.0  # dolgu bolgesi
    maske = np.array([False, False, True, True])
    en_iyi, skor, _ = tune_power_mean(tahminler, y_bozuk, weights=_KUVVET_AGIRLIK, covered=maske)
    assert en_iyi == pytest.approx(2.0)
    assert skor == pytest.approx(0.0, abs=1e-12)


def test_tune_kuvvet_giris_hatalari():
    with pytest.raises(ValueError, match="uzunluklari"):
        tune_power_mean(_KUVVET_TAHMIN, np.ones(5), weights=_KUVVET_AGIRLIK)
    with pytest.raises(ValueError, match="uzunluklari"):
        tune_power_mean(
            _KUVVET_TAHMIN,
            np.ones(2),
            weights=_KUVVET_AGIRLIK,
            covered=np.ones(5, dtype=bool),
        )
    with pytest.raises(ValueError, match="kalmadi"):
        tune_power_mean(
            _KUVVET_TAHMIN,
            np.ones(2),
            weights=_KUVVET_AGIRLIK,
            covered=np.zeros(2, dtype=bool),
        )


# --------------------------------------------------------------------------
# Exposure-offset: init_score / base_margin (sigorta pratigi)
# --------------------------------------------------------------------------


def _cv_verisi():
    rng = np.random.default_rng(0)
    n = 600
    tarih = pd.Series(np.tile(pd.date_range("2025-01-01", periods=200), 3))
    x = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    y = (x.a * 3 + rng.normal(0, 0.5, n)).to_numpy()
    folds = purged_time_series_split(
        tarih,
        embargo=pd.Timedelta(days=5),
        n_splits=2,
        test_span=pd.Timedelta(days=30),
        verbose=False,
    )
    return x, y, folds


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["lightgbm", "xgboost"])
def test_offset_kimlik_linkte_tahmine_geri_ekleniyor(kind):
    """PLUMBING KANITI: sabit 1000 offset tahminde geri eklenmeli.

    Model offset'li uzayda kalintiyi ogrenir; offset geri eklenmezse tum
    tahminler ~1000 kayar ve MAE ~1000 olur. Dogru baglanmissa MAE kucuktur.
    """
    x, y, folds = _cv_verisi()
    params = {"n_estimators": 80} if kind == "xgboost" else {"n_estimators": 80, "verbose": -1}
    sonuc = cross_validate(
        x,
        y,
        folds,
        kind=kind,
        metric="mae",
        params=params,
        init_score=np.full(len(x), 1000.0),
        verbose=False,
    )
    assert sonuc.overall_score < 50.0, (
        f"MAE {sonuc.overall_score:.1f}: offset tahmine geri eklenmemis olabilir"
    )


@pytest.mark.slow
def test_offset_poisson_maruziyet_kazanci():
    """log(maruziyet) offset'i, maruziyeti feature'suz modele VERILI yapar.

    y = 2 * maruziyet ve feature'lar saf gurultu: offset'siz model maruziyeti
    bilemez; offset'li model sabit orani ogrenip cok daha iyi tahmin eder.
    """
    x, _, folds = _cv_verisi()
    rng = np.random.default_rng(1)
    maruziyet = rng.choice([1.0, 5.0, 25.0], size=len(x))
    y = maruziyet * 2.0
    params = {"n_estimators": 80, "verbose": -1, "objective": "poisson"}

    offsetli = cross_validate(
        x,
        y,
        folds,
        kind="lightgbm",
        metric="mae",
        params=params,
        init_score=np.log(maruziyet),
        verbose=False,
    )
    offsetsiz = cross_validate(
        x, y, folds, kind="lightgbm", metric="mae", params=params, verbose=False
    )
    assert offsetli.overall_score < offsetsiz.overall_score * 0.8, (
        f"offset'li {offsetli.overall_score:.2f} >= offset'siz {offsetsiz.overall_score:.2f} * 0.8"
    )
    # Poisson tahmini exp ile geri gelmeli: negatif tahmin imkansiz.
    kapsanan = offsetli.oof_predictions[offsetli.oof_covered]
    assert (kapsanan >= 0).all()


def test_offset_catboost_acikca_reddediliyor():
    x, y, folds = _cv_verisi()
    with pytest.raises(NotImplementedError, match="CatBoost"):
        cross_validate(
            x,
            y,
            folds,
            kind="catboost",
            metric="mae",
            init_score=np.zeros(len(x)),
            verbose=False,
        )


def test_offset_siniflandirma_objektifi_reddediliyor():
    """Bilinmeyen/siniflandirma linkinde offset sessizce yanlis eklenmemeli."""
    x, y, folds = _cv_verisi()
    with pytest.raises(NotImplementedError, match="regresyon|link"):
        cross_validate(
            x,
            (y > 0).astype(int),
            folds,
            kind="lightgbm",
            metric="mae",
            params={"n_estimators": 20, "verbose": -1, "objective": "binary"},
            init_score=np.zeros(len(x)),
            verbose=False,
        )


def test_offset_giris_hatalari():
    x, y, folds = _cv_verisi()
    with pytest.raises(ValueError, match="init_score"):
        cross_validate(
            x,
            y,
            folds,
            kind="lightgbm",
            metric="mae",
            init_score=np.zeros(3),
            verbose=False,
        )
    bozuk = np.zeros(len(x))
    bozuk[0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        cross_validate(
            x,
            y,
            folds,
            kind="lightgbm",
            metric="mae",
            init_score=bozuk,
            verbose=False,
        )


def test_offset_test_tutarliligi_zorunlu():
    """Offset ya iki tarafa birden uygulanir ya hic -- tek tarafli olcek kaydirir."""
    x, y, folds = _cv_verisi()
    test = x.head(10).copy()
    with pytest.raises(ValueError, match="test_init_score yok"):
        cross_validate(
            x,
            y,
            folds,
            kind="lightgbm",
            metric="mae",
            test=test,
            init_score=np.zeros(len(x)),
            verbose=False,
        )
    with pytest.raises(ValueError, match="init_score yok"):
        cross_validate(
            x,
            y,
            folds,
            kind="lightgbm",
            metric="mae",
            test=test,
            test_init_score=np.zeros(10),
            verbose=False,
        )
    with pytest.raises(ValueError, match="test frame'i yok"):
        cross_validate(
            x,
            y,
            folds,
            kind="lightgbm",
            metric="mae",
            init_score=np.zeros(len(x)),
            test_init_score=np.zeros(10),
            verbose=False,
        )


# --------------------------------------------------------------------------
# Hawkes-esinli ustel-bozunumlu recency
# --------------------------------------------------------------------------

_UFUK = 3


def _bozunum_paneli() -> pd.DataFrame:
    """Tek grup, 8 gun, olaylar 0/3/7. Yari omur 1 gun -> alpha = 0.5."""
    return pd.DataFrame(
        {
            "gun": pd.date_range("2024-03-01", periods=8, freq="D"),
            "ilce": "bornova",
            "kesinti": [1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0],
        }
    )


def test_bozunum_el_hesabi_ve_ufuk_kaydirmasi():
    """SIZINTI TESTI: d gunundeki bozunum d - horizon'a kadarki gozlemlerden.

    Ham seri uzerinde D[j] = x[j] + 0.5*D[j-1]:
      deger: [1, .5, .25, 2.125, 1.0625, ...] -> satir i degeri D[i-3]
      olay : [1, .5, .25, 1.125, 0.5625, ...]
    Ilk 3 satirda gorulebilir gecmis yok -> NaN.
    """
    sonuc = add_event_decay_features(
        _bozunum_paneli(),
        "kesinti",
        time_column="gun",
        horizon=_UFUK,
        group_columns=["ilce"],
        half_lives=(1.0,),
    )
    deger = sonuc[f"kesinti_ufuk{_UFUK}_bozunum1g_deger"].to_numpy()
    olay = sonuc[f"kesinti_ufuk{_UFUK}_bozunum1g_olay"].to_numpy()

    assert np.isnan(deger[:_UFUK]).all(), "gecmis yokken NaN olmali"
    np.testing.assert_allclose(deger[_UFUK:], [1.0, 0.5, 0.25, 2.125, 1.0625], rtol=1e-6)
    np.testing.assert_allclose(olay[_UFUK:], [1.0, 0.5, 0.25, 1.125, 0.5625], rtol=1e-6)


def test_bozunum_son_ufuk_gunleri_feature_i_degistirmiyor():
    """YAPISAL SIZINTI KANITI: son `horizon` gunun degerleri hicbir satirin
    feature'ini etkileyemez -- etkileseydi gelecek sizmis olurdu."""
    panel = _bozunum_paneli()
    bozuk = panel.copy()
    bozuk.loc[bozuk.index[-_UFUK:], "kesinti"] = 999.0  # "gelecek" kurcalandi

    temiz = add_event_decay_features(
        panel,
        "kesinti",
        time_column="gun",
        horizon=_UFUK,
        group_columns=["ilce"],
        half_lives=(1.0, 14.0),
    )
    kurcalanmis = add_event_decay_features(
        bozuk,
        "kesinti",
        time_column="gun",
        horizon=_UFUK,
        group_columns=["ilce"],
        half_lives=(1.0, 14.0),
    )
    for kolon in temiz.filter(like="bozunum").columns:
        np.testing.assert_array_equal(
            temiz[kolon].to_numpy(), kurcalanmis[kolon].to_numpy(), err_msg=kolon
        )


def test_bozunum_gruplar_bagimsiz_ve_sira_korunuyor():
    """Iki grup + karisik satir sirasi: bozunum grup ICINDEN, satir sirasinda."""
    gunler = pd.date_range("2024-03-01", periods=5, freq="D")
    panel = (
        pd.concat(
            [
                pd.DataFrame({"gun": gunler, "ilce": "b", "y": [8.0, 0.0, 0.0, 0.0, 0.0]}),
                pd.DataFrame({"gun": gunler, "ilce": "a", "y": [0.0, 4.0, 0.0, 0.0, 0.0]}),
            ],
            ignore_index=True,
        )
        .sample(frac=1, random_state=5)
        .reset_index(drop=True)
    )

    sonuc = add_event_decay_features(
        panel, "y", time_column="gun", horizon=1, group_columns=["ilce"], half_lives=(1.0,)
    )
    assert list(sonuc["gun"]) == list(panel["gun"])  # sira korundu
    beklenen = {
        ("b", 1): 8.0,
        ("b", 2): 4.0,
        ("b", 3): 2.0,
        ("b", 4): 1.0,
        ("a", 1): 0.0,
        ("a", 2): 4.0,
        ("a", 3): 2.0,
        ("a", 4): 1.0,
    }
    for _, satir in sonuc.iterrows():
        gun_no = (satir["gun"] - gunler[0]).days
        deger = satir["y_bozunum1g_deger"]
        if gun_no == 0:
            assert np.isnan(deger)
        else:
            assert deger == pytest.approx(beklenen[(satir["ilce"], gun_no)])


def test_bozunum_ufuk1_kolon_adi_etiketsiz():
    sonuc = add_event_decay_features(
        _bozunum_paneli(),
        "kesinti",
        time_column="gun",
        horizon=1,
        group_columns=["ilce"],
        half_lives=(3.0, 14.0),
    )
    for ad in (
        "kesinti_bozunum3g_olay",
        "kesinti_bozunum3g_deger",
        "kesinti_bozunum14g_olay",
        "kesinti_bozunum14g_deger",
    ):
        assert ad in sonuc.columns


def test_bozunum_parametre_hatalari():
    panel = _bozunum_paneli()
    with pytest.raises(ValueError, match="horizon"):
        add_event_decay_features(
            panel, "kesinti", time_column="gun", horizon=0, group_columns=["ilce"]
        )
    with pytest.raises(ValueError, match="yari omur"):
        add_event_decay_features(
            panel,
            "kesinti",
            time_column="gun",
            horizon=1,
            group_columns=["ilce"],
            half_lives=(),
        )
    with pytest.raises(ValueError, match="pozitif"):
        add_event_decay_features(
            panel,
            "kesinti",
            time_column="gun",
            horizon=1,
            group_columns=["ilce"],
            half_lives=(3.0, -1.0),
        )
    with pytest.raises(KeyError, match="yok"):
        add_event_decay_features(
            panel, "olmayan", time_column="gun", horizon=1, group_columns=["ilce"]
        )


# --------------------------------------------------------------------------
# Son olaydan gecen gun (Sivas tezi #3)
# --------------------------------------------------------------------------


def test_son_olay_el_hesabi_ve_ufuk_kaydirmasi():
    """SIZINTI TESTI: d gunundeki satir yalnizca <= d - horizon olaylarini gorur.

    Olaylar 0. ve 3. gunde (deger 1 ve 2). Satir i, i-3'e kadar gorur:
      i=3: son olay gun 0 -> 3    i=6: son olay gun 3 -> 3
      i=7: son olay gun 3 -> 4 (7. gunun KENDI olayi gorunmez -- sizinti olurdu)
    """
    sonuc = add_days_since_event_features(
        _bozunum_paneli(),
        "kesinti",
        time_column="gun",
        horizon=_UFUK,
        group_columns=["ilce"],
    )
    gecen = sonuc[f"kesinti_ufuk{_UFUK}_son_olaydan_gun"].to_numpy()
    bayrak = sonuc[f"kesinti_ufuk{_UFUK}_hic_olay_yok"].to_numpy()

    assert np.isnan(gecen[:_UFUK]).all()
    np.testing.assert_array_equal(bayrak[:_UFUK], [1, 1, 1])
    np.testing.assert_allclose(gecen[_UFUK:], [3.0, 4.0, 5.0, 3.0, 4.0])
    np.testing.assert_array_equal(bayrak[_UFUK:], [0, 0, 0, 0, 0])


def test_son_olay_son_ufuk_gunleri_feature_i_degistirmiyor():
    """YAPISAL SIZINTI KANITI: son `horizon` gun kurcalanirsa cikti degismez."""
    panel = _bozunum_paneli()
    bozuk = panel.copy()
    bozuk.loc[bozuk.index[-_UFUK:], "kesinti"] = 999.0

    temiz = add_days_since_event_features(
        panel, "kesinti", time_column="gun", horizon=_UFUK, group_columns=["ilce"]
    )
    kurcalanmis = add_days_since_event_features(
        bozuk, "kesinti", time_column="gun", horizon=_UFUK, group_columns=["ilce"]
    )
    for kolon in ("kesinti_ufuk3_son_olaydan_gun", "kesinti_ufuk3_hic_olay_yok"):
        np.testing.assert_array_equal(
            temiz[kolon].to_numpy(), kurcalanmis[kolon].to_numpy(), err_msg=kolon
        )


def test_son_olay_hic_olaysiz_grup():
    """Hic olay gormemis grup: deger NaN, bayrak 1 -- '0 gun once' ile karismaz."""
    panel = pd.DataFrame(
        {
            "gun": pd.date_range("2024-03-01", periods=6, freq="D"),
            "ilce": "sessiz",
            "kesinti": 0.0,
        }
    )
    sonuc = add_days_since_event_features(
        panel, "kesinti", time_column="gun", horizon=1, group_columns=["ilce"]
    )
    assert sonuc["kesinti_son_olaydan_gun"].isna().all()
    assert (sonuc["kesinti_hic_olay_yok"] == 1).all()


def test_son_olay_gruplar_bagimsiz():
    gunler = pd.date_range("2024-03-01", periods=4, freq="D")
    panel = pd.concat(
        [
            pd.DataFrame({"gun": gunler, "ilce": "b", "y": [5.0, 0.0, 0.0, 0.0]}),
            pd.DataFrame({"gun": gunler, "ilce": "a", "y": [0.0, 0.0, 7.0, 0.0]}),
        ],
        ignore_index=True,
    )
    sonuc = add_days_since_event_features(
        panel, "y", time_column="gun", horizon=1, group_columns=["ilce"]
    )
    b = sonuc[sonuc["ilce"] == "b"]["y_son_olaydan_gun"].to_numpy()
    a = sonuc[sonuc["ilce"] == "a"]["y_son_olaydan_gun"].to_numpy()
    np.testing.assert_allclose(b[1:], [1.0, 2.0, 3.0])
    assert np.isnan(b[0])
    assert np.isnan(a[:3]).all()  # a'nin olayi 2. gunde, ufuk 1 -> 3. gunde gorunur
    assert a[3] == pytest.approx(1.0)


def test_son_olay_parametre_hatalari():
    panel = _bozunum_paneli()
    with pytest.raises(ValueError, match="horizon"):
        add_days_since_event_features(
            panel, "kesinti", time_column="gun", horizon=0, group_columns=["ilce"]
        )
    with pytest.raises(KeyError, match="yok"):
        add_days_since_event_features(
            panel, "olmayan", time_column="gun", horizon=1, group_columns=["ilce"]
        )


# --------------------------------------------------------------------------
# Mesafe-agirlikli komsu + genis komsu istatistikleri (KDD Cup 2018)
# --------------------------------------------------------------------------


def _komsu_paneli() -> tuple[pd.DataFrame, pd.DataFrame]:
    """3 varlik, 2 gun; 'a'nin komsulari b (10 km) ve c (40 km)."""
    gunler = pd.date_range("2024-05-01", periods=2, freq="D")
    panel = pd.concat(
        [
            pd.DataFrame({"gun": gunler, "yer": "a", "deger": [1.0, 5.0]}),
            pd.DataFrame({"gun": gunler, "yer": "b", "deger": [2.0, 6.0]}),
            pd.DataFrame({"gun": gunler, "yer": "c", "deger": [10.0, 30.0]}),
        ],
        ignore_index=True,
    )
    komsular = pd.DataFrame(
        [
            {"yer": "a", "komsu": "b", "mesafe_km": 10.0, "komsu_sirasi": 1},
            {"yer": "a", "komsu": "c", "mesafe_km": 40.0, "komsu_sirasi": 2},
            {"yer": "b", "komsu": "a", "mesafe_km": 10.0, "komsu_sirasi": 1},
            {"yer": "c", "komsu": "a", "mesafe_km": 40.0, "komsu_sirasi": 1},
        ]
    )
    return panel, komsular


def test_agirliksiz_varsayilan_eski_davranisla_ayni():
    """GERIYE UYUM: bayrak verilmemis ve False cagrilari bit-esit; mean duz."""
    panel, komsular = _komsu_paneli()
    ortak = {
        "key_column": "yer",
        "time_column": "gun",
        "value_columns": ["deger"],
        "target_column": None,
    }
    varsayilan = add_neighbour_feature_mean(panel, komsular, **ortak)
    kapali = add_neighbour_feature_mean(panel, komsular, **ortak, weight_by_distance=False)
    pd.testing.assert_frame_equal(varsayilan, kapali)
    # a'nin 1. gun duz ortalamasi: (2 + 10) / 2 = 6.
    a_gun1 = varsayilan[(varsayilan["yer"] == "a") & (varsayilan["gun"] == "2024-05-01")]
    assert a_gun1["komsu_deger_mean"].iloc[0] == pytest.approx(6.0)


def test_mesafe_agirlikli_ortalama_el_hesabi():
    """a: w_b = 1/10, w_c = 1/40 -> (0.1*2 + 0.025*10) / 0.125 = 3.6."""
    panel, komsular = _komsu_paneli()
    sonuc = add_neighbour_feature_mean(
        panel,
        komsular,
        key_column="yer",
        time_column="gun",
        value_columns=["deger"],
        target_column=None,
        weight_by_distance=True,
    )
    a = sonuc[sonuc["yer"] == "a"].sort_values("gun")["komsu_deger_mean"].to_numpy()
    np.testing.assert_allclose(a, [3.6, (0.1 * 6.0 + 0.025 * 30.0) / 0.125])
    # max agirliktan ETKILENMEZ: agirlikli max diye bir sey yok.
    a_max = sonuc[sonuc["yer"] == "a"].sort_values("gun")["komsu_deger_max"].to_numpy()
    np.testing.assert_allclose(a_max, [10.0, 30.0])


def test_mesafe_agirlikli_hedef_lag_gecmisten_geliyor():
    """Agirlikli komsu HEDEF ortalamasi da ufuk kadar geriden gelmeli."""
    panel, komsular = _komsu_paneli()
    sonuc = add_neighbour_target_lag(
        panel,
        komsular,
        key_column="yer",
        time_column="gun",
        target_column="deger",
        horizon=1,
        statistics=("mean",),
        weight_by_distance=True,
    )
    a = sonuc[sonuc["yer"] == "a"].sort_values("gun")["komsu_deger_ufuk1_mean"]
    assert np.isnan(a.iloc[0])  # 1. gun: komsularin gecmisi yok
    assert a.iloc[1] == pytest.approx(3.6)  # 2. gun: 1. gunun agirlikli ortalamasi


def test_sifir_mesafe_epsilon_korumasi():
    """mesafe 0 -> 1/0 inf olurdu; taban MIN_NEIGHBOUR_DISTANCE_KM devrede."""
    panel, komsular = _komsu_paneli()
    sifirli = komsular.copy()
    sifirli.loc[0, "mesafe_km"] = 0.0
    sonuc = add_neighbour_feature_mean(
        panel,
        sifirli,
        key_column="yer",
        time_column="gun",
        value_columns=["deger"],
        target_column=None,
        weight_by_distance=True,
    )
    degerler = sonuc["komsu_deger_mean"].to_numpy()
    assert np.isfinite(degerler[~np.isnan(degerler)]).all()
    # Sifir mesafeli komsu (b, deger 2) baskin agirlik alir: 1/eps >> 1/40.
    a_gun1 = sonuc[(sonuc["yer"] == "a") & (sonuc["gun"] == "2024-05-01")]
    assert a_gun1["komsu_deger_mean"].iloc[0] == pytest.approx(2.0, abs=1e-3)
    assert MIN_NEIGHBOUR_DISTANCE_KM > 0


def test_genis_komsu_istatistikleri_el_hesabi():
    """KDD Cup 2018 deseni: min/max/std bolgesel gradyani yakalar."""
    panel, komsular = _komsu_paneli()
    sonuc = add_neighbour_feature_mean(
        panel,
        komsular,
        key_column="yer",
        time_column="gun",
        value_columns=["deger"],
        target_column=None,
        statistics=("mean", "min", "max", "std"),
    )
    a_gun1 = sonuc[(sonuc["yer"] == "a") & (sonuc["gun"] == "2024-05-01")]
    assert a_gun1["komsu_deger_mean"].iloc[0] == pytest.approx(6.0)
    assert a_gun1["komsu_deger_min"].iloc[0] == pytest.approx(2.0)
    assert a_gun1["komsu_deger_max"].iloc[0] == pytest.approx(10.0)
    assert a_gun1["komsu_deger_std"].iloc[0] == pytest.approx(np.std([2.0, 10.0], ddof=1))


def test_mesafe_agirlik_parametre_hatalari():
    panel, komsular = _komsu_paneli()
    with pytest.raises(ValueError, match="mesafe_km"):
        add_neighbour_feature_mean(
            panel,
            komsular.drop(columns=["mesafe_km"]),
            key_column="yer",
            time_column="gun",
            value_columns=["deger"],
            target_column=None,
            weight_by_distance=True,
        )
    with pytest.raises(ValueError, match="mean"):
        add_neighbour_feature_mean(
            panel,
            komsular,
            key_column="yer",
            time_column="gun",
            value_columns=["deger"],
            target_column=None,
            statistics=("max",),
            weight_by_distance=True,
        )


# --------------------------------------------------------------------------
# Monotonik kisit yardimcisi (arXiv 2512.17945)
# --------------------------------------------------------------------------


def test_monoton_kisit_uc_kutuphane_formati():
    sonuc = monotone_constraints_for(
        ["ruzgar", "nem", "sicaklik"], increasing=("ruzgar",), decreasing=("sicaklik",)
    )
    assert sonuc["lightgbm"] == [1, 0, -1]
    assert sonuc["xgboost"] == "(1,0,-1)"
    assert sonuc["catboost"] == [1, 0, -1]


def test_monoton_kisitsiz_hepsi_sifir():
    sonuc = monotone_constraints_for(["a", "b"])
    assert sonuc["lightgbm"] == [0, 0]
    assert sonuc["xgboost"] == "(0,0)"


def test_monoton_bilinmeyen_kolon_hatasi():
    """Yazim hatasi sessizce 'kisit yok' olurdu -- hata firlatilmali."""
    with pytest.raises(ValueError, match="feature_columns icinde yok"):
        monotone_constraints_for(["a", "b"], increasing=("ruzgarr",))


def test_monoton_cakisan_yon_hatasi():
    with pytest.raises(ValueError, match="iki yonde birden"):
        monotone_constraints_for(["a"], increasing=("a",), decreasing=("a",))


@pytest.mark.slow
def test_monoton_kisit_lightgbm_gercekten_uyguluyor():
    """Format kabul KANITI: artan kisitli feature'da tahmin monoton artmali."""
    import lightgbm as lgb

    rng = np.random.default_rng(2)
    n = 400
    x = pd.DataFrame({"a": rng.uniform(0, 1, n), "b": rng.normal(0, 1, n)})
    y = 3.0 * x["a"].to_numpy() + rng.normal(0, 0.3, n)

    kisit = monotone_constraints_for(["a", "b"], increasing=("a",))
    model = lgb.LGBMRegressor(n_estimators=60, verbose=-1, monotone_constraints=kisit["lightgbm"])
    model.fit(x, y)

    izgara = pd.DataFrame({"a": np.linspace(0, 1, 50), "b": 0.0})
    tahmin = model.predict(izgara)
    assert (np.diff(tahmin) >= -1e-9).all(), "artan kisit ihlal edildi"
