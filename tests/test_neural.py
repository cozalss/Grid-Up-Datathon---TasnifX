"""Varlik gomulu sinir agi testleri.

En kritik testler on-isleme SIZINTISINA dairdir: olcekleyici, kategori
sozlugu ve hedef istatistikleri yalnizca fold'un EGITIM tarafindan
ogrenilmelidir. Bunlar tum veri uzerinde hesaplanirsa CV sessizce iyimser
olur -- ve bu, yarismada leaderboard'da gorulen ama lokalde gorulmeyen
hayal kirikliginin en yaygin sebebidir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.metrics import mape
from gridup.models import CVResult
from gridup.neural import UNKNOWN_INDEX, NeuralConfig, _FoldPreprocessor, neural_cross_validate
from gridup.validation import purged_time_series_split

torch = pytest.importorskip("torch")

HIZLI = NeuralConfig(hidden_sizes=(32,), max_epochs=8, patience=3, batch_size=256)


def _panel(n_gun: int = 200, n_ilce: int = 8, seed: int = 3):
    rng = np.random.default_rng(seed)
    gunler = pd.date_range("2024-01-01", periods=n_gun, freq="D")
    tarih = pd.Series(np.tile(gunler, n_ilce))
    adlar = [f"ilce_{i}" for i in range(n_ilce)]
    ilce = np.repeat(adlar, n_gun)
    n = len(tarih)
    etki = dict(zip(adlar, rng.normal(0, 10, n_ilce), strict=True))
    frame = pd.DataFrame(
        {
            "ilce": ilce,
            "haftagun": tarih.dt.dayofweek.to_numpy().astype(str),
            "sicaklik": rng.normal(18, 8, n),
            "nem": rng.normal(60, 12, n),
        }
    )
    hedef = (
        150 + np.array([etki[i] for i in ilce]) + 2.0 * frame.sicaklik + rng.normal(0, 4, n)
    ).to_numpy()
    folds = purged_time_series_split(
        tarih, embargo=pd.Timedelta(days=7), n_splits=2,
        test_span=pd.Timedelta(days=30), verbose=False,
    )
    return frame, hedef, folds


# --------------------------------------------------------------------------
# Sizinti disiplini -- bu dosyanin varlik sebebi
# --------------------------------------------------------------------------


def test_olcekleyici_yalnizca_egitim_tarafindan_ogreniliyor():
    """Dogrulama tarafina asiri buyuk deger koymak istatistikleri DEGISTIRMEMELI."""
    egitim = pd.DataFrame({"x": [1.0, 2.0, 3.0], "k": ["a", "b", "a"]})
    dogrulama = pd.DataFrame({"x": [1e9, 1e9], "k": ["a", "b"]})

    on = _FoldPreprocessor(["k"], ["x"]).fit(egitim, np.array([10.0, 20.0, 30.0]))
    beklenen_ortalama = float(on.means[0])
    on.transform(dogrulama)

    assert float(on.means[0]) == beklenen_ortalama == pytest.approx(2.0)
    assert float(on.stds[0]) == pytest.approx(np.std([1.0, 2.0, 3.0]))


def test_hedef_istatistikleri_egitimden_geliyor():
    on = _FoldPreprocessor([], ["x"]).fit(
        pd.DataFrame({"x": [1.0, 2.0]}), np.array([100.0, 200.0])
    )
    assert on.target_mean == pytest.approx(150.0)
    # Olcekle-tersine cevir kimlik olmali.
    ham = np.array([100.0, 150.0, 200.0])
    assert on.unscale_target(on.scale_target(ham)) == pytest.approx(ham, abs=1e-4)


def test_gorulmemis_kategori_unk_indeksine_dusuyor():
    """Dogrulamada YENI bir ilce cikarsa patlamamali, UNK olmali."""
    egitim = pd.DataFrame({"k": ["a", "b"]})
    dogrulama = pd.DataFrame({"k": ["a", "ZZZ_hic_gorulmedi"]})

    on = _FoldPreprocessor(["k"], []).fit(egitim, np.array([1.0, 2.0]))
    kodlar, _ = on.transform(dogrulama)

    assert kodlar[1, 0] == UNKNOWN_INDEX
    assert kodlar[0, 0] != UNKNOWN_INDEX
    # Sozluk 0'i UNK'ye ayirdigi icin kardinalite kategori sayisi + 1.
    assert on.cardinalities == [3]


def test_sabit_kolon_sifira_bolmuyor():
    on = _FoldPreprocessor([], ["sabit"]).fit(
        pd.DataFrame({"sabit": [5.0, 5.0, 5.0]}), np.array([1.0, 2.0, 3.0])
    )
    _, olcekli = on.transform(pd.DataFrame({"sabit": [5.0, 7.0]}))
    assert np.isfinite(olcekli).all()
    assert olcekli[0, 0] == pytest.approx(0.0)


def test_nan_ortalamaya_dusuyor_sonsuz_uretmiyor():
    on = _FoldPreprocessor([], ["x"]).fit(
        pd.DataFrame({"x": [1.0, 2.0, 3.0]}), np.array([1.0, 2.0, 3.0])
    )
    _, olcekli = on.transform(pd.DataFrame({"x": [np.nan, np.inf, 2.0]}))
    assert np.isfinite(olcekli).all()
    assert olcekli[0, 0] == pytest.approx(0.0)  # NaN -> olcekli 0 = ortalama


def test_sabit_hedefte_std_sifira_bolmuyor():
    on = _FoldPreprocessor([], ["x"]).fit(
        pd.DataFrame({"x": [1.0, 2.0]}), np.array([7.0, 7.0])
    )
    assert on.target_std == 1.0
    assert np.isfinite(on.scale_target(np.array([7.0, 7.0]))).all()


# --------------------------------------------------------------------------
# CVResult uyumu -- harman makinesine takilabilmeli
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_cvresult_donduruyor_ve_alanlari_dolu():
    X, y, folds = _panel()
    sonuc = neural_cross_validate(
        X, y, folds, cat_columns=["ilce", "haftagun"], metric="mape",
        config=HIZLI, verbose=False,
    )
    assert isinstance(sonuc, CVResult)
    assert sonuc.model_kind == "neural"
    assert len(sonuc.fold_scores) == len(folds)
    assert len(sonuc.oof_predictions) == len(y)
    assert np.isfinite(sonuc.overall_score)
    assert set(sonuc.feature_importance["feature"]) == set(X.columns)


@pytest.mark.slow
def test_oof_yalnizca_valid_satirlarinda_dolu():
    X, y, folds = _panel()
    sonuc = neural_cross_validate(
        X, y, folds, cat_columns=["ilce", "haftagun"], config=HIZLI, verbose=False
    )
    kapsam = np.zeros(len(y), dtype=bool)
    for _, valid in folds:
        kapsam[valid] = True
    # Kapsam disindaki satirlar hic tahmin almamis olmali (0 kalir).
    assert np.all(sonuc.oof_predictions[~kapsam] == 0.0)
    assert np.isfinite(sonuc.oof_predictions[kapsam]).all()


@pytest.mark.slow
def test_test_tahminleri_fold_ortalamasi():
    X, y, folds = _panel()
    test = X.iloc[:50].copy()
    sonuc = neural_cross_validate(
        X, y, folds, cat_columns=["ilce", "haftagun"], test=test,
        config=HIZLI, verbose=False,
    )
    assert sonuc.test_predictions is not None
    assert len(sonuc.test_predictions) == len(test)
    assert np.isfinite(sonuc.test_predictions).all()


@pytest.mark.slow
def test_gbdt_ile_ayni_foldlarda_harmanlanabiliyor():
    """Ayni fold'lar verildiginde OOF vektorleri hizali olmali ve harman calismali."""
    from gridup.models import cross_validate

    X, y, folds = _panel()
    Xg = X.copy()
    for kolon in ("ilce", "haftagun"):
        Xg[kolon] = Xg[kolon].astype("category")

    agac = cross_validate(
        Xg, y, folds, kind="lightgbm", metric="mape",
        params={"n_estimators": 80, "learning_rate": 0.1, "verbose": -1}, verbose=False,
    )
    sinir = neural_cross_validate(
        X, y, folds, cat_columns=["ilce", "haftagun"], metric="mape",
        config=HIZLI, verbose=False,
    )

    kapsam = np.zeros(len(y), dtype=bool)
    for _, valid in folds:
        kapsam[valid] = True

    harman = 0.5 * agac.oof_predictions[kapsam] + 0.5 * sinir.oof_predictions[kapsam]
    assert np.isfinite(mape(y[kapsam], harman))
    assert len(agac.oof_predictions) == len(sinir.oof_predictions)


@pytest.mark.slow
def test_kategorik_otomatik_tespit_ediliyor():
    """cat_columns verilmezse metin kolonlari kategorik sayilmali (pandas 3.0 str dtype)."""
    X, y, folds = _panel()
    sonuc = neural_cross_validate(X, y, folds, config=HIZLI, verbose=False)
    assert np.isfinite(sonuc.overall_score)


def test_bos_fold_reddediliyor():
    X, y, _ = _panel()
    with pytest.raises(ValueError, match="En az bir fold"):
        neural_cross_validate(X, y, [], config=HIZLI, verbose=False)


def test_olmayan_kategorik_kolon_reddediliyor():
    X, y, folds = _panel()
    with pytest.raises(KeyError, match="frame'de yok"):
        neural_cross_validate(X, y, folds, cat_columns=["yok_boyle_kolon"], config=HIZLI)


# --------------------------------------------------------------------------
# Bozuk girdi ACIK hata vermeli -- sessizce egitilmemis model dondurmemeli
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bozuk_deger", [np.nan, np.inf, -np.inf])
def test_hedefte_gecersiz_deger_acik_hata_veriyor(bozuk_deger):
    """NaN/inf hedef, on-isleyicinin istatistiklerini bozar; kayip NaN cikar,
    hicbir epok 'iyilesme' sayilmaz ve model EGITILMEMIS kalir.

    Onceki surumde hata ancak SKORLAMA asamasinda sklearn'in
    'Input contains NaN' mesajiyla ortaya cikiyordu -- nereden geldigi
    belirsizdi. Artik erken ve acik.
    """
    X, y, folds = _panel()
    bozuk = np.array(y, copy=True)
    bozuk[3] = bozuk_deger

    with pytest.raises(ValueError, match="NaN/sonsuz"):
        neural_cross_validate(X, bozuk, folds, config=HIZLI, verbose=False)


def test_saglikli_hedef_hala_calisiyor():
    """Koruma yanlis pozitif uretmemeli."""
    X, y, folds = _panel()
    sonuc = neural_cross_validate(X, y, folds, config=HIZLI, verbose=False)
    assert np.isfinite(sonuc.overall_score)


def test_egitim_hic_iyilesmezse_uyariyor():
    """best_state None kalirsa model egitilmemis demektir -- sessiz kalmamali."""
    import warnings as _w

    from gridup.neural import _train_one_fold

    torch_mod = pytest.importorskip("torch")
    ag = torch_mod.nn.Sequential(torch_mod.nn.Linear(2, 1))
    ag.mlp = ag  # _train_one_fold state_dict bekliyor

    class _SabitKayip(torch_mod.nn.Module):
        """Kayip her zaman NaN -> hicbir epok iyilesmez."""

        def forward(self, *_):
            return torch_mod.tensor(float("nan"), requires_grad=True)

    kodlar = torch_mod.zeros((4, 0), dtype=torch_mod.int64)
    sayisal = torch_mod.zeros((4, 2))
    hedef = torch_mod.zeros(4)

    class _Ag(torch_mod.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch_mod.nn.Sequential(torch_mod.nn.Linear(2, 1))
            self.embeddings = torch_mod.nn.ModuleList()

        def forward(self, _codes, numeric):
            return self.mlp(numeric).squeeze(-1)

    ag = _Ag()
    yapilandirma = NeuralConfig(max_epochs=3, patience=2, batch_size=2)

    with _w.catch_warnings(record=True) as yakalanan:
        _w.simplefilter("always")
        # Kayip NaN oldugu icin hicbir epok iyilesme saymayacak.
        orijinal = torch_mod.nn.MSELoss
        torch_mod.nn.MSELoss = _SabitKayip
        try:
            _train_one_fold(
                ag, (kodlar, sayisal, hedef), (kodlar, sayisal, hedef),
                yapilandirma, torch_mod,
            )
        finally:
            torch_mod.nn.MSELoss = orijinal

    assert any("iyilesmedi" in str(u.message) for u in yakalanan), (
        "egitim hic iyilesmedigi halde uyari verilmedi"
    )
