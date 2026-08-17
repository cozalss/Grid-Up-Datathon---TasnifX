"""SIZINTI AVI TUR 8 -- FEATURE SECIMI + GRUP AGREGASYONU.

SINIF: "koruma var ama opt-in" ve "raporlanan sayi odenen isle ortusmuyor".

Bu turun ortak temasi, kodun DOGRU seyi yapabiliyor ama VARSAYILAN olarak
yapmiyor olmasi. Bir koruma yalnizca kullanici onu hatirlarsa calisiyorsa,
yarismanin uckuncu gununde saat ikide calismayacaktir.

Her test ``t8_repro.py`` ile ONCE-SONRA olculmus bir bulguya karsilik gelir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.features.aggregate import add_group_statistics, add_target_free_aggregates
from gridup.selection import null_importance_filter, shap_backward_selection


def _grup_verisi(n: int = 2000, tohum: int = 0):
    """Ilce etkisi TASIYAN bir hedef -- grup ortalamasi hedefi ele verir."""
    rng = np.random.default_rng(tohum)
    ilce = rng.choice([f"i{k:02d}" for k in range(20)], n)
    etki = pd.Series(ilce).map({f"i{k:02d}": k * 3.0 for k in range(20)}).to_numpy()
    hedef = etki + rng.normal(0, 5, n)
    frame = pd.DataFrame({"ilce": ilce, "nufus": rng.normal(size=n), "hedef": hedef})
    return frame, hedef


# --------------------------------------------------------------------------
# B45 -- hedef korumasi OPT-IN idi
# --------------------------------------------------------------------------


def test_add_group_statistics_target_column_olmadan_calismiyor():
    """OLCULDU: ``target_column`` varsayilani ``None`` oldugu icin koruma
    pratikte HIC calismiyordu. Hedef ``value_columns``a girince 16 kolon
    uretiliyor ve dordu hedefle **+0.96** korelasyonlu oluyordu:

        hedef_bazinda_ilce_mean     +0.9612
        hedef_bazinda_ilce_median   +0.9610
        hedef_bazinda_ilce_max      +0.9586
        hedef_bazinda_ilce_min      +0.9544

    Bu, ``oof_target_encode``in onlemek icin var oldugu sizintinin ta kendisi --
    baska bir kapidan. Artik ``target_column`` ACIKCA verilmek zorunda:
    "vermedim" ile "hedef yok" ayirt edilebilsin diye.
    """
    frame, _ = _grup_verisi()
    with pytest.raises(TypeError, match="ACIKCA verilmelidir"):
        add_group_statistics(frame, ["ilce"], ["nufus", "hedef"])


def test_hedef_value_columns_icindeyse_hala_reddediliyor():
    frame, _ = _grup_verisi()
    with pytest.raises(ValueError, match="fold-disi DEGILDIR"):
        add_group_statistics(frame, ["ilce"], ["nufus", "hedef"], target_column="hedef")


def test_hedef_yok_diye_bilincli_karar_verilirse_calisiyor():
    """``target_column=None`` bilincli bir karardir ve gecmelidir."""
    frame, _ = _grup_verisi()
    sonuc = add_group_statistics(frame, ["ilce"], ["nufus"], target_column=None)
    assert len(sonuc.columns) > len(frame.columns)
    assert sonuc["nufus"].equals(frame["nufus"]), "girdi degistirilmemeli"


def test_add_target_free_aggregates_de_ayni_korumayi_uyguluyor():
    """Ayni koruma ikiz fonksiyonda da zorunlu olmali."""
    frame, _ = _grup_verisi(400)
    train, test = frame.iloc[:300], frame.iloc[300:]
    with pytest.raises(TypeError, match="ACIKCA verilmelidir"):
        add_target_free_aggregates(train, test, ["ilce"], ["nufus"])


# --------------------------------------------------------------------------
# B3 -- catboost yolu her zaman cokuyordu
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["lightgbm", "catboost"])
def test_null_importance_filter_her_model_tipinde_calisiyor(kind: str):
    """OLCULDU: catboost yolu HER ZAMAN cokuyordu --

        CatBoostError: only one of the parameters iterations, n_estimators,
                       num_boost_round, num_trees should be initialized

    Sebep: agac sayisi her modele ``n_estimators`` diye yaziliyordu ama
    CatBoost'un baslangic parametrelerinde ``iterations`` var ve CatBoost
    ikisini ayni anda kabul etmiyor. Ne eski (setdefault) ne yeni (acik atama)
    surumde calisiyordu.
    """
    rng = np.random.default_rng(0)
    m = 800
    frame = pd.DataFrame({f"g{i}": rng.normal(size=m) for i in range(8)})
    frame["sinyal"] = rng.normal(size=m)
    y = frame["sinyal"] * 3 + rng.normal(0, 0.5, m)

    sonuc = null_importance_filter(frame, y, kind=kind, n_runs=2, verbose=False)
    assert "sinyal" in sonuc["keep"], f"{kind}: gercek sinyal elenmemeli"
    assert len(sonuc["keep"]) <= 3, f"{kind}: gurultunun cogu elenmelı"


# --------------------------------------------------------------------------
# B55 -- patience ile durunca son adim kayboluyordu
# --------------------------------------------------------------------------


def _secim_verisi(n: int = 1200, n_feature: int = 60, tohum: int = 0):
    rng = np.random.default_rng(tohum)
    frame = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(n_feature)})
    y = frame["f0"] * 2 + frame["f1"] - frame["f2"] + rng.normal(0, 0.5, n)
    folds = [
        (np.arange(0, 800), np.arange(800, 1000)),
        (np.arange(0, 1000), np.arange(1000, 1200)),
    ]
    return frame, y.to_numpy(), folds


def test_patience_ile_durunca_son_adim_history_ye_yaziliyor():
    """OLCULDU: 3 CV kosusu yapildi, history'de 2 adim vardi ve summary()
    "Toplam 2 adim" diyordu.

        ONCE : [60, 55]
        SONRA: [60, 55, 50]

    Kaybolan adim, bedeli ODENMIS bir CV kosusuydu -- ustelik tam da durma
    kararini gerekcelendiren adim. Juriye sunulan eleme egrisinin son noktasi
    eksik kaliyordu.
    """
    frame, y, folds = _secim_verisi()
    sonuc = shap_backward_selection(
        frame,
        y,
        folds,
        drop_per_step=5,
        min_features=20,
        max_steps=8,
        patience=1,
        progress=None,
    )
    assert [adim.n_features for adim in sonuc.history] == [60, 55, 50]
    assert f"Toplam {len(sonuc.history)} adim" in sonuc.summary()


def test_history_deki_her_adim_gercek_bir_cv_kosusuna_karsilik_geliyor():
    """Egri, odenen isin TAMAMINI gostermeli -- eksigi de fazlasi da olmamali."""
    frame, y, folds = _secim_verisi()
    sonuc = shap_backward_selection(
        frame,
        y,
        folds,
        drop_per_step=10,
        min_features=20,
        max_steps=8,
        patience=2,
        progress=None,
    )
    egri = sonuc.curve()
    assert len(egri) == len(sonuc.history)
    # best_score history'de GERCEKTEN bulunmali (aksi halde nereden geldigi belirsiz)
    assert any(abs(adim.score - sonuc.best_score) < 1e-12 for adim in sonuc.history)


# --------------------------------------------------------------------------
# B52 -- secim yanliligi
# --------------------------------------------------------------------------


def test_secim_yanliligi_acikca_raporlaniyor():
    """OLCULDU: 6 adimda best 0.620982, adim ortalamasi 0.627206.

    ``best_score`` AYNI fold'lar uzerinde kosulan N korele denemenin
    MINIMUMUDUR. Bunu "modelin skoru" diye raporlamak, 200 noktali izgarada
    esik secip ayni veride skor bildirmekle ayni hatadir. Sayi yanlis degil,
    SUNUMU yaniltici -- ve juriye giden slayt bu.
    """
    frame, y, folds = _secim_verisi()
    sonuc = shap_backward_selection(
        frame,
        y,
        folds,
        drop_per_step=10,
        min_features=20,
        max_steps=8,
        patience=2,
        progress=None,
    )
    assert len(sonuc.history) >= 2
    assert sonuc.selection_optimism > 0.0
    ozet = sonuc.summary()
    assert "korele denemenin EN IYISIDIR" in ozet
    assert "bagimsiz bir kumede dogrula" in ozet


def test_tek_adimda_yanlilik_uyarisi_cikmiyor():
    """Yanlis pozitif korumasi: tek deneme yapildiysa secim yanliligi yoktur."""
    frame, y, folds = _secim_verisi(n_feature=25)
    sonuc = shap_backward_selection(
        frame,
        y,
        folds,
        drop_per_step=10,
        min_features=20,
        max_steps=1,
        patience=1,
        progress=None,
    )
    assert len(sonuc.history) == 1
    assert sonuc.selection_optimism == 0.0
    assert "korele denemenin" not in sonuc.summary()
