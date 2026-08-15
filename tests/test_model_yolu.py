"""models.py egitim yolunun UC sessiz kusuru icin regresyon testleri.

Her test, duzeltmeyi doguran OLCUMU docstring'inde ismen tasir. Uc bulgu:

1. ``_prepare_categoricals`` train/test kategori ortusmesini hic olcmuyordu.
2. ``_extract_importance`` LightGBM'de 'split' donduruyordu ('gain' olmali).
3. ``cross_validate`` erken durdurma yanliligini hicbir yerde raporlamiyordu.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
import pytest

from gridup.models import (
    EARLY_STOPPING_BIAS_NOTE,
    MIN_CATEGORY_OVERLAP,
    CVResult,
    _extract_importance,
    _prepare_categoricals,
    cross_validate,
)

ILCELER = [f"ILCE_{i:02d}" for i in range(30)]


def _ortusme_ciktisi(train: pd.DataFrame, test: pd.DataFrame | None) -> str:
    """``_prepare_categoricals``i kosar ve bastigi metni dondurur."""
    yakala = io.StringIO()
    with contextlib.redirect_stdout(yakala):
        _prepare_categoricals(train, test, "lightgbm")
    return yakala.getvalue()


def _ilce_frame(degerler: list[str], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "ilce": degerler,
        "sicaklik": rng.normal(15, 5, len(degerler)),
    })


# --------------------------------------------------------------------------
# 1. Kategori ortusmesi
# --------------------------------------------------------------------------


def test_sifir_kategori_ortusmesi_artik_uyari_basiyor():
    """REGRESYON: sifir ortusme SESSIZCE geciyordu.

    OLCULDU (30 ilce, test tarafinda basta TEK BOSLUK -> ortak kategori 0):
      basilan UYARI satiri sayisi: 0 -> 1
      raporlanan ortusme         : %0.0
    Sessizligin bedeli ayni veride olculmustu:
      CV rmse (train ici) 3.0139 iken gercek test RMSE 28.5162 --
      hep-ortalama baseline (25.1324) bile daha iyi.
    """
    rng = np.random.default_rng(7)
    train = _ilce_frame(list(rng.choice(ILCELER, 2700)), seed=1)
    # Basta tek bosluk: gozle ayirt edilemez, kategori olarak TAMAMEN farkli.
    test = _ilce_frame([" " + s for s in rng.choice(ILCELER, 900)], seed=2)

    assert not set(train["ilce"]) & set(test["ilce"]), "kurgu bozuldu: ortusme var"

    cikti = _ortusme_ciktisi(train, test)
    uyari_satirlari = [satir for satir in cikti.splitlines() if "UYARI" in satir]

    assert len(uyari_satirlari) == 1
    assert "ilce" in uyari_satirlari[0]
    assert "%0.0" in uyari_satirlari[0]
    # Kullaniciya ne yapacagini da soylemeli -- ciplak uyari is gormez.
    assert "ortak kategori 0" in uyari_satirlari[0]
    assert "normalize" in cikti


def test_tam_ortusmede_hic_uyari_basilmiyor_yanlis_pozitif_korumasi():
    """YANLIS-POZITIF KORUMASI: masum durum sessiz kalmali.

    OLCULDU: ayni 30 ilce iki tarafta da -> ortusme %100.0, UYARI satiri 0.
    Uyari her kosuda atesleseydi kullanici onu ogrenip gormezden gelirdi.
    """
    rng = np.random.default_rng(7)
    train = _ilce_frame(list(rng.choice(ILCELER, 2700)), seed=1)
    test = _ilce_frame(list(rng.choice(ILCELER, 900)), seed=2)

    assert _ortusme_ciktisi(train, test) == ""


def test_esigin_ustundeki_kismi_ortusme_sessiz_kalir():
    """YANLIS-POZITIF KORUMASI: bir miktar YENI kategori normaldir.

    OLCULDU: test satirlarinin %80'i train'de gorulmus, %20'si yeni ->
    UYARI satiri 0 (esik MIN_CATEGORY_OVERLAP = 0.50).
    Gercek panelde test doneminde birkac yeni trafo/ilce kodu cikmasi
    beklenen bir seydir; alarm yalnizca kodlama semasi KAYDIGINDA calmali.
    """
    tanidik = ILCELER[:1] * 800
    yeni = ["YENI_KOD"] * 200
    train = _ilce_frame(ILCELER[:1] * 1000, seed=1)
    test = _ilce_frame(tanidik + yeni, seed=2)

    assert MIN_CATEGORY_OVERLAP == 0.50
    assert _ortusme_ciktisi(train, test) == ""


def test_esigin_altindaki_ortusme_uyariyor():
    """Esik gercekten uygulaniyor mu -- kenar durum.

    OLCULDU: test satirlarinin %30'u tanidik, %70'i yeni -> UYARI satiri 1
    ve mesajda "%30.0" geciyor.
    """
    train = _ilce_frame(ILCELER[:1] * 1000, seed=1)
    test = _ilce_frame(ILCELER[:1] * 300 + ["YENI_KOD"] * 700, seed=2)

    cikti = _ortusme_ciktisi(train, test)
    assert "UYARI" in cikti
    assert "%30.0" in cikti


def test_prepare_categoricals_sozlesmesi_bozulmadi():
    """selection/refit/two_stage AYNI fonksiyonu cagiriyor -- imza korunmali.

    Bu uc modul ``prepared, _, categorical = _prepare_categoricals(...)``
    yaziyor; donen deger sayisi degisirse UCU DE kirilir. Test frame'i
    ``None`` verilen yol da (selection.py:317, two_stage.py:547) sessiz
    kalmali -- karsilastirilacak bir test tarafi yoktur.
    """
    train = _ilce_frame(list(ILCELER) * 10, seed=1)

    yakala = io.StringIO()
    with contextlib.redirect_stdout(yakala):
        donen = _prepare_categoricals(train, None, "lightgbm")

    assert len(donen) == 3
    hazir, test_hazir, kategorik = donen
    assert test_hazir is None
    assert kategorik == ["ilce"]
    assert isinstance(hazir["ilce"].dtype, pd.CategoricalDtype)
    assert len(hazir) == len(train)
    assert yakala.getvalue() == ""


# --------------------------------------------------------------------------
# 2. Feature onemi: split -> gain
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sinyal_ve_gurultu():
    """3 gercek sinyal + 40 saf gurultu, N=4000 -- bulgunun olcum kurgusu."""
    rng = np.random.default_rng(11)
    n = 4000
    frame = pd.DataFrame({f"gurultu_{i}": rng.normal(0, 1, n) for i in range(40)})
    for i in range(3):
        frame[f"gercek_{i}"] = rng.normal(0, 1, n)
    hedef = (sum(0.5 * frame[f"gercek_{i}"] for i in range(3)) + rng.normal(0, 1, n)).to_numpy()
    return frame, hedef


@pytest.mark.slow
def test_lightgbm_onemi_modelin_importance_type_ini_dinlemez_gain_dondurur(sinyal_ve_gurultu):
    """REGRESYON: params disaridan gelince model 'split'e dusuyordu.

    ``LGB_DEFAULTS['importance_type'] == 'gain'`` olmasina ragmen
    ``feature_importances_`` yalnizca modelin KURULUS parametresini uygular.
    Optuna'nin best_params'i veya ``selection.null_importance_filter`` o
    anahtari tasimaz -> LightGBM varsayilani 'split' devreye girerdi.

    OLCULDU (3 gercek + 40 gurultu, 1000 agac, N=4000, importance_type='split'
    ile kurulmus LGBMRegressor):
      _extract_importance == split : True  -> False
      _extract_importance == gain  : False -> True
      gercek/gurultu onem orani    : 1.31x -> 12.89x
    """
    import lightgbm as lgb

    frame, hedef = sinyal_ve_gurultu
    kolonlar = list(frame.columns)
    model = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, verbose=-1, n_jobs=-1
    ).fit(frame, hedef)

    assert model.importance_type == "split", "kurgu bozuldu: model zaten gain veriyor"

    cikan = _extract_importance(model, kolonlar)
    split = np.asarray(model.booster_.feature_importance("split"), dtype="float64")
    gain = np.asarray(model.booster_.feature_importance("gain"), dtype="float64")

    assert not np.allclose(cikan, split)
    assert np.allclose(cikan, gain)

    gercek = [kolonlar.index(f"gercek_{i}") for i in range(3)]
    gurultu = [i for i in range(len(kolonlar)) if i not in gercek]
    oran = float(cikan[gercek].mean() / cikan[gurultu].mean())
    split_oran = float(split[gercek].mean() / split[gurultu].mean())

    assert split_oran < 2.0, f"kurgu bozuldu: split zaten ayiriyor ({split_oran:.2f}x)"
    assert oran > 8.0, f"gain gercek sinyali ayirmali, olculen {oran:.2f}x"


def test_lightgbm_disi_model_eski_yoldan_okunuyor_yanlis_pozitif_korumasi():
    """YANLIS-POZITIF KORUMASI: gain dali yalnizca LightGBM'i yakalamali.

    XGBoost/CatBoost'ta ``feature_importance`` diye bir metot yoktur; onlar
    ``feature_importances_`` / ``get_feature_importance`` uzerinden okunur ve
    o yol DEGISMEDI. Sahte modelle dogruluyoruz -- agir kutuphane gerekmez.
    """

    class SahteXgb:
        feature_importances_ = np.array([0.7, 0.2, 0.1])

    class SahteCat:
        def get_feature_importance(self):
            return np.array([1.0, 2.0, 3.0])

    kolonlar = ["a", "b", "c"]
    assert np.allclose(_extract_importance(SahteXgb(), kolonlar), [0.7, 0.2, 0.1])
    assert np.allclose(_extract_importance(SahteCat(), kolonlar), [1.0, 2.0, 3.0])


def test_onem_cikarilamayan_model_hala_sesli_sifir_donduruyor(capsys):
    """YANLIS-POZITIF KORUMASI: 'sessiz kalma' sozu bozulmadi.

    Gain dali eklendikten sonra da taninmayan model SIFIR vektoru + UYARI
    uretmeli; sessiz sifir tablosu "hicbir feature ise yaramiyor" diye
    yanlis okunur.
    """

    class Tanimsiz:
        pass

    sonuc = _extract_importance(Tanimsiz(), ["a", "b"])

    assert np.array_equal(sonuc, np.zeros(2))
    assert "UYARI" in capsys.readouterr().out


def test_yanlis_boyutlu_gain_vektoru_gain_dalini_atlatir(capsys):
    """Boyut uyusmazliginda gain dali sessizce YANLIS tablo dondurmemeli.

    Feature sayisi 2 iken 5 elemanli bir onem vektoru gelirse hizalama
    kaybolur ve tablo tamamen anlamsizlasir. Bu durumda eski yola dusup
    (o da bulamayinca) UYARI basmali.
    """

    class BozukBooster:
        def feature_importance(self, importance_type="split"):
            return np.arange(5.0)

    sonuc = _extract_importance(BozukBooster(), ["a", "b"])

    assert np.array_equal(sonuc, np.zeros(2))
    assert "UYARI" in capsys.readouterr().out


@pytest.mark.slow
def test_null_importance_filter_gain_ile_de_gercek_sinyali_tutuyor():
    """UYUMLULUK: selection.py ayni ``_extract_importance``i kullaniyor.

    Olcu split'ten gain'e gecti; null-importance karsilastirmasi hem gercek
    hem karistirilmis kosuda AYNI olcuyu kullandigi icin karar kurali
    gecerli kalir. OLCULDU: gercek sinyalin 3/3'u tutuluyor.
    """
    from gridup.selection import null_importance_filter

    rng = np.random.default_rng(5)
    n = 1200
    frame = pd.DataFrame({
        "gercek_a": rng.normal(size=n),
        "gercek_b": rng.normal(size=n),
    })
    for i in range(6):
        frame[f"gurultu_{i}"] = rng.normal(size=n)
    hedef = (4 * frame["gercek_a"] - 3 * frame["gercek_b"] + rng.normal(0, 0.3, n)).to_numpy()

    sonuc = null_importance_filter(
        frame, hedef, params={"n_estimators": 120, "verbose": -1}, n_runs=2, verbose=False
    )

    assert "gercek_a" in sonuc["keep"]
    assert "gercek_b" in sonuc["keep"]
    assert len(sonuc["keep"]) + len(sonuc["drop"]) == frame.shape[1]


# --------------------------------------------------------------------------
# 3. Erken durdurma yanliligi raporlaniyor
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_cross_validate_erken_durdurma_yanliligini_raporluyor():
    """REGRESYON: agac sayisi SKORLANAN fold'da seciliyor, kimse soylemiyordu.

    Mimari BILEREK degistirilmedi -- olculen buyukluk kucuk:
    OLCULDU (N=3000, 12 feature, KFold(4), lr=0.05, n_estimators=3000, esr=200;
    referans = AYNI fold'larda sabit agac sayisi, eval_set YOK):
      erken durdurmali CV : 2.118068   (fold agaclari 86/88/70/94)
      sabit  87 agac      : 2.121460   -> fark 0.003392 (%0.16)
      sabit 200 agac      : 2.153995   -> fark 0.035927 (%1.67)
    CVResult.warnings alani: YOK -> 1 kayit.
    """
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(3)
    n = 800
    X = pd.DataFrame({f"x_{i}": rng.normal(0, 1, n) for i in range(6)})
    y = (2.0 * X["x_0"] - 1.5 * X["x_1"] + rng.normal(0, 2, n)).to_numpy()
    folds = list(KFold(n_splits=3, shuffle=True, random_state=0).split(X))

    sonuc = cross_validate(
        X, y, folds, kind="lightgbm", metric="rmse",
        params={"n_estimators": 400, "learning_rate": 0.05, "verbose": -1},
        verbose=False,
    )

    assert sonuc.warnings == [EARLY_STOPPING_BIAS_NOTE]
    assert "IYIMSER" in sonuc.warnings[0]
    # Olculen sayi uyarinin ICINDE olmali; ciplak "dikkat et" is gormez.
    assert "2.118068" in sonuc.warnings[0]
    assert "%0.16" in sonuc.warnings[0]
    assert "UYARI:" in sonuc.summary()


def test_elle_kurulan_cvresult_uyarisiz_ve_sessiz_yanlis_pozitif_korumasi():
    """YANLIS-POZITIF KORUMASI: yeni alan geriye uyumlulugu bozmamali.

    ``neural.py`` ve testler CVResult'i elle kuruyor; ``warnings`` varsayilani
    BOS liste olmali ve ``summary()`` o durumda hicbir UYARI satiri basmamali
    -- aksi halde uyari anlamsizlasir.
    """
    elle = CVResult(
        oof_predictions=np.arange(5.0),
        test_predictions=None,
        fold_scores=[1.0],
        overall_score=1.0,
        feature_importance=pd.DataFrame({"feature": [], "importance": []}),
    )

    assert elle.warnings == []
    assert "UYARI:" not in elle.summary()
    # Eski alanlar da yerinde -- alan ekleme sirasi bozmadi.
    assert elle.coverage == 1.0
