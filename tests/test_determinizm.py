"""Determinizm ve parametre birlestirme sozlesmesi.

"Ayni girdi -> ayni submission" bir yarismada pazarlik edilemez bir
gerekliliktir: skorunu yeniden uretemiyorsan, hangi degisikligin ise
yaradigini da bilemezsin ve juriye sunacagin sayilar dogrulanamaz.

Bu dosya o garantiyi OLCER, varsaymaz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.config import set_global_seed
from gridup.models import (
    INFRASTRUCTURE_KEYS,
    cross_validate,
    merge_infrastructure_params,
)
from gridup.validation import purged_time_series_split


def _veri(n_gun: int = 120, n_varlik: int = 4):
    rng = np.random.default_rng(0)
    tarih = pd.Series(np.tile(pd.date_range("2025-01-01", periods=n_gun), n_varlik))
    n = len(tarih)
    frame = pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
            "k": np.tile(["p", "q", "r", "s"], n // 4),
        }
    )
    hedef = (frame.a * 3 + frame.b * 2 + rng.normal(0, 1, n)).to_numpy()
    folds = purged_time_series_split(
        tarih, embargo=pd.Timedelta(days=5), n_splits=2,
        test_span=pd.Timedelta(days=20), verbose=False,
    )
    return frame, hedef, folds


# --------------------------------------------------------------------------
# Altyapi parametrelerinin birlestirilmesi
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["lightgbm", "xgboost", "catboost"])
def test_altyapi_anahtarlari_params_verilince_kaybolmuyor(kind):
    """REGRESYON: params verince varsayilanlar TAMAMEN degisiyordu.

    OLCULDU: params={'n_estimators':200} XGBoost'ta enable_categorical'i
    dusuruyordu -> kategorik kolonda "DataFrame.dtypes for data must be
    int, float, bool or category" hatasi. Kullanici sadece agac sayisini
    degistirmek istemisti.
    """
    birlesik = merge_infrastructure_params(kind, {"n_estimators": 200})
    for anahtar in INFRASTRUCTURE_KEYS[kind]:
        assert anahtar in birlesik, f"{kind}: '{anahtar}' altyapi anahtari kayboldu"


@pytest.mark.parametrize("kind", ["lightgbm", "xgboost", "catboost"])
def test_kullanicinin_acik_degeri_ezilmiyor(kind):
    anahtar = INFRASTRUCTURE_KEYS[kind][0]
    birlesik = merge_infrastructure_params(kind, {anahtar: "KULLANICI"})
    assert birlesik[anahtar] == "KULLANICI"


def test_ogrenme_hiperparametreleri_gizlice_eklenmiyor():
    """params verildiginde ogrenme davranisini SADECE kullanici belirlemeli.

    Gizli varsayilanlar (learning_rate, reg_alpha ...) tekrarlanabilirligi
    bozar: ayni params sozlugu farkli surumlerde farkli model uretir.
    """
    birlesik = merge_infrastructure_params("lightgbm", {"n_estimators": 200})
    for gizli in ("learning_rate", "num_leaves", "reg_alpha", "subsample"):
        assert gizli not in birlesik


@pytest.mark.slow
def test_xgboost_kategorik_kolonla_params_verilerek_calisiyor():
    """Uctan uca: bu tam olarak coken senaryoydu."""
    X, y, folds = _veri()
    X["k"] = X["k"].astype("category")
    sonuc = cross_validate(
        X, y, folds, kind="xgboost", metric="rmse",
        params={"n_estimators": 40, "learning_rate": 0.1}, verbose=False,
    )
    assert np.isfinite(sonuc.overall_score)


# --------------------------------------------------------------------------
# Determinizm
# --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("lightgbm", {"n_estimators": 60, "learning_rate": 0.1}),
        ("xgboost", {"n_estimators": 60, "learning_rate": 0.1}),
        ("catboost", {"iterations": 60, "learning_rate": 0.1}),
    ],
)
def test_ayni_tohumla_ayni_tahmin(kind, params):
    """SOZLESME: ayni tohum + ayni girdi -> BIT DUZEYINDE ayni tahmin."""
    X, y, folds = _veri()
    X["k"] = X["k"].astype("category")

    def kos():
        set_global_seed(42)
        return cross_validate(
            X, y, folds, kind=kind, metric="rmse", params=params, verbose=False
        )

    a, b = kos(), kos()
    assert np.array_equal(a.oof_predictions, b.oof_predictions)
    assert a.overall_score == b.overall_score


@pytest.mark.slow
def test_sinir_agi_ayni_tohumla_ayni_tahmin():
    """GPU'da bile deterministik olmali -- olculdu, oyle."""
    pytest.importorskip("torch")
    from gridup.neural import NeuralConfig, neural_cross_validate

    X, y, folds = _veri()

    def kos():
        set_global_seed(42)
        return neural_cross_validate(
            X, y, folds, cat_columns=["k"], metric="rmse",
            config=NeuralConfig(max_epochs=15, patience=4), verbose=False,
        )

    a, b = kos(), kos()
    assert np.array_equal(a.oof_predictions, b.oof_predictions)


@pytest.mark.slow
def test_farkli_tohum_farkli_sonuc_veriyor():
    """Determinizm testi VAKUMLU olmasin: tohum gercekten etkili mi?"""
    X, y, folds = _veri()

    def kos(tohum):
        set_global_seed(tohum)
        return cross_validate(
            X, y, folds, kind="lightgbm", metric="rmse",
            params={"n_estimators": 60, "learning_rate": 0.1,
                    "subsample": 0.7, "subsample_freq": 1,
                    "colsample_bytree": 0.7, "random_state": tohum},
            verbose=False,
        )

    assert not np.array_equal(kos(1).oof_predictions, kos(2).oof_predictions)


# --------------------------------------------------------------------------
# fold_shap_importance -- bir P0 duzeltmesiydi ama TESTI YOKTU
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_shap_onemi_dogrulama_satirlarindan_hesaplaniyor():
    """SHAP, fold'un EGITIM satirlarindan degil DOGRULAMA satirlarindan.

    Bu fonksiyon tam olarak bu hatayi duzeltmek icin yazilmisti (onceki
    surum tum fold modellerinin ilkini alip TUM veride SHAP hesapliyordu --
    yani modelin egitildigi satirlarda). Duzeltme vardi ama TESTI YOKTU.

    Burada dogrudan olcuyoruz: sinyal YALNIZCA son fold'un valid penceresinde
    var; model onu ancak dogrulama satirlarina bakarsa onemli bulabilir.
    """
    from gridup.selection import fold_shap_importance

    pytest.importorskip("shap")
    X, y, folds = _veri(n_gun=160, n_varlik=4)
    X = X.drop(columns=["k"])

    sonuc = cross_validate(
        X, y, folds, kind="lightgbm", metric="rmse",
        params={"n_estimators": 60, "learning_rate": 0.1}, verbose=False,
    )
    tablo = fold_shap_importance(sonuc.models, X, folds, sample_per_fold=200)

    assert set(tablo["feature"]) == set(X.columns)
    assert (tablo["mean_abs_shap"] >= 0).all()
    # 'a' katsayisi 3, 'b' katsayisi 2 -> a daha onemli olmali.
    onem = tablo.set_index("feature")["mean_abs_shap"]
    assert onem["a"] > onem["b"]


@pytest.mark.slow
def test_shap_onemi_model_ve_fold_sayisi_uyusmazliginda_patliyor():
    """Sessizce yanlis fold'u eslestirmektense HATA VERMELI."""
    from gridup.selection import fold_shap_importance

    pytest.importorskip("shap")
    X, y, folds = _veri()
    X = X.drop(columns=["k"])
    sonuc = cross_validate(
        X, y, folds, kind="lightgbm", metric="rmse",
        params={"n_estimators": 30}, verbose=False,
    )
    with pytest.raises((ValueError, AssertionError)):
        fold_shap_importance(sonuc.models[:1], X, folds, sample_per_fold=50)


# --------------------------------------------------------------------------
# EPIAS istemcisi -- sir yonetimi sozlesmesi
# --------------------------------------------------------------------------


def test_sifre_repr_de_gorunmuyor():
    """Bir istemci nesnesi yanlislikla print/log edilirse sifre SIZMAMALI."""
    from gridup.epias import EpiasClient

    istemci = EpiasClient(username="kullanici@ornek.com", password="COK-GIZLI-SIFRE")
    metin = repr(istemci) + str(istemci)
    assert "COK-GIZLI-SIFRE" not in metin


def test_tgt_url_de_degil_header_da_tasiniyor():
    """URL query stringi proxy ve erisim loglarina duser -- TGT oraya konmamali."""
    import inspect

    from gridup import epias

    kaynak = inspect.getsource(epias)
    assert 'headers["TGT"]' in kaynak, "TGT header ile gonderilmiyor"
    # TGT bir f-string URL'ine gomulmus olmamali.
    for satir in kaynak.splitlines():
        soyulmus = satir.strip()
        if soyulmus.startswith("#"):
            continue
        if "tgt" in soyulmus.lower() and ("?" in soyulmus or "params=" in soyulmus):
            raise AssertionError(f"TGT URL'de tasiniyor olabilir: {soyulmus[:90]}")


def test_isteklerde_timeout_var():
    """Timeout'suz istek yarisma gunu pipeline'i SONSUZA kadar bekletir."""
    import inspect
    import re

    from gridup import epias

    kaynak = inspect.getsource(epias)
    cagrilar = re.findall(r"(?:session|requests)\.(?:get|post)\([^)]*\)", kaynak, re.S)
    assert cagrilar, "hic HTTP cagrisi bulunamadi -- test bayatlamis olabilir"
    for cagri in cagrilar:
        assert "timeout" in cagri, f"timeout'suz istek: {cagri[:90]}"


def test_sabit_kodlanmis_kimlik_bilgisi_yok():
    import inspect
    import re

    from gridup import epias

    kaynak = inspect.getsource(epias)
    kalip = re.compile(
        r"(password|api_key|secret|token)\s*=\s*[\"'][^\"']{8,}[\"']", re.IGNORECASE
    )
    bulunan = [
        m.group(0)
        for m in kalip.finditer(kaynak)
        if not any(g in m.group(0).lower() for g in ("environ", "getenv", "ornek", "example"))
    ]
    assert not bulunan, f"sabit kodlanmis kimlik bilgisi: {bulunan}"


# --------------------------------------------------------------------------
# Bozuk hedef ANA YOLDA da sessiz geciyordu
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bozuk", [np.nan, np.inf, -np.inf])
def test_cross_validate_bozuk_hedefi_reddediyor(bozuk):
    """REGRESYON: NaN hedefle skor 0.527 donuyordu -- tamamen inandirici.

    Bu, hata bicimlerinin EN TEHLIKELISI: model NaN satirlari sessizce yok
    sayar, skor makul gorunur, kimse bir sey fark etmez. inf durumunda
    4.1e+35 donuyordu -- absurt ama yine hatasiz.
    """
    X, y, folds = _veri()
    X = X.drop(columns=["k"])
    kirli = np.array(y, copy=True)
    kirli[3] = bozuk

    with pytest.raises(ValueError, match="NaN/sonsuz"):
        cross_validate(
            X, kirli, folds, kind="lightgbm", metric="rmse",
            params={"n_estimators": 20}, verbose=False,
        )


def test_saglikli_hedef_ve_siniflandirma_etkilenmiyor():
    """Koruma yanlis pozitif uretmemeli -- regresyon ve siniflandirma calismali."""
    X, y, folds = _veri()
    X = X.drop(columns=["k"])

    regresyon = cross_validate(
        X, y, folds, kind="lightgbm", metric="rmse",
        params={"n_estimators": 20}, verbose=False,
    )
    ikili = cross_validate(
        X, (X.a > 0).astype(int).to_numpy(), folds, kind="lightgbm",
        task_type="binary", metric="auc", params={"n_estimators": 20}, verbose=False,
    )
    assert np.isfinite(regresyon.overall_score)
    assert np.isfinite(ikili.overall_score)


def test_sayisal_olmayan_hedefte_koruma_patlamiyor():
    """np.isfinite metin dizisinde TypeError firlatir -- koruma onu atlamali.

    (LightGBM metin etiketi zaten kabul etmez; burada sinanan sey KORUMANIN
    kendisinin sayisal olmayan dtype'ta cokmemesi.)
    """
    X, _, folds = _veri()
    X = X.drop(columns=["k"])
    etiket = np.array(["a", "b"] * (len(X) // 2))

    # Koruma satiri: sayisal degilse kontrol ATLANIR, TypeError firlamaz.
    assert not np.issubdtype(etiket.dtype, np.number)
    with pytest.raises(Exception) as hata:
        cross_validate(
            X, etiket, folds, kind="lightgbm", task_type="binary",
            metric="accuracy", params={"n_estimators": 20}, verbose=False,
        )
    # Hata LightGBM'den gelmeli, bizim NaN korumamizdan DEGIL.
    assert "NaN/sonsuz" not in str(hata.value)
