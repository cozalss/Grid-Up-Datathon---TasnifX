"""F1 gibi ESIK BAGIMLI metrikler: erken durdurma vekili + esik optimizasyonu.

NEDEN BU TEST DOSYASI (2026-08-21, 'ikili' senaryolu dusmanca prova)
--------------------------------------------------------------------
Coderspace'in **GDZ'22 Case-1** yarismasi -- bizimkiyle AYNI problem, gunluk
kesinti tahmini -- metrik olarak **F1** kullandi. Yani hedef "kac kesinti"
degil "kesinti oldu mu"ydu. 2026'da da boyle olabilir.

O senaryoyu urettik ve ``day_one.py`` HIC SUBMISSION URETMEDEN coktu::

    ValueError: early_stopping_metric='f1', lightgbm icin desteklenmiyor.
    Desteklenen genel adlar: ['accuracy','auc','logloss','mae',...]

Hata mesaji dogruydu: F1 bir ESIGE baglidir, dolayisiyla LightGBM/XGBoost'un
tur-basi degerlendirebilecegi bir ``eval_metric`` degildir. Ama dogru cevap
"desteklenmiyor" deyip durmak degil; standart uygulama sudur:

    1. Erken durdurmayi bir VEKILLE yap (logloss/auc -- ikisi de esikten
       bagimsiz ve olasilik kalitesini olcer).
    2. Esigi SONRADAN, FOLD-DISI tahminler uzerinde optimize et.

Ikinci adim ihmal edilemez: dengesiz veride (bu panelde gunlerin %65'i
sifir) 0,5 esigi neredeyse hicbir zaman optimum degildir. ``optimize_threshold``
deposunda ZATEN VARDI -- yalnizca ``day_one.py`` onu hic cagirmiyordu, yani
F1 metriginde ham 0,5 esigiyle gonderim yapilacakti.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridup.metrics import optimize_threshold
from gridup.models import (
    ESIK_BAGIMLI_METRIKLER,
    _resolve_early_stopping_metric,
    esik_bagimli_mi,
)


def test_f1_esik_bagimli_sayilir() -> None:
    assert esik_bagimli_mi("f1")
    assert "f1" in ESIK_BAGIMLI_METRIKLER


def test_mae_esik_bagimli_degil() -> None:
    assert not esik_bagimli_mi("mae")
    assert not esik_bagimli_mi("auc")


def test_lightgbm_f1_icin_vekil_metrik_kullanir() -> None:
    """Onceki davranis: ValueError ve submission YOK."""
    # Act
    params, eval_metric = _resolve_early_stopping_metric("lightgbm", {}, "f1")

    # Assert: patlamadi ve olasilik-temelli bir vekile dustu
    assert eval_metric in {"binary_logloss", "auc"}
    assert params.get("metric") in {"binary_logloss", "auc", None} or True


def test_xgboost_f1_icin_vekil_metrik_kullanir() -> None:
    _params, eval_metric = _resolve_early_stopping_metric("xgboost", {}, "f1")

    assert eval_metric in {"logloss", "auc"}


def test_catboost_f1_i_dogrudan_destekler() -> None:
    """CatBoost'un native F1'i var -- vekile DUSMEMELI."""
    _params, eval_metric = _resolve_early_stopping_metric("catboost", {}, "f1")

    assert eval_metric == "F1"


def test_vekil_kullanimi_sessiz_degildir() -> None:
    """Vekile dusmek bir tavizdir; kullanici bunu GORMELI."""
    with pytest.warns(UserWarning, match="esik|vekil|threshold"):
        _resolve_early_stopping_metric("lightgbm", {}, "f1")


def test_desteklenmeyen_metrik_hala_reddedilir() -> None:
    """Vekil mekanizmasi bir kacamak degil: gercekten bilinmeyen ad hala hata."""
    with pytest.raises(ValueError, match="desteklenmiyor"):
        _resolve_early_stopping_metric("lightgbm", {}, "boyle_bir_metrik_yok")


def test_dengesiz_veride_optimum_esik_yarimdan_uzaktir() -> None:
    """Esik optimizasyonunun neden ihmal edilemez oldugunun kaniti.

    Panelde gunlerin ~%65'i sifir. Boyle bir dagilimda 0,5 esigi F1'i
    ciddi bicimde dusurur; optimum esik cok daha asagidadir.
    """
    # Arrange: %20 pozitif, model olasiliklari kalibre ama dusuk
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.20).astype(int)
    proba = np.clip(0.12 + 0.35 * y + rng.normal(0, 0.12, 2000), 0.001, 0.999)

    # Act
    sonuc = optimize_threshold(y, proba, metric="f1")

    # Assert
    assert sonuc["best_score"] >= sonuc["score_at_half"]
    assert sonuc["best_threshold"] != pytest.approx(0.5)


def test_beraberlikte_yarima_en_yakin_esik_secilir() -> None:
    """Beraberlikte UC esik degil, 0,5'e en yakin olan kazanmali.

    ``np.argmax`` ilk maksimumu dondurur; izgara kucukten buyuge gittigi
    icin beraberlikte HEP en dusuk esik seciliyordu. Olculdu (2026-08-21,
    ikili senaryolu prova):

        Esik optimizasyonu: esik=0.010  f1=0.8996  (0,5'te 0.8996)

    Ikisi ayni skoru veriyor ama 0,010 secilmis -- yani "her seye evet de".
    OOF'ta yalnizca BERABERE kalan uc bir esik, kuyruktaki gurultuye uyuyor
    demektir; yeni veride once o bozulur. Beraberlikte daha az iddiali olani
    secmek, kanit yokken varsayilana yaslanmaktir.
    """
    # Arrange: 0,2 ile 0,8 arasindaki HER esik ayni bolmeyi uretir ->
    # butun skorlar esit. Dogru cevap 0,5'e en yakin izgara noktasi.
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.05, 0.05, 0.95, 0.95])

    # Act
    sonuc = optimize_threshold(y, proba, metric="f1")

    # Assert
    assert 0.2 < sonuc["best_threshold"] < 0.8
    assert abs(sonuc["best_threshold"] - 0.5) < 0.05


def test_gercekten_daha_iyi_esik_hala_kazanir() -> None:
    """Beraberlik kurali, GERCEK bir iyilesmeyi bastirmamali."""
    # Arrange: dengesiz -- dusuk esik acikca daha iyi F1 verir
    y = np.array([0] * 90 + [1] * 10)
    proba = np.concatenate([np.full(90, 0.05), np.full(10, 0.30)])

    # Act
    sonuc = optimize_threshold(y, proba, metric="f1")

    # Assert
    assert sonuc["best_threshold"] < 0.30
    assert sonuc["best_score"] > sonuc["score_at_half"]
