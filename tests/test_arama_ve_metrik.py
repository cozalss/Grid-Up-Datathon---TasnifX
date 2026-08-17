"""SIZINTI AVI -- ARAMA (tuning) + METRIK YONU.

SINIF: "YON HATASI ve SECIM YANLILIGI".

Bu turun ortak temasi, kodun bir metrigin YONUNU (buyuk-daha-iyi mi,
kucuk-daha-iyi mi) sormadan tek yone gore karar vermesi. Tek yone bakan bir
karar, metriklerin yarisinda sessizce TERSINE doner: en kotuyu "en iyi" diye
raporlar veya sizinti sezgisini tamamen kapatir.

Ikinci tema, N denemenin EN IYISININ "modelin skoru" gibi sunulmasi. Sayi
yanlis degil, SUNUMU yaniltici -- ve juriye giden slayt bu.

Her test ``repro_tuning_metrics.py`` ile ONCE-SONRA olculmus bir bulguya
karsilik gelir; olculen sayilar docstring'lerde ismen tasinir.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from gridup.metrics import SUSPICIOUS_LOW_SCORE, SUSPICIOUS_SCORE, optimize_threshold
from gridup.tuning import TuningResult, tune_with_optuna

#: Iki objective ailesi. A ailesinin EN IYI denemesi (0.91) B'nin hepsinden
#: iyidir; ama A'nin en KOTU denemesi (0.60) B'nin en kotusunden (0.55) kotudur.
#: Yon hatasi tam bu araliktan sizar.
_A_SKORLARI = [0.60, 0.62, 0.91]
_B_SKORLARI = [0.55, 0.58, 0.59]


def _iki_aileli_gecmis() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial": list(range(6)),
            "skor": _A_SKORLARI + _B_SKORLARI,
            "objective": ["A"] * 3 + ["B"] * 3,
        }
    )


def _sonuc(*, greater_is_better: bool, best_score: float) -> TuningResult:
    return TuningResult(
        best_params={},
        best_score=best_score,
        n_trials=6,
        history=_iki_aileli_gecmis(),
        greater_is_better=greater_is_better,
    )


# --------------------------------------------------------------------------
# tuning.py:69 -- objective_comparison HER ZAMAN min ile hesapliyordu
# --------------------------------------------------------------------------


def test_objective_karsilastirmasi_buyuk_daha_iyi_metrikte_kazanani_basa_koyuyor():
    """OLCULDU: r2/auc/f1/accuracy'de tablo KAYBEDEN aileyi basa koyuyordu.

    Ayni gecmis (A ailesi [0.60, 0.62, 0.91], B ailesi [0.55, 0.58, 0.59]),
    metric='r2' (greater_is_better=True):

        ONCE : basa konan aile B, A'nin 'en_iyi' sutunu 0.60
        SONRA: basa konan aile A, A'nin 'en_iyi' sutunu 0.91

    Gercek kazanan A'dir (0.91). Docstring bu tabloyu "juri sunumunda tek
    satirlik gerekce" diye tanitiyor -- yani yanlis kayip fonksiyonu onerilir.
    """
    tablo = _sonuc(greater_is_better=True, best_score=max(_A_SKORLARI)).objective_comparison()

    assert tablo.iloc[0]["objective"] == "A", "buyuk-daha-iyi metrikte kazanan basa gelmeli"
    a_satiri = tablo.loc[tablo["objective"] == "A"].iloc[0]
    assert a_satiri["en_iyi"] == pytest.approx(0.91), "'en_iyi' ailenin EN IYISI olmali"
    assert list(tablo["en_iyi"]) == sorted(tablo["en_iyi"], reverse=True), (
        "buyuk-daha-iyi metrikte tablo AZALAN siralanmali"
    )


def test_objective_karsilastirmasi_kucuk_daha_iyi_metrikte_ayni_kaliyor():
    """YANLIS-POZITIF KORUMASI: rmse/mae yolu zaten dogruydu, bozulmamali.

    OLCULDU (greater_is_better=False, ayni gecmis):
        ONCE : basa B (en_iyi 0.55), A'nin en_iyi'si 0.60
        SONRA: basa B (en_iyi 0.55), A'nin en_iyi'si 0.60   -- DEGISMEDI

    Bulgunun testlerden kacmasinin sebebi buydu: varsayilan metrik rmse ve o
    yolda min dogru cevabi veriyor.
    """
    tablo = _sonuc(greater_is_better=False, best_score=min(_B_SKORLARI)).objective_comparison()

    assert tablo.iloc[0]["objective"] == "B"
    assert tablo.iloc[0]["en_iyi"] == pytest.approx(0.55)
    a_satiri = tablo.loc[tablo["objective"] == "A"].iloc[0]
    assert a_satiri["en_iyi"] == pytest.approx(0.60)
    assert list(tablo["en_iyi"]) == sorted(tablo["en_iyi"]), (
        "kucuk-daha-iyi metrikte tablo ARTAN siralanmali"
    )


def test_en_iyi_sutunu_her_zaman_best_score_ile_ayni_yonde():
    """OZELLIK: tablonun ilk satiri, raporlanan ``best_score``i icermeli.

    Ilk satirin 'en_iyi'si best_score'dan farkliysa, tablo ile ozet birbirini
    yalanliyor demektir -- juri sunumunda iki farkli sayi gorunur.
    """
    for gib, beklenen in [(True, 0.91), (False, 0.55)]:
        sonuc = _sonuc(greater_is_better=gib, best_score=beklenen)
        tablo = sonuc.objective_comparison()
        assert tablo.iloc[0]["en_iyi"] == pytest.approx(sonuc.best_score), (
            f"greater_is_better={gib}: tablo basi ile best_score uyusmuyor"
        )


def test_gecmis_bossa_bos_tablo_donuyor():
    """YANLIS-POZITIF KORUMASI: gecmis yoksa hata degil, bos tablo."""
    bos = TuningResult(best_params={}, best_score=1.0, n_trials=0, greater_is_better=True)
    assert bos.objective_comparison().empty


# --------------------------------------------------------------------------
# tuning.py:263 -- best_score N denemenin en iyisi, hicbir yerde soylenmiyordu
# --------------------------------------------------------------------------


def _gurultu_verisi(n: int = 900, p: int = 12, tohum: int = 0):
    """SIFIR sinyal: hedef saf gurultu. Ogrenilecek hicbir sey YOK."""
    rng = np.random.default_rng(tohum)
    train = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = rng.normal(0, 2.0, n)
    folds = list(KFold(n_splits=4, shuffle=True, random_state=0).split(train))
    return train, y, folds


@pytest.fixture(scope="module")
def gurultude_arama() -> TuningResult:
    """SIFIR sinyalli veride gercek bir Optuna aramasi (modul basina bir kez)."""
    train, y, folds = _gurultu_verisi()
    return tune_with_optuna(
        train,
        y,
        folds,
        metric="rmse",
        n_trials=12,
        early_stopping_rounds=30,
        verbose=False,
    )


def test_arama_skoru_secim_yanliligini_acikca_soyluyor(gurultude_arama: TuningResult):
    """OLCULDU: SIFIR sinyalde raporlanan skor teorik TAVANI "asiyor".

        hedefin kendi std'si (teorik tavan) : 1.952159
        best_score (RAPORLANAN)             : 1.947733   <-- tavanin ALTINDA
        deneme ortalamasi                   : 1.951498
        iyimserlik (fark)                   : 0.003766

        ONCE : selection_optimism alani YOK, summary()'de uyari YOK (False)
        SONRA: selection_optimism = 0.003766, summary() uyariyor (True)

    Ogrenilecek hicbir sey yokken "en iyi skor" hedefin kendi std'sinin altina
    inemez; bu tamamen secim artefaktidir. ``selection.SelectionResult`` ayni
    yanliligi tur 8'de zaten raporluyordu -- ayni desen buraya tasindi.
    """
    assert len(gurultude_arama.history) >= 2
    assert gurultude_arama.selection_optimism > 0.0, (
        "12 korele denemenin en iyisi deneme ortalamasina esit olamaz"
    )
    ozet = gurultude_arama.summary()
    assert "korele denemenin EN IYISIDIR" in ozet
    assert "bagimsiz bir kumede dogrula" in ozet


def test_arama_sonucu_metrigin_yonunu_tasiyor(gurultude_arama: TuningResult):
    """``tune_with_optuna`` yonu ``get_metric``ten DOLDURMALI.

    Aksi halde objective_comparison varsayilan ``False``ta kalir ve r2/auc
    aramasinda yine kaybedeni basa koyar -- duzeltme cagri yerine baglidir.
    """
    assert gurultude_arama.greater_is_better is False, "rmse kucuk-daha-iyidir"

    train, y, folds = _gurultu_verisi(n=300, p=5, tohum=1)
    r2_sonuc = tune_with_optuna(
        train,
        y,
        folds,
        metric="r2",
        n_trials=3,
        early_stopping_rounds=20,
        verbose=False,
    )
    assert r2_sonuc.greater_is_better is True, "r2 buyuk-daha-iyidir"


def test_tek_denemede_yanlilik_uyarisi_cikmiyor():
    """YANLIS-POZITIF KORUMASI: tek deneme yapildiysa secim yanliligi YOKTUR.

    Tek satirlik gecmiste best_score ile ortalama ayni sayidir; buna
    "iyimserlik" demek kullaniciyi bos yere korkutur.
    """
    tek = TuningResult(
        best_params={},
        best_score=1.5,
        n_trials=1,
        history=pd.DataFrame({"trial": [0], "skor": [1.5], "objective": ["regression"]}),
    )
    assert tek.selection_optimism == 0.0
    assert "korele denemenin" not in tek.summary()


def test_gecmissiz_sonucta_yanlilik_hesabi_patlamiyor():
    """YANLIS-POZITIF KORUMASI: gecmis hic yoksa (bos DataFrame) 0.0 donmeli."""
    bos = TuningResult(best_params={}, best_score=0.5, n_trials=0)
    assert bos.selection_optimism == 0.0
    assert "korele denemenin" not in bos.summary()


# --------------------------------------------------------------------------
# metrics.py:206 -- supheli-skor sezgisi yalnizca buyuk-daha-iyi metriklerde
# --------------------------------------------------------------------------


def _sizintili_ve_gercekci(n: int = 800, tohum: int = 3):
    """``y_proba = y`` (tam sizinti) ve gercekci bir OOF dizisi."""
    rng = np.random.default_rng(tohum)
    y = (rng.random(n) < 0.3).astype(int)
    tam_sizintili = y.astype(float)
    gercekci = np.clip(y * 0.25 + rng.normal(0.35, 0.2, n), 0, 1)
    return y, tam_sizintili, gercekci


def _supheli_uyarilar(y, proba, metrik: str) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as yakalanan:
        warnings.simplefilter("always")
        optimize_threshold(y, proba, metric=metrik)
    return [uyari for uyari in yakalanan if "supheli" in str(uyari.message)]


@pytest.mark.parametrize("metrik", ["logloss", "mae", "rmse"])
def test_kucuk_daha_iyi_metrikte_supheli_mukemmellik_uyari_uretiyor(metrik: str):
    """OLCULDU: kucuk-daha-iyi metrikte sezgi HIC devrede degildi.

    TAM SIZINTILI dizide (``y_proba = y``, n=800) uyari sayisi:

        metrik    best_score   ONCE   SONRA
        f1          1.000000     1      1
        accuracy    1.000000     1      1
        logloss     0.000000     0      1
        mae         0.000000     0      1
        rmse        0.000000     0      1

    Sebep ``greater_is_better and best_score > 0.99`` kapisiydi: mukemmellik
    kucuk-daha-iyi metrikte yukaridan degil ASAGIDAN gelir.
    """
    y, tam_sizintili, _ = _sizintili_ve_gercekci()

    uyarilar = _supheli_uyarilar(y, tam_sizintili, metrik)

    assert uyarilar, f"{metrik}: tam sizintili dizide uyari YOK"
    assert "fold-disi" in str(uyarilar[0].message)
    assert "dusuk" in str(uyarilar[0].message), "yon mesajda dogru yazilmali"


@pytest.mark.parametrize("metrik", ["logloss", "mae", "rmse"])
def test_gercekci_oof_dizisinde_kucuk_metrikte_yanlis_pozitif_yok(metrik: str):
    """YANLIS-POZITIF KORUMASI: gercekci OOF skorlari esikten UZAK.

    OLCULDU (ayni n=800, gercekci OOF dizisi):
        logloss = 7.884549   mae = 0.218750   rmse = 0.467707
    Hepsi 0.01 sinirinin cok ustunde -- yeni uyari gunluk kullanimda otmez.
    """
    y, _, gercekci = _sizintili_ve_gercekci()

    assert not _supheli_uyarilar(y, gercekci, metrik), (
        f"{metrik}: gercekci OOF skorunda yanlis pozitif uyari"
    )


@pytest.mark.parametrize("metrik", ["f1", "accuracy"])
def test_buyuk_daha_iyi_metrikte_eski_davranis_korunuyor(metrik: str):
    """REGRESYON: f1/accuracy yolu ZATEN calisiyordu, aynen kalmali.

    OLCULDU: tam sizintili dizide her ikisi de best_score=1.000000 ve 1 uyari;
    gercekci OOF dizisinde f1=0.609642 / accuracy=0.781250 ve 0 uyari.
    """
    y, tam_sizintili, gercekci = _sizintili_ve_gercekci()

    assert _supheli_uyarilar(y, tam_sizintili, metrik), f"{metrik}: sizintida uyari kayboldu"
    assert not _supheli_uyarilar(y, gercekci, metrik), f"{metrik}: yanlis pozitif"


def test_iki_supheli_sinir_birbirinin_aynasi():
    """Esikler simetrik secildi: 0.99 ustu ve 0.01 alti.

    Bu bir sozlesme: birini degistiren digerini de degistirmeli, aksi halde
    metrigin yonune gore farkli hassasiyet olusur ve sezgi yine tek yana kayar.
    """
    assert pytest.approx(1.0) == SUSPICIOUS_SCORE + SUSPICIOUS_LOW_SCORE
