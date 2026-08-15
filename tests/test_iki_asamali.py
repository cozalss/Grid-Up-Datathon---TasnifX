"""SIZINTI AVI TUR 7 -- IKI ASAMALI MODEL (``two_stage.py``).

SINIF: "dogrulanmamis satirlar skora giriyor" + "matematiksel olarak yanlis
dagilim kullaniliyor".

Bu tur, sizintinin daha sinsi bir bicimine bakiyor: kod calisir, hata vermez,
sayilar makul gorunur -- ama SKORLANAN satirlarin bir kismi hic dogrulanmamis,
ve "MAE-optimal" diye sunulan yol yanlis dagilimin kuantilini kullaniyor.

Her test ``t7_repro.py`` ile ONCE-SONRA olculmus bir bulguya karsilik gelir.
"""

from __future__ import annotations

import typing

import numpy as np
import pandas as pd
import pytest

from gridup.metrics import get_metric
from gridup.two_stage import (
    CONDITIONAL_LADDER_LEVELS,
    PredictMode,
    conditional_quantile_from_hurdle,
    fit_conditional_quantile_ladder,
    fit_quantile_ladder,
    fit_two_stage,
    mae_optimal_quantile,
)
from gridup.validation import purged_time_series_split

MAE = get_metric("mae")[0]


def _hurdle(n: int = 1500, tohum: int = 5):
    """Bilinen bir hurdle sureci: p = sigmoid(1.2x), pozitifse lognormal."""
    rng = np.random.default_rng(tohum)
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(1.2 * x)))
    y = np.where(rng.random(n) < p, np.exp(rng.normal(2.0 + 0.5 * x, 0.6)), 0.0)
    frame = pd.DataFrame({"x": x, "g": rng.normal(size=n)})
    zaman = pd.Series(pd.date_range("2024-01-01", periods=n, freq="h"))
    folds = purged_time_series_split(
        zaman, embargo=pd.Timedelta(hours=6), n_splits=3, verbose=False
    )
    return frame, y, folds


# --------------------------------------------------------------------------
# B21 -- tip imzasi ile gercek davranis
# --------------------------------------------------------------------------


def test_predict_mode_ilan_ettigi_her_modu_gercekten_destekliyor():
    """OLCULDU: PredictMode 'mae_optimal' ilan ediyordu ama _combine onu
    reddediyordu -- tip denetleyicisi cagriyi GECERLI sayarken calisma
    zamaninda ValueError aliniyordu."""
    frame, y, folds = _hurdle(600, tohum=0)
    sonuc = fit_two_stage(frame, y, folds, verbose=False)
    for mod in typing.get_args(PredictMode):
        tahmin = sonuc.predict_oof(mode=mod)
        assert len(tahmin) == len(y)
        assert np.isfinite(tahmin).all()


# --------------------------------------------------------------------------
# B19 -- OOF kapsami
# --------------------------------------------------------------------------


def test_skorlar_dogrulanmamis_satirlari_saymiyor():
    """OLCULDU (600 satir, kapsam %75): hic dogrulanmamis 150 satirin
    oof_probability'si TEK deger (0.0) iken gercek hedef ortalamalari 6.086.

    Onlari skora katmak, gercek pozitif satirlari "sifir tahmin edildi" diye
    puanlamaktir:
        TUM satirlarda  -> mae 5.4283
        yalniz kapsanan -> mae 5.2092   (%4.2 sisme)
    """
    frame, y, folds = _hurdle(600, tohum=0)
    sonuc = fit_two_stage(frame, y, folds, verbose=False)

    kapsanan = sonuc.covered()
    assert sonuc.oof_covered is not None
    assert 0.0 < sonuc.coverage < 1.0, "bu veride kismi kapsam bekleniyor"

    # Kapsanmayan satirlar gercekten DOLGU mu?
    assert np.allclose(sonuc.oof_probability[~kapsanan], 0.0)

    # Raporlanan skor kapsanan satirlarla ayni olmali, TUM satirlarla degil.
    from gridup.two_stage import tune_threshold

    beklenen = tune_threshold(
        y[kapsanan], sonuc.oof_probability[kapsanan],
        sonuc.oof_magnitude[kapsanan], metric="mae",
    )
    assert sonuc.diagnostics["esikli_mod_mae"] == pytest.approx(
        round(beklenen["best_score"], 6)
    )
    assert sonuc.diagnostics["oof_kapsami"] == pytest.approx(
        round(float(kapsanan.mean()), 4)
    )
    assert "not" in sonuc.diagnostics


def test_tam_kapsamda_kapsam_notu_cikmiyor():
    """Yanlis pozitif korumasi: KFold'da her satir dogrulanir."""
    frame, y, _ = _hurdle(600, tohum=0)
    n = len(y)
    kesim = n // 2
    folds = [
        (np.arange(kesim, n), np.arange(0, kesim)),
        (np.arange(0, kesim), np.arange(kesim, n)),
    ]
    sonuc = fit_two_stage(frame, y, folds, verbose=False)
    assert sonuc.coverage == 1.0
    assert "not" not in sonuc.diagnostics


# --------------------------------------------------------------------------
# B18 -- KOSULLU vs MARJINAL merdiven
# --------------------------------------------------------------------------


def test_kosullu_merdiven_marjinal_olandan_ve_esikli_moddan_iyi():
    """OLCULDU -- bu turun ana bulgusu.

    ``q* = 1 - 0.5/p`` turetimi ``F(y|y>0)`` KOSULLU dagilimin kuantilini
    ister. ``fit_quantile_ladder`` ise hedefin TAMAMIYLA egitilir ve MARJINAL
    kuantil ogrenir:

        marjinal q=0.50 tahmin ort : 3.700
        KOSULLU  q=0.50 tahmin ort : 6.907
        gercek kosullu medyan      : 9.328

    Sonuc, "MAE-optimal" diye sunulan yolun basit yoldan DAHA KOTU olmasiydi:

        hep sifir          : 6.4595
        marjinal merdiven  : 5.2678   <- eski yol
        esikli mod         : 5.0914   <- basit yol, DAHA IYI
        KOSULLU merdiven   : 4.9203   <- dogru yol, en iyi

    Ayrica dogru kullanim public API ile URETILEMIYORDU: cross_validate OOF'u
    yalnizca egitilen satirlar icin uretir, oysa merdiven TUM satirlara tahmin
    vermek zorunda.
    """
    frame, y, folds = _hurdle(1500, tohum=5)
    sonuc = fit_two_stage(frame, y, folds, verbose=False)
    kapsanan = sonuc.covered()
    seviyeler = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)

    marjinal = {
        q: r.oof_predictions
        for q, r in fit_quantile_ladder(
            frame, y, folds, quantiles=seviyeler, verbose=False
        ).items()
    }
    kosullu = fit_conditional_quantile_ladder(
        frame, y, folds, quantiles=seviyeler, verbose=False
    )

    # Kosullu merdiven pozitif buyuklukleri ogrenmeli, sifiri degil.
    assert kosullu[0.5].mean() > 1.8 * marjinal[0.5].mean()

    mae_marjinal = MAE(
        y[kapsanan],
        conditional_quantile_from_hurdle(
            sonuc.oof_probability, marjinal, verbose=False
        )[kapsanan],
    )
    mae_kosullu = MAE(
        y[kapsanan],
        conditional_quantile_from_hurdle(
            sonuc.oof_probability, kosullu, verbose=False
        )[kapsanan],
    )
    mae_esikli = MAE(y[kapsanan], sonuc.predict_oof(mode="thresholded")[kapsanan])

    assert mae_kosullu < mae_esikli < mae_marjinal, (
        f"kosullu={mae_kosullu:.4f} esikli={mae_esikli:.4f} "
        f"marjinal={mae_marjinal:.4f}"
    )


def test_kosullu_merdiven_tum_satirlara_tahmin_uretiyor():
    """Merdiven pozitiflerle EGITILIR ama TUM satirlara tahmin vermelidir --
    yoksa conditional_quantile_from_hurdle'a verilemez."""
    frame, y, folds = _hurdle(800, tohum=2)
    merdiven = fit_conditional_quantile_ladder(
        frame, y, folds, quantiles=(0.1, 0.3, 0.5), verbose=False
    )
    assert set(merdiven) == {0.1, 0.3, 0.5}
    for tahminler in merdiven.values():
        assert len(tahminler) == len(y)
        assert np.isfinite(tahminler).all()


def test_pozitif_satir_yoksa_kosullu_merdiven_acik_hata_veriyor():
    frame, _, folds = _hurdle(600, tohum=0)
    with pytest.raises(ValueError, match="Pozitif satir yok"):
        fit_conditional_quantile_ladder(
            frame, np.zeros(600), folds, quantiles=(0.5,), verbose=False
        )


# --------------------------------------------------------------------------
# B22 -- q* kirpilmasi ve bosa egitilen seviyeler
# --------------------------------------------------------------------------


def test_q_yildiz_asla_yarimi_asmaz_ve_varsayilan_merdiven_bosa_seviye_icermez():
    """``q* = 1 - 0.5/p`` matematiksel olarak (0, 0.5] araligindadir.

    Eski varsayilan merdivende 0.6/0.7/0.8/0.9 vardi -- 10 seviyenin 4'u,
    yani CV kosusunun %40'i HIC kullanilmadan egitiliyordu.
    """
    olasilik = np.linspace(0.5001, 1.0, 500)
    q = mae_optimal_quantile(olasilik)
    gecerli = ~np.isnan(q)
    assert q[gecerli].max() <= 0.5 + 1e-12
    assert q[gecerli].min() > 0.0
    assert max(CONDITIONAL_LADDER_LEVELS) <= 0.5


def test_merdiven_altina_dusen_q_yildiz_uyari_uretiyor(capsys):
    """OLCULDU: [0.05, 0.5] merdiveninde 646 satirin 56'si (%8.7) alta dusup
    en dusuk seviyeye kirpilmisti -- sistematik ASIRI TAHMIN, sessizce."""
    olasilik = np.array([0.5001, 0.51, 0.9, 0.99])  # ilk ikisi cok kucuk q*
    merdiven = {
        0.2: np.array([1.0, 1.0, 1.0, 1.0]),
        0.5: np.array([5.0, 5.0, 5.0, 5.0]),
    }
    conditional_quantile_from_hurdle(olasilik, merdiven, verbose=True)
    cikti = capsys.readouterr().out
    assert "merdivenin altinda" in cikti


def test_marjinal_gorunumlu_merdiven_uyari_uretiyor(capsys):
    """En dusuk seviyenin tahminleri cogunlukla sifirsa merdiven MARJINAL
    ogrenmis olabilir -- olculdu: marjinal q=0.05 tahminlerinin %100'u sifir."""
    olasilik = np.full(100, 0.9)
    merdiven = {
        0.05: np.zeros(100),          # marjinal merdivenin imzasi
        0.5: np.full(100, 4.0),
    }
    conditional_quantile_from_hurdle(olasilik, merdiven, verbose=True)
    assert "MARJINAL" in capsys.readouterr().out
