from __future__ import annotations

import numpy as np


def test_gram_coz_etiketsiz_olarak_izdusum_optimumunu_bulur():
    from scripts.gram_ansambl import gram_coz

    gercek = np.array([1.0, 2.0, 3.0])
    taban = np.zeros(3)
    adaylar = {
        "p1": np.array([1.0, 0.0, 0.0]),
        "p2": np.array([0.0, 1.0, 0.0]),
    }
    skorlar = {
        "taban": float(np.sqrt(np.mean((taban - gercek) ** 2))),
        "p1": float(np.sqrt(np.mean((adaylar["p1"] - gercek) ** 2))),
        "p2": float(np.sqrt(np.mean((adaylar["p2"] - gercek) ** 2))),
    }

    sonuc = gram_coz(taban, adaylar, skorlar, taban_adi="taban")

    np.testing.assert_allclose(sonuc.katsayilar, np.array([1.0, 2.0]))
    np.testing.assert_allclose(sonuc.log_tahmin, np.array([1.0, 2.0, 0.0]))
    assert np.isclose(sonuc.tahmini_rmsle, np.sqrt(3.0))


def test_prob_kappa_tek_gonderimden_yon_optimumunu_cozer():
    from scripts.gram_ansambl import prob_kappa

    # Taban MSE=4, adım Q=1 ve gerçek optimum k=2 ise prob k=1 MSE=1 olur.
    assert np.isclose(prob_kappa(2.0, 1.0, q=1.0), 2.0)


def test_skor_araligi_merkez_tahmini_icerir():
    from scripts.gram_ansambl import aday_skor_araligi, gram_coz

    taban = np.zeros(4)
    adaylar = {"p1": np.ones(4)}
    skorlar = {"taban": 2.0, "p1": 1.0}
    sonuc = gram_coz(taban, adaylar, skorlar, taban_adi="taban")

    alt, ust = aday_skor_araligi(
        sonuc.gram,
        sonuc.katsayilar,
        skorlar,
        aday_adlari=("p1",),
        taban_adi="taban",
        yarim_adim=5e-6,
    )

    assert alt <= sonuc.tahmini_rmsle <= ust
    assert ust - alt > 0


def test_prob_skor_araligi_olcekli_adayi_kullanir():
    from scripts.gram_ansambl import prob_skor_araligi

    alt, ust = prob_skor_araligi(
        taban_skor=2.0,
        prob_skor=1.0,
        q=1.0,
        kappa=2.0,
        yarim_adim=5e-6,
    )

    assert alt == 0.0
    assert ust > alt


def test_afin_agirliklar_kappa_ile_taban_etrafinda_olceklenir():
    from scripts.gram_ansambl import afin_agirliklari

    agirliklar = afin_agirliklari(
        np.array([0.25, -0.5]),
        ("a", "b"),
        taban_adi="taban",
        kappa=2.0,
    )

    assert agirliklar == {"a": 0.5, "b": -1.0, "taban": 1.5}
    assert np.isclose(sum(agirliklar.values()), 1.0)
