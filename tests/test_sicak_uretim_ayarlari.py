from __future__ import annotations

import numpy as np


def test_sicak_adaylar_uretim_tabanini_korur_ve_tek_eksenlidir():
    from scripts.deney_sicak_uretim_ayarlari import sicak_adaylar

    adaylar = sicak_adaylar()

    assert adaylar[0].ad == "TABAN"
    assert adaylar[0].parametreler == {
        "depth": 6,
        "l2_leaf_reg": 1.0,
        "random_strength": 4.0,
    }
    assert len({aday.ad for aday in adaylar}) == len(adaylar)
    assert any(aday.parametreler.get("learning_rate") == 0.03 for aday in adaylar)
    assert any(aday.parametreler.get("rsm") == 1.0 for aday in adaylar)
    assert any(aday.parametreler.get("bootstrap_type") == "Bernoulli" for aday in adaylar)


def test_log_harmani_uretim_agirliklarini_uygular():
    from scripts.deney_sicak_uretim_ayarlari import log_harmani

    tahminler = {
        "cat": np.array([1.0, 2.0]),
        "xgb": np.array([2.0, 4.0]),
        "lgbm": np.array([3.0, 6.0]),
        "sinir_agi": np.array([4.0, 8.0]),
    }
    agirliklar = {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0, "sinir_agi": 1.4}

    sonuc = log_harmani(tahminler, agirliklar)
    beklenen = (
        3.0 * tahminler["cat"] + tahminler["xgb"] + tahminler["lgbm"] + 1.4 * tahminler["sinir_agi"]
    ) / 6.4

    np.testing.assert_allclose(sonuc, beklenen)


def test_blok_hukmu_her_blokta_pozitif_kazanc_ister():
    from scripts.deney_sicak_uretim_ayarlari import blok_hukmu

    assert blok_hukmu({"yaz25": 0.01, "kis26": 0.02}) is True
    assert blok_hukmu({"yaz25": 0.01, "kis26": -0.001}) is False
    assert blok_hukmu({}) is False


def test_elemede_ilk_blok_kaybedilirse_ikinci_fit_atlanir():
    from scripts.deney_sicak_uretim_ayarlari import devam_etmeli

    assert devam_etmeli("eleme", "yaz25", "yaz25", []) is True
    assert devam_etmeli("eleme", "kis26", "yaz25", [0.01]) is True
    assert devam_etmeli("eleme", "kis26", "yaz25", [-0.001]) is False
    assert devam_etmeli("dogrula", "kis26", "yaz25", [-0.001]) is True


def test_torbali_log_harmani_once_aileleri_sonra_tohumlari_ortalar():
    from scripts.deney_sicak_uretim_ayarlari import torbali_log_harmani

    tohum_tahminleri = [
        {"cat": np.array([1.0]), "xgb": np.array([3.0])},
        {"cat": np.array([5.0]), "xgb": np.array([7.0])},
    ]

    sonuc = torbali_log_harmani(tohum_tahminleri, {"cat": 3.0, "xgb": 1.0})

    # Tohum harmanları 1.5 ve 5.5; üretim bunları log uzayında ortalar.
    np.testing.assert_allclose(sonuc, np.array([3.5]))


def test_kuadratik_optimum_tam_aday_otesindeki_en_iyi_olcegi_bulur():
    from scripts.deney_sicak_uretim_ayarlari import kuadratik_optimum

    gercek = np.array([2.0, 4.0])
    taban = np.array([0.0, 0.0])
    aday = np.array([1.0, 2.0])

    kappa, kazanc = kuadratik_optimum(gercek, taban, aday, np.ones(2))

    assert np.isclose(kappa, 2.0)
    assert np.isclose(kazanc, -10.0)
