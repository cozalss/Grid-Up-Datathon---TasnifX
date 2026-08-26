from __future__ import annotations

import numpy as np
import pytest


def test_prob_cozucu_kuadratik_optimumu_geri_bulur():
    from scripts.yarin_coz import prob_coz

    taban_mse = 1.0
    adim = 0.08
    gercek_optimum_delta = 0.12
    q = adim**2
    ic_carpim = adim * gercek_optimum_delta
    prob_mse = taban_mse + q - 2 * ic_carpim

    sonuc = prob_coz(np.sqrt(taban_mse), np.sqrt(prob_mse), q=q, adim=adim)

    assert np.isclose(sonuc.katsayi, 1.5)
    assert np.isclose(sonuc.optimum_ek_delta, gercek_optimum_delta)
    assert np.isclose(sonuc.kazanc_mse, -(gercek_optimum_delta**2))


def test_prob_cozucu_notr_probda_sifir_delta_dondurur():
    from scripts.yarin_coz import prob_coz

    taban_mse = 1.0270038
    q = 0.0047143
    sonuc = prob_coz(np.sqrt(taban_mse), np.sqrt(taban_mse + q), q=q, adim=0.08)

    assert abs(sonuc.optimum_ek_delta) < 1e-12
    assert abs(sonuc.kazanc_mse) < 1e-12


def test_soguk_optimumu_duzeltilmis_ortak_denklemden_cozulur():
    from scripts.yarin_coz import ortak_denklemden_soguk_coz

    p_soguk = 158_369 / 714_688
    p_kuyruk = 29_873 / 714_688
    b_soguk = 0.1046
    b_kuyruk = 0.166441
    l_ortak = p_soguk * 0.22 * b_soguk + p_kuyruk * 0.35 * b_kuyruk

    bulunan = ortak_denklemden_soguk_coz(
        b_kuyruk,
        l_ortak=l_ortak,
        p_soguk=p_soguk,
        p_kuyruk=p_kuyruk,
    )

    assert np.isclose(bulunan, b_soguk)


def test_uc_rejim_deltasi_birbirine_karismadan_uygulanir():
    from scripts.yarin_coz import uc_rejim_deltasi_uygula

    taban = np.array([0.0, 9.0, 99.0, 999.0])
    soguk = np.array([True, False, False, False])
    kuyruk = np.array([False, True, False, False])

    sonuc = uc_rejim_deltasi_uygula(
        taban,
        soguk=soguk,
        kuyruk=kuyruk,
        b_soguk=0.10,
        b_kuyruk=0.20,
        b_sicak_cekirdek=0.30,
    )
    fark = np.log1p(sonuc) - np.log1p(taban)

    assert np.allclose(fark, [0.10, 0.20, 0.30, 0.30])


def test_negatif_delta_sifir_tahmini_kirpacaksa_sessizce_devam_etmez():
    from scripts.yarin_coz import uc_rejim_deltasi_uygula

    with pytest.raises(ValueError, match="kırpma"):
        uc_rejim_deltasi_uygula(
            np.array([0.0, 10.0]),
            soguk=np.array([True, False]),
            kuyruk=np.array([False, False]),
            b_soguk=-0.10,
            b_kuyruk=0.0,
            b_sicak_cekirdek=0.0,
        )


def test_kuyruk_skoru_yokken_sicak_optimumu_ayni_gun_bankalanabilir():
    from scripts.yarin_coz import ProbSonucu, nihai_cozum

    sicak = ProbSonucu(
        katsayi=1.5,
        optimum_ek_delta=0.12,
        kazanc_mse=-0.010,
        gerceklesen_dmse=-0.005,
    )

    cozum = nihai_cozum(1.02, sicak, kuyruk_sonuc=None)

    assert np.isclose(cozum.b_sicak_cekirdek, 0.12)
    assert np.isclose(cozum.b_soguk, 0.1046)
    assert np.isclose(cozum.b_kuyruk, 0.1664)
    assert np.isclose(cozum.tahmini_mse, 1.02**2 - 0.010)
