from __future__ import annotations

import numpy as np
import pandas as pd


def test_uretim_kolonlari_yalin_filtreyi_birebir_uygular(monkeypatch):
    from scripts import deney_uretim_ayarlari as dua

    ham = ["sicaklik", "tk_ay", "g_ilce", "t_log_son7", "nufus", "ufuk_gun"]
    monkeypatch.setattr(dua.tm, "oznitelikler", lambda _frame: ham)
    egitim = pd.DataFrame({k: [1.0] for k in ham})
    test = pd.DataFrame({k: [1.0] for k in ham if k != "ufuk_gun"})

    assert dua.uretim_kolonlari(egitim, test) == ["sicaklik", "t_log_son7"]


def test_rejim_kaynagi_sogukta_dar_sicakta_genis_seti_secer():
    from scripts.deney_uretim_ayarlari import rejim_kaynagi

    genis = object()
    dar = object()

    assert rejim_kaynagi("soguk", genis, dar) is dar
    assert rejim_kaynagi("sicak", genis, dar) is genis


def test_eslestirilmis_ozet_mse_farkini_ve_standart_hatayi_hesaplar():
    from scripts.deney_uretim_ayarlari import eslestirilmis_ozet

    taban = np.array([1.00, 1.10, 1.20, 1.30])
    aday = np.array([0.98, 1.09, 1.18, 1.31])

    ozet = eslestirilmis_ozet(taban, aday)
    farklar = taban**2 - aday**2

    assert np.isclose(ozet.kazanc_mse, farklar.mean())
    assert np.isclose(ozet.standart_hata, farklar.std(ddof=1) / np.sqrt(len(farklar)))
    assert ozet.kazanan_cift == 3
    assert ozet.toplam_cift == 4


def test_soguk_adaylar_uretim_tabanini_ilk_ve_degisimsiz_tutar():
    from scripts.deney_uretim_ayarlari import soguk_adaylar

    adaylar = soguk_adaylar()

    assert adaylar[0].ad == "TABAN"
    assert adaylar[0].parametreler == {"depth": 7}
    assert any(a.parametreler.get("learning_rate") == 0.03 for a in adaylar)
    assert any(a.parametreler.get("rsm") == 1.0 for a in adaylar)
    assert any(
        a.parametreler.get("learning_rate") == 0.03 and a.parametreler.get("random_strength") == 4.0
        for a in adaylar
    )
    assert len({a.ad for a in adaylar}) == len(adaylar)
