"""OZELLIK TABANLI TESTLER -- kenar durumlari BEN degil, makine uretsin.

NEDEN BU DOSYA VAR
------------------
Bu depoda bulunan ciddi hatalarin hepsi, o sirada gecmekte olan 400+ testin
ALTINDAN gecti. Sebep hep ayniydi: **test fixture'lari benim hayal ettigim
durumlari iceriyordu.**

  * Grup kaydirma hatasi gorunmedi, cunku fixture'lar 'TR001','TR002'
    kullaniyordu -- gorunum sirasi zaten alfabetikti.
  * Gunes takasi gorunmedi, cunku fixture'lar hep benzersiz index kuruyordu.
  * Bozuk hedef gorunmedi, cunku hicbir fixture'da NaN hedef yoktu.

Ornek tabanli testin tavani budur: yalnizca dusunulmus durumlari kapsar.
Ozellik tabanli test bu tavani kaldirir -- **degismesi gereken sey degil,
DEGISMEMESI gereken sey** yazilir ve girdiyi hypothesis uretir. Bir kars
ornek bulursa onu en kucuk haline indirger (shrinking) ve gosterir.

BURADA SINANAN OZELLIKLER
-------------------------
Hepsi "her gecerli girdi icin dogru olmali" bicimindedir::

    tr_lower(tr_upper(x)) == tr_lower(x)         her Turkce metin icin
    fold'lar asla cakismaz                       her zaman serisi + ambargo icin
    downcast degeri bozmaz                       her float dizisi icin
    feature girdiyi degistirmez                  her panel icin
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from gridup.compat import downcast_numeric
from gridup.features.temporal import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
)
from gridup.metrics import mape, mape_coverage, rmse, rmsle, smape
from gridup.turkish import join_key, tr_lower, tr_sorted, tr_upper
from gridup.two_stage import mae_optimal_quantile
from gridup.validation import assert_folds_align, purged_time_series_split

# DIKKAT: burada ``max_examples`` VERMIYORUZ.
#
# hypothesis'te test uzerindeki @settings, yuklenmis PROFILI ezer. Ilk
# surumde max_examples=150 yazmistim ve HYPOTHESIS_PROFILE=derin ile
# calistirdigimda hicbir sey degismiyordu -- "derin tarama" cephe koddu.
# Ornek sayisi artik yalnizca conftest.py'daki profilden gelir:
#   varsayilan -> 150   (her kosu)
#   derin      -> 2000  (veri gelmeden once bir kez)
AYAR = settings(
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)

#: Turkce metin: ASCII + Turkce'ye ozgu harfler + bosluk.
TURKCE_HARF = st.sampled_from(list("abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ _-"))
TURKCE_METIN = st.text(alphabet=TURKCE_HARF, min_size=1, max_size=25)


# --------------------------------------------------------------------------
# Turkce metin -- bu projenin en sik kirilan yeri
# --------------------------------------------------------------------------


@AYAR
@given(TURKCE_METIN)
def test_tr_lower_sabit_nokta(metin: str):
    """OZELLIK: kucultmeyi iki kez uygulamak bir kez uygulamakla ayni olmali.

    Bu, birlesik nokta (U+0307) uretilmedigini de kanitlar: uretilseydi
    ikinci cagri farkli bir sonuc verirdi.
    """
    bir = tr_lower(metin)
    assert tr_lower(bir) == bir


@AYAR
@given(TURKCE_METIN)
def test_tr_lower_birlesik_nokta_uretmiyor(metin: str):
    """OZELLIK: cikti ASLA U+0307 (birlesik nokta) icermemeli.

    Duz ``.lower()`` 'İ' icin bunu uretir ve tum join'leri sessizce bozar.
    """
    assert "̇" not in tr_lower(metin)


@AYAR
@given(TURKCE_METIN)
def test_join_key_kararli_ve_diyakritiksiz(metin: str):
    """OZELLIK: join_key idempotenttir ve Turkce'ye ozgu harf birakmaz."""
    anahtar = join_key(metin)
    assert join_key(anahtar) == anahtar
    assert not (set(anahtar) & set("çğıöşüÇĞIİÖŞÜ"))
    assert "̇" not in anahtar


@AYAR
@given(TURKCE_METIN)
def test_buyuk_kucuk_don_gidis_kararli(metin: str):
    """OZELLIK: tr_lower(tr_upper(x)) == tr_lower(x).

    Buyuk harfe cevirip geri donmek bilgi KAYBETMEMELI.
    """
    assert tr_lower(tr_upper(metin)) == tr_lower(metin)


@AYAR
@given(st.lists(TURKCE_METIN, min_size=0, max_size=12))
def test_tr_sorted_permutasyon_uretiyor(kelimeler: list[str]):
    """OZELLIK: siralama eleman EKLEMEZ, SILMEZ, DEGISTIRMEZ."""
    sirali = tr_sorted(kelimeler)
    assert sorted(sirali) == sorted(kelimeler)


# --------------------------------------------------------------------------
# Metrikler -- her sonlu girdi icin tanimli davranis
# --------------------------------------------------------------------------

SONLU = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


@AYAR
@given(st.lists(SONLU, min_size=1, max_size=60))
def test_rmse_kendisiyle_sifir(degerler: list[float]):
    """OZELLIK: mukemmel tahminin hatasi sifirdir."""
    dizi = np.array(degerler)
    assert rmse(dizi, dizi) == pytest.approx(0.0, abs=1e-9)


@AYAR
@given(st.lists(SONLU, min_size=1, max_size=60), st.lists(SONLU, min_size=1, max_size=60))
def test_rmse_simetrik(a: list[float], b: list[float]):
    """OZELLIK: RMSE argumanlarin sirasindan bagimsizdir."""
    n = min(len(a), len(b))
    x, y = np.array(a[:n]), np.array(b[:n])
    assert rmse(x, y) == pytest.approx(rmse(y, x), rel=1e-9)


@AYAR
@given(st.lists(st.floats(min_value=0.0, max_value=1e6, allow_nan=False), min_size=1, max_size=60))
def test_rmsle_negatif_olmayan(degerler: list[float]):
    """OZELLIK: RMSLE her zaman >= 0 ve sonlu."""
    dizi = np.array(degerler)
    sonuc = rmsle(dizi, dizi)
    assert sonuc >= 0.0
    assert np.isfinite(sonuc)


@AYAR
@given(
    st.lists(st.floats(min_value=1.0, max_value=1e5, allow_nan=False), min_size=1, max_size=60),
    st.lists(st.floats(min_value=1.0, max_value=1e5, allow_nan=False), min_size=1, max_size=60),
)
def test_smape_ust_sinirli(a: list[float], b: list[float]):
    """OZELLIK: SMAPE tanimi geregi [0, 200] araligindadir."""
    n = min(len(a), len(b))
    sonuc = smape(np.array(a[:n]), np.array(b[:n]))
    assert 0.0 <= sonuc <= 200.0 + 1e-9


@AYAR
@given(st.lists(SONLU, min_size=1, max_size=60))
def test_mape_kapsami_orani_dogru(degerler: list[float]):
    """OZELLIK: kapsam = sifir olmayan satirlarin orani."""
    dizi = np.array(degerler)
    beklenen = float(np.mean(np.abs(dizi) >= 1e-9))
    assert mape_coverage(dizi) == pytest.approx(beklenen)


@AYAR
@given(st.lists(st.floats(min_value=0.1, max_value=1e4, allow_nan=False), min_size=1, max_size=60))
def test_mape_kendisiyle_sifir(degerler: list[float]):
    dizi = np.array(degerler)
    assert mape(dizi, dizi) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# MAE-optimal kuantil -- matematiksel ozellik
# --------------------------------------------------------------------------


@AYAR
@given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=40))
def test_mae_optimal_kuantil_gecerli_aralikta(olasiliklar: list[float]):
    """OZELLIK: q* = 1 - 0.5/p ya (0,1) araligindadir ya da NaN'dir.

    p <= 0.5 iken formul negatif verir -- o durumda dagilimin medyani
    sifirdir ve kosullu kuantil TANIMSIZDIR, NaN dondurulmelidir.
    """
    p = np.array(olasiliklar)
    q = mae_optimal_quantile(p)

    gecerli = ~np.isnan(q)
    assert np.all(q[gecerli] > 0.0)
    assert np.all(q[gecerli] < 1.0)
    # p <= 0.5 olan her yerde NaN olmali.
    assert np.all(np.isnan(q[p <= 0.5]))


# --------------------------------------------------------------------------
# downcast -- degeri bozmama sozlesmesi
# --------------------------------------------------------------------------


@AYAR
@given(
    st.lists(
        st.floats(min_value=-1e12, max_value=1e12, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=60,
    )
)
def test_downcast_degeri_bagil_olarak_koruyor(degerler: list[float]):
    """OZELLIK: dtype degisebilir ama BAGIL hata 1e-6'yi asamaz."""
    seri = pd.Series(degerler, dtype="float64")
    inen = downcast_numeric(seri)

    a = seri.to_numpy(dtype="float64")
    b = inen.to_numpy(dtype="float64")
    olcek = float(np.abs(a).max()) or 1.0
    assert float(np.abs(a - b).max()) / olcek < 1e-6


@AYAR
@given(st.lists(st.integers(min_value=-(2**40), max_value=2**40), min_size=1, max_size=60))
def test_downcast_tamsayiyi_birebir_koruyor(degerler: list[int]):
    """OZELLIK: tamsayi indirgemesi ASLA deger degistirmez."""
    seri = pd.Series(degerler, dtype="int64")
    inen = downcast_numeric(seri)
    assert np.array_equal(seri.to_numpy(), inen.to_numpy())


# --------------------------------------------------------------------------
# Fold'lar -- dogrulama semasinin degismez kurallari
# --------------------------------------------------------------------------


@AYAR
@given(
    n_gun=st.integers(min_value=40, max_value=300),
    n_varlik=st.integers(min_value=1, max_value=6),
    n_splits=st.integers(min_value=1, max_value=4),
    ambargo_gun=st.integers(min_value=0, max_value=20),
    pencere_gun=st.integers(min_value=3, max_value=40),
)
def test_foldlar_asla_cakismaz(n_gun, n_varlik, n_splits, ambargo_gun, pencere_gun):
    """OZELLIK: hicbir satir ayni fold'da hem train hem valid olamaz.

    Bu, sizintinin en dogrudan bicimi. Her gecerli parametre kombinasyonu
    icin gecerli olmali.
    """
    gunler = pd.date_range("2025-01-01", periods=n_gun, freq="D")
    zaman = pd.Series(np.repeat(gunler, n_varlik))

    try:
        folds = purged_time_series_split(
            zaman,
            embargo=pd.Timedelta(days=ambargo_gun),
            n_splits=n_splits,
            test_span=pd.Timedelta(days=pencere_gun),
            verbose=False,
        )
    except ValueError:
        # Ambargo/pencere veri araligina sigmiyor -- ACIK hata, kabul.
        return

    assert_folds_align(len(zaman), folds)
    for train_idx, valid_idx in folds:
        assert np.intersect1d(train_idx, valid_idx).size == 0


@AYAR
@given(
    n_gun=st.integers(min_value=60, max_value=300),
    n_varlik=st.integers(min_value=1, max_value=6),
    n_splits=st.integers(min_value=1, max_value=3),
    ambargo_gun=st.integers(min_value=1, max_value=15),
    pencere_gun=st.integers(min_value=5, max_value=30),
)
def test_ambargo_her_zaman_korunuyor(n_gun, n_varlik, n_splits, ambargo_gun, pencere_gun):
    """OZELLIK: train'in son ani ile valid'in ilk ani arasinda ambargo KADAR bosluk."""
    gunler = pd.date_range("2025-01-01", periods=n_gun, freq="D")
    zaman = pd.Series(np.repeat(gunler, n_varlik))
    ambargo = pd.Timedelta(days=ambargo_gun)

    try:
        folds = purged_time_series_split(
            zaman,
            embargo=ambargo,
            n_splits=n_splits,
            test_span=pd.Timedelta(days=pencere_gun),
            verbose=False,
        )
    except ValueError:
        return

    degerler = zaman.to_numpy()
    for train_idx, valid_idx in folds:
        bosluk = pd.Timestamp(degerler[valid_idx].min()) - pd.Timestamp(degerler[train_idx].max())
        assert bosluk > ambargo


@AYAR
@given(
    n_gun=st.integers(min_value=60, max_value=250),
    n_varlik=st.integers(min_value=1, max_value=5),
    pencere_gun=st.integers(min_value=5, max_value=30),
)
def test_foldlar_gecmisten_gelecege_egitiliyor(n_gun, n_varlik, pencere_gun):
    """OZELLIK: train HER ZAMAN valid'den ONCE olmali -- gelecek sizamaz."""
    gunler = pd.date_range("2025-01-01", periods=n_gun, freq="D")
    zaman = pd.Series(np.repeat(gunler, n_varlik))

    try:
        folds = purged_time_series_split(
            zaman,
            embargo=pd.Timedelta(days=1),
            n_splits=2,
            test_span=pd.Timedelta(days=pencere_gun),
            verbose=False,
        )
    except ValueError:
        return

    degerler = zaman.to_numpy()
    for train_idx, valid_idx in folds:
        assert degerler[train_idx].max() < degerler[valid_idx].min()


# --------------------------------------------------------------------------
# Feature sozlesmesi -- her panel icin
# --------------------------------------------------------------------------


@st.composite
def panel(draw):
    """Rastgele bir varlik-zaman paneli uretir.

    Yer adlari KASITLI olarak alfabetik olmayan sirada secilebilir; grup
    kaydirma hatalari ancak boyle gorunur.
    """
    n_gun = draw(st.integers(min_value=3, max_value=40))
    yerler = draw(
        st.lists(
            st.sampled_from(["zeytinburnu", "aliaga", "menemen", "bornova", "efeler"]),
            min_size=1,
            max_size=5,
            unique=True,
        )
    )
    baslangic = draw(st.sampled_from(["2024-01-01", "2025-06-15", "2026-02-01"]))
    gunler = pd.date_range(baslangic, periods=n_gun, freq="D")

    degerler = draw(
        st.lists(
            st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
            min_size=n_gun * len(yerler),
            max_size=n_gun * len(yerler),
        )
    )
    frame = pd.DataFrame(
        {
            "tarih": np.tile(gunler, len(yerler)),
            "yer": np.repeat(yerler, n_gun),
            "hedef": degerler,
        }
    )
    karistir = draw(st.booleans())
    if karistir:
        frame = frame.sample(frac=1.0, random_state=draw(st.integers(0, 999))).reset_index(
            drop=True
        )
    return frame


@AYAR
@given(panel())
def test_takvim_feature_girdiyi_degistirmiyor(frame: pd.DataFrame):
    """OZELLIK: her panel icin girdi frame'i DEGISMEZ."""
    onceki = frame.copy(deep=True)
    add_calendar_features(frame, "tarih")
    pd.testing.assert_frame_equal(frame, onceki)


@AYAR
@given(panel())
def test_lag_satir_sayisini_ve_sirasini_koruyor(frame: pd.DataFrame):
    """OZELLIK: lag ekleme satir SAYISINI ve SIRASINI korur."""
    cikti = add_lag_features(
        frame, "hedef", shifts=[1], time_column="tarih", horizon=1, group_columns=["yer"]
    )
    assert len(cikti) == len(frame)
    assert list(cikti["yer"]) == list(frame["yer"])
    assert list(cikti["tarih"]) == list(frame["tarih"])


@AYAR
@given(panel())
def test_kayan_pencere_dogru_gruptan_hesapliyor(frame: pd.DataFrame):
    """OZELLIK: her grup KENDI gecmisinden hesaplanir -- baskasindan degil.

    Bu, bu depoda bulunan en ciddi P0 hatanin ozellik hali. Ornek tabanli
    test onu ancak grup adlari alfabetik OLMAYAN sirada oldugunda
    yakalayabiliyordu; burada hypothesis o durumu kendisi uretiyor.
    """
    assume(len(frame) > 0)
    cikti = add_rolling_features(
        frame, "hedef", [2], time_column="tarih", horizon=1, group_columns=["yer"]
    )

    for yer in frame["yer"].unique():
        grup = frame[frame.yer == yer].sort_values("tarih")
        beklenen = grup["hedef"].shift(1).rolling(2, min_periods=1).mean().to_numpy()
        gercek = cikti[cikti.yer == yer].sort_values("tarih")["hedef_kayan2_mean"].to_numpy()
        # Tolerans float32'ye gore: kayan pencere ciktisi bellek icin
        # float32'ye indiriliyor, dolayisiyla ~7 anlamli basamak beklenir.
        # rtol=1e-9 istemek KODU degil TESTI yanlis yapar.
        assert np.allclose(beklenen, gercek, equal_nan=True, rtol=1e-6, atol=1e-6), (
            f"'{yer}' kayan ortalamasi baska gruptan geliyor\n"
            f"  beklenen: {beklenen}\n  gercek  : {gercek}"
        )
