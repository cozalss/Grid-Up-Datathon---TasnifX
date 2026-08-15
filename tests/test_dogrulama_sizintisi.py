"""SIZINTI AVI TUR 4 -- CV semasinin kendisi (``validation.py``).

NEDEN AYRI DOSYA
----------------
Sizinti avinin en kritik yuzeyi model degil, DOGRULAMA SEMASIDIR. Bir feature
sizintisi tek bir kolonu bozar; bir sema sizintisi TUM skorlari anlamsiz kilar
ve fark edilmesi haftalar surer.

Buradaki her test, ``t4_repro.py`` ile ONCE-SONRA olculmus somut bir bulguya
karsilik gelir ve olculen sayiyi ISMEN icerir. Sayi degisirse test kirilir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.validation import (
    MAX_UNPARSED_TIME_RATIO,
    adversarial_validation,
    leakage_report,
    parse_time_series,
    purged_time_series_split,
    suggest_scheme,
)

# --------------------------------------------------------------------------
# B33 -- Turkce "gg.aa.yyyy" tarih kolonu
# --------------------------------------------------------------------------


def test_tr_gun_once_tarih_fold_kronolojisini_bozmuyor():
    """OLCULDU: duz ``pd.to_datetime`` 366 gunun 222'sini NaT yapiyor, 354'unu
    ise YANLIS tarihe cozuyordu (NaT + gun/ay takasi).

    Hayatta kalan azinlik siralamayi yonetiyor, NaT'lar sona gidiyor ve olculen
    sonuc soydu:

        fold: train_son=2024-12-06  valid_ilk=2024-01-09
              -> 73 train satiri GERCEK takvimde valid'in GELECEGINDE

    Ambargo 30 gun olmasina ragmen hicbir sey yapmiyordu.
    """
    gunler = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    tr_metin = pd.Series([g.strftime("%d.%m.%Y") for g in gunler])

    # Once tuzagin hala orada oldugunu dogrula -- pandas surumu degisirse bu
    # test yanlis nedenle gecmesin.
    ham = pd.to_datetime(tr_metin, errors="coerce")
    assert ham.isna().sum() == 222
    assert int((ham.dt.date != pd.Series(gunler).dt.date).sum()) == 354

    folds = purged_time_series_split(
        tr_metin, embargo=pd.Timedelta(days=30), n_splits=3, verbose=False
    )
    ihlal = sum(
        int((gunler[train_idx] >= gunler[valid_idx].min()).sum())
        for train_idx, valid_idx in folds
    )
    assert ihlal == 0, f"{ihlal} train satiri gercek takvimde valid'in gelecegine bakiyor"


def test_iso_tarih_dayfirst_ile_bozulmuyor():
    """pandas 3.0.3'te ``dayfirst=True`` ISO tarihleri de ceviriyor (olculdu):

        pd.to_datetime("2024-01-02", format="mixed", dayfirst=True) -> 2024-02-01

    Yani "olmazsa dayfirst dene" mantigi ISO veriyi SESSIZCE bozar. Bicimi
    veriden kanitliyoruz; bu test o korumanin bekcisidir.
    """
    iso = pd.Series(["2024-01-02", "2024-03-04", "2024-11-30"])
    cozulen = parse_time_series(iso)
    assert list(cozulen.dt.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-03-04", "2024-11-30"]


def test_belirsiz_tarih_bicimi_tahmin_edilmiyor_hata_firlatiyor():
    """Her iki bilesen de <=12 ise gun-once mu ay-once mi KANITLANAMAZ.

    Tahmin etmek yerine hata firlatiyoruz: sessizce yanlis kronoloji, bu
    modulun onlemek icin var oldugu seyin ta kendisi.
    """
    belirsiz = pd.Series(["01.02.2024", "03.04.2024", "05.06.2024"] * 20)
    with pytest.raises(ValueError, match="BELIRSIZ"):
        parse_time_series(belirsiz)


def test_gun_once_kanitlandiginda_dogru_cozuluyor():
    """Tek bir gun>12 kaydi bicimi KANITLAR."""
    kanitli = pd.Series(["01.02.2024", "25.03.2024", "05.06.2024"])
    cozulen = parse_time_series(kanitli)
    assert list(cozulen.dt.strftime("%Y-%m-%d")) == ["2024-02-01", "2024-03-25", "2024-06-05"]


# --------------------------------------------------------------------------
# B38 -- kismi NaT
# --------------------------------------------------------------------------


def test_ayristirilamayan_tarihler_son_foldun_valid_setine_yigilmiyor():
    """OLCULDU: 5 gecersiz tarihin 5'i de SON fold'un valid setine gidiyordu.

    ``np.argsort`` NaT'lari dizinin sonuna atar ve pencere onlari gercek
    tarihmis gibi kullanir -- tek bir uyari cikmadan.
    """
    tarihler = [f"2024-01-{i:02d}" for i in range(1, 29)] * 7 + ["GECERSIZ"]
    seri = pd.Series(tarihler)
    gecersiz_konum = len(seri) - 1
    assert 1 / len(seri) < MAX_UNPARSED_TIME_RATIO  # sinirin altinda, tolere edilir

    folds = purged_time_series_split(
        seri, embargo=pd.Timedelta(days=1), n_splits=2, verbose=False
    )
    kullanilan: set[int] = set()
    for train_idx, valid_idx in folds:
        kullanilan |= set(train_idx.tolist()) | set(valid_idx.tolist())
    assert gecersiz_konum not in kullanilan


def test_asiri_ayristirilamayan_tarih_hata_firlatiyor():
    """%2 sinirinin ustunde sessiz kalmak yerine duruyoruz (olculdu: %15.2)."""
    seri = pd.Series([f"2024-01-{i:02d}" for i in range(1, 29)] + ["GECERSIZ"] * 5)
    with pytest.raises(ValueError, match="ayristirilamadi"):
        purged_time_series_split(seri, embargo=pd.Timedelta(days=1), n_splits=2)


# --------------------------------------------------------------------------
# B37 -- negatif ambargo
# --------------------------------------------------------------------------


def test_negatif_ambargo_reddediliyor():
    """OLCULDU: embargo=-40 gun kabul ediliyordu ve train'in son ani valid'in
    ilk anindan 39 gun SONRA oluyordu -- dogrudan sizinti, sessizce."""
    zaman = pd.Series(pd.date_range("2024-01-01", periods=300, freq="D"))
    with pytest.raises(ValueError, match="negatif olamaz"):
        purged_time_series_split(zaman, embargo=pd.Timedelta(days=-40), n_splits=3)


# --------------------------------------------------------------------------
# B46 -- grup kolonu sinir hatasi
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tekrar", [2, 3, 5])
def test_her_varlik_tam_iki_kez_gorulse_bile_grup_kolonu_bulunuyor(tekrar: int):
    """OLCULDU: varlik basina TAM 2 satir -> sema='KFold', grup=None.

    Sebep sinir kosuluydu: 2 tekrarda unique/row_count TAM 0.5 ve kod
    ``< 0.5`` istiyordu. Panel veri 2 donemlik bir pencereye kirpildiginda
    (yarismada siradan bir sey) grup sizintisi geri geliyordu.
    """
    n_ilce = 20
    rng = np.random.default_rng(0)
    cerceve = pd.DataFrame(
        {
            "ilce": np.tile(np.arange(n_ilce), tekrar),
            "x": rng.normal(size=n_ilce * tekrar),
            "hedef": rng.normal(size=n_ilce * tekrar),
        }
    )
    oneri = suggest_scheme(cerceve, target="hedef")
    assert oneri.group_column == "ilce"
    assert oneri.scheme == "GroupKFold"


# --------------------------------------------------------------------------
# B32 -- leakage_report zaman kolonunu kendisi bulmali
# --------------------------------------------------------------------------


def test_leakage_report_time_column_olmadan_da_zaman_ortusmesini_buluyor():
    """OLCULDU: time_column'suz cagri '0 kritik', ayni veri time_column ile
    '1 kritik' veriyordu. README quickstart tam bu sekilde cagiriyordu --
    raporun en agir bulgusu bir arguman atlandigi icin kayboluyordu."""
    rng = np.random.default_rng(0)
    train = pd.DataFrame(
        {
            "tarih": pd.date_range("2024-01-01", periods=200, freq="D"),
            "x": rng.normal(size=200),
            "hedef": rng.normal(size=200),
        }
    )
    test = pd.DataFrame(
        {
            "tarih": pd.date_range("2024-03-01", periods=100, freq="D"),
            "x": rng.normal(size=100),
        }
    )
    rapor = leakage_report(train, "hedef", test=test)
    assert any("Zaman ortusmesi" in bulgu for bulgu in rapor["critical"])


# --------------------------------------------------------------------------
# B36 / B44 -- korelasyon kontrolunun iki korlugu
# --------------------------------------------------------------------------


def test_hedefin_monoton_donusumu_yakalaniyor():
    """OLCULDU: log1p(hedef) -> Pearson 0.8841 (esigin ALTINDA), Spearman 1.0000.

    Hedefin tersinir monoton donusumu sizintinin en saf halidir ama Pearson
    onu goremez.
    """
    rng = np.random.default_rng(0)
    hedef = rng.gamma(2.0, 50.0, size=400)
    train = pd.DataFrame({"hedef": hedef, "log_hedef": np.log1p(hedef)})

    pearson = abs(train["log_hedef"].corr(train["hedef"]))
    assert pearson < 0.95, "tuzak kayboldu: Pearson zaten esigi asiyor"

    rapor = leakage_report(train, "hedef")
    assert any("log_hedef" in bulgu for bulgu in rapor["critical"])


def test_hedeften_turemis_metin_kolonu_yakalaniyor():
    """OLCULDU: metin kolonlari TAMAMEN atlaniyordu; hedefi kovalara bolen bir
    kolon 'tertemiz' raporlaniyordu (0 kritik)."""
    rng = np.random.default_rng(0)
    hedef = rng.gamma(2.0, 50.0, size=400)
    train = pd.DataFrame(
        {
            "hedef": hedef,
            "hedef_metin": [f"kova_{int(v // 10)}" for v in hedef],
            "gurultu": rng.normal(size=400),
        }
    )
    rapor = leakage_report(train, "hedef")
    assert any("hedef_metin" in bulgu for bulgu in rapor["critical"])
    assert not any("gurultu" in bulgu for bulgu in rapor["critical"])


def test_masum_kategorik_kolon_yanlis_alarm_uretmiyor():
    """Yanlis pozitif korumasi: hedefle ilgisiz bir kategorik kolon temiz kalmali."""
    rng = np.random.default_rng(1)
    train = pd.DataFrame(
        {
            "hedef": rng.normal(size=400),
            "ilce": rng.choice(["konak", "bornova", "cesme"], size=400),
        }
    )
    rapor = leakage_report(train, "hedef")
    assert not rapor["critical"], rapor["critical"]


# --------------------------------------------------------------------------
# B43 / B40 / B34 / B35 / B42 -- adversarial_validation
# --------------------------------------------------------------------------


def test_adversarial_validation_ham_datetime_kolonuyla_cokmuyor():
    """OLCULDU: DTypePromotionError ile coküyordu.

    Oysa train/test'i en cok ayiran kolon tam olarak tarihtir -- cokmek,
    fonksiyonun isini hic yapmamasi demekti. Belgelenen gun-1 cagrisi buydu.
    """
    rng = np.random.default_rng(0)
    train = pd.DataFrame(
        {"tarih": pd.date_range("2024-01-01", periods=300, freq="D"),
         "x": rng.normal(size=300)}
    )
    test = pd.DataFrame(
        {"tarih": pd.date_range("2024-10-27", periods=150, freq="D"),
         "x": rng.normal(size=150)}
    )
    sonuc = adversarial_validation(train, test, n_splits=3)
    assert sonuc["auc"] > 0.9
    assert sonuc["top_features"][0][0] == "tarih"


def test_top_features_gercek_ayiriciyi_acik_ara_one_koyuyor():
    """OLCULDU: varsayilan importance_type='split' gercek ayiriciyi gurultuye
    yalnizca 2.2 kat (896.7 / 413.0) onde gosteriyordu; gain ayni veride
    495 kat (4008.7 / 8.1) fark uretiyor.

    Split "kac kez bolundu"yu sayar, "ne kadar ayirdi"yi degil -- cok kolonlu
    gercek veride suclu gurultunun icinde kaybolur.
    """
    rng = np.random.default_rng(0)
    train = pd.DataFrame(
        {"gercek_ayirici": rng.normal(0, 1, 400), "gurultu": rng.normal(size=400)}
    )
    test = pd.DataFrame(
        {"gercek_ayirici": rng.normal(30, 1, 200), "gurultu": rng.normal(size=200)}
    )
    sonuc = adversarial_validation(train, test, n_splits=3)
    siralama = dict(sonuc["top_features"])
    assert sonuc["top_features"][0][0] == "gercek_ayirici"
    assert siralama["gercek_ayirici"] > 50 * siralama["gurultu"]


def test_auc_bire_yakinken_sample_weights_kullanma_uyarisi_veriyor():
    """OLCULDU: 400 train satiri, agirlik toplami 2.004, etkin ornek
    buyuklugu (Kish) 2.0 -- yani '%0.5'.

    Eski verdict tam o durumda 'sample_weights ile agirliklandir' diyordu;
    tavsiyeye uyan biri modeli 2 satirla egitmis olurdu.
    """
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"x": rng.normal(0, 1, 400)})
    test = pd.DataFrame({"x": rng.normal(50, 1, 200)})
    sonuc = adversarial_validation(train, test, n_splits=3)

    assert sonuc["auc"] > 0.99
    assert sonuc["sample_weight_ess_ratio"] < 0.05
    assert "KULLANMA" in sonuc["verdict"]


def test_ortak_olmayan_feature_columns_sessizce_dusurulmuyor():
    """Kullanici o kolonu bilerek istedi; test'te olmamasi baslibasina bulgudur."""
    rng = np.random.default_rng(0)
    train = pd.DataFrame(
        {"a": rng.normal(size=200), "sadece_train": rng.normal(size=200)}
    )
    test = pd.DataFrame({"a": rng.normal(size=100)})
    sonuc = adversarial_validation(
        train, test, feature_columns=["a", "sadece_train"], n_splits=3
    )
    assert any("sadece_train" in not_ for not_ in sonuc["notes"])
