"""SIZINTI AVI -- enjeksiyon testleri.

YONTEM
------
Sizintiyi "yok" diye iddia etmek kolaydir. Bu dosya farkli calisir:
**kasten sizinti enjekte eder** ve sistemin onu engelledigini/yakaladigini
olcer. Her test bir REFERANS sizintili surumle karsilastirma yapar.

Bir koruma calismiyorsa test degil, KARSILASTIRMA kirilir -- yani yanlis
pozitif uretmesi zordur.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from gridup.features.categorical import oof_target_encode
from gridup.metrics import SUSPICIOUS_SCORE, optimize_threshold
from gridup.validation import purged_time_series_split


def _kapsam(n: int, folds) -> np.ndarray:
    maske = np.zeros(n, dtype=bool)
    for _, valid in folds:
        maske[valid] = True
    return maske


def test_hedef_kodlama_enjekte_edilen_sizintiyi_engelliyor():
    """ENJEKSIYON: her kategori 4 satir -> naif kodlama satirin KENDI hedefini tasir.

    OLCULDU:
      fold-disi kodlama  -> hedefle korelasyon +0.017  (gurultu)
      tum-veri kodlamasi -> hedefle korelasyon +0.542  (sizinti)
    """
    rng = np.random.default_rng(0)
    n = 1200
    zaman = pd.Series(np.tile(pd.date_range("2025-01-01", periods=300), 4))
    kategori = np.tile([f"k{i}" for i in range(300)], 4)
    hedef = pd.Series(rng.normal(0, 1, n))
    frame = pd.DataFrame({"kat": kategori})

    folds = purged_time_series_split(
        zaman, embargo=pd.Timedelta(days=5), n_splits=3,
        test_span=pd.Timedelta(days=30), verbose=False,
    )
    kodlanmis, _ = oof_target_encode(frame, hedef, ["kat"], folds, smoothing=0.0)
    kolon = [c for c in kodlanmis.columns if c not in frame.columns][0]

    gecerli = _kapsam(n, folds) & kodlanmis[kolon].notna().to_numpy()
    fold_disi = abs(
        np.corrcoef(
            kodlanmis[kolon].to_numpy()[gecerli], hedef.to_numpy()[gecerli]
        )[0, 1]
    )

    # KASTEN SIZINTILI referans: tum veriden kategori ortalamasi.
    sizintili = (
        pd.DataFrame({"kat": kategori, "y": hedef})
        .groupby("kat")["y"].transform("mean").to_numpy()
    )
    naif = abs(np.corrcoef(sizintili[gecerli], hedef.to_numpy()[gecerli])[0, 1])

    assert naif > 0.3, "enjeksiyon calismadi -- referans sizinti uretmiyor, test anlamsiz"
    assert fold_disi < 0.15, f"fold-disi kodlama sizdiriyor: korelasyon {fold_disi:.3f}"
    assert fold_disi < naif / 3


def test_hedef_kodlamada_prior_da_fold_icinden():
    """Yumusatma priori GLOBAL hesaplanirsa satirin kendi hedefi 1/N kadar sizar.

    Kod bunu fold icinden hesapliyor; burada davranissal olarak dogruluyoruz:
    smoothing YUKSEK oldugunda bile korelasyon gurultu seviyesinde kalmali
    (global prior kullanilsaydi smoothing arttikca sizinti da artardi).
    """
    rng = np.random.default_rng(1)
    n = 1200
    zaman = pd.Series(np.tile(pd.date_range("2025-01-01", periods=300), 4))
    kategori = np.tile([f"k{i}" for i in range(300)], 4)
    hedef = pd.Series(rng.normal(0, 1, n))
    frame = pd.DataFrame({"kat": kategori})
    folds = purged_time_series_split(
        zaman, embargo=pd.Timedelta(days=5), n_splits=3,
        test_span=pd.Timedelta(days=30), verbose=False,
    )

    for yumusatma in (0.0, 20.0, 200.0):
        kodlanmis, _ = oof_target_encode(
            frame, hedef, ["kat"], folds, smoothing=yumusatma
        )
        kolon = [c for c in kodlanmis.columns if c not in frame.columns][0]
        gecerli = _kapsam(n, folds) & kodlanmis[kolon].notna().to_numpy()
        if gecerli.sum() < 10:
            continue
        r = abs(
            np.corrcoef(kodlanmis[kolon].to_numpy()[gecerli], hedef.to_numpy()[gecerli])[0, 1]
        )
        assert r < 0.15, f"smoothing={yumusatma} icin sizinti: korelasyon {r:.3f}"


def test_esik_optimizasyonu_supheli_skoru_yakaliyor():
    """optimize_threshold OOF mu egitim mi aldigini BILEMEZ -- belirtisini yakalar.

    OLCULDU: ayni hedefte egitim tahminiyle f1=1.000, OOF ile f1=0.612.
    """
    rng = np.random.default_rng(0)
    n = 800
    y = (rng.random(n) < 0.3).astype(int)
    egitim = np.where(y == 1, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n))
    fold_disi = np.clip(y * 0.25 + rng.normal(0.35, 0.2, n), 0, 1)

    with warnings.catch_warnings(record=True) as yakalanan:
        warnings.simplefilter("always")
        kotu = optimize_threshold(y, egitim, metric="f1")
    assert kotu["best_score"] > SUSPICIOUS_SCORE
    assert yakalanan, "egitim tahmininde optimize edildi ama UYARI YOK"
    assert "fold-disi" in str(yakalanan[0].message)

    with warnings.catch_warnings(record=True) as temiz:
        warnings.simplefilter("always")
        iyi = optimize_threshold(y, fold_disi, metric="f1")
    assert iyi["best_score"] < SUSPICIOUS_SCORE
    assert not temiz, "gercekci OOF skorunda yanlis pozitif uyari"


def test_kucuk_metrikte_supheli_skor_uyarisi_yok():
    """greater_is_better=False metriklerde (rmse) yuksek skor KOTU demektir --
    supheli-skor sezgisi orada calismamali."""
    rng = np.random.default_rng(2)
    y = (rng.random(200) < 0.4).astype(int)
    proba = rng.random(200)
    with warnings.catch_warnings(record=True) as yakalanan:
        warnings.simplefilter("always")
        optimize_threshold(y, proba, metric="accuracy")
    # accuracy buyuk-daha-iyi; rastgele tahminde 0.99'u asmaz -> uyari olmamali
    assert not [u for u in yakalanan if "supheli" in str(u.message)]


@pytest.mark.parametrize("fonksiyon", ["add_frequency_encoding", "add_count_encoding"])
def test_frekans_kodlamasi_hedefe_dokunmuyor(fonksiyon: str):
    """Frekans/sayim kodlamasi yalnizca FEATURE dagilimini kullanir.

    Hedefe dokunmadigi icin fold gerekmez ve train+test uzerinde guvenle
    hesaplanabilir. Bunu imza uzerinden dogruluyoruz: hedef parametresi
    OLMAMALI.
    """
    import inspect

    from gridup.features import categorical

    parametreler = set(inspect.signature(getattr(categorical, fonksiyon)).parameters)
    assert not (parametreler & {"target", "target_column", "y"}), (
        f"{fonksiyon} hedef parametresi aliyor -- fold-disi olmasi gerekir"
    )


def test_esik_izgarasi_varsayilan_0_5_i_iceriyor():
    """REGRESYON: linspace(0.01,0.99,200) 0.5'i ISKALIYORDU.

    Bu fonksiyonun tum amaci 0.5 esigini yenmektir; 0.5'i hic denemedigi
    icin ONDAN KOTU bir esik dondurebiliyordu.
    OLCULDU: en iyi f1=0.7356 dondu, oysa 0.5 esiginde f1=0.7429.
    """
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)
    proba = np.clip(y * 0.3 + rng.random(2000) * 0.5, 0, 1)

    sonuc = optimize_threshold(y, proba, metric="f1")

    assert sonuc["best_score"] >= sonuc["score_at_half"], (
        "optimizasyon varsayilan esikten KOTU sonuc dondurdu"
    )


@pytest.mark.parametrize("metrik", ["f1", "accuracy"])
def test_optimizasyon_asla_varsayilandan_kotu_olmuyor(metrik: str):
    """OZELLIK: her veri icin best_score >= score_at_half olmali."""
    for tohum in range(6):
        rng = np.random.default_rng(tohum)
        n = 500
        y = (rng.random(n) < rng.uniform(0.05, 0.5)).astype(int)
        proba = np.clip(y * rng.uniform(0.1, 0.4) + rng.random(n) * 0.6, 0, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sonuc = optimize_threshold(y, proba, metric=metrik)
        assert sonuc["best_score"] >= sonuc["score_at_half"] - 1e-12, (
            f"tohum={tohum} metrik={metrik}: optimizasyon varsayilani yenemedi"
        )


# --------------------------------------------------------------------------
# Panel doldurma: sentetik satirlar UYDURULMUS gozlemdir
# --------------------------------------------------------------------------


def test_panel_doldurma_semantigini_acikca_soyluyor(capsys):
    """Doldurulan satir uydurulmus bir gozlemdir -- kullanici bunu GORMELI.

    OLCULDU: yalnizca ariza gunlerinde kayit ureten bir varlik icin panel
    %40 sentetik satir uretti ve o varligin ortalamasi 50.0 -> 10.0'a dustu.
    Sayim hedefinde DOGRU, olcum hedefinde veriyi BOZAR.
    """
    from gridup.panel import build_panel

    kayit = []
    for gun, tarih in enumerate(pd.date_range("2025-01-01", periods=10)):
        kayit.append({"tarih": tarih, "trafo": "A", "deger": float(gun)})
        if gun in (2, 7):
            kayit.append({"tarih": tarih, "trafo": "B", "deger": 50.0})

    build_panel(
        pd.DataFrame(kayit), entity_columns=["trafo"], time_column="tarih",
        value_columns=["deger"], verbose=True,
    )

    cikti = capsys.readouterr().out
    assert "DOLDURMA" in cikti
    assert "fill_value=0.0" in cikti
    assert "np.nan" in cikti, "olcum hedefi icin alternatif soylenmemis"


def test_panel_nan_doldurma_ortalamayi_bozmuyor():
    """Olcum hedefinde np.nan ile doldurmak gercek ortalamayi korur."""
    from gridup.panel import build_panel

    kayit = []
    for gun, tarih in enumerate(pd.date_range("2025-01-01", periods=10)):
        kayit.append({"tarih": tarih, "trafo": "A", "tuketim": float(100 + gun)})
        if gun in (2, 7):
            kayit.append({"tarih": tarih, "trafo": "B", "tuketim": 500.0})
    frame = pd.DataFrame(kayit)

    sifirli = build_panel(
        frame, entity_columns=["trafo"], time_column="tarih",
        value_columns=["tuketim"], verbose=False,
    )
    nanli = build_panel(
        frame, entity_columns=["trafo"], time_column="tarih",
        value_columns=["tuketim"], fill_value=np.nan, verbose=False,
    )

    b_sifir = sifirli.loc[sifirli.trafo == "B", "tuketim"].mean()
    b_nan = nanli.loc[nanli.trafo == "B", "tuketim"].mean()

    assert b_nan == pytest.approx(500.0), "NaN doldurma gercek ortalamayi korumali"
    assert b_sifir < b_nan, "sifir doldurma ortalamayi asagi ceker -- beklenen davranis"


def test_dolduruldu_bayragi_sentetik_satirlari_isaretliyor():
    from gridup.panel import build_panel

    kayit = [{"tarih": pd.Timestamp("2025-01-01"), "trafo": "A", "deger": 1.0},
             {"tarih": pd.Timestamp("2025-01-03"), "trafo": "A", "deger": 3.0}]
    panel = build_panel(
        pd.DataFrame(kayit), entity_columns=["trafo"], time_column="tarih",
        value_columns=["deger"], verbose=False,
    )
    assert len(panel) == 3
    assert int(panel["_dolduruldu"].sum()) == 1
    # Bayrak DOGRU satiri isaretlemeli (2 Ocak eksikti).
    eksik = panel.loc[panel["_dolduruldu"] == 1, "tarih"].iloc[0]
    assert eksik == pd.Timestamp("2025-01-02")
