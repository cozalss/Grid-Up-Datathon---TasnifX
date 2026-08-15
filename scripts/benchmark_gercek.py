"""GERCEK GDZ VERISINDE MODEL KARSILASTIRMA -- hangi recete kazaniyor?

NEDEN BU BETIK VAR
------------------
2023 GDZ Datathon birincisi CatBoost'u MAE kaybiyla kullandi. Ama o secim o
yilin verisinde yapildi; elimizde ayni aileden GERCEK bir kayit var: 68.257
GDZ kesintisi (Izmir+Manisa, 47 ilce, 2021-05..2022-08). Bu betik alti
receteyi AYNI fold'larda, AYNI feature setiyle ve AYNI agac butcesiyle
kosturur -- "hangisi kazanir" sorusu tahminle degil olcumle kapanir.

ADIL KARSILASTIRMA SARTLARI
---------------------------
* Ayni fold'lar : purged_time_series_split(embargo=31g, test_span=31g, 4 bolme)
  -- provayla (real_data_rehearsal.py) birebir ayni sema.
* Ayni feature seti: takvim + tatil + hava + gunes + lag(31/62/93, ufuk=31)
  + frekans + 3. dalga (Hawkes bozunumu 3g/14g + toplu-olay payi, ikisi de
  ufuk=31 kaydirmali). Ayni gunun reason/effectedsubscribers/hourlyloadavg
  kolonlari FEATURE DEGIL (tahmin aninda bilinmez); yalnizca ufuk=31
  kaydirilmis lag'leri mesru. (Ilk provanin MAE=266.60'i bu kurala uymuyordu
  -- ayni gunun effectedsubscribers'ini feature aliyordu; buradaki sayilar o
  yuzden provayla KIYASLANMAZ, kendi hep-sifir baseline'iyla kiyaslanir.)
* Ornek agirligi YOK -- olculmus bir catisma karari (2026-08-15):
  recency_activity_weights tek basina kazandiriyordu (lgb_mae 323.13 ->
  313.63) ama ayni yenilik sinyalini feature olarak tasiyan Hawkes
  bozunumuyla CATISIYOR: bozunum+agirlik birlikte lgb_mae'yi 335.30'a itti
  (bozunum tek basina 309.92, ikisi feature setinde agirliksiz 310.14).
  Yumusak rampa (326.51), yalniz-aktiflik (322.45) ve yalniz-rampa (324.97)
  varyantlari da kurtaramadi. Ders: ayni bilgiyi hem kayip agirligi hem
  feature kanalindan vermek kaybettirir; feature kanali kazandi cunku model
  eski veriyi ATMAK yerine rejime KOSULLANIYOR. Boru hatti duruyor:
  fonksiyonlar ``agirliklar`` alir, fit_two_stage/merdiven sample_weight
  gecirir -- 2026 verisinde yeniden olcmek tek satir.
* Ayni butce: her modele 2000 agac/iterasyon, erken durdurma 100 tur.
  CatBoost'a 5000 vermek toplam koşuyu 25 dk hedefinin uzerine tasiyordu
  (olcek provasi: CatBoost 500 iter/100k satir = 37.6 sn, LightGBM = 4.8 sn).

SKORLAR HEP KAPSANAN (covered) SATIRLARDA
-----------------------------------------
purged bolme ilk donemi hicbir fold'un valid tarafina koymaz; o satirlarin
OOF degeri dolgudur. Harman/stack de zoo.oof_covered desenindeki gibi ortak
maskeyle kurulur -- maskesiz harman skoru %24.5'e kadar sapar (olculdu,
ensemble.py docstring).

KULLANIM
    python scripts/benchmark_gercek.py
Cikti: experiments/benchmark_gercek.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gridup import (  # noqa: E402
    build_panel,
    cross_validate,
    fit_two_stage,
    read_table,
    set_global_seed,
)
from gridup.ensemble import hill_climb_weights, stack_oof  # noqa: E402
from gridup.features import (  # noqa: E402
    add_calendar_features,
    add_event_decay_features,
    add_frequency_encoding,
    add_lag_features,
    add_mass_event_features,
    add_turkish_holiday_features,
)
from gridup.features.solar import add_solar_features  # noqa: E402
from gridup.metrics import (  # noqa: E402
    get_metric,
    inverse_sqrt_transform,
    sqrt_transform_target,
)
from gridup.models import starter_params  # noqa: E402
from gridup.panel import PANEL_FLAG_COLUMN  # noqa: E402
from gridup.turkish import join_key, strip_qualifier  # noqa: E402
from gridup.two_stage import (  # noqa: E402
    calibrate_positive_probability,
    conditional_quantile_from_hurdle,
    fit_conditional_quantile_ladder,
)
from gridup.validation import purged_time_series_split  # noqa: E402

KOK = Path(__file__).resolve().parents[1]
VERI = KOK / "data" / "prior" / "ayna" / "MANISA_IZMIR_PLANSIZ_KESINTILER.csv"
REFERANS = KOK / "data" / "reference" / "ilceler_gdz_adm.parquet"
HAVA = KOK / "data" / "external" / "hava_gunluk.parquet"
CIKTI = KOK / "experiments" / "benchmark_gercek.json"

HEDEF = "kesinti_dk"
ZAMAN = "gun"
GRUP = "ilce_key"
# Ayni gunun bilgisi -- feature OLMAZ, yalnizca ufuk kaydirilmis lag'leri mesru.
AYNI_GUN_KOLONLARI = ("effectedsubscribers", "hourlyloadavg")

#: Ham olay kaydinin TUM kolonlari: hepsi ayni gunun bilgisidir ve hicbiri
#: dogrudan feature olamaz. build_panel bunlari 'first' ile tasidigi icin
#: dolgu satirlarinda NaN kalirlar -- yani NaN desenleri _dolduruldu
#: bayraginin proxy'sidir (id ile olculdu: uyum 1.000000).
HAM_KOLONLAR = frozenset({
    "id", "il", "ilce", "date", "starttime", "endtime", "reason",
    "effectedsubscribers", "hourlyloadavg", "effectedneighbourhoods",
    "distributioncompanyname",
})

UFUK = 31           # test bloğu 31 gun -> lag'ler en az 31 gun geriden gelmeli
LAGLAR = (31, 62, 93)
ORTAK_BUTCE = 2000  # agac/iterasyon -- TUM modellere ayni; adil karsilastirma sarti
ERKEN_DURDURMA = 100
#: Hawkes bozunumunun yari omurleri: 3g = gecen haftanin izleri, 14g = ayin
#: rejimi. Tek basina olculdu: lgb_mae 323.13 -> 309.92 (docs/10 bolum 3).
YARI_OMURLER = (3.0, 14.0)
#: Harman tirmanmasinin kararlilik cezasi (Home Credit 2024 + M5 1.si):
#: objektif = ortalama(fold MAE) + ceza * std(fold MAE). Tek fold'un
#: hediyesiyle parlayan agirlik LB'de geri teper; 0.5 near-free olculdu.
KARARLILIK_CEZASI = 0.5


def panel_kur() -> pd.DataFrame:
    """Olay kaydini gunluk ilce paneline cevirir -- provanin kanitli recetesi."""
    ham = read_table(VERI, verbose=False)

    bas = pd.to_datetime(ham["starttime"], utc=True, format="mixed")
    bit = pd.to_datetime(ham["endtime"], utc=True, format="mixed")
    ham[HEDEF] = (bit - bas).dt.total_seconds() / 60.0
    ham[ZAMAN] = (
        pd.to_datetime(ham["date"], utc=True, format="mixed")
        .dt.tz_convert("Europe/Istanbul")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    ham = ham[ham[HEDEF] >= 0]  # bitis < baslangic olan kayitlar disari

    # 'Koprubasi / Manisa' kurtarmasi -- provada 284 satir kazandirdi.
    ham[GRUP] = ham["ilce"].map(lambda x: join_key(strip_qualifier(str(x))))

    return build_panel(
        ham, entity_columns=[GRUP], time_column=ZAMAN,
        value_columns=[HEDEF, *AYNI_GUN_KOLONLARI],
        verbose=False,
    )


def ozellik_kur(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Feature seti: takvim + tatil + hava + gunes + lag + frekans + 3. dalga.

    Butun donusumler satir sirasini korur (kutuphane sozlesmesi) -- fold
    indeksleri panelle ayni kalir.
    """
    ozellik = panel
    # Lag'ler: hedef VE ayni-gun kolonlarinin ufuk=31 kaydirilmis gecmisi.
    # Ayni gunun degeri sizinti; 31+ gun onceki degeri mesru sinyaldir.
    for kolon in (HEDEF, *AYNI_GUN_KOLONLARI):
        ozellik = add_lag_features(
            ozellik, kolon, LAGLAR,
            time_column=ZAMAN, horizon=UFUK, group_columns=[GRUP],
        )

    # 3. dalga -- iki aile de hedeften turer ama YALNIZCA ufuk=31 kaydirmali
    # yayin yapar (fonksiyonlar horizon<1'i zaten reddeder), yani sizinti
    # duvarinin ARKASINDA kalirlar; HAM_KOLONLAR dus kumesi degismez.
    # Hawkes bozunumu: art arda ariza kumelenir -- tek basina en buyuk
    # olculmus kazanc (lgb_mae 323.13 -> 309.92).
    ozellik = add_event_decay_features(
        ozellik, HEDEF, time_column=ZAMAN, horizon=UFUK,
        group_columns=[GRUP], half_lives=YARI_OMURLER,
    )
    # Toplu-olay payi: firtina gunu ilcelerin buyuk kismi ayni gun kesintili
    # (M5 out-of-stock analogu; tek basina 320.37).
    ozellik = add_mass_event_features(
        ozellik, HEDEF, time_column=ZAMAN, horizon=UFUK, group_columns=[GRUP],
    )

    # include_year=False: test donemi train'den sonra -- yil ekstrapolasyon riski.
    ozellik = add_calendar_features(ozellik, ZAMAN, include_year=False)
    ozellik = add_turkish_holiday_features(ozellik, ZAMAN)

    hava = pd.read_parquet(HAVA)
    oncesi = len(ozellik)
    ozellik = ozellik.merge(
        hava.drop(columns=[c for c in ("konum", "konum_key", "il_key") if c in hava.columns]),
        left_on=[GRUP, ZAMAN], right_on=["ilce_key", "tarih"],
        how="left", validate="many_to_one",
    )
    if len(ozellik) != oncesi:
        raise RuntimeError("hava merge satir sayisini degistirdi -- join anahtari bozuk.")

    # Gunes: saf astronomik geometri (gun uzunlugu, deklinasyon). geometry_only
    # cunku hava zaten OLCULMUS gunes_radyasyon tasiyor; pvlib acik-gokyuzu
    # modeli onunla buyuk olcude ortusur, geometri ise mevsim sinyalini verir.
    ref = pd.read_parquet(REFERANS)
    koordinatlar = {
        satir.ilce_key: (float(satir.lat), float(satir.lon))
        for satir in ref.itertuples()
    }
    ozellik = add_solar_features(
        ozellik, time_column=ZAMAN, location_column=GRUP,
        coordinates=koordinatlar, geometry_only=True,
    )

    ozellik = add_frequency_encoding(ozellik, [GRUP])

    # SIZINTI DUVARI: hedef, HAM OLAY KAYDININ TUM KOLONLARI, panel dolgu
    # bayragi ve anahtar kolonlar feature olamaz. Gerisi sayisal ise feature'dir.
    #
    # NEDEN TUM HAM KOLONLAR (cekismeli denetim yakaladi, olculdu):
    # Ilk surum yalnizca effectedsubscribers/hourlyloadavg'i disliyordu ama
    # 'id' de ham kaydin sayisal bir kolonu ve build_panel onu 'first' ile
    # tasiyip dolgu satirlarinda NaN birakiyor. id'nin NaN deseni boylece
    # _dolduruldu bayraginin BIREBIR kopyasi oluyor:
    #     id NaN orani = 0.3475 = _dolduruldu orani (uyum 1.000000)
    #     y==0 ile uyum 0.9975
    #     lgb_tweedie gain'de id 1./50 (ikincinin ~13 kati)
    #     id cikarilinca lgb_tweedie 260.21 -> 325.54
    # Tek tek kolon dislamak bu sinifa karsi kirilgandir; ham kaydin TAMAMI
    # ayni gunun bilgisidir ve kara listeye toptan girer.
    dus = {HEDEF, ZAMAN, GRUP, PANEL_FLAG_COLUMN, "tarih", *HAM_KOLONLAR}
    kolonlar = [
        c for c in ozellik.columns
        if c not in dus and pd.api.types.is_numeric_dtype(ozellik[c])
    ]
    yasak_kacak = [c for c in kolonlar if c in HAM_KOLONLAR]
    if yasak_kacak:  # pragma: no cover - savunma
        raise RuntimeError(f"Ham kolon feature listesine sizdi: {yasak_kacak}")
    return ozellik, kolonlar


def _butceli(kind: str, params: dict[str, Any]) -> dict[str, Any]:
    """Ortak agac butcesini uygular -- hangi kutuphane olursa olsun."""
    sonuc = dict(params)
    anahtar = "iterations" if kind == "catboost" else "n_estimators"
    sonuc[anahtar] = ORTAK_BUTCE
    return sonuc


def tek_modelleri_kos(
    x: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    agirliklar: np.ndarray | None,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Bes tek modeli ayni fold'larda kosturur; skor + OOF + kapsam dondurur.

    ``agirliklar`` verilirse her modelin egitimine gecirilir (cross_validate
    yalnizca train dilimini olcekler); skorlar agirliksiz OOF uzerinde kalir.
    Kanonik kosu None gecer -- bkz. modul docstring'indeki catisma olcumu.
    """
    tweedie = starter_params("lightgbm", "regression", objective="tweedie")
    tweedie["tweedie_variance_power"] = 1.3
    catboost = starter_params("catboost", "regression", objective="mae")
    catboost["eval_metric"] = "MAE"  # 2023 birinci recetesi: loss=MAE, eval=MAE

    tarifler: dict[str, tuple[str, dict[str, Any]]] = {
        "lgb_l2": ("lightgbm", starter_params("lightgbm", "regression")),
        "lgb_mae": ("lightgbm", starter_params("lightgbm", "regression", objective="mae")),
        "lgb_tweedie": ("lightgbm", tweedie),
        "catboost_mae": ("catboost", catboost),
        "xgb": ("xgboost", starter_params("xgboost", "regression")),
    }

    skorlar: dict[str, dict[str, float]] = {}
    oof: dict[str, np.ndarray] = {}
    kapsam: dict[str, np.ndarray] = {}
    for ad, (kind, params) in tarifler.items():
        print(f"  {ad} kosuyor...")
        sonuc = cross_validate(
            x, y, folds, kind=kind, metric="mae",  # type: ignore[arg-type]
            params=_butceli(kind, params), sample_weight=agirliklar,
            early_stopping_rounds=ERKEN_DURDURMA, verbose=False,
        )
        skorlar[ad] = {
            "mae": float(sonuc.overall_score),
            "fold_std": float(sonuc.fold_std),
            "sure_sn": float(sonuc.elapsed_seconds),
        }
        oof[ad] = sonuc.oof_predictions
        kapsam[ad] = sonuc.oof_covered
        print(f"    mae={sonuc.overall_score:.2f}  fold_std={sonuc.fold_std:.2f}  "
              f"sure={sonuc.elapsed_seconds:.0f} sn")

    # sqrt recetesi -- Rohlik Sales v2'nin 2. ve 3.'sunden BAGIMSIZ cifte kanit:
    # sqrt(y) uzayinda L2, ham MAE'yi VE yerli Tweedie'yi gecti. Karsi kanit da
    # var (Rohlik Orders 3.: log1p CV'yi bozdu) -- yani donusum teoriden
    # okunamaz, burada OLCULUR. Skor HAM uzayda: once geri-kare, sonra MAE.
    print("  lgb_sqrt kosuyor...")
    sonuc = cross_validate(
        x, sqrt_transform_target(y), folds, kind="lightgbm", metric="mae",
        params=_butceli("lightgbm", starter_params("lightgbm", "regression")),
        sample_weight=agirliklar,
        early_stopping_rounds=ERKEN_DURDURMA, verbose=False,
    )
    geri = inverse_sqrt_transform(sonuc.oof_predictions)
    maske = sonuc.oof_covered
    mae_fn, _, _ = get_metric("mae")
    sqrt_mae = float(mae_fn(y[maske], geri[maske]))
    fold_skorlari = []
    for _, valid_idx in folds:
        gecerli = valid_idx[maske[valid_idx]]
        if gecerli.size:
            fold_skorlari.append(float(mae_fn(y[gecerli], geri[gecerli])))
    skorlar["lgb_sqrt"] = {
        "mae": sqrt_mae,
        "fold_std": float(np.std(fold_skorlari)) if fold_skorlari else 0.0,
        "sure_sn": float(sonuc.elapsed_seconds),
    }
    oof["lgb_sqrt"] = geri
    kapsam["lgb_sqrt"] = maske
    print(f"    mae={sqrt_mae:.2f}  fold_std={skorlar['lgb_sqrt']['fold_std']:.2f}  "
          f"sure={sonuc.elapsed_seconds:.0f} sn")
    return skorlar, oof, kapsam


def iki_asama_kos(
    x: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    agirliklar: np.ndarray | None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Iki asamali (hurdle) modeli kosturur; birlesik OOF ve sifir oranini dondurur.

    Sifir orani ~%35 -- fit_two_stage'in kendi dokumani %40 altinda duz
    regresyonun daha iyi olmasini bekler. Bunu VARSAYMIYORUZ, olcuyoruz.
    """
    print("  iki_asama kosuyor...")
    baslangic = time.perf_counter()
    sonuc = fit_two_stage(
        x, y, folds, kind="lightgbm", metric="mae",
        classifier_params=_butceli("lightgbm", starter_params("lightgbm", "binary")),
        regressor_params=_butceli(
            "lightgbm", starter_params("lightgbm", "regression", objective="mae")
        ),
        sample_weight=agirliklar,
        early_stopping_rounds=ERKEN_DURDURMA, verbose=False,
    )
    sure = time.perf_counter() - baslangic

    mae_fn, _, _ = get_metric("mae")
    maske = sonuc.covered()
    birlesik = sonuc.predict_oof(mode="thresholded")
    mae = float(mae_fn(y[maske], birlesik[maske]))
    esik = float(sonuc.best_threshold if sonuc.best_threshold is not None else 0.5)

    # Tek modellerle ayni tanimda fold_std: her fold'un valid dilimindeki MAE.
    fold_skorlari = []
    for _, valid_idx in folds:
        gecerli = valid_idx[maske[valid_idx]]
        if gecerli.size:
            fold_skorlari.append(float(mae_fn(y[gecerli], birlesik[gecerli])))
    fold_std = float(np.std(fold_skorlari)) if fold_skorlari else 0.0

    print(f"    mae={mae:.2f}  fold_std={fold_std:.2f}  sure={sure:.0f} sn  "
          f"(esik={esik:.3f})")
    skor = {"mae": mae, "fold_std": fold_std, "sure_sn": float(sure)}
    return skor, birlesik, maske, float((y == 0).mean()), esik, sonuc.oof_probability


def medyan_kurali_kos(
    x: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    olasilik: np.ndarray,
    maske: np.ndarray,
    agirliklar: np.ndarray | None,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray], dict[str, Any]]:
    """MAE-optimal medyan kurali: kosullu kuantil merdiveni + q* = 1 - 0.5/p.

    NEDEN (2024-2026 arastirma taramasi, #1 oneri)
    ----------------------------------------------
    MAE'nin optimal nokta tahmini kosullu MEDYANDIR; iki asamanin 'expected'
    modu (p*mu) RMSE'nin optimalidir, 'thresholded' modu ise sabit esikli bir
    yaklasiklamadir. Kural kutuphanede zaten vardi
    (``conditional_quantile_from_hurdle``); burada GERCEK veride, ayni fold ve
    butceyle OLCULUYOR.

    Kalibre varyantin sorusu ayri ve net: OOF-ayarli esigin 0.5'ten sapmasi
    (0.606 olculmustu) siniflandiricinin kalibrasyon hatasi mi? Izotonik
    kalibrasyon sonrasi ayni kural daha iyiyse cevap evettir.
    """
    print("  kosullu kuantil merdiveni egitiliyor (11 seviye x 4 fold)...")
    basla = time.perf_counter()
    merdiven = fit_conditional_quantile_ladder(
        x, y, folds,
        params=_butceli("lightgbm", starter_params("lightgbm", "regression")),
        sample_weight=agirliklar,
        early_stopping_rounds=ERKEN_DURDURMA, verbose=False,
    )
    kalibrasyon = calibrate_positive_probability(
        olasilik, y, folds, covered=maske, verbose=False
    )
    mae_fn, _, _ = get_metric("mae")

    skorlar: dict[str, dict[str, float]] = {}
    ooflar: dict[str, np.ndarray] = {}
    secenekler = {
        "iki_asama_medyan": olasilik,
        "iki_asama_medyan_kalibre": kalibrasyon.calibrated,
    }
    sure = time.perf_counter() - basla
    for ad, p in secenekler.items():
        tahmin = conditional_quantile_from_hurdle(p, merdiven, verbose=False)
        mae = float(mae_fn(y[maske], tahmin[maske]))
        fold_skorlari = []
        for _, valid_idx in folds:
            gecerli = valid_idx[maske[valid_idx]]
            if gecerli.size:
                fold_skorlari.append(float(mae_fn(y[gecerli], tahmin[gecerli])))
        # Merdiven iki varyantin ORTAK maliyetidir; sure ikisine de yazilir.
        skorlar[ad] = {
            "mae": mae,
            "fold_std": float(np.std(fold_skorlari)) if fold_skorlari else 0.0,
            "sure_sn": float(sure),
        }
        ooflar[ad] = tahmin
        print(f"    {ad}: mae={mae:.2f}  fold_std={skorlar[ad]['fold_std']:.2f}")

    kalibrasyon_ozeti = {
        "brier_once": kalibrasyon.brier_before,
        "brier_sonra": kalibrasyon.brier_after,
        "iyilesti": bool(kalibrasyon.improved),
    }
    print(f"    kalibrasyon: Brier {kalibrasyon.brier_before:.4f} -> "
          f"{kalibrasyon.brier_after:.4f} "
          f"({'iyilesti' if kalibrasyon.improved else 'IYILESMEDI'})")
    return skorlar, ooflar, kalibrasyon_ozeti


def harman_ve_stack(
    modeller: dict[str, dict[str, float]],
    oof: dict[str, np.ndarray],
    kapsam: dict[str, np.ndarray],
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, Any], float]:
    """TUM uyelerin hill-climb harmani + ridge stacking.

    Iki adim da ORTAK kapsam maskesiyle calisir (zoo.oof_covered deseni):
    kapsanmayan satirlarin OOF'u dolgudur, skora girerse sayiyi bozar.

    NEDEN "EN IYI 3" DEGIL DE HEPSI (olculdu, 2026-08-15)
    -----------------------------------------------------
    Onceki surum en dusuk MAE'li 3 uyeyi secip harmanliyordu. Medyan-kurali
    varyantlari eklenince "en iyi 3", birbirinin kopyasi iki medyan cikti
    (ayni merdiven, neredeyse ayni olasilik) + lgb_sqrt oldu ve harman
    311.83'e GERILEDI -- eski cesitli uclu (iki_asama + catboost_mae +
    lgb_mae) 308.27 veriyordu. Ders: harmani uye KALITESI degil hata
    CESITLILIGI tasir. Simdi tum uyeler hill-climb'e girer; ise yaramayanlara
    zaten ~0 agirlik verir, agirligi 0 cikanlar rapordan dusulur.

    Tirmanma KARARLILIK CEZALI kosulur (stability_penalty=0.5): objektif tum-
    OOF MAE degil, fold MAE'lerinin ortalama + 0.5*std'sidir -- tek fold'un
    hediyesiyle parlayan agirlik burada kazanamaz.
    """
    uyeler = sorted(modeller, key=lambda ad: modeller[ad]["mae"])

    ortak_maske = np.ones(len(y), dtype=bool)
    for ad in uyeler:
        ortak_maske &= kapsam[ad]
    indeks = np.flatnonzero(ortak_maske)

    # Kararlilik cezasi fold-bazli skor ister: her fold'un valid indeksleri
    # ortak kapsama indirgenir ve MASKELI dizinin konumsal indeksine cevrilir.
    # indeks sirali ve kapsanan_valid onun alt kumesi oldugu icin searchsorted
    # birebir konumu verir.
    dilimler = []
    for _, valid_idx in folds:
        kapsanan_valid = valid_idx[ortak_maske[valid_idx]]
        if kapsanan_valid.size:
            dilimler.append(np.searchsorted(indeks, kapsanan_valid))

    maskeli = {ad: oof[ad][indeks] for ad in uyeler}
    agirliklar = hill_climb_weights(
        maskeli, y[indeks], metric="mae",
        covered=np.ones(indeks.size, dtype=bool),  # onceden maskelendi
        stability_penalty=KARARLILIK_CEZASI, fold_slices=dilimler,
        verbose=False,
    )
    mae_fn, _, _ = get_metric("mae")
    harman_tahmin = np.zeros(indeks.size)
    for ad, agirlik in agirliklar.items():
        harman_tahmin += agirlik * maskeli[ad]
    # Agirligi 0 cikanlar harmana katki vermiyor -- raporda yer almasinlar.
    secilen = {ad: w for ad, w in agirliklar.items() if w > 0}
    harman = {
        "mae": float(mae_fn(y[indeks], harman_tahmin)),
        "uyeler": sorted(secilen, key=lambda ad: -secilen[ad]),
        "agirliklar": {ad: round(float(w), 4) for ad, w in secilen.items()},
    }
    print(f"  harman: mae={harman['mae']:.2f}  uyeler={harman['uyeler']}")

    stack = stack_oof(
        {ad: oof[ad] for ad in uyeler}, y, folds,
        base_covered=ortak_maske, meta="ridge", metric="mae", verbose=False,
    )
    stack_mae = float(stack["score"])
    print(f"  stack : mae={stack_mae:.2f}")
    return harman, stack_mae


def recete_yaz(
    modeller: dict[str, dict[str, float]],
    harman: dict[str, Any],
    stack_mae: float,
    kazanan: str,
    sifir_baseline: float,
    sifir_orani: float,
    esik: float,
) -> str:
    """Gun-1 karari: olculen sayilardan tek paragraf -- tahmin yok.

    DIKKAT: iki_asama kendisiyle DEGIL, duz regresyonlarin en iyisiyle
    kiyaslanir -- ilk surum en iyi tek model iki_asama olunca kendi kendine
    'gecmiyor' diyordu (olculdu ve duzeltildi).
    """
    tek = min(modeller, key=lambda ad: modeller[ad]["mae"])
    tek_mae = modeller[tek]["mae"]
    # "duz" = hurdle olmayan tek modeller; iki_asama TUM varyantlariyla haric.
    duzler = {
        ad: bilgi["mae"] for ad, bilgi in modeller.items()
        if not ad.startswith("iki_asama")
    }
    duz_ad = min(duzler, key=lambda ad: duzler[ad])
    sira = sorted(modeller, key=lambda ad: modeller[ad]["mae"])
    cat_sira = sira.index("catboost_mae") + 1
    cat_mae = modeller["catboost_mae"]["mae"]
    iki_mae = modeller["iki_asama"]["mae"]
    cat_hukmu = (
        "recete dogrudan tasinabilir" if cat_sira == 1
        else "recete iyi bir baslangic ama tek basina kazanmiyor, kor kopyalanmamali"
    )
    iki_kiyas = (
        "geciyor -- dokumanin %40-alti-sifir beklentisinin aksine"
        if iki_mae < duzler[duz_ad]
        else "gecmiyor; dokumanin %40-alti-sifir uyarisiyla tutarli"
    )
    iki_hukmu = f"duz modellerin en iyisini ({duz_ad} {duzler[duz_ad]:.2f}) {iki_kiyas}"
    esik_notu = (
        f" (optimum esik {esik:.2f} tabana dayandi: siniflandirici fiilen devre "
        "disi, kazanc MAE ile egitilmis buyukluk modelinden geliyor)"
        if esik <= 0.02 else f" (optimum esik {esik:.2f})"
    )
    medyan_mae = modeller["iki_asama_medyan"]["mae"]
    kalibre_mae = modeller["iki_asama_medyan_kalibre"]["mae"]
    sqrt_mae = modeller["lgb_sqrt"]["mae"]
    # Harmanin agirlik verdigi DIGER uyeler onerilir; kazanan zaten baslangic.
    ek_uyeler = [ad for ad in harman["uyeler"] if ad != tek]
    ek_metin = " ve ".join(ek_uyeler) if ek_uyeler else "catboost_mae ve lgb_tweedie"
    return (
        f"Veri gununde ilk kosulacak tek model {tek} (MAE {tek_mae:.2f}; hep-sifir "
        f"baseline {sifir_baseline:.2f}, sifir orani %{sifir_orani * 100:.1f}). "
        f"2023 birincisinin recetesi catboost_mae bu veride MAE {cat_mae:.2f} ile "
        f"{cat_sira}. sirada -- {cat_hukmu}. Iki asamali model (MAE {iki_mae:.2f}) "
        f"{iki_hukmu}{esik_notu}. MAE-optimal medyan kurali {medyan_mae:.2f}, "
        f"kalibre olasilikla {kalibre_mae:.2f}; sqrt donusumu (Rohlik recetesi) "
        f"{sqrt_mae:.2f}. Tum uyeler uzerinde hill-climb harmani MAE "
        f"{harman['mae']:.2f} (agirlik alan uyeler: {', '.join(harman['uyeler'])}), "
        f"ridge stacking {stack_mae:.2f} (purged semada ilk "
        f"fold'lar meta-egitim kapsami disinda kaliyor); genel kazanan '{kazanan}'. "
        f"Oneri: gun-1'de {tek} ile basla, ayni fold'larda {ek_metin} uyelerini "
        f"ekleyip hill-climb harmanini kur; stacking'e fold kapsami "
        f"genislemeden donme."
    )


def main() -> int:
    if not VERI.exists():
        print(f"HATA: {VERI} yok. Indir: kaggle datasets download -d "
              "tmlalper/manisa-izmir-plansiz-elektrik-kesintileri --unzip")
        return 1

    set_global_seed(42)
    baslangic = time.perf_counter()

    print("1/4 panel + feature kuruluyor...")
    panel = panel_kur()
    ozellik, kolonlar = ozellik_kur(panel)
    y = ozellik[HEDEF].to_numpy()
    print(f"  panel {panel.shape[0]:,} satir, {len(kolonlar)} sayisal feature")

    # ORNEK AGIRLIGI KARARI (olculdu, modul docstring'inde dokum): Hawkes
    # bozunumu feature setine girince recency_activity_weights ZARARLI --
    # ayni yenilik sinyali iki kanaldan verilince lgb_mae 310.14 -> 335.30.
    # Kanonik kosu agirliksiz; 2026 verisinde yeniden olcmek icin buraya
    # recency_activity_weights(ozellik, HEDEF, time_column=ZAMAN,
    # group_columns=[GRUP]) gecir.
    agirliklar = None

    folds = purged_time_series_split(
        ozellik[ZAMAN], embargo=pd.Timedelta(days=UFUK),
        n_splits=4, test_span=pd.Timedelta(days=UFUK), verbose=False,
    )

    print("2/4 tek modeller (ortak butce: "
          f"{ORTAK_BUTCE} agac, erken durdurma {ERKEN_DURDURMA})...")
    modeller, oof, kapsam = tek_modelleri_kos(ozellik[kolonlar], y, folds, agirliklar)

    print("3/4 iki asamali model...")
    iki_skor, iki_oof, iki_maske, sifir_orani, esik, olasilik = iki_asama_kos(
        ozellik[kolonlar], y, folds, agirliklar
    )
    modeller["iki_asama"] = iki_skor
    oof["iki_asama"] = iki_oof
    kapsam["iki_asama"] = iki_maske

    print("3b/4 MAE-optimal medyan kurali (ham + kalibre olasilik)...")
    medyan_skorlar, medyan_oof, kalibrasyon_ozeti = medyan_kurali_kos(
        ozellik[kolonlar], y, folds, olasilik, iki_maske, agirliklar
    )
    modeller.update(medyan_skorlar)
    for ad, tahminler in medyan_oof.items():
        oof[ad] = tahminler
        kapsam[ad] = iki_maske

    print("4/4 harman + stacking...")
    harman, stack_mae = harman_ve_stack(modeller, oof, kapsam, y, folds)

    # Baseline ve kazanan -- hepsi ayni kapsanan satir kumesinde.
    ortak_maske = kapsam["lgb_mae"]
    sifir_baseline = float(np.abs(y[ortak_maske]).mean())
    adaylar = {ad: bilgi["mae"] for ad, bilgi in modeller.items()}
    adaylar["harman"] = harman["mae"]
    adaylar["stack"] = stack_mae
    kazanan = min(adaylar, key=lambda ad: adaylar[ad])

    sonuc = {
        "modeller": modeller,
        "harman": harman,
        "stack_mae": stack_mae,
        "kazanan": kazanan,
        "sifir_baseline": sifir_baseline,
        "sifir_orani": sifir_orani,
        # Feature listesi JSON'a yazilir ki test 'yasak kolon sizdi mi'
        # sorusunu MAKINE ile sorabilsin -- denetim, MAE esigine bakan
        # testin id sizintisini ayirt edemedigini gosterdi.
        "feature_kolonlari": list(kolonlar),
        "kalibrasyon": kalibrasyon_ozeti,
        "gun1_recetesi": recete_yaz(
            modeller, harman, stack_mae, kazanan, sifir_baseline, sifir_orani, esik
        ),
    }
    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    CIKTI.write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nKAZANAN: {kazanan}  (mae={adaylar[kazanan]:.2f}, "
          f"hep-sifir {sifir_baseline:.2f}, sifir orani %{sifir_orani * 100:.1f})")
    print(f"Sonuc: {CIKTI}")
    print(f"Toplam sure: {time.perf_counter() - baslangic:.0f} sn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
