"""GERCEK GDZ VERISINDE MODEL KARSILASTIRMA -- kanit kazanan icin yeterli mi?

NEDEN BU BETIK VAR
------------------
2023 GDZ Datathon birincisi CatBoost'u MAE kaybiyla kullandi. Ama o secim o
yilin verisinde yapildi; elimizde ayni aileden GERCEK bir kayit var: 68.257
GDZ kesintisi (Izmir+Manisa, 47 ilce, 2021-05..2022-08). Bu betik alti
receteyi AYNI fold'larda, AYNI feature setiyle ve AYNI agac butcesiyle
kosturur. Ayni OOF'taki siralama aday uretir; "kazanan" karari ancak en az 6
bagimsiz, eslestirilmis outer anchor kanitiyla kapanir.

ADIL KARSILASTIRMA SARTLARI
---------------------------
* Ayni fold'lar : purged_time_series_split(embargo=31g, test_span=31g, 4 bolme)
  -- provayla (real_data_rehearsal.py) birebir ayni sema. Bunlar IC OOF
  siralamasi icindir; sayi ve bagimsizlik olarak kazanan kaniti degildir.
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

import argparse
import json
import sys
import time
from collections.abc import Sequence
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
from gridup.ensemble import hill_climb_weights, median_blend, stack_oof  # noqa: E402
from gridup.evaluation import (  # noqa: E402
    OuterAnchor,
    OuterEvidence,
    paired_model_decision,
)
from gridup.features import (  # noqa: E402
    add_calendar_features,
    add_event_decay_features,
    add_expanding_features,
    add_lag_features,
    add_mass_event_features,
    add_turkish_holiday_features,
)
from gridup.features.solar import add_solar_features  # noqa: E402
from gridup.io_utils import atomic_write_bytes  # noqa: E402
from gridup.metrics import get_metric  # noqa: E402
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
HAM_KOLONLAR = frozenset(
    {
        "id",
        "il",
        "ilce",
        "date",
        "starttime",
        "endtime",
        "reason",
        "effectedsubscribers",
        "hourlyloadavg",
        "effectedneighbourhoods",
        "distributioncompanyname",
    }
)

UFUK = 31  # test bloğu 31 gun -> lag'ler en az 31 gun geriden gelmeli
SHIFT_OFSETLERI = (31, 62, 93)
ORTAK_BUTCE = 2000  # agac/iterasyon -- TUM modellere ayni; adil karsilastirma sarti
ERKEN_DURDURMA = 100
#: Hawkes bozunumunun yari omurleri: 3g = gecen haftanin izleri, 14g = ayin
#: rejimi. Tek basina olculdu: lgb_mae 323.13 -> 309.92 (docs/10 bolum 3).
YARI_OMURLER = (3.0, 14.0)
#: Harman tirmanmasinin kararlilik cezasi (Home Credit 2024 + M5 1.si):
#: objektif = ortalama(fold MAE) + ceza * std(fold MAE). Tek fold'un
#: hediyesiyle parlayan agirlik LB'de geri teper; 0.5 near-free olculdu.
KARARLILIK_CEZASI = 0.5
MIN_OUTER_ANCHORS = 6


def _sonucsuz_karar(
    apparent_oof_best: str,
    reason: str,
    *,
    n_anchors: int = 0,
    pairwise_decisions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Eksik/gecersiz kanitta bilimsel karar kapisini kapali tutar."""
    return {
        "apparent_oof_best": apparent_oof_best,
        "winner": None,
        "statistically_conclusive": False,
        "decision_reason": reason,
        "n_anchors": n_anchors,
        "required_anchors": MIN_OUTER_ANCHORS,
        "pairwise_decisions": pairwise_decisions or [],
    }


def _outer_anchor_kanitini_dogrula(
    candidates: set[str],
    outer_anchor_scores: dict[str, Sequence[float]],
) -> tuple[dict[str, np.ndarray] | None, int, str | None]:
    """Outer skorlarini eslesme, kapsam, sayi ve sonluluk acisindan dogrular."""
    missing = sorted(candidates - set(outer_anchor_scores))
    if missing:
        return (
            None,
            0,
            (
                f"Outer kanit tum adaylari kapsamiyor; eksik adaylar: {missing}. "
                "Kazanan ilan edilmedi."
            ),
        )
    try:
        anchors = {
            name: np.asarray(outer_anchor_scores[name], dtype="float64") for name in candidates
        }
    except (TypeError, ValueError):
        return None, 0, "Outer anchor skorlarinin sayisal oldugu kanitlanamadi."
    shapes = {values.shape for values in anchors.values()}
    if len(shapes) != 1 or any(values.ndim != 1 for values in anchors.values()):
        return (
            None,
            0,
            (
                "Outer anchor skorlarinin her aday icin ayni uzunlukta 1B ve "
                "eslestirilmis oldugu kanitlanamadi; kazanan ilan edilmedi."
            ),
        )
    n_anchors = len(next(iter(anchors.values())))
    if n_anchors < MIN_OUTER_ANCHORS:
        return (
            None,
            n_anchors,
            (
                f"Yalnizca {n_anchors} bagimsiz eslestirilmis outer anchor var; "
                "kazanan icin en az 6 gerekir."
            ),
        )
    if not all(np.isfinite(values).all() for values in anchors.values()):
        return (
            None,
            n_anchors,
            (
                "Outer anchor skorlarinda NaN/sonsuz deger var; kanit gecersiz ve "
                "kazanan ilan edilmedi."
            ),
        )
    return anchors, n_anchors, None


def bilimsel_kazanan_karari(
    oof_scores: dict[str, float],
    *,
    outer_evidence: OuterEvidence | None = None,
    outer_anchor_scores: dict[str, Sequence[float]] | None = None,
    independent_outer: bool = False,
    practical_effect: float = 0.0,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """OOF siralamasini ancak bagimsiz, eslesmis outer kanitla karara cevirir.

    ``apparent_oof_best`` yalnizca ic OOF'ta onceden secilen adaydir. Ozellikle
    hill-climb harmani ayni OOF'ta optimize edildigi icin bu skor tek basina
    kazanan kaniti OLAMAZ. Aday, en az alti bagimsiz outer anchor'da butun
    rakiplerini eslesmis guven araligiyla gecmedikce ``winner=None`` kalir.
    Coklu rakiplerde Bonferroni-duzeltilmis guven seviyesi kullanilir.

    ``independent_outer=True`` bir provenance beyanidir: her anchor'da model
    secimi ve harman agirliklari yalnizca o anchor'in training/inner-CV
    bolumunde yeniden fit edilmis, outer bolum hicbir ayara dokunmamis olmali.
    Hazir ayni-OOF tahminini parcalara bolmek bagimsiz outer kanit SAYILMAZ.
    """
    if len(oof_scores) < 2:
        raise ValueError("Bilimsel benchmark karari icin en az iki aday gerekli.")
    if not all(np.isfinite(float(score)) for score in oof_scores.values()):
        raise ValueError("OOF skorlarinda NaN veya sonsuz deger olamaz.")

    apparent = min(oof_scores, key=lambda name: (float(oof_scores[name]), name))
    if outer_evidence is None and outer_anchor_scores is None:
        return _sonucsuz_karar(
            apparent,
            "Ayni OOF uzerinde secilen/optimize edilen aday bilimsel kazanan "
            "ilan edilmedi: en az 6 bagimsiz, eslestirilmis outer fold/anchor "
            "kaniti yok.",
        )
    if outer_evidence is None:
        return _sonucsuz_karar(
            apparent,
            "Outer skorlar ve independent_outer beyanı structured provenance "
            "yerine gecmez. OuterEvidence(anchor_id, zaman siniri, recipe/fold "
            "fingerprint) olmadan bagimsizlik kanitlanamaz.",
        )

    outer_anchor_scores = outer_evidence.score_map()

    anchors, n_anchors, invalid_reason = _outer_anchor_kanitini_dogrula(
        set(oof_scores), outer_anchor_scores
    )
    if anchors is None:
        return _sonucsuz_karar(
            apparent,
            invalid_reason or "Outer anchor kaniti gecersiz; kazanan ilan edilmedi.",
            n_anchors=n_anchors,
        )

    rivals = [name for name in oof_scores if name != apparent]
    adjusted_confidence = 1.0 - (1.0 - confidence) / len(rivals)
    comparisons = [
        paired_model_decision(
            candidate_scores=anchors[apparent],
            baseline_scores=anchors[rival],
            candidate_name=apparent,
            baseline_name=rival,
            practical_effect=practical_effect,
            confidence=adjusted_confidence,
            n_bootstrap=n_bootstrap,
            seed=seed + index,
        )
        for index, rival in enumerate(rivals)
    ]
    serialized = [comparison.to_dict() for comparison in comparisons]
    if all(comparison.winner == apparent for comparison in comparisons):
        return {
            "apparent_oof_best": apparent,
            "winner": apparent,
            "statistically_conclusive": True,
            "decision_reason": ("preselected_oof_candidate_better_on_independent_outer_anchors"),
            "n_anchors": n_anchors,
            "required_anchors": MIN_OUTER_ANCHORS,
            "pairwise_decisions": serialized,
        }
    return _sonucsuz_karar(
        apparent,
        "Bagimsiz outer anchor guven araliklari OOF'ta secilen adayin tum "
        "rakiplerini pratik etki esigiyle gectigini gostermiyor; sonuc "
        "istatistiksel olarak kararsiz.",
        n_anchors=n_anchors,
        pairwise_decisions=serialized,
    )


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
        ham,
        entity_columns=[GRUP],
        time_column=ZAMAN,
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
            ozellik,
            kolon,
            shifts=SHIFT_OFSETLERI,
            time_column=ZAMAN,
            horizon=UFUK,
            group_columns=[GRUP],
        )

    # 3. dalga -- iki aile de hedeften turer ama YALNIZCA ufuk=31 kaydirmali
    # yayin yapar (fonksiyonlar horizon<1'i zaten reddeder), yani sizinti
    # duvarinin ARKASINDA kalirlar; HAM_KOLONLAR dus kumesi degismez.
    # Hawkes bozunumu: art arda ariza kumelenir -- tek basina en buyuk
    # olculmus kazanc (lgb_mae 323.13 -> 309.92).
    ozellik = add_event_decay_features(
        ozellik,
        HEDEF,
        time_column=ZAMAN,
        horizon=UFUK,
        group_columns=[GRUP],
        half_lives=YARI_OMURLER,
    )
    # Toplu-olay payi: firtina gunu ilcelerin buyuk kismi ayni gun kesintili
    # (M5 out-of-stock analogu; tek basina 320.37).
    ozellik = add_mass_event_features(
        ozellik,
        HEDEF,
        time_column=ZAMAN,
        horizon=UFUK,
        group_columns=[GRUP],
    )

    # ILCE KIMLIGI + ILCENIN KENDI GECMIS SEVIYESI (2026-08-18 denetimi, P1-2)
    # ---------------------------------------------------------------------
    # Onceki feature setinde ilceyi AYIRT EDEN hicbir kolon yoktu: frekans
    # kodlamasi dolu izgarada sabit (1/47), model ilceleri yalnizca lag
    # DEGERLERINDEN ayirt edebiliyordu. Expanding istatistik "bu ilce genelde
    # ne kadar kesinti yasar" sorusunu ufuk-kaydirmali (shift=UFUK) yanitlar;
    # ayni fonksiyon zaten sizinti duvarinin arkasindadir.
    # Olculdu (ayni fold'lar, catboost_mae): 304.30 -> 299.31 (-4.99),
    # lgb_mae 310.58 -> 306.35. Repodaki en buyuk tekil feature kazanci.
    ozellik = add_expanding_features(
        ozellik,
        HEDEF,
        time_column=ZAMAN,
        horizon=UFUK,
        group_columns=[GRUP],
        aggregations=("mean", "median", "std"),
    )
    # Sifir payi: "bu ilce gunlerin kacinda hic kesinti yasamamis" -- iki
    # asamali modelin sifir kutlesi bilgisini tek modele de tasir.
    sifir_gostergesi = f"{HEDEF}_sifir_mi"
    ozellik = ozellik.assign(**{sifir_gostergesi: (ozellik[HEDEF] == 0).astype("float64")})
    ozellik = add_expanding_features(
        ozellik,
        sifir_gostergesi,
        time_column=ZAMAN,
        horizon=UFUK,
        group_columns=[GRUP],
        aggregations=("mean",),
    )
    ozellik = ozellik.drop(columns=[sifir_gostergesi])

    # include_year=False: test donemi train'den sonra -- yil ekstrapolasyon riski.
    ozellik = add_calendar_features(ozellik, ZAMAN, include_year=False)
    ozellik = add_turkish_holiday_features(ozellik, ZAMAN)

    hava = pd.read_parquet(HAVA)
    oncesi = len(ozellik)
    ozellik = ozellik.merge(
        hava.drop(columns=[c for c in ("konum", "konum_key", "il_key") if c in hava.columns]),
        left_on=[GRUP, ZAMAN],
        right_on=["ilce_key", "tarih"],
        how="left",
        validate="many_to_one",
    )
    if len(ozellik) != oncesi:
        raise RuntimeError("hava merge satir sayisini degistirdi -- join anahtari bozuk.")

    # Gunes: saf astronomik geometri (gun uzunlugu, deklinasyon). geometry_only
    # cunku hava zaten OLCULMUS gunes_radyasyon tasiyor; pvlib acik-gokyuzu
    # modeli onunla buyuk olcude ortusur, geometri ise mevsim sinyalini verir.
    ref = pd.read_parquet(REFERANS)
    koordinatlar = {
        satir.ilce_key: (float(satir.lat), float(satir.lon)) for satir in ref.itertuples()
    }
    ozellik = add_solar_features(
        ozellik,
        time_column=ZAMAN,
        location_column=GRUP,
        coordinates=koordinatlar,
        geometry_only=True,
    )

    # Dagilim/frekans ailesi temporal fold icinde fit edilmedikce kullanilmaz.

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
        c for c in ozellik.columns if c not in dus and pd.api.types.is_numeric_dtype(ozellik[c])
    ]
    # ILCE KIMLIGI kategorik feature olarak: GBDT'ler (ozellikle CatBoost)
    # yuksek kardinaliteli kategoriyi sizintisiz kodlar. GRUP kolonunun
    # KOPYASI eklenir -- orijinal anahtar kolonu join/fold icin dokunulmaz
    # kalsin. Kategorik oldugu icin yukaridaki sayisal filtreye takilmaz.
    ozellik = ozellik.assign(ilce_kimlik=ozellik[GRUP].astype("category"))
    kolonlar.append("ilce_kimlik")
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
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray]]:
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

    skorlar: dict[str, dict[str, Any]] = {}
    oof: dict[str, np.ndarray] = {}
    kapsam: dict[str, np.ndarray] = {}
    for ad, (kind, params) in tarifler.items():
        print(f"  {ad} kosuyor...")
        sonuc = cross_validate(
            x,
            y,
            folds,
            kind=kind,
            metric="mae",
            params=_butceli(kind, params),
            sample_weight=agirliklar,
            early_stopping_rounds=ERKEN_DURDURMA,
            early_stopping_metric="mae",
            verbose=False,
        )
        skorlar[ad] = {
            "mae": float(sonuc.overall_score),
            "fold_std": float(sonuc.fold_std),
            # Fold skorlari JSON'a YAZILIR: eslestirilmis karsilastirma (ayni
            # fold'da A-B farki) fold_std'den cok daha keskin bir kanittir ve
            # artefaktlardan yeniden hesaplanabilir olmali (denetim P1-13).
            "fold_scores": [float(v) for v in sonuc.fold_scores],
            "sure_sn": float(sonuc.elapsed_seconds),
        }
        oof[ad] = sonuc.oof_predictions
        kapsam[ad] = sonuc.oof_covered
        print(
            f"    mae={sonuc.overall_score:.2f}  fold_std={sonuc.fold_std:.2f}  "
            f"sure={sonuc.elapsed_seconds:.0f} sn"
        )

    # sqrt recetesi -- Rohlik Sales v2'nin 2. ve 3.'sunden BAGIMSIZ cifte kanit:
    # sqrt(y) uzayinda L2, ham MAE'yi VE yerli Tweedie'yi gecti. Karsi kanit da
    # var (Rohlik Orders 3.: log1p CV'yi bozdu) -- yani donusum teoriden
    # okunamaz, burada OLCULUR. Skor HAM uzayda: once geri-kare, sonra MAE.
    # sqrt + FIT UZAYI erken durdurma (2026-08-18 denetimi, P1-4). Onceki
    # surumde ham-uzay esdegeri olmadigi icin guard erken durdurmayi
    # KAPATIYORDU ve 2000 sabit agac kosuluyordu -> 393.00 MAE, yani "sqrt
    # kotu" sonucu bir ARTEFAKTTI. early_stopping_space="fit" ile durdurma
    # sqrt uzayinda (l2) yapilir, skor yine HAM uzayda MAE'dir.
    print("  lgb_sqrt kosuyor...")
    sonuc = cross_validate(
        x,
        y,
        folds,
        kind="lightgbm",
        metric="mae",
        params=_butceli("lightgbm", starter_params("lightgbm", "regression")),
        sample_weight=agirliklar,
        target_transform="sqrt",
        early_stopping_metric="rmse",
        early_stopping_space="fit",
        early_stopping_rounds=ERKEN_DURDURMA,
        verbose=False,
    )
    maske = sonuc.oof_covered
    skorlar["lgb_sqrt"] = {
        "mae": float(sonuc.overall_score),
        "fold_std": float(sonuc.fold_std),
        "fold_scores": [float(v) for v in sonuc.fold_scores],
        "sure_sn": float(sonuc.elapsed_seconds),
    }
    oof["lgb_sqrt"] = sonuc.oof_predictions
    kapsam["lgb_sqrt"] = maske
    print(
        f"    mae={sonuc.overall_score:.2f}  fold_std={sonuc.fold_std:.2f}  "
        f"sure={sonuc.elapsed_seconds:.0f} sn"
    )
    return skorlar, oof, kapsam


def tohum_kararliligi(
    x: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    tohumlar: Sequence[int] = (42, 1, 2, 3, 4),
) -> dict[str, Any]:
    """Kazanan tekil modeli birden fazla tohumla kosar: gurultu ve ortalama kazanci.

    NEDEN (2026-08-18 denetimi, P1-1): "harman 0.55 MAE kazandiriyor" iddiasini
    degerlendirmek icin GURULTUNUN buyuklugu bilinmeli. Ayrica tohum ortalamasi
    (multi_seed_refit'in yaptigi sey) harmandan farkli olarak yapisal bir
    yanlilik tasimaz -- ne kadar kazandirdigini OLCUP kaydediyoruz.

    Her tohum ayni fold'larda kosar; OOF tahminleri ortalanip tek bir "tohum
    ortalamasi" skoru da hesaplanir.
    """
    catboost = starter_params("catboost", "regression", objective="mae")
    catboost["eval_metric"] = "MAE"
    mae_fn, _, _ = get_metric("mae")
    skorlar: list[float] = []
    oof_yigin: list[np.ndarray] = []
    maske: np.ndarray | None = None
    for tohum in tohumlar:
        params = _butceli("catboost", dict(catboost))
        params["random_seed"] = int(tohum)
        sonuc = cross_validate(
            x,
            y,
            folds,
            kind="catboost",
            metric="mae",
            params=params,
            early_stopping_rounds=ERKEN_DURDURMA,
            early_stopping_metric="mae",
            verbose=False,
        )
        skorlar.append(float(sonuc.overall_score))
        oof_yigin.append(sonuc.oof_predictions)
        maske = sonuc.oof_covered if maske is None else (maske & sonuc.oof_covered)
        print(f"    tohum {tohum}: mae={sonuc.overall_score:.2f}")

    assert maske is not None
    ortalama_oof = np.mean(np.vstack(oof_yigin), axis=0)
    ortalama_mae = float(mae_fn(y[maske], ortalama_oof[maske]))
    # TOHUM EGRISI (2026-08-20, docs/18 bolum B3): "5 mi 25 mi 100 mu" sorusu
    # kanitla cevaplanir. Ilk k tohumun ortalamasi k=1..N icin olculur; kazanc
    # ~1/sqrt(k) ile doyar. Egri, tohum sayisini artirmanin nerede tohum
    # gurultusunun altina dustugunu GOSTERIR -- playbook'un "100 tohum" tavsiyesi
    # bu veride dogrulanmadan alinmaz.
    egri: list[dict[str, float]] = []
    for k in range(1, len(oof_yigin) + 1):
        kismi = float(mae_fn(y[maske], np.mean(np.vstack(oof_yigin[:k]), axis=0)[maske]))
        egri.append({"tohum_sayisi": k, "mae": round(kismi, 3)})

    return {
        "tohumlar": list(tohumlar),
        "tekil_mae": [round(v, 2) for v in skorlar],
        "tohum_yayilimi": round(float(np.std(skorlar)), 3),
        "tohum_araligi": round(float(max(skorlar) - min(skorlar)), 2),
        "tohum_ortalamasi_mae": round(ortalama_mae, 2),
        "ortalama_kazanci": round(float(np.mean(skorlar)) - ortalama_mae, 3),
        "tohum_egrisi": egri,
        "son_tohumun_katkisi": round(egri[-2]["mae"] - egri[-1]["mae"], 3)
        if len(egri) >= 2
        else None,
    }


def iki_asama_kos(
    x: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    agirliklar: np.ndarray | None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Iki asamali (hurdle) modeli kosturur; birlesik OOF ve sifir oranini dondurur.

    Sifir orani ~%35 -- fit_two_stage'in kendi dokumani %40 altinda duz
    regresyonun daha iyi olmasini bekler. Bunu VARSAYMIYORUZ, olcuyoruz.
    """
    print("  iki_asama kosuyor...")
    baslangic = time.perf_counter()
    sonuc = fit_two_stage(
        x,
        y,
        folds,
        kind="lightgbm",
        metric="mae",
        classifier_params=_butceli("lightgbm", starter_params("lightgbm", "binary")),
        regressor_params=_butceli(
            "lightgbm", starter_params("lightgbm", "regression", objective="mae")
        ),
        sample_weight=agirliklar,
        early_stopping_rounds=ERKEN_DURDURMA,
        verbose=False,
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

    print(f"    mae={mae:.2f}  fold_std={fold_std:.2f}  sure={sure:.0f} sn  (esik={esik:.3f})")
    skor = {
        "mae": mae,
        "fold_std": fold_std,
        "fold_scores": fold_skorlari,
        "sure_sn": float(sure),
    }
    return skor, birlesik, maske, float((y == 0).mean()), esik, sonuc.oof_probability


def medyan_kurali_kos(
    x: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    olasilik: np.ndarray,
    maske: np.ndarray,
    agirliklar: np.ndarray | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
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
        x,
        y,
        folds,
        params=_butceli("lightgbm", starter_params("lightgbm", "regression")),
        sample_weight=agirliklar,
        early_stopping_rounds=ERKEN_DURDURMA,
        verbose=False,
    )
    kalibrasyon = calibrate_positive_probability(olasilik, y, folds, covered=maske, verbose=False)
    mae_fn, _, _ = get_metric("mae")

    skorlar: dict[str, dict[str, Any]] = {}
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
            "fold_scores": fold_skorlari,
            "sure_sn": float(sure),
        }
        ooflar[ad] = tahmin
        print(f"    {ad}: mae={mae:.2f}  fold_std={skorlar[ad]['fold_std']:.2f}")

    kalibrasyon_ozeti = {
        "brier_once": kalibrasyon.brier_before,
        "brier_sonra": kalibrasyon.brier_after,
        "iyilesti": bool(kalibrasyon.improved),
    }
    print(
        f"    kalibrasyon: Brier {kalibrasyon.brier_before:.4f} -> "
        f"{kalibrasyon.brier_after:.4f} "
        f"({'iyilesti' if kalibrasyon.improved else 'IYILESMEDI'})"
    )
    return skorlar, ooflar, kalibrasyon_ozeti


def nested_harman(
    uyeler: Sequence[str],
    oof: dict[str, np.ndarray],
    y: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    ortak_maske: np.ndarray,
    *,
    toplayici: str = "ortalama",
) -> dict[str, Any]:
    """Harman agirligini GECMIS fold'larda ogrenip SONRAKI fold'da skorlar.

    NEDEN (2026-08-18 denetimi, P1-1): ``harman_ve_stack`` agirliklari TUM
    OOF uzerinde tirmanip ayni OOF'ta skorluyor. Bu, harmanin lehine
    yapisal bir yanliliktir: uye sayisi kadar serbestlik dereceli bir
    optimizasyon, skorlandigi veriye bakarak yapiliyor. Olculdu (bagimsiz
    denetim): ornek-ici harman 303.44, ayni fold'larda YUVALANMIS harman
    305.49, tek basina catboost_mae 304.30 -- yani "harman kazandi" sonucu
    olcum yontemi degisince kayboluyordu. Tohum gurultusu ~4 MAE oldugu icin
    0.55'lik "kazanc" zaten gurultunun altindaydi.

    Yontem (rolling-origin blend): k'nci fold icin agirliklar 1..k-1
    fold'larinin valid satirlarindan ogrenilir, skor YALNIZCA k'nci fold'un
    valid satirlarinda alinir. Ilk fold agirlik ogrenemez (gecmisi yok),
    disarida kalir. Boylece rapor edilen sayi, "yarin bu harmani kurup
    kullansam ne olurdu" sorusunun cevabidir.

    TOPLAYICI (2026-08-20, docs/18 bolum B2): ``"ortalama"`` agirlikli
    aritmetik ortalama, ``"medyan"`` agirlikli medyandir. Metrik MAE oldugu
    icin ikincisi teorik olarak dogru toplayicidir (MAE'yi minimize eden
    tahmin medyandir). Ilk denetimde YALNIZCA ortalama denenmis ve harman
    reddedilmisti; bu parametre ayni fold'larda ikisini de olcup karari
    toplayici seciminden BAGIMSIZ hale getirir.

    Args:
        toplayici: ``"ortalama"`` | ``"medyan"``. Agirliklar her iki durumda
            da ayni sekilde (MAE hedefli hill-climb) ogrenilir; degisen
            yalnizca uyelerin nasil birlestirildigidir.

    Returns:
        ``fold_mae`` (fold basina yuvalanmis harman skoru), ``mae``
        (agirlikli ortalama), ``agirliklar`` (fold basina) ve karsilastirma
        icin ayni fold'lardaki en iyi TEK uye.

    Raises:
        ValueError: taninmayan ``toplayici``.
    """
    if toplayici not in ("ortalama", "medyan"):
        raise ValueError(f"Bilinmeyen toplayici: {toplayici!r} (ortalama|medyan)")
    mae_fn, _, _ = get_metric("mae")
    fold_kayitlari: list[dict[str, Any]] = []
    toplam_hata, toplam_satir = 0.0, 0
    tekil_hata: dict[str, float] = dict.fromkeys(uyeler, 0.0)

    for sira in range(1, len(folds)):
        gecmis = np.concatenate([folds[onceki][1] for onceki in range(sira)])
        gecmis = gecmis[ortak_maske[gecmis]]
        simdiki = folds[sira][1]
        simdiki = simdiki[ortak_maske[simdiki]]
        if gecmis.size == 0 or simdiki.size == 0:
            continue

        agirliklar = hill_climb_weights(
            {ad: oof[ad][gecmis] for ad in uyeler},
            y[gecmis],
            metric="mae",
            covered=np.ones(gecmis.size, dtype=bool),
            verbose=False,
        )
        if toplayici == "medyan":
            tahmin = median_blend({ad: oof[ad][simdiki] for ad in uyeler}, agirliklar)
        else:
            tahmin = np.zeros(simdiki.size)
            for ad, w in agirliklar.items():
                tahmin += w * oof[ad][simdiki]
        skor = float(mae_fn(y[simdiki], tahmin))
        toplam_hata += skor * simdiki.size
        toplam_satir += simdiki.size
        for ad in uyeler:
            tekil_hata[ad] += float(mae_fn(y[simdiki], oof[ad][simdiki])) * simdiki.size
        fold_kayitlari.append(
            {
                "fold": sira,
                "n_valid": int(simdiki.size),
                "mae": skor,
                "agirliklar": {ad: round(float(w), 4) for ad, w in agirliklar.items() if w > 0},
            }
        )

    if not fold_kayitlari:
        return {"mae": None, "aciklama": "yuvalanmis harman icin yeterli fold yok"}

    tekil = {ad: tekil_hata[ad] / toplam_satir for ad in uyeler}
    en_iyi_tekil = min(tekil, key=lambda ad: tekil[ad])
    harman_mae = toplam_hata / toplam_satir
    return {
        "mae": harman_mae,
        "fold_kayitlari": fold_kayitlari,
        "ayni_satirlarda_en_iyi_tekil": en_iyi_tekil,
        "ayni_satirlarda_tekil_mae": {ad: round(v, 2) for ad, v in tekil.items()},
        "toplayici": toplayici,
        "harman_tekilden_iyi_mi": bool(harman_mae < tekil[en_iyi_tekil]),
        "fark": round(harman_mae - tekil[en_iyi_tekil], 3),
    }


def harman_ve_stack(
    modeller: dict[str, dict[str, Any]],
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
        maskeli,
        y[indeks],
        metric="mae",
        covered=np.ones(indeks.size, dtype=bool),  # onceden maskelendi
        stability_penalty=KARARLILIK_CEZASI,
        fold_slices=dilimler,
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
        {ad: oof[ad] for ad in uyeler},
        y,
        folds,
        base_covered=ortak_maske,
        meta="ridge",
        metric="mae",
        verbose=False,
    )
    stack_mae = float(stack["score"])
    print(f"  stack : mae={stack_mae:.2f}")

    # YUVALANMIS KONTROL: yukaridaki harman skoru ornek-icidir (agirliklar ayni
    # OOF'ta ogrenildi). Gonderim karari bu sayiya degil, asagidakine bakar.
    # IKI TOPLAYICI birden, AYNI fold'larda (docs/18 B2). Metrik MAE oldugu
    # icin medyan teorik olarak dogru toplayici; ilk denetimde yalnizca
    # ortalama denenmis ve harman o yuzden reddedilmis olabilir. Karar artik
    # toplayici seciminden bagimsiz veriliyor -- ikisi de gecemezse harman
    # gercekten gecmiyordur.
    nested = nested_harman(uyeler, oof, y, folds, ortak_maske, toplayici="ortalama")
    nested_medyan = nested_harman(uyeler, oof, y, folds, ortak_maske, toplayici="medyan")
    harman["yuvalanmis"] = nested
    harman["yuvalanmis_medyan"] = nested_medyan
    for etiket, kayit in (("ortalama", nested), ("medyan", nested_medyan)):
        if kayit.get("mae") is None:
            continue
        durum = "GECIYOR" if kayit["harman_tekilden_iyi_mi"] else "GECMIYOR"
        en_iyi = kayit["ayni_satirlarda_en_iyi_tekil"]
        print(
            f"  yuvalanmis harman ({etiket}): mae={kayit['mae']:.2f} vs "
            f"tek uye {en_iyi} {kayit['ayni_satirlarda_tekil_mae'][en_iyi]:.2f} "
            f"-> harman {durum}"
        )
    return harman, stack_mae


def recete_yaz(
    modeller: dict[str, dict[str, Any]],
    harman: dict[str, Any],
    stack_mae: float,
    kazanan: str | None,
    karar_gerekcesi: str,
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
    duzler = {ad: bilgi["mae"] for ad, bilgi in modeller.items() if not ad.startswith("iki_asama")}
    duz_ad = min(duzler, key=lambda ad: duzler[ad])
    sira = sorted(modeller, key=lambda ad: modeller[ad]["mae"])
    cat_sira = sira.index("catboost_mae") + 1
    cat_mae = modeller["catboost_mae"]["mae"]
    iki_mae = modeller["iki_asama"]["mae"]
    cat_hukmu = (
        "ic OOF'ta ilk aday, ama outer kanit olmadan kazanan sayilmaz"
        if cat_sira == 1
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
        if esik <= 0.02
        else f" (optimum esik {esik:.2f})"
    )
    medyan_mae = modeller["iki_asama_medyan"]["mae"]
    kalibre_mae = modeller["iki_asama_medyan_kalibre"]["mae"]
    sqrt_mae = modeller["lgb_sqrt"]["mae"]
    # Harmanin agirlik verdigi DIGER uyeler onerilir; kazanan zaten baslangic.
    ek_uyeler = [ad for ad in harman["uyeler"] if ad != tek]
    ek_metin = " ve ".join(ek_uyeler) if ek_uyeler else "catboost_mae ve lgb_tweedie"
    kazanan_metni = (
        f"Bagimsiz outer kanitla genel kazanan '{kazanan}'."
        if kazanan is not None
        else f"Bilimsel kazanan ilan edilmedi: {karar_gerekcesi}"
    )
    # IKI TOPLAYICI: karar, ikisinin IYISINE bakar (docs/18 B2). Metrik MAE
    # oldugu icin medyan toplayici teoride dogru olandir; ortalama uzerinden
    # verilen eski "harman gecmiyor" hukmu, yanlis adayin reddi olabilirdi.
    # Ikisi de gecemezse harman gercekten gecmiyordur.
    nested_adaylar = [
        kayit
        for kayit in (harman.get("yuvalanmis", {}), harman.get("yuvalanmis_medyan", {}))
        if kayit.get("mae") is not None
    ]
    nested = min(nested_adaylar, key=lambda k: float(k["mae"])) if nested_adaylar else {}
    toplayici_adi = nested.get("toplayici", "ortalama")
    if nested.get("mae") is not None:
        tekil_ad = nested["ayni_satirlarda_en_iyi_tekil"]
        tekil_mae = nested["ayni_satirlarda_tekil_mae"][tekil_ad]
        if nested["harman_tekilden_iyi_mi"]:
            harman_hukmu = (
                f"YUVALANMIS kontrolde de geciyor ({toplayici_adi} toplayiciyla; "
                f"agirliklar gecmis fold'larda ogrenilip sonraki fold'da skorlandi: "
                f"{nested['mae']:.2f} vs {tekil_ad} {tekil_mae:.2f}) -- gun-1'de "
                "kurmaya deger"
            )
            harman_onerisi = (
                f"Oneri: gun-1'de {tek} ile basla, ayni fold'larda {ek_metin} "
                f"uyelerini ekleyip hill-climb harmanini {toplayici_adi.upper()} "
                "toplayiciyla kur; agirliklari HER ZAMAN gecmis fold'larda ogren "
                "(nested), ayni OOF'ta tirmanip ayni OOF'ta skorlama."
            )
        else:
            harman_hukmu = (
                f"ama YUVALANMIS kontrolde GECMIYOR (ortalama VE medyan toplayici, "
                f"ikisi de): agirliklar gecmis fold'larda "
                f"ogrenilip sonraki fold'da skorlanunca en iyisi {nested['mae']:.2f}, ayni "
                f"satirlarda tek basina {tekil_ad} {tekil_mae:.2f} "
                f"(fark {nested['fark']:+.2f}). Ornek-ici harman skoru, uye sayisi "
                "kadar serbestlik dereceli bir optimizasyonun kendi verisinde "
                "olculmesidir; tohum gurultusu ~4 MAE oldugu icin bu 'kazanc' "
                "zaten gurultunun altinda"
            )
            harman_onerisi = (
                f"Oneri: gun-1'de {tek} modelini 5 TOHUMLA yeniden egitip "
                "(multi_seed_refit, tohum ortalamasi) gonder; harmani ancak "
                "yuvalanmis kontrolu gecerse ekle. Tohum ortalamasi olculmus ve "
                "bedava bir varyans dususudur, harman ise bu veride degildir."
            )
    else:
        harman_hukmu = "yuvalanmis kontrol icin yeterli fold yok"
        harman_onerisi = f"Oneri: gun-1'de {tek} ile basla ve 5 tohumla yeniden egit."

    return (
        f"Veri gununde ilk kosulacak tek model {tek} (MAE {tek_mae:.2f}; hep-sifir "
        f"baseline {sifir_baseline:.2f}, sifir orani %{sifir_orani * 100:.1f}). "
        f"2023 birincisinin recetesi catboost_mae bu veride MAE {cat_mae:.2f} ile "
        f"{cat_sira}. sirada -- {cat_hukmu}. Iki asamali model (MAE {iki_mae:.2f}) "
        f"{iki_hukmu}{esik_notu}. MAE-optimal medyan kurali {medyan_mae:.2f}, "
        f"kalibre olasilikla {kalibre_mae:.2f}; sqrt donusumu (Rohlik recetesi, "
        f"fit-uzayi erken durdurmayla) {sqrt_mae:.2f}. Tum uyeler uzerinde "
        f"hill-climb harmani MAE {harman['mae']:.2f} (agirlik alan uyeler: "
        f"{', '.join(harman['uyeler'])}) -- {harman_hukmu}; ridge stacking "
        f"{stack_mae:.2f} (purged semada ilk fold'lar meta-egitim kapsami disinda "
        f"kaliyor). {kazanan_metni} {harman_onerisi}"
    )


def outer_kanit_yukle(yol: Path) -> OuterEvidence:
    """``outer_anchor_kosusu.py`` ciktisini ``OuterEvidence``e cevirir.

    Kapinin istedigi yapisal provenance burada kurulur: her anchor kendi
    zaman sinirini ve recipe/fold parmak izini tasir. Dosya bozuksa ya da
    anchor'lar ayni aday kumesini paylasmiyorsa ``OuterEvidence`` kendisi
    reddeder -- sessiz kabul yok.
    """
    veri = json.loads(yol.read_text(encoding="utf-8"))
    anchors = tuple(
        OuterAnchor(
            anchor_id=str(kayit["anchor_id"]),
            train_end=str(kayit["train_end"]),
            validation_start=str(kayit["validation_start"]),
            validation_end=str(kayit["validation_end"]),
            scores={ad: float(deger) for ad, deger in kayit["scores"].items()},
            recipe_fingerprint=str(kayit["recipe_fingerprint"]),
            fold_fingerprint=str(kayit["fold_fingerprint"]),
        )
        for kayit in veri["anchors"]
    )
    return OuterEvidence(anchors=anchors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outer",
        default=None,
        help="outer_anchor_kosusu.py ciktisi (JSON); kazanan kapisini atesler",
    )
    parser.add_argument(
        "--tohum",
        type=int,
        default=5,
        help="Tohum kararliligi kac tohumla olculsun (egri de bu kadar uzar)",
    )
    args = parser.parse_args()
    if args.tohum < 2:
        parser.error("--tohum en az 2 olmali (yayilim tek tohumla olculemez)")

    if not VERI.exists():
        print(
            f"HATA: {VERI} yok. Indir: kaggle datasets download -d "
            "tmlalper/manisa-izmir-plansiz-elektrik-kesintileri --unzip"
        )
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
        ozellik[ZAMAN],
        embargo=pd.Timedelta(days=UFUK),
        n_splits=4,
        test_span=pd.Timedelta(days=UFUK),
        verbose=False,
    )

    print(f"2/4 tek modeller (ortak butce: {ORTAK_BUTCE} agac, erken durdurma {ERKEN_DURDURMA})...")
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

    print("3c/4 tohum kararliligi (catboost_mae x5)...")
    # Tohum listesi 42 ile baslar (repodaki kanonik tohum), sonra 0,1,2,...
    # boylece --tohum buyutuldugunde onceki kosunun ilk k tohumu KORUNUR ve
    # egri kosular arasinda karsilastirilabilir kalir.
    tohumlar = (42, *range(args.tohum - 1))
    tohum_ozeti = tohum_kararliligi(ozellik[kolonlar], y, folds, tohumlar=tohumlar)
    print(
        f"  tohum yayilimi {tohum_ozeti['tohum_yayilimi']:.2f} MAE, "
        f"aralik {tohum_ozeti['tohum_araligi']:.2f}; 5-tohum ortalamasi "
        f"{tohum_ozeti['tohum_ortalamasi_mae']:.2f} "
        f"(tekil ortalamasindan {tohum_ozeti['ortalama_kazanci']:+.2f})"
    )

    print("4/4 harman + stacking...")
    harman, stack_mae = harman_ve_stack(modeller, oof, kapsam, y, folds)

    # Baseline ve kazanan -- hepsi ayni kapsanan satir kumesinde.
    ortak_maske = kapsam["lgb_mae"]
    sifir_baseline = float(np.abs(y[ortak_maske]).mean())
    adaylar = {ad: bilgi["mae"] for ad, bilgi in modeller.items()}
    adaylar["harman"] = harman["mae"]
    adaylar["stack"] = stack_mae
    # Bu kosuda harman agirliklari ve aday sirasi AYNI OOF uzerinden geliyor;
    # dolayisiyla minimum skor, ozellikle harman icin, dis kanit degildir.
    # Nested/rolling outer anchor kosusu henuz yok: karar kapisi bilincli olarak
    # kapali kalir ve OOF minimumu yalnizca ``apparent_oof_best`` diye kaydedilir.
    outer_kanit = None
    if args.outer:
        outer_yolu = Path(args.outer)
        if not outer_yolu.exists():
            print(f"UYARI: {outer_yolu} yok; kazanan kapisi kapali kalacak.")
        else:
            outer_kanit = outer_kanit_yukle(outer_yolu)
            print()
            print(
                f"Dis capa kaniti yuklendi: {len(outer_kanit.anchors)} anchor ({outer_yolu.name})"
            )
            # Kapi YALNIZCA outer'da olculen adaylari karsilastirabilir.
            olculen = set(outer_kanit.anchors[0].scores)
            adaylar = {ad: skor for ad, skor in adaylar.items() if ad in olculen}
    karar = bilimsel_kazanan_karari(adaylar, outer_evidence=outer_kanit)
    kazanan = karar["winner"]

    sonuc = {
        "modeller": modeller,
        "harman": harman,
        "stack_mae": stack_mae,
        "kazanan": kazanan,
        "statistically_conclusive": karar["statistically_conclusive"],
        "decision_reason": karar["decision_reason"],
        "benchmark_decision": karar,
        "sifir_baseline": sifir_baseline,
        "sifir_orani": sifir_orani,
        # Feature listesi JSON'a yazilir ki test 'yasak kolon sizdi mi'
        # sorusunu MAKINE ile sorabilsin -- denetim, MAE esigine bakan
        # testin id sizintisini ayirt edemedigini gosterdi.
        "feature_kolonlari": list(kolonlar),
        "kalibrasyon": kalibrasyon_ozeti,
        "tohum_kararliligi": tohum_ozeti,
        "gun1_recetesi": recete_yaz(
            modeller,
            harman,
            stack_mae,
            kazanan,
            karar["decision_reason"],
            sifir_baseline,
            sifir_orani,
            esik,
        ),
    }
    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(
        CIKTI,
        (json.dumps(sonuc, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )

    if kazanan is None:
        print(f"\nKAZANAN: YOK -- {karar['decision_reason']}")
        print(
            f"OOF'ta gorunen en iyi: {karar['apparent_oof_best']} "
            f"(mae={adaylar[karar['apparent_oof_best']]:.2f})"
        )
    else:
        print(
            f"\nKAZANAN: {kazanan}  (mae={adaylar[kazanan]:.2f}, "
            f"hep-sifir {sifir_baseline:.2f}, "
            f"sifir orani %{sifir_orani * 100:.1f})"
        )
    print(f"Sonuc: {CIKTI}")
    print(f"Toplam sure: {time.perf_counter() - baslangic:.0f} sn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
