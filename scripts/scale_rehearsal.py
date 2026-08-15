"""Yarisma olceginde zaman ve bellek provasi -- 12 gunluk butceyi cikarir.

NEDEN
-----
``smoke_test.py`` ve ``full_pipeline.py`` **kucuk** veride her seyin
CALISTIGINI kanitlar. Bu betik farkli bir soruyu cevaplar:

    "Yarisma boyutunda bunlarin her biri NE KADAR SURER?"

Bu, 12 gunluk planin temelidir. Optuna'ya 100 deneme mi 20 deneme mi
ayirabilecegini, model zoo'sunu kac kez kosabilecegini, SHAP secimini hic
yapip yapamayacagini bilmeden plan yapilamaz. Veri geldiginde bunu olcmek
icin vakit yoktur -- SIMDI olculur.

OLCEKLER (Grid Up icin makul araliklar)
---------------------------------------
    100k satir  = 96 ilce x ~3 yil GUNLUK panel        <- en olasi
    500k satir  = 96 ilce x ~3 yil gunluk x 5 varlik   <- trafo/fider kirilimi
    2.5M satir  = 96 ilce x ~3 yil SAATLIK             <- 2023 formati saatlikti

Ilk ikisi OLCULUR, ucuncusu olculen egimden kestirilir. Kestirim acikca
"tahmin" olarak isaretlenir -- olculmus gibi sunulmaz.

KULLANIM
--------
::

    python scripts/scale_rehearsal.py             # 100k + 500k olcer
    python scripts/scale_rehearsal.py --hizli     # sadece 100k
    python scripts/scale_rehearsal.py --agir      # 2.5M'i da GERCEKTEN olcer (yavas)
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridup.validation import purged_time_series_split  # noqa: E402

#: Gunluk submission siniri (GDZ'nin uc yarismasinda da 3'tu).
GUNLUK_SUBMISSION = 3

#: Yarisma suresi (21 Agustos - 1 Eylul 2026).
YARISMA_GUNU = 12

#: Gunde kac saat gercekci calisma varsayiyoruz (tek kisi, is/okul disi).
GUNLUK_CALISMA_SAATI = 5


@dataclass
class Olcum:
    """Tek bir islemin olcumu."""

    ad: str
    saniye: float
    bellek_mb: float = 0.0
    not_: str = ""
    tahmin: bool = False


@dataclass
class OlcekSonucu:
    satir: int
    kolon: int
    olcumler: list[Olcum] = field(default_factory=list)

    def tablo(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "islem": o.ad,
                    "saniye": round(o.saniye, 1),
                    "dakika": round(o.saniye / 60, 2),
                    "bellek_mb": round(o.bellek_mb, 0),
                    "tahmin": o.tahmin,
                    "not": o.not_,
                }
                for o in self.olcumler
            ]
        )


def _bellek_mb() -> float:
    """Surecin anlik bellek kullanimi (MB). psutil yoksa 0 doner."""
    try:
        import psutil
    except ImportError:
        return 0.0
    return psutil.Process().memory_info().rss / 1024 / 1024


class _Kronometre:
    """Sure ve bellek artisini olcer."""

    def __init__(self, ad: str, sonuc: OlcekSonucu, not_: str = "") -> None:
        self.ad, self.sonuc, self.not_ = ad, sonuc, not_

    def __enter__(self):
        gc.collect()
        self.baslangic_bellek = _bellek_mb()
        self.baslangic = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        sure = time.perf_counter() - self.baslangic
        artis = max(0.0, _bellek_mb() - self.baslangic_bellek)
        self.sonuc.olcumler.append(Olcum(self.ad, sure, artis, self.not_))
        print(f"    {self.ad:<34} {sure:7.1f} sn   (+{artis:5.0f} MB)")


def panel_uret(n_satir: int, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Yarisma benzeri bir panel uretir: ~60 feature, karisik dtype.

    Feature sayisi kasitli olarak gercekci: takvim (12) + lag/rolling (20) +
    hava (12) + gunes (7) + mekansal (5) + kategorik (4). Az feature ile
    olculen sure yaniltici derecede iyimser olur.
    """
    rng = np.random.default_rng(seed)
    n_varlik = 96
    n_gun = max(60, n_satir // n_varlik)

    gunler = pd.date_range("2023-01-01", periods=n_gun, freq="D")
    tarih = pd.Series(np.tile(gunler, n_varlik))
    varlik = np.repeat([f"ilce_{i:03d}" for i in range(n_varlik)], n_gun)
    n = len(tarih)

    veri: dict[str, np.ndarray] = {
        "ilce": varlik,
        "ay": tarih.dt.month.to_numpy().astype("int8"),
        "haftagun": tarih.dt.dayofweek.to_numpy().astype("int8"),
        "yilgun": tarih.dt.dayofyear.to_numpy().astype("int16"),
        "hafta": tarih.dt.isocalendar().week.to_numpy().astype("int8"),
    }
    for i in range(8):
        veri[f"takvim_{i}"] = rng.normal(0, 1, n).astype("float32")
    for i in range(20):
        veri[f"lag_{i}"] = rng.normal(0, 1, n).astype("float32")
    for i in range(12):
        veri[f"hava_{i}"] = rng.normal(15, 8, n).astype("float32")
    for i in range(7):
        veri[f"gunes_{i}"] = rng.uniform(0, 8, n).astype("float32")
    for i in range(5):
        veri[f"komsu_{i}"] = rng.normal(0, 1, n).astype("float32")
    for i in range(3):
        veri[f"kat_{i}"] = rng.integers(0, 12, n).astype("int8")

    frame = pd.DataFrame(veri)
    frame["ilce"] = frame["ilce"].astype("category")

    hedef = (
        120
        + 3.0 * frame["hava_0"].to_numpy()
        + 5.0 * frame["lag_0"].to_numpy()
        + 2.0 * frame["gunes_0"].to_numpy()
        + rng.normal(0, 4, n)
    ).astype("float64")

    return frame, hedef, tarih


def olcek_kos(n_satir: int, *, tam: bool) -> OlcekSonucu:
    """Tek bir olcek icin tum islemleri olcer."""
    from gridup.ensemble import hill_climb_weights, stack_oof
    from gridup.models import cross_validate
    from gridup.neural import NeuralConfig, neural_cross_validate
    from gridup.selection import fold_shap_importance
    from gridup.zoo import make_model_zoo

    print(f"\n{'=' * 68}")
    print(f"OLCEK: {n_satir:,} satir")
    print("=" * 68)

    X, y, tarih = panel_uret(n_satir)
    sonuc = OlcekSonucu(satir=len(X), kolon=X.shape[1])
    print(f"  panel: {len(X):,} satir x {X.shape[1]} kolon "
          f"({X.memory_usage(deep=True).sum() / 1024 / 1024:.0f} MB)")

    folds = purged_time_series_split(
        tarih, embargo=pd.Timedelta(days=30), n_splits=3,
        test_span=pd.Timedelta(days=31), verbose=False,
    )
    print(f"  fold: {len(folds)} x {len(folds[0][1]):,} valid satir\n")

    hizli = {"n_estimators": 500, "learning_rate": 0.05, "verbose": -1}

    with _Kronometre("LightGBM tek CV (500 agac)", sonuc):
        lgbm = cross_validate(X, y, folds, kind="lightgbm", metric="mape",
                              params=hizli, verbose=False)

    with _Kronometre("CatBoost tek CV (500 iter)", sonuc):
        cat = cross_validate(
            X, y, folds, kind="catboost", metric="mape",
            params={"iterations": 500, "learning_rate": 0.05, "verbose": 0,
                    "allow_writing_files": False},
            verbose=False,
        )

    with _Kronometre("Sinir agi CV (60 epok)", sonuc):
        nn = neural_cross_validate(
            X, y, folds, cat_columns=["ilce"], metric="mape",
            config=NeuralConfig(max_epochs=60, patience=8), verbose=False,
        )

    # Harman icin IKI ayri gorunum gerekiyor:
    #   * hill_climb yalnizca OOF kapsamindaki satirlari ister (skor orada olculur)
    #   * stack_oof fold indeksleriyle calisir, yani TAM uzunlukta dizi ister
    kapsam = np.zeros(len(y), dtype=bool)
    for _, v in folds:
        kapsam[v] = True
    tam = {
        "lgbm": lgbm.oof_predictions,
        "cat": cat.oof_predictions,
        "nn": nn.oof_predictions,
    }
    kapsamli = {ad: dizi[kapsam] for ad, dizi in tam.items()}

    with _Kronometre("hill_climb_weights (3 uye)", sonuc):
        hill_climb_weights(kapsamli, y[kapsam], metric="mape", verbose=False)

    with _Kronometre("stack_oof (meta model)", sonuc):
        stack_oof(tam, y, folds, metric="mape", verbose=False)

    with _Kronometre("SHAP onem (fold basi 2000)", sonuc):
        fold_shap_importance(lgbm.models, X, folds, sample_per_fold=2000)

    if tam:
        with _Kronometre("model zoo (3 model x 3 fold)", sonuc,
                         "zoo = lgbm + xgb + cat, ayni fold'lar"):
            make_model_zoo(X, y, folds, metric="mape", verbose=False)

    # Optuna: TEK deneme olculur, kullanici deneme sayisiyla carpar.
    with _Kronometre("Optuna TEK deneme", sonuc, "n_trials ile carp"):
        from gridup.tuning import tune_with_optuna

        tune_with_optuna(X, y, folds, kind="lightgbm", metric="mape",
                         n_trials=1, verbose=False)

    return sonuc


#: Bir hipotezi kurmak, kodlamak ve sonucunu YORUMLAMAK icin gereken insan
#: suresi (dakika). Model kosma suresinden bagimsizdir ve genelde ondan
#: BUYUKTUR -- butcenin gercek kisiti budur.
HIPOTEZ_BASINA_DAKIKA = 25


def butce_raporu(sonuclar: dict[int, OlcekSonucu], hedef_olcek: int) -> str:
    """12 gunluk yarismada gercek kisit hangisi: hesap mi, insan mi?

    ONCEKI SURUMUN HATASI
    ---------------------
    Ilk versiyon butun calisma suresini hesaplamaya bolup "7778 tam CV
    yapabilirsin" diyordu. Bu sayi sacmadir ve modelin yanlis oldugunu
    gosterir: 7778 deneyi kimse KURAMAZ, KODLAYAMAZ ve YORUMLAYAMAZ.

    Dogru model uc kisiti ayri ayri hesaplayip **en dar olani** bulmaktir:
      1. Insan   -- kac hipotez kurup yorumlayabilirsin
      2. Hesap   -- makine kac kosu cikarabilir
      3. Geri bildirim -- gunde 3 submission, toplam 36

    Kucuk veride (100k) insan kisiti baglar, hesap bedavadir.
    Buyuk veride (2.5M) hesap baglar ve strateji tumuyle degisir.
    """
    sonuc = sonuclar[hedef_olcek]
    sureler = {o.ad: o.saniye for o in sonuc.olcumler}

    toplam_saat = YARISMA_GUNU * GUNLUK_CALISMA_SAATI
    hesap_saniye = toplam_saat * 3600 * 0.5  # yarisi makineye, yarisi dusunmeye

    lgbm_tek = sureler.get("LightGBM tek CV (500 agac)", 0.0) or 1.0
    optuna_tek = sureler.get("Optuna TEK deneme", 0.0) or 1.0
    zoo = sureler.get("model zoo (3 model x 3 fold)", 0.0)

    insan_limiti = int(toplam_saat * 60 / HIPOTEZ_BASINA_DAKIKA)
    hesap_limiti = int(hesap_saniye / lgbm_tek)
    geri_bildirim = YARISMA_GUNU * GUNLUK_SUBMISSION

    baglayan, deger = min(
        [("INSAN", insan_limiti), ("HESAP", hesap_limiti)], key=lambda p: p[1]
    )

    satirlar = [
        "",
        "=" * 68,
        f"12 GUNLUK BUTCE  ({hedef_olcek:,} satir)",
        "=" * 68,
        f"  Toplam calisma  : {YARISMA_GUNU} gun x {GUNLUK_CALISMA_SAATI} sa = {toplam_saat} saat",
        "",
        "  UC KISIT (en dar olani baglar):",
        f"    insan     : ~{insan_limiti:>5} hipotez  "
        f"(hipotez basi {HIPOTEZ_BASINA_DAKIKA} dk kurma+yorumlama)",
        f"    hesap     : ~{hesap_limiti:>5} CV kosusu "
        f"(kosu basi {lgbm_tek:.0f} sn, butcenin yarisi)",
        f"    geri bildirim: {geri_bildirim:>4} submission (gunde {GUNLUK_SUBMISSION})",
        "",
        f"  >>> BAGLAYAN KISIT: {baglayan} (~{deger} deney)",
    ]

    if baglayan == "INSAN":
        satirlar += [
            "",
            "  YORUM: Hesap BEDAVA. Makineyi bosta birakma:",
            f"    * Optuna'yi gece boyu kos ({optuna_tek:.0f} sn/deneme -> "
            f"8 saatte ~{int(8 * 3600 / optuna_tek)} deneme)",
            "    * Model zoo'yu her feature setinde tekrar kos, elemeyi CV'ye yaptir",
            "    * Cok tohumlu refit'i kis ma -- ucuz varyans azaltmadir",
            "  Asil kit kaynak SENIN DIKKATIN. Hipotezleri sirala, ilk 5'i iyi sec.",
        ]
    else:
        satirlar += [
            "",
            "  YORUM: Hesap SIKISIK. Once maliyeti dusur:",
            "    * Feature secimini erken yap (SHAP geri eleme), kolonu yarila",
            "    * Fold sayisini 5'ten 3'e indir, sonda 5'e cik",
            "    * Optuna'yi alt-ornekte (%20 satir) kos, kazanani tam veride dogrula",
            f"    * CatBoost {sureler.get('CatBoost tek CV (500 iter)', 0):.0f} sn -- "
            f"LightGBM'in {sureler.get('CatBoost tek CV (500 iter)', 0) / lgbm_tek:.0f}x'i. "
            "Arastirmayi LightGBM'le yap, CatBoost'u sadece final harmana kat.",
        ]

    darbogaz = max(sonuc.olcumler, key=lambda o: o.saniye)
    en_agir_bellek = max(sonuc.olcumler, key=lambda o: o.bellek_mb)
    satirlar += [
        "",
        f"  EN YAVAS   : {darbogaz.ad} ({darbogaz.saniye:.0f} sn)",
        f"  EN OBUR    : {en_agir_bellek.ad} (+{en_agir_bellek.bellek_mb:.0f} MB)",
    ]
    if zoo > 0:
        satirlar.append(
            f"  Model zoo bir kosuda {zoo / 60:.1f} dk -- gunde en fazla "
            f"{int(GUNLUK_CALISMA_SAATI * 3600 * 0.5 / zoo)} kez."
        )
    satirlar.append("=" * 68)
    return "\n".join(satirlar)


def main() -> int:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--hizli", action="store_true", help="sadece 100k olc")
    ayristirici.add_argument("--agir", action="store_true", help="2.5M'i da gercekten olc")
    args = ayristirici.parse_args()

    olcekler = [100_000]
    if not args.hizli:
        olcekler.append(500_000)
    if args.agir:
        olcekler.append(2_500_000)

    print("=" * 68)
    print("YARISMA OLCEGINDE ZAMAN VE BELLEK PROVASI")
    print("=" * 68)
    try:
        import psutil

        toplam = psutil.virtual_memory().total / 1024**3
        print(f"  makine: {psutil.cpu_count()} cekirdek, {toplam:.0f} GB RAM")
    except ImportError:
        print("  (psutil yok -- bellek olcumu atlanacak)")

    sonuclar: dict[int, OlcekSonucu] = {}
    for olcek in olcekler:
        sonuclar[olcek] = olcek_kos(olcek, tam=(olcek <= 500_000))
        gc.collect()

    print("\n" + "=" * 68)
    print("SONUC TABLOSU")
    print("=" * 68)
    for olcek, sonuc in sonuclar.items():
        print(f"\n--- {olcek:,} satir ---")
        print(sonuc.tablo().to_string(index=False))

    # Olculmemis olcegi dogrusal kestir (agac modelleri ~O(n log n), pratikte
    # bu araliklarda dogrusala yakin). ACIKCA TAHMIN olarak isaretlenir.
    if 2_500_000 not in sonuclar and len(sonuclar) >= 2:
        kucuk, buyuk = sorted(sonuclar)[:2]
        oran = 2_500_000 / buyuk
        kestirim = OlcekSonucu(satir=2_500_000, kolon=sonuclar[buyuk].kolon)
        for olcum in sonuclar[buyuk].olcumler:
            kestirim.olcumler.append(
                Olcum(olcum.ad, olcum.saniye * oran, olcum.bellek_mb * oran,
                      "dogrusal kestirim", tahmin=True)
            )
        sonuclar[2_500_000] = kestirim
        print(f"\n--- 2,500,000 satir (KESTIRIM, {buyuk:,}'den dogrusal) ---")
        print(kestirim.tablo().to_string(index=False))

    hedef = 500_000 if 500_000 in sonuclar else min(sonuclar)
    print(butce_raporu(sonuclar, hedef))

    cikti = ROOT / "experiments" / "olcek_provasi.json"
    cikti.parent.mkdir(exist_ok=True)
    cikti.write_text(
        json.dumps(
            {
                str(k): {
                    "satir": v.satir,
                    "kolon": v.kolon,
                    "olcumler": [
                        {"ad": o.ad, "saniye": o.saniye, "bellek_mb": o.bellek_mb,
                         "tahmin": o.tahmin}
                        for o in v.olcumler
                    ],
                }
                for k, v in sonuclar.items()
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nKaydedildi: {cikti}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
