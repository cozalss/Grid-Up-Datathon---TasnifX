"""Gercek GDZ verisinde ZAMAN BUTCELI hiperparametre aramasi (P1-15).

NEDEN BU BETIK
--------------
2026-08-18 denetimi: benchmark'in tum uyeleri ``starter_params`` varsayilaniyla
kosuyordu; hiperparametre gercek veride HIC ayarlanmamisti. 2024 GDZ birincisi
(Pikachow) objective'i bile Optuna arama uzayina koymustu.

KARAR KURALI -- GURULTUYU GECMEYEN AYAR ALINMAZ
-----------------------------------------------
Tohum gurultusu OLCULDU: ayni CatBoost 5 tohumda 301,21-304,80 (yayilim 1,24).
Bu yuzden Optuna'nin buldugu ayar, varsayilani **yalnizca eslestirilmis fold
farki gurultuyu asiyorsa** kazanmis sayilir. Betik iki sayiyi da yazar:
  * ham fark        : en iyi deneme - varsayilan
  * eslestirilmis   : fold basina fark, ortalama +- std
Karar cumlesi JSON'a girer; "en dusuk skoru al" kestirmesi YAPILMAZ.

SURE BUTCESI
------------
``--dakika`` ile sinirlanir (varsayilan 40). Olcek provasi: 100k satirda bir
Optuna denemesi ~22 sn -> 40 dk ~ 100 deneme. Yarisma gunu bu butce
pazarlik konusudur; betik butceyi ASMAZ.

KULLANIM
    python scripts/tune_gercek.py                    # catboost, 40 dk
    python scripts/tune_gercek.py --model lightgbm --dakika 20
Cikti: experiments/tuning_gercek_<model>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import benchmark_gercek as bench  # noqa: E402

from gridup import cross_validate, set_global_seed  # noqa: E402
from gridup.io_utils import atomic_write_bytes  # noqa: E402
from gridup.models import starter_params  # noqa: E402
from gridup.tuning import tune_with_optuna  # noqa: E402
from gridup.validation import purged_time_series_split  # noqa: E402

#: Tohum gurultusu (benchmark_gercek "tohum_kararliligi" ile olculdu, MAE dk).
#: Bu esigin altindaki kazanc, ayarin degil tohumun eseridir.
TOHUM_GURULTUSU = 1.24


def _varsayilan(model: str) -> dict[str, Any]:
    if model == "catboost":
        params = starter_params("catboost", "regression", objective="mae")
        params["eval_metric"] = "MAE"
        return params
    return starter_params("lightgbm", "regression", objective="mae")


def _fold_skorlari(
    x: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    model: str,
    params: dict[str, Any],
) -> list[float]:
    sonuc = cross_validate(
        x,
        y,
        folds,
        kind=model,
        metric="mae",
        params=bench._butceli(model, params),
        early_stopping_rounds=bench.ERKEN_DURDURMA,
        early_stopping_metric="mae",
        verbose=False,
    )
    return [float(v) for v in sonuc.fold_scores]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("catboost", "lightgbm"), default="catboost")
    parser.add_argument("--dakika", type=int, default=40, help="Arama sure butcesi")
    parser.add_argument("--deneme", type=int, default=200, help="Ust sinir (sure once biter)")
    args = parser.parse_args()

    if not bench.VERI.exists():
        print(f"HATA: {bench.VERI} yok.")
        return 1

    set_global_seed(42)
    baslangic = time.perf_counter()

    print("1/3  panel + feature")
    panel = bench.panel_kur()
    ozellik, kolonlar = bench.ozellik_kur(panel)
    y = ozellik[bench.HEDEF].to_numpy()
    folds = purged_time_series_split(
        ozellik[bench.ZAMAN],
        embargo=pd.Timedelta(0),
        n_splits=4,
        test_span=pd.Timedelta(days=bench.UFUK),
        verbose=False,
    )
    x = ozellik[kolonlar]
    print(f"  {len(ozellik):,} satir, {len(kolonlar)} feature, {len(folds)} fold")

    print("\n2/3  varsayilan taban")
    varsayilan = _varsayilan(args.model)
    taban_foldlar = _fold_skorlari(x, y, folds, args.model, varsayilan)
    taban = float(np.mean(taban_foldlar))
    yuvarlak = [round(v, 1) for v in taban_foldlar]
    print(f"  {args.model} varsayilan: MAE {taban:.2f}  foldlar {yuvarlak}")

    print(f"\n3/3  Optuna ({args.dakika} dk butce, en fazla {args.deneme} deneme)")
    sonuc = tune_with_optuna(
        x,
        y,
        folds,
        kind=args.model,
        metric="mae",
        n_trials=args.deneme,
        timeout=args.dakika * 60,
        search_objective=False,  # objective zaten MAE; metrikle sabit
        early_stopping_rounds=bench.ERKEN_DURDURMA,
        seed=42,
        verbose=False,
    )
    en_iyi = dict(varsayilan)
    en_iyi.update(sonuc.best_params)
    aday_foldlar = _fold_skorlari(x, y, folds, args.model, en_iyi)
    aday = float(np.mean(aday_foldlar))

    farklar = np.array(taban_foldlar) - np.array(aday_foldlar)  # + = aday daha iyi
    ort, sapma = float(farklar.mean()), float(farklar.std())
    gecti = bool(ort > TOHUM_GURULTUSU and ort > sapma)
    karar = (
        f"KABUL: eslestirilmis kazanc {ort:+.2f} +- {sapma:.2f} MAE, tohum gurultusunun "
        f"({TOHUM_GURULTUSU}) USTUNDE."
        if gecti
        else (
            f"RED: eslestirilmis kazanc {ort:+.2f} +- {sapma:.2f} MAE; tohum gurultusu "
            f"{TOHUM_GURULTUSU} MAE. Ayar varsayilani gecmiyor -- varsayilanla devam."
        )
    )
    print(f"  en iyi deneme    : MAE {sonuc.best_score:.2f} ({sonuc.n_trials} deneme)")
    print(f"  yeniden olculdu  : MAE {aday:.2f}  foldlar {[round(v, 1) for v in aday_foldlar]}")
    print(f"  {karar}")

    cikti = KOK / "experiments" / f"tuning_gercek_{args.model}.json"
    atomic_write_bytes(
        cikti,
        (
            json.dumps(
                {
                    "model": args.model,
                    "n_trials": int(sonuc.n_trials),
                    "sure_dk": round((time.perf_counter() - baslangic) / 60, 1),
                    "varsayilan_mae": taban,
                    "varsayilan_fold": taban_foldlar,
                    "aday_mae": aday,
                    "aday_fold": aday_foldlar,
                    "optuna_best_score": float(sonuc.best_score),
                    "best_params": sonuc.best_params,
                    "eslestirilmis_kazanc": ort,
                    "eslestirilmis_sapma": sapma,
                    "tohum_gurultusu": TOHUM_GURULTUSU,
                    "kabul": gecti,
                    "karar": karar,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        ).encode("utf-8"),
    )
    print(f"\nYazildi: {cikti}")
    print(f"Toplam sure: {(time.perf_counter() - baslangic) / 60:.1f} dk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
