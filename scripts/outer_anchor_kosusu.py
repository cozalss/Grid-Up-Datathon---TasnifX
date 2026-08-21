"""BAGIMSIZ dis capa (outer anchor) kosusu -- kazanan kapisini ateslemek icin.

NEDEN BU BETIK
--------------
2026-08-18 denetimi (P1-13): ``bilimsel_kazanan_karari`` en az ALTI bagimsiz,
eslestirilmis outer anchor istiyor ama hicbir betik bunlari URETMIYORDU.
Sonuc: ``kazanan=null``, her koşuda. Yani "hangi model gercekten en iyi"
sorusu hep acik kaliyordu ve ekip ic OOF siralamasina bakmak zorundaydi --
tam da kapinin engellemeye calistigi sey.

BAGIMSIZLIK NE DEMEK (kapinin sordugu sey)
------------------------------------------
Her capada model secimi ve harman agirliklari YALNIZCA o capanin egitim
bolumunde (inner CV) belirlenir; outer dilim hicbir ayara dokunmaz. Hazir
tek bir OOF dizisini parcalara bolmek bagimsiz kanit SAYILMAZ -- cunku o OOF
zaten tum veriyi gormus bir secim sureciyle uretilmistir.

TASARIM (rolling origin, ic ice)
--------------------------------
Capa k icin:
  1. ``outer_valid`` = son dilimden k adim geriye 31 gunluk pencere,
     ``outer_train`` = onundeki TUM gunler.
  2. ``outer_train`` icinde 2 fold'luk INNER purged CV kosulur; harman
     agirliklari YALNIZCA burada tirmanilir.
  3. Her aday ``outer_train``in tamamiyla egitilir, ``outer_valid``de
     skorlanir. Harman, inner agirliklarla kurulur.
  4. Capanin skorlari (aday -> MAE) kaydedilir; fingerprint'ler capanin
     kendi fold yapisindan turetilir.

Cikti dogrudan ``OuterEvidence``e cevrilebilir bicimdedir ve
``benchmark_gercek.py --outer experiments/outer_anchors.json`` ile tuketilir.

KULLANIM
    python scripts/outer_anchor_kosusu.py              # 8 capa
    python scripts/outer_anchor_kosusu.py --capa 6     # daha hizli
Cikti: experiments/outer_anchors.json
"""

from __future__ import annotations

import argparse
import hashlib
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

import benchmark_gercek as bench  # noqa: E402  (ayni panel + feature receti)

from gridup import cross_validate, set_global_seed  # noqa: E402
from gridup.ensemble import hill_climb_weights  # noqa: E402
from gridup.io_utils import atomic_write_bytes  # noqa: E402
from gridup.metrics import get_metric  # noqa: E402
from gridup.models import starter_params  # noqa: E402
from gridup.reporting import satir_tamponlu_cikti  # noqa: E402
from gridup.validation import purged_time_series_split  # noqa: E402

CIKTI = KOK / "experiments" / "outer_anchors.json"

#: Capa basina dogrulama penceresi (gun) -- test blogu uzunlugunu taklit eder.
CAPA_PENCERESI = 31
#: Her capanin egitim bolumunde kosulan ic fold sayisi (agirlik/secim icin).
IC_FOLD = 2
#: Adaylar: tekil modeller + inner-agirlikli harman. Az ve CESITLI tutuldu --
#: her capa bunlarin hepsini yeniden egitir, maliyet capa sayisiyla carpilir.
ADAYLAR: tuple[str, ...] = ("catboost_mae", "lgb_mae", "lgb_tweedie", "harman")


def _tarifler() -> dict[str, tuple[str, dict[str, Any]]]:
    tweedie = starter_params("lightgbm", "regression", objective="tweedie")
    tweedie["tweedie_variance_power"] = 1.3
    catboost = starter_params("catboost", "regression", objective="mae")
    catboost["eval_metric"] = "MAE"
    return {
        "catboost_mae": ("catboost", catboost),
        "lgb_mae": ("lightgbm", starter_params("lightgbm", "regression", objective="mae")),
        "lgb_tweedie": ("lightgbm", tweedie),
    }


def _parmak_izi(*parcalar: object) -> str:
    ozet = hashlib.sha256()
    for parca in parcalar:
        ozet.update(str(parca).encode("utf-8"))
    # TAM SHA-256: OuterAnchor kisaltilmis parmak izini REDDEDER (64 hane
    # sart). Kisaltma "provenance var" gorunumu verip carpisma alanini
    # kucultur; kapi bilerek katidir.
    return ozet.hexdigest()


def capa_kos(
    x: pd.DataFrame,
    y: np.ndarray,
    zaman: pd.Series,
    *,
    egitim_maske: np.ndarray,
    valid_maske: np.ndarray,
) -> dict[str, float]:
    """Tek capa: inner CV ile agirlik/secim, outer dilimde skor.

    Outer dilim hicbir secime katilmaz -- ne model, ne harman agirligi.
    """
    mae_fn, _, _ = get_metric("mae")
    egitim_idx = np.flatnonzero(egitim_maske)
    valid_idx = np.flatnonzero(valid_maske)
    x_egitim, y_egitim = x.iloc[egitim_idx], y[egitim_idx]
    x_valid, y_valid = x.iloc[valid_idx], y[valid_idx]

    ic_foldlar = purged_time_series_split(
        zaman.iloc[egitim_idx].reset_index(drop=True),
        embargo=pd.Timedelta(0),
        n_splits=IC_FOLD,
        test_span=pd.Timedelta(days=CAPA_PENCERESI),
        verbose=False,
    )

    tarifler = _tarifler()
    ic_oof: dict[str, np.ndarray] = {}
    ic_kapsam: dict[str, np.ndarray] = {}
    outer_tahmin: dict[str, np.ndarray] = {}
    skorlar: dict[str, float] = {}

    for ad, (kind, params) in tarifler.items():
        # 1) IC CV -- yalnizca agirlik/secim icin (outer'a dokunmaz).
        ic = cross_validate(
            x_egitim,
            y_egitim,
            ic_foldlar,
            kind=kind,
            metric="mae",
            params=bench._butceli(kind, params),
            early_stopping_rounds=bench.ERKEN_DURDURMA,
            early_stopping_metric="mae",
            verbose=False,
        )
        ic_oof[ad] = ic.oof_predictions
        ic_kapsam[ad] = ic.oof_covered

        # 2) OUTER: ayni tarif, egitim boluminde CV ile fit; test = outer dilim.
        dis = cross_validate(
            x_egitim,
            y_egitim,
            ic_foldlar,
            kind=kind,
            metric="mae",
            params=bench._butceli(kind, params),
            test=x_valid,
            early_stopping_rounds=bench.ERKEN_DURDURMA,
            early_stopping_metric="mae",
            verbose=False,
        )
        assert dis.test_predictions is not None
        outer_tahmin[ad] = dis.test_predictions
        skorlar[ad] = float(mae_fn(y_valid, dis.test_predictions))

    # 3) HARMAN: agirliklar YALNIZCA inner OOF'ta tirmanilir.
    ortak = np.ones(len(y_egitim), dtype=bool)
    for ad in tarifler:
        ortak &= ic_kapsam[ad]
    indeks = np.flatnonzero(ortak)
    agirliklar = hill_climb_weights(
        {ad: ic_oof[ad][indeks] for ad in tarifler},
        y_egitim[indeks],
        metric="mae",
        covered=np.ones(indeks.size, dtype=bool),
        verbose=False,
    )
    harman = np.zeros(len(y_valid))
    for ad, agirlik in agirliklar.items():
        harman += agirlik * outer_tahmin[ad]
    skorlar["harman"] = float(mae_fn(y_valid, harman))
    skorlar["_agirliklar"] = agirliklar  # type: ignore[assignment]
    return skorlar


def main() -> int:
    satir_tamponlu_cikti()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capa", type=int, default=8, help="Capa sayisi (en az 6)")
    parser.add_argument("--out", default=str(CIKTI))
    args = parser.parse_args()

    if args.capa < 6:
        print(f"UYARI: {args.capa} capa, kapinin istedigi 6'nin altinda; kazanan ilan edilemez.")

    if not bench.VERI.exists():
        print(f"HATA: {bench.VERI} yok.")
        return 1

    set_global_seed(42)
    baslangic = time.perf_counter()

    print("1/3  panel + feature")
    panel = bench.panel_kur()
    ozellik, kolonlar = bench.ozellik_kur(panel)
    y = ozellik[bench.HEDEF].to_numpy()
    zaman = pd.to_datetime(ozellik[bench.ZAMAN])
    print(f"  {len(ozellik):,} satir, {len(kolonlar)} feature")

    son_gun = zaman.max().normalize()
    print(f"\n2/3  {args.capa} capa (her biri {CAPA_PENCERESI} gun, ic {IC_FOLD} fold)")
    capalar: list[dict[str, Any]] = []
    for k in range(args.capa):
        valid_son = son_gun - pd.Timedelta(days=k * CAPA_PENCERESI)
        valid_bas = valid_son - pd.Timedelta(days=CAPA_PENCERESI - 1)
        valid_maske = (zaman >= valid_bas) & (zaman <= valid_son)
        egitim_maske = zaman < valid_bas
        if int(egitim_maske.sum()) < 5000 or int(valid_maske.sum()) == 0:
            print(f"  capa {k + 1}: yetersiz egitim verisi, durduruldu")
            break

        skorlar = capa_kos(
            ozellik[kolonlar],
            y,
            zaman,
            egitim_maske=egitim_maske.to_numpy(),
            valid_maske=valid_maske.to_numpy(),
        )
        agirliklar = skorlar.pop("_agirliklar")
        capalar.append(
            {
                "anchor_id": f"capa{k + 1:02d}",
                "train_end": str((valid_bas - pd.Timedelta(days=1)).date()),
                "validation_start": str(valid_bas.date()),
                "validation_end": str(valid_son.date()),
                "scores": {ad: round(float(skorlar[ad]), 4) for ad in ADAYLAR},
                "recipe_fingerprint": _parmak_izi(sorted(kolonlar), sorted(ADAYLAR)),
                "fold_fingerprint": _parmak_izi(
                    valid_bas, valid_son, int(egitim_maske.sum()), IC_FOLD
                ),
                "harman_agirliklari": {
                    a: round(float(w), 4) for a, w in agirliklar.items() if w > 0
                },
                "n_egitim": int(egitim_maske.sum()),
                "n_valid": int(valid_maske.sum()),
            }
        )
        en_iyi = min(ADAYLAR, key=lambda ad: skorlar[ad])
        print(
            f"  capa {k + 1}/{args.capa} [{valid_bas.date()}..{valid_son.date()}] "
            f"en iyi={en_iyi} " + " ".join(f"{ad}={skorlar[ad]:.2f}" for ad in ADAYLAR)
        )

    print("\n3/3  ozet")
    if not capalar:
        print("  Hicbir capa uretilemedi.")
        return 1
    matris = {ad: np.array([c["scores"][ad] for c in capalar]) for ad in ADAYLAR}
    for ad in sorted(ADAYLAR, key=lambda a: matris[a].mean()):
        kazanma = int(sum(1 for c in capalar if min(c["scores"], key=c["scores"].get) == ad))
        print(
            f"  {ad:<16} ortalama {matris[ad].mean():7.2f}  medyan {np.median(matris[ad]):7.2f}  "
            f"capa kazanci {kazanma}/{len(capalar)}"
        )

    # KRONOLOJIK SIRA: capalar sondan geriye uretilir ama OuterEvidence
    # kronolojik siralilik ister (kapinin dogrulamalarindan biri). Sirala.
    capalar.sort(key=lambda c: c["validation_start"])
    sonuc = {
        "capa_sayisi": len(capalar),
        "capa_penceresi_gun": CAPA_PENCERESI,
        "ic_fold": IC_FOLD,
        "adaylar": list(ADAYLAR),
        "hedef": bench.HEDEF,
        "anchors": capalar,
    }
    atomic_write_bytes(
        Path(args.out), (json.dumps(sonuc, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )
    print(f"\nYazildi: {args.out}")
    print(f"  benchmark_gercek.py --outer {args.out} ile kazanan kapisi ateslenir")
    print(f"Toplam sure: {time.perf_counter() - baslangic:.0f} sn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
