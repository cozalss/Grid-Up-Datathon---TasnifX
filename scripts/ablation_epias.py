"""96 ILCELIK EPIAS panelinde harici veri ablasyonu -- ADM dahil ilk prova.

NEDEN AYRI BETIK (ablation_gercek.py yaninda)
---------------------------------------------
``ablation_gercek.py`` 2021-22 GDZ aynasinda, 47 ILCEDE kosar. O prova
degerlidir ama iki sey eksiktir: **ADM'nin 49 ilcesi** ve **2023-26 rejimi**.
Yarisma bolgesi 96 ilcedir ve test donemi 2026'dir. Bu betik ayni ablasyonu
EPIAS panelinde kosar (2022-01..2026-08, 92 ilce kayitli, 1284 kapsanan gun).

IKI PANEL, IKI HEDEF -- IKISI DE GERCEK
---------------------------------------
* ``kesinti_adet``  : gunluk kesinti SAYISI  (2024 GDZ hedefi bu tipteydi)
* ``kesinti_dk``    : gunluk toplam SURE     (ayna provasinin hedefi)
Varsayilan ``adet``; ``--hedef dk`` ile digeri olculur. Ayni feature seti,
ayni fold semasi -- boylece "hedef tipi degisince siralama degisiyor mu"
sorusu OLCUMLE yanitlanir.

KAPSAMA
-------
Panel yalnizca EPIAS'in yayimladigi gunleri icerir (``epias_panel.py``
kapsanan_gun=1). Bosluklar sahte sifir olarak girmez; ama pencere
sureklidir DIYE VARSAYILMAZ: lag/rolling satir kaydirmasi kullandigi icin
panel gunluk IZGARAYA tamamlanir ve bosluk gunleri feature uretiminden
SONRA atilir (pipeline.build_paired_history_features ile ayni yaklasim).

KULLANIM
    python scripts/ablation_epias.py               # hedef: kesinti_adet
    python scripts/ablation_epias.py --hedef dk    # hedef: kesinti_dk
Cikti: experiments/ablasyon_epias_<hedef>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))

from gridup import cross_validate, set_global_seed  # noqa: E402
from gridup.ablation import FeatureGroup, leave_one_group_out  # noqa: E402
from gridup.features import (  # noqa: E402
    add_calendar_features,
    add_event_decay_features,
    add_expanding_features,
    add_lag_features,
    add_mass_event_features,
    add_turkish_holiday_features,
)
from gridup.features.external import attach_external  # noqa: E402
from gridup.io_utils import atomic_write_bytes  # noqa: E402
from gridup.models import starter_params  # noqa: E402
from gridup.validation import purged_time_series_split  # noqa: E402

#: TAM IZGARA panel (bosluk gunleri dahil, kapsanan_gun bayrakli). Feature
#: uretimi bunun uzerinde yapilir; skorlama YALNIZCA kapsanan gunlerde.
PANEL = KOK / "data" / "external" / "epias" / "panel_ilce_gun_tam.parquet"
ZAMAN = "gun"
GRUP = "ilce_key"
UFUK = 31
SHIFT_OFSETLERI = (31, 62, 93)
YARI_OMURLER = (3.0, 14.0)
#: Ayni gunun bilgisi -- feature olamaz.
AYNI_GUN = ("kesinti_adet", "kesinti_dk", "etkilenen_abone", "kapsanan_gun")


def cekirdek_feature(panel: pd.DataFrame, hedef: str) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Takvim/tatil/lag/bozunum/kitle-olay/genisleyen aileleri kurar."""
    aileler: dict[str, list[str]] = {}
    frame = panel

    def kaydet(ad: str, once: set[str]) -> None:
        aileler[ad] = [
            k
            for k in frame.columns
            if k not in once and k not in AYNI_GUN and pd.api.types.is_numeric_dtype(frame[k])
        ]

    once = set(frame.columns)
    frame = add_calendar_features(frame, ZAMAN, include_year=False)
    kaydet("takvim", once)

    once = set(frame.columns)
    frame = add_turkish_holiday_features(frame, ZAMAN)
    kaydet("tatil", once)

    once = set(frame.columns)
    frame = add_lag_features(
        frame, hedef, shifts=SHIFT_OFSETLERI, time_column=ZAMAN, horizon=UFUK, group_columns=[GRUP]
    )
    frame = add_event_decay_features(
        frame, hedef, time_column=ZAMAN, horizon=UFUK, group_columns=[GRUP], half_lives=YARI_OMURLER
    )
    frame = add_mass_event_features(
        frame, hedef, time_column=ZAMAN, horizon=UFUK, group_columns=[GRUP]
    )
    kaydet("lag", once)

    once = set(frame.columns)
    frame = add_expanding_features(
        frame,
        hedef,
        time_column=ZAMAN,
        horizon=UFUK,
        group_columns=[GRUP],
        aggregations=("mean", "median", "std"),
    )
    kaydet("ilce_gecmisi", once)

    return frame, aileler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hedef", choices=("adet", "dk"), default="adet")
    parser.add_argument("--hizli", action="store_true", help="3 fold (varsayilan 4)")
    args = parser.parse_args()

    if not PANEL.exists():
        print(f"HATA: {PANEL} yok. Once scripts/epias_panel.py calistir.")
        return 1

    set_global_seed(42)
    baslangic = time.perf_counter()
    hedef = "kesinti_adet" if args.hedef == "adet" else "kesinti_dk"

    print("1/4  PANEL")
    panel = pd.read_parquet(PANEL)
    panel = panel.sort_values([GRUP, ZAMAN]).reset_index(drop=True)
    if "kapsanan_gun" not in panel.columns:
        raise RuntimeError(
            f"{PANEL.name}: kapsanan_gun kolonu yok. "
            "epias_panel.py --tum-gunler ile uretilmis TAM IZGARA gerekiyor."
        )
    kapsanan_maske = panel["kapsanan_gun"].to_numpy() == 1
    # BOSLUK GUNLERININ HEDEFI BILINMIYOR: 0 degil NaN. Lag/rolling satir
    # kaydirdigi icin izgara SUREKLI olmali (yoksa "31 satir once" 40 gun
    # oncesine denk gelir); ama o gunlerin hedefi feature'a SIZMAMALI.
    panel.loc[~kapsanan_maske, ["kesinti_adet", "kesinti_dk", "etkilenen_abone"]] = float("nan")
    print(
        f"  {len(panel):,} satir = {panel[GRUP].nunique()} ilce x "
        f"{panel[ZAMAN].nunique()} gun  |  hedef={hedef}"
    )
    print(
        f"  kapsanan: {int(kapsanan_maske.sum()):,} satir "
        f"(%{100 * kapsanan_maske.mean():.1f}); skorlama YALNIZCA bunlarda"
    )
    print(f"  sifir orani (kapsanan) %{(panel.loc[kapsanan_maske, hedef] == 0).mean() * 100:.1f}")

    print("\n2/4  AILELER")
    ozellik, aileler = cekirdek_feature(panel, hedef)
    ek = attach_external(ozellik, key_column=GRUP, time_column=ZAMAN, horizon=UFUK, root=KOK)
    if len(ek.frame) != len(panel):
        raise RuntimeError("attach_external satir sayisini degistirdi")
    ozellik = ek.frame
    for ad, kolonlar in ek.families.items():
        sayisal = [
            k for k in kolonlar if k not in AYNI_GUN and pd.api.types.is_numeric_dtype(ozellik[k])
        ]
        if sayisal:
            aileler[ad] = sayisal
    for ad, kolonlar in aileler.items():
        print(f"  {ad:<14} {len(kolonlar):>3} kolon")
    for ad, neden in ek.skipped.items():
        print(f"  {ad:<14} ATLANDI: {neden}")

    kolonlar = [k for kols in aileler.values() for k in kols]
    if len(kolonlar) != len(set(kolonlar)):
        raise RuntimeError("Aileler arasinda kolon cakismasi var")
    sizinti = [k for k in kolonlar if k in AYNI_GUN]
    if sizinti:
        raise RuntimeError(f"Ayni gun kolonu feature listesine girdi: {sizinti}")
    print(f"  toplam {len(kolonlar)} feature")

    # Bosluk satirlari feature URETIMINE girdi (izgarayi surekli kilmak icin),
    # egitim/skorlamaya GIRMEZ: hedefleri bilinmiyor.
    ozellik = ozellik.loc[kapsanan_maske].reset_index(drop=True)
    print(f"  kapsanan satirlara indirgendi: {len(ozellik):,}")

    folds = purged_time_series_split(
        ozellik[ZAMAN],
        embargo=pd.Timedelta(0),
        n_splits=3 if args.hizli else 4,
        test_span=pd.Timedelta(days=UFUK),
        verbose=False,
    )

    print("\n3/4  TAM MODEL")
    params = starter_params("lightgbm", "regression", objective="mae")
    y = ozellik[hedef].to_numpy()
    tam = cross_validate(
        ozellik[kolonlar], y, folds, kind="lightgbm", metric="mae", params=params, verbose=False
    )
    kapsanan, _ = tam.covered_predictions()
    sifir_baseline = float(np.abs(y[kapsanan]).mean())
    print(f"  tam MAE       : {tam.overall_score:.4f}")
    print(f"  sifir-baseline: {sifir_baseline:.4f}")
    print(f"  fold skorlari : {[round(v, 3) for v in tam.fold_scores]}")

    print("\n4/4  LEAVE-ONE-GROUP-OUT")
    gruplar = [FeatureGroup(ad, tuple(kols)) for ad, kols in aileler.items()]
    tablo = leave_one_group_out(
        ozellik[kolonlar],
        y,
        folds,
        groups=gruplar,
        kind="lightgbm",
        metric="mae",
        params=params,
        verbose=False,
    )
    # delta = mae_ailesiz - mae_tam; POZITIF delta = aile katki veriyor.
    olculen = {
        satir.grup: {
            "mae_ailesiz": round(float(satir.skor_grupsuz), 4),
            "delta": round(float(satir.skor_grupsuz) - float(tam.overall_score), 4),
            "kolon_sayisi": int(satir.feature_sayisi),
        }
        for satir in tablo.itertuples()
    }
    siralama = sorted(olculen, key=lambda ad: -olculen[ad]["delta"])
    print()
    print(f"{'aile':<16}{'delta':>10}{'ailesiz MAE':>14}{'kolon':>7}")
    for ad in siralama:
        bilgi = olculen[ad]
        print(
            f"{ad:<16}{bilgi['delta']:>+10.4f}{bilgi['mae_ailesiz']:>14.4f}"
            f"{bilgi['kolon_sayisi']:>7}"
        )

    sonuc = {
        "hedef": hedef,
        "panel_satir": int(len(ozellik)),
        "izgara_satir": int(len(panel)),
        "ilce": int(ozellik[GRUP].nunique()),
        "gun": int(ozellik[ZAMAN].nunique()),
        "sifir_orani": float((ozellik[hedef] == 0).mean()),
        "tam_mae": float(tam.overall_score),
        "fold_scores": [float(v) for v in tam.fold_scores],
        "sifir_baseline": sifir_baseline,
        "aileler": olculen,
        "siralama": siralama,
        "atlanan": ek.skipped,
    }
    cikti = KOK / "experiments" / f"ablasyon_epias_{args.hedef}.json"
    atomic_write_bytes(cikti, (json.dumps(sonuc, ensure_ascii=False, indent=2) + "\n").encode())
    print(f"\nYazildi: {cikti}")
    print(f"TAMAM ({time.perf_counter() - baslangic:.0f} sn)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
