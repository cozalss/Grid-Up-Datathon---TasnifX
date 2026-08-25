# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""KUVVET ORTALAMASI = GIZLI SEVIYE KAYDIRMASI MI?

Soguk tarafta q>0 tekdüze kazandiriyor. Iki alternatif aciklama var:
  A) birlestiricinin BICIMI gercekten daha iyi (kuvvet ortalamasi),
  B) q>0 sadece tahmini YUKARI itiyor ve soguk uzman kis26'da sistematik
     olarak ASAGI tahmin ediyor -- yani kazanc SEVIYE ekseninden.

B dogruysa bulgu tasinamaz: test Nisan-Temmuz, kis26 kis. Seviye
yanliligi mevsime baglidir.

    python scripts/deney_kuvvet_seviye.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import olcut as ol  # noqa: E402
import tuketim_model as tm  # noqa: E402
from deney_kuvvet_ekseni import AGIRLIK, ONB, SOG, SOGUK_KATSAYI, TOHUMLAR, kuvvet_ort  # noqa: E402

BETA = 0.60


def main() -> int:
    egitim, test = d.cerceveleri_kur()

    # --- A) SICAK: blok bazinda SEVIYE YANLILIGI (mevsim isareti) ---
    guc_kenar = ol.guc_kenarlari(test)
    te_s = test[test["soguk_mu"] != 1]
    aileler = list(AGIRLIK)
    agir = [AGIRLIK[a] for a in aileler]
    print("  SICAK: agirlikli ortalama artik  (gercek - tahmin, log birimi)")
    print(f"  {'blok':8}{'ay araligi':>14}{'ort artik':>12}{'>0 pay':>9}")
    for b in tm.BLOKLAR:
        _, dogrulama, gercek, soguk = di.blok_parcalari(egitim, b.ad)
        sicak = ~soguk
        dg = dogrulama[sicak]
        w, _ = ol.test_agirliklari(dg, te_s, guc_kenar, eksenler=("bayatlik",))
        y = np.load(ONB / f"{b.ad}_gercek.npy").astype("float64")
        lt = np.mean(
            [
                kuvvet_ort(
                    [
                        np.load(ONB / f"{b.ad}_{t}_{a}_uretim.npy").astype("float64")
                        for a in aileler
                    ],
                    agir,
                    0.0,
                )
                for t in TOHUMLAR
            ],
            axis=0,
        )
        r = np.log1p(np.clip(y, 0, None)) - lt
        ay = pd.to_datetime(dg["tarih"]).dt.month
        print(
            f"  {b.ad:8}{f'{ay.min()}-{ay.max()}':>14}"
            f"{float(np.dot(w, r) / w.sum()):+12.5f}{float(np.dot(w, r > 0) / w.sum()):9.3f}"
        )

    # --- B) SOGUK kis26: q ile sabit kaymayi YARISTIR ---
    _, dogrulama, gercek, soguk = di.blok_parcalari(egitim, "kis26")
    dg = dogrulama[soguk]
    y = gercek[soguk]
    log_guc = np.log1p(dg["guc"].to_numpy(dtype="float64"))
    te_c = test[test["soguk_mu"] == 1]
    w, _ = ol.test_agirliklari(dg, te_c, ol.guc_kenarlari(te_c), eksenler=("guc",))
    tah = [np.load(SOG / f"kis26_{t}_taban.npy").astype("float64") for t in TOHUMLAR]

    def buz(log_t):
        r = log_t - log_guc
        return np.clip(np.expm1(r.mean() + BETA * (r - r.mean()) + log_guc), 0.0, None)

    taban_l = kuvvet_ort(tah, [1.0] * 3, 0.0)
    taban = ol.agirlikli_rmsle(y, buz(taban_l), w)
    r0 = np.log1p(np.clip(y, 0, None)) - np.log1p(buz(taban_l))
    print(
        f"\n  SOGUK kis26: agirlikli ort artik {float(np.dot(w, r0) / w.sum()):+.5f}  "
        f"(pozitif = model ASAGI tahmin ediyor)"
    )
    print(f"  URETIM (q=0) {taban:.5f}")
    print(
        f"\n  {'q':>7}{'ort kayma':>11}{'skor':>11}{'fark':>11}   "
        f"{'ayni kaymali SABIT':>20}{'fark':>10}"
    )
    for q in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        lq = kuvvet_ort(tah, [1.0] * 3, q)
        kay = float(np.dot(w, lq - taban_l) / w.sum())
        sq = ol.agirlikli_rmsle(y, buz(lq), w)
        sc = ol.agirlikli_rmsle(y, buz(taban_l + kay), w)
        print(f"  {q:+7.1f}{kay:+11.5f}{sq:11.5f}{taban - sq:+11.5f}{sc:20.5f}{taban - sc:+10.5f}")
    lmax = np.max(np.stack(tah), axis=0)
    kay = float(np.dot(w, lmax - taban_l) / w.sum())
    smax = ol.agirlikli_rmsle(y, buz(lmax), w)
    sc = ol.agirlikli_rmsle(y, buz(taban_l + kay), w)
    print(
        f"  {'maks':>7}{kay:+11.5f}{smax:11.5f}{taban - smax:+11.5f}{sc:20.5f}{taban - sc:+10.5f}"
    )

    print("\n  SABIT KAYMA TARAMASI (kuvvet ortalamasi YOK)")
    print(f"  {'c':>7}{'skor':>11}{'fark':>11}{'genele':>10}")
    for c in (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
        s = ol.agirlikli_rmsle(y, buz(taban_l + c), w)
        print(f"  {c:+7.2f}{s:11.5f}{taban - s:+11.5f}{-(taban - s) * SOGUK_KATSAYI:+10.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
