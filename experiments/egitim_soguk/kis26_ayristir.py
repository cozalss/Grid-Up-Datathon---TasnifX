"""KAZANAN ADAYLARI kis26'da TRAFO BAZINDA AYRISTIR + KIRPMA TABLOSU.

Kalici kural 1 (docs/47): soguk taraftaki her kazanc trafo bazinda ayristirilir;
kirpma tablosu K = 0,1,5,10,25,50 verilmeden kabul edilmez. Gerekce: 1.223
trafolu bir katta tek bir olu trafo t=13,71 uretebiliyor (tuketim_model.py:990).

Tahminleri onbellege yazar; ikinci calistirmada fit yapmaz.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import deney_uretim_ayarlari as rig  # noqa: E402
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

BLOK = "kis26"
TOHUMLAR = (1000, 1001, 1002, 1003, 1004)
P_SOGUK = 0.22159
ADAYLAR = {
    "TABAN_d7": {"depth": 7},
    "random_4": {"depth": 7, "random_strength": 4.0},
    "depth_5": {"depth": 5},
    "lr003_random4": {"depth": 7, "learning_rate": 0.03, "iterations": 400, "random_strength": 4.0},
}
ONBELLEK = KOK / "experiments" / "egitim_soguk" / f"ayristir_{BLOK}.npz"


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    dar, test = d.cerceveleri_kur()
    kolonlar = rig.uretim_kolonlari(dar, test)
    tm.kategorik_kodla(dar, test)
    kalan, dogrulama, gercek, soguk = di.blok_parcalari(dar, BLOK)
    dg = dogrulama[soguk]
    tanim = dg[tm.GRUP].to_numpy()
    y = gercek[soguk]

    ham: dict[str, np.ndarray] = {}
    if ONBELLEK.exists():
        z = np.load(ONBELLEK)
        ham = {k: z[k] for k in z.files}
        print(f"  onbellek okundu: {len(ham)} anahtar")
    for tohum in TOHUMLAR:
        eksik = [a for a in ADAYLAR if f"{tohum}_{a}" not in ham]
        if not eksik:
            continue
        maskeli = d.soguk_maskele(kalan, kolonlar, 1.0, tohum)
        for ad in eksik:
            ham[f"{tohum}_{ad}"] = di.egit_tahmin(
                "cat", maskeli, dogrulama, kolonlar, tohum, **ADAYLAR[ad]
            )[soguk]
            print(f"  {tohum}_{ad} hazir ({time.time() - t0:.0f} sn)", flush=True)
            np.savez_compressed(ONBELLEK, **ham)
        del maskeli

    lgy = np.log1p(np.clip(y, 0, None))

    def kare(ad: str) -> np.ndarray:
        """Tohum ortalamali tahminin satir bazinda kareli hatasi (uretim olcutu)."""
        lg = np.mean([ham[f"{t}_{ad}"] for t in TOHUMLAR], axis=0)
        p = np.clip(np.expm1(lg), 0.0, None)
        return (np.log1p(p) - lgy) ** 2

    tb = kare("TABAN_d7")
    print(f"\n{BLOK} soguk {len(y):,} satir, {pd.Series(tanim).nunique():,} trafo")
    print(f"TABAN_d7 MSE = {tb.mean():.6f}  (RMSLE {np.sqrt(tb.mean()):.6f})\n")
    print(f"{'aday':16}{'dMSE':>11}{'test dMSE':>12}   kirpma tablosu K=0,1,5,10,25,50 (test dMSE)")
    for ad in ADAYLAR:
        if ad == "TABAN_d7":
            continue
        ay = kare(ad)
        fark = ay - tb  # negatif = kazanc
        g = pd.DataFrame({"t": tanim, "f": fark}).groupby("t")["f"].sum().sort_values()
        satir = f"{ad:16}{fark.mean():>+11.6f}{P_SOGUK * fark.mean():>+12.6f}   "
        tablo = []
        for K in (0, 1, 5, 10, 25, 50):
            atil = set(g.index[:K])  # EN COK KAZANDIRAN K trafo atilir
            m = ~pd.Series(tanim).isin(atil).to_numpy()
            tablo.append(f"{P_SOGUK * fark[m].mean():+.5f}")
        satir += " ".join(tablo)
        print(satir)
        kaz = int((g < 0).sum())
        print(
            f"{'':16}kazanan trafo {kaz}/{len(g)} (%{100 * kaz / len(g):.1f})   "
            f"kazanan satir %{100 * (fark < 0).mean():.1f}   "
            f"en iyi tek trafo payi %{100 * g.iloc[0] / fark.sum():.1f}   "
            f"ilk 5 %{100 * g.head(5).sum() / fark.sum():.1f}"
        )
    print(f"\nTAMAM {(time.time() - t0) / 60:.1f} dk  | {ONBELLEK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
