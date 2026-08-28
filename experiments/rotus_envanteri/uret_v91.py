"""v91 -- GRUP B KALDIRMASI. v83 tabani + 93 grup-B trafosuna sabit log kaldirma.

OLCUM (reports/g3g_ofset.json, sizintisiz ileri pencere, uc kesme):

    kesme        trafo  satir  sifir%  gercek_ofs  v83_ofs  delta*
    2025-06-30     4     284    0.0     +1.0547    -0.5508  +1.6055
    2025-08-31     5     298    0.0     +0.4621    -0.5508  +1.0129
    2025-11-30     4     216    0.0     +1.6418    -0.5508  +2.1926

Isaret uc kesmede de AYNI (+); 798 satirin HICBIRI sifir degil; trafo bazinda
12/13 pozitif. Ama orneklem KUCUK (9 essiz trafo) ve kesmeler trafo paylasiyor.

SECILEN KATSAYI: +0.50 -- olculen EN KUCUK delta*'in (+1.0129) %49'u.
Projedeki buzme gelenegiyle (s=0.6-0.7) uyumlu, asagi riski yariya indirir.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from ortak import KOK, SUB, hizala, test

TABAN = "tuketim_v83_sicak_optimum.csv"
BEKLENEN_TRAFO = 93
BEKLENEN_SATIR = 7149


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=0.50)
    ap.add_argument("--taban", default=TABAN)
    ap.add_argument("--cikis", default="tuketim_v91_grupb_kaldirma.csv")
    ar = ap.parse_args()
    if not 0.0 < ar.delta <= 1.0:
        raise SystemExit(f"delta araligi disinda: {ar.delta}")

    te = test()
    taban = hizala(ar.taban, te)
    B = set((KOK / "experiments/rotus_envanteri/grup_b.txt").read_text(encoding="utf-8").split())
    sel = te["tanim"].isin(B).to_numpy()

    # ---- KAPILAR (uretimden ONCE)
    if len(B) != BEKLENEN_TRAFO:
        raise SystemExit(f"KAPI: grup B {len(B)} trafo, beklenen {BEKLENEN_TRAFO}")
    if int(sel.sum()) != BEKLENEN_SATIR:
        raise SystemExit(f"KAPI: grup B {int(sel.sum())} satir, beklenen {BEKLENEN_SATIR}")

    lp = np.log1p(np.clip(taban, 0.0, None))
    yeni_lp = lp + np.where(sel, ar.delta, 0.0)
    yeni = np.maximum(np.expm1(yeni_lp), 0.0)

    # ---- KAPILAR (uretimden SONRA)
    if np.isnan(yeni).any() or (yeni < 0).any():
        raise SystemExit("KAPI: NaN/negatif")
    sap = float(np.abs(yeni[~sel] - taban[~sel]).max())
    if sap > 1e-9:
        raise SystemExit(f"KAPI: dokunulmayan satirlar degismis ({sap:.2e})")
    uyg = float((np.log1p(yeni[sel]) - lp[sel]).mean())
    if abs(uyg - ar.delta) > 1e-12:
        raise SystemExit(f"KAPI: uygulanan kayma {uyg:.9f} != {ar.delta}")

    yol = SUB / ar.cikis
    if yol.exists():
        raise SystemExit(f"KAPI: {yol.name} ZATEN VAR -- mevcut dosya bozulmaz")
    pd.DataFrame({"id": te["id"], "tuketim": yeni}).to_csv(yol, index=False)

    d = yeni_lp - lp
    print(f"taban            : {ar.taban}")
    print(f"grup B           : {len(B)} trafo / {int(sel.sum()):,} satir")
    print(f"delta            : {ar.delta:+.4f} (log1p uzayi)")
    print(f"ort log1p        : {lp.mean():.6f} -> {yeni_lp.mean():.6f}")
    print(f"grup B ort log1p : {lp[sel].mean():.4f} -> {yeni_lp[sel].mean():.4f}")
    print(f"grup B ort kWh   : {taban[sel].mean():.1f} -> {yeni[sel].mean():.1f}")
    print(f"yon enerjisi Q   : {float(d @ d / len(te)):.6f}")
    print(f"yazildi          : {yol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
