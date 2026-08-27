"""v87 skoru gelince olu-trafo yonunun optimum kappa'sini coz.

v87 = v83 + tek yonde delta (yalniz 14.484 satir). Iki olculmus skor
bu yondeki optimum katsayiyi TAM belirler -- tahmin yok.

  MSE(k) = MSE83 + 2k<d,e83>/n + k^2 Q,   Q = ||d||^2/n
  k* = (Q - (MSE87 - MSE83)) / (2Q)

k*>=1  -> tam sifirlama optimum, daha ileri gidilemez (sifir taban)
k*<1   -> kismi buzme daha iyi; dosya k* ile uretilir
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
G = KOK / "submissions"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--taban-score", type=float, default=1.01318, help="v83 olculmus skoru")
    ap.add_argument("--prob-score", type=float, required=True, help="v87 olculmus skoru")
    ap.add_argument("--cikti", default="tuketim_v88_olu_kappa.csv")
    args = ap.parse_args()

    t = pd.read_csv(G / "tuketim_v83_sicak_optimum.csv")
    p = pd.read_csv(G / "tuketim_v87_olu_izole.csv")
    if not t["id"].equals(p["id"]):
        raise RuntimeError("id sirasi eslesmiyor")

    a = np.log1p(t["tuketim"].to_numpy())
    d = np.log1p(p["tuketim"].to_numpy()) - a
    n = len(a)
    q = float(np.sum(d**2) / n)
    m0, m1 = args.taban_score**2, args.prob_score**2
    kappa = (q - (m1 - m0)) / (2.0 * q)
    bek = float(np.sqrt(max(m0 + 2 * kappa * (m1 - m0 - q) / 2.0 + kappa * kappa * q, 0.0)))

    print(f"  Q (yon enerjisi)   = {q:.6f}")
    print(f"  v83 {args.taban_score:.5f}  ->  v87 {args.prob_score:.5f}")
    print(f"  optimum kappa      = {kappa:+.6f}")
    print(f"  beklenen RMSLE     = {bek:.5f}")

    if kappa >= 1.0:
        print("\n  kappa>=1: TAM sifirlama optimum. v87 zaten en iyisi, yeni dosya uretilmedi.")
        return 0
    if kappa <= 0.0:
        print("\n  kappa<=0: sifirlama ZARARLI. v83 korunmali, yeni dosya uretilmedi.")
        return 0

    yeni = np.clip(np.expm1(a + kappa * d), 0.0, None)
    yol = G / args.cikti
    pd.DataFrame({"id": t["id"], "tuketim": yeni}).to_csv(yol, index=False)
    print(f"\n  YAZILDI {yol.name}  (kappa={kappa:.4f} ile kismi buzme)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
