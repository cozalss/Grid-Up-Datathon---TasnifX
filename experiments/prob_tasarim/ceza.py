"""PROB TASARIMI -- adim 7: DURUSTLUK KAPISI, public alt kumede asiri uydurma cezasi.

Public/private orani BILINMIYOR (docs/31 §2). Prob kampanyasi LB'ye k tane
SKALER uydurur (yon basina bir kappa). Rastgele bolmede beklenen private
kaybi yon basina:

    ceza = (1-f) * s^2 / (f * N * Q),      s^2 = Var(d_i * eps_i)

Turetme: kappa = L_pub/Q ile private kazanc
    E[2 kappa L_priv - kappa^2 Q] = L^2/Q - (1-f) s^2 / (f N Q).

s^2 = E[d^2 eps^2] - L^2. Eger d ile eps^2 bagimsiz olsaydi s^2 = Q*sigma^2
olurdu; gercekte artik varyansi tahmin seviyesine bagli. Bu betik SISME
CARPANINI  c = E[d^2 e^2] / (E[d^2] E[e^2])  yaz25 blogunda OLCER ve cezayi
o carpanla verir.

Not: satir duzeyi bir tasarim N tane serbest parametre yakardi; oradaki ceza
(1-f)/f * sigma^2 mertebesinde, yani ~20 birim MSE. Grup duzeyinde kalmanin
sebebi budur.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
BURA = Path(__file__).resolve().parent
sys.path.insert(0, str(BURA))

from tavan import SIC, _soguk_modul, desil  # noqa: E402

N = 714_688
M_V93 = 1.0167414


def sisme(e: np.ndarray, d: np.ndarray) -> float:
    d = d - d.mean()
    e = e - e.mean()
    ust = float(np.mean((d * e) ** 2))
    alt = float(np.mean(d**2) * np.mean(e**2))
    return ust / alt if alt > 0 else float("nan")


def main() -> None:
    c_list = []

    bl = SIC.bloklari_kur()
    b = bl["yaz25"]
    r0 = SIC.taban_r(b)
    lgp = np.maximum(r0 + b.lgc, 0.0)
    e = b.lgy - lgp
    for ad, g in {
        "sicak ilce": b.cerceve["ilce"].to_numpy(),
        "sicak seviye10": desil(lgp, 10),
    }.items():
        gs = pd.Series(np.asarray(g).astype(str))
        des = pd.Series(e - e.mean()).groupby(gs).mean()
        d = gs.map(des).to_numpy(dtype="float64")
        c = sisme(e, d)
        c_list.append(c)
        print(f"  {ad:18s} sisme carpani c = {c:.3f}")
    del bl

    SOG = _soguk_modul()
    sb = SOG.tum_bloklar()
    b = sb["yaz25"]
    r0 = SOG.taban_r(b)
    lgp = r0 + b.lgc
    e = b.lgy - lgp
    gs = pd.Series(desil(lgp, 10).astype(str))
    des = pd.Series(e - e.mean()).groupby(gs).mean()
    d = gs.map(des).to_numpy(dtype="float64")
    c = sisme(e, d)
    c_list.append(c)
    print(f"  {'soguk seviye10':18s} sisme carpani c = {c:.3f}")

    c_max = float(np.nanmax(c_list))
    print(f"\nen kotu sisme carpani c = {c_max:.3f}   (sigma^2 = M(v93) = {M_V93:.4f})")

    print("\nYON BASINA ASIRI UYDURMA CEZASI (MSE)")
    print(
        f"{'f (public payi)':>16s}{'ceza/yon':>14s}{'3 yon':>12s}{'6 yon':>12s}{'v93 16 + 6':>14s}"
    )
    tablo = []
    for f in (0.05, 0.20, 0.30, 0.50):
        ceza = (1 - f) * c_max * M_V93 / (f * N)
        tablo.append(
            {"f": f, "ceza_yon": ceza, "ceza_3": 3 * ceza, "ceza_6": 6 * ceza, "ceza_22": 22 * ceza}
        )
        print(f"{f:>16.2f}{ceza:>14.2e}{3 * ceza:>12.2e}{6 * ceza:>12.2e}{22 * ceza:>14.2e}")

    print("\nKARSILASTIRMA: satir duzeyi tasarim (N serbest parametre)")
    for f in (0.05, 0.20, 0.30, 0.50):
        print(
            f"   f={f:.2f}  ceza = {(1 - f) * c_max * M_V93 / f:.3f} MSE  -> YASAK BOLGE (docs/48)"
        )

    (BURA / "ceza.json").write_text(
        json.dumps(
            {"c_max": c_max, "M_v93": M_V93, "N": N, "tablo": tablo}, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    print("\nyazildi: ceza.json")


if __name__ == "__main__":
    main()
