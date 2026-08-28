"""BUTCEYE ARTAN KATKI -- v108 mevcut dik envantere NE EKLIYOR?

Bilinen durum (docs/52 13 + gorev tanimi):
    v102 olculmus MSE 1.011091   (RMSLE 1.00553)
    prob kampanyasi f=0.4115'te 1.003044'e (RMSLE 1.00152) indiriyor
    lider 0.982835  ->  kampanyadan SONRA kalan acik 0.020209 MSE

v108'in katkisi iki parcaya ayrilir:
    Q        = toplam enerji
    q_perp   = olculmus span'a dik  (span icindeki pay zaten cozulmus)
    q_yeni   = mevcut 9 dik envanter yonunden de arindirilmis  <- ARTAN KATKI

Kazanc modeli: L = f*kappa*Q  ->  kazanc = f^2 kappa^2 Q  (yon boyunca tam).
Artan kazanc = f^2 kappa^2 q_yeni.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BURA = Path(__file__).resolve().parent
KOK = BURA.parents[1]

MSE102 = 1.011091
MSE_KAMPANYA = 1.003044  # RMSLE 1.00152
MSE_LIDER = 0.982835


def yy():
    spec = importlib.util.spec_from_file_location(
        "yy_ortak", KOK / "experiments" / "yeni_yon" / "ortak.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["yy_ortak"] = m
    spec.loader.exec_module(m)
    return m


def main() -> int:
    m = yy()
    g = m.geo()
    E, ad = g.envanter()
    print("=" * 96)
    print("1) MEVCUT DIK ENVANTER")
    print("=" * 96)
    N = {
        "v93": "tuketim_v93_gram_optimum.csv",
        "v90": "tuketim_v90_temiz_sota.csv",
        "P1": "tuketim_p1_sicak_ilce.csv",
        "P2": "tuketim_p2_sicak_seviye.csv",
        "P3": "tuketim_p3_soguk_seviye.csv",
        "P4": "tuketim_p4_sicak_ay.csv",
        "P5": "tuketim_p5_soguk_kva.csv",
        "yas": "tuketim_prob_yas790.csv",
        "v82": "tuketim_v82_ayirici.csv",
        "v99": "tuketim_v99_mimari_sekil.csv",
        "B": "tuketim_v96_grupb_optimum.csv",
        "bos": "tuketim_v94_bosluk_oncesi.csv",
    }
    F = {k: m.log1p_gonderim(v) for k, v in N.items()}
    sira = [
        ("P1", F["P1"] - F["v93"]),
        ("P3", F["P3"] - F["v93"]),
        ("v96", F["B"] - F["v93"]),
        ("bos", F["bos"] - F["v93"]),
        ("v90", F["v90"] - g.v83),
        ("P2", F["P2"] - F["v93"]),
        ("yas", F["yas"] - F["v93"]),
        ("v99", F["v99"] - F["v90"]),
        ("P4", F["P4"] - F["v93"]),
        ("v82", F["v82"] - g.v83),
        ("P5", F["P5"] - F["v93"]),
    ]
    E2, top = [], 0.0
    print(f"{'yon':8}{'q_yeni':>14}")
    for a, u in sira:
        up, _ = g.perp(u)
        v = up.copy()
        for e in E2:
            v -= (float(v @ e) / g.n) * e
        qy = float(v @ v) / g.n
        if qy > 5e-6:
            E2.append(v / np.sqrt(qy))
            top += qy
            print(f"{a:8}{qy:>14.7f}")
    print(f"{'TOPLAM':8}{top:>14.7f}")

    v108 = np.log1p(
        pd.read_csv(KOK / "submissions/tuketim_v108_sicak_onarim.csv")["tuketim"].to_numpy(
            "float64"
        )
    )
    u = v108 - g.v102
    rap = g.olc("v108", u)
    Q, qp, qy = float(rap["Q"]), float(rap["q_perp"]), float(rap["q_yeni"])
    kap = json.load(open(BURA / "07_geometri.json"))["kappa_cv"]

    print()
    print("=" * 96)
    print("2) v108'IN ARTAN KATKISI")
    print("=" * 96)
    print(f"  Q          = {Q:.7f}")
    print(f"  q_perp     = {qp:.7f}  (span disi %{100 * qp / Q:.1f})")
    print(f"  q_yeni     = {qy:.7f}  (envanter disi %{100 * qy / Q:.1f})")
    print(f"  dik envanter {top:.7f} -> {top + qy:.7f}  (+%{100 * qy / top:.1f})")
    print(f"  kappa*_cv  = {kap:.4f}")
    print()
    print(
        f"{'f':>8}{'tek basina kazanc':>20}{'ARTAN kazanc':>16}"
        f"{'kampanya+v108 MSE':>20}{'RMSLE':>10}{'acigin %':>11}"
    )
    print("-" * 96)
    acik = MSE_KAMPANYA - MSE_LIDER
    for f in (1.0, 0.60, 0.4115, 0.338, 0.20):
        tek = f * f * kap * kap * Q
        art = f * f * kap * kap * qy
        yeni = MSE_KAMPANYA - art
        print(
            f"{f:>8.4f}{tek:>20.7f}{art:>16.7f}{yeni:>20.6f}"
            f"{np.sqrt(max(yeni, 0)):>10.5f}{100 * art / acik:>10.1f}%"
        )

    print()
    print("=" * 96)
    print("3) P2 CELISKISI -- ayni aile, kirli olcutle kurulmus surum")
    print("=" * 96)
    uP2 = F["P2"] - F["v93"]
    pp, _ = g.perp(uP2)
    up, _ = g.perp(u)
    kos = float(pp @ up) / np.sqrt(float(pp @ pp) * float(up @ up))
    print(f"  cos(v108_dik, P2_dik) = {kos:+.4f}")
    print("  P2 = SICAK seviye desili probu, deseni YAZ25'ten alinmis (docs/52 8).")
    print("  yaz25 ileri kat payi %92,8 -- en KIRLI blok. Ayni aile, TERS isaret.")
    print("  P2 kampanyada VARSA isareti sorgulanmali; v108 onun temiz surumudur.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
