"""v108'in GEOMETRISI ve BUTCEYE KATKISI.

1. ``q_perp``  : olculmus 21 gonderimlik span'a DIK enerji.
2. ``q_yeni``  : mevcut dik envanterden de arindirildiktan sonra kalan.
3. CV'den ``L``, ``kappa*``, ``L^2/Q``  -- ``f`` icin tek on gosterge.
4. f varsayimlariyla LB beklentisi ve 1. sira acigina katki.

Kaggle'a HICBIR SEY gonderilmez.
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
sys.path.insert(0, str(BURA))
sys.path.insert(0, str(BURA.parent / "sicak_kaldirac"))
from olcut import delta_coz, hazirla, zincir  # noqa: E402
from ortak import SICAK_PAY, bloklari_kur  # noqa: E402

MSE102 = 1.011091  # docs/52 13.2 -- OLCULMUS
ACIK = 1.011091 - 0.982835
KOVA = 10
N0 = 200.0
KAT = 0.5


def sira_kovasi(p, m, k=KOVA):
    out = np.full(len(p), -1, dtype="int64")
    idx = np.flatnonzero(m)
    sira = np.argsort(np.argsort(p[idx], kind="stable"), kind="stable")
    out[idx] = np.minimum((sira * k) // len(idx), k - 1)
    return out


def ofset_ogren(e, kov, k=KOVA, n0=N0):
    o = np.zeros(k)
    n = np.zeros(k)
    em = e - e.mean()
    for j in range(k):
        s = kov == j
        n[j] = s.sum()
        if n[j]:
            o[j] = em[s].mean() * n[j] / (n[j] + n0)
    return o - float((o * n).sum() / n.sum())


def yy_geo():
    spec = importlib.util.spec_from_file_location(
        "yy_ortak", KOK / "experiments" / "yeni_yon" / "ortak.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["yy_ortak"] = m
    spec.loader.exec_module(m)
    return m.geo()


def main() -> int:
    g = yy_geo()
    v108 = np.log1p(
        pd.read_csv(KOK / "submissions/tuketim_v108_sicak_onarim.csv")["tuketim"].to_numpy(
            "float64"
        )
    )
    u = v108 - g.v102
    rap = g.olc("v108", u)
    print("=" * 96)
    print("1) GEOMETRI -- yon u = log1p(v108) - log1p(v102)")
    print("=" * 96)
    for k, v in rap.items():
        print(f"  {k:16} {v}")

    print()
    print("=" * 96)
    print("2) CV'DEN L, kappa*, L^2/Q   (kis26 SINA = 2026-02-01..2026-03-31)")
    print("=" * 96)
    bl = bloklari_kur()
    b = bl["kis26"]
    hazirla(b)
    r = zincir(b)
    tar = pd.to_datetime(b.cerceve["tarih"]).to_numpy()
    m_og = tar < np.datetime64("2026-02-01")
    m_si = ~m_og

    r0_og = r + delta_coz(b, r, m_og)
    p_og = r0_og + b.lgc
    kov_og = sira_kovasi(p_og, m_og)
    o = ofset_ogren((b.lgy - np.maximum(p_og, 0.0))[m_og], kov_og[m_og])

    r0_si = r + delta_coz(b, r, m_si)
    p_si = r0_si + b.lgc
    kov_si = sira_kovasi(p_si, m_si)
    d = np.zeros(b.n)
    d[m_si] = KAT * o[kov_si[m_si]]

    lg0 = np.maximum(p_si, 0.0)
    lg1 = np.maximum(p_si + d, 0.0)
    dd = (lg1 - lg0)[m_si]
    res = (b.lgy - lg0)[m_si]
    nsi = int(m_si.sum())
    Qcv = float(dd @ dd) / nsi
    Lcv = float(res @ dd) / nsi
    kap = Lcv / Qcv
    print(
        f"  n={nsi:,}   Q_cv={Qcv:.7f}   L_cv={Lcv:.7f}   kappa*_cv={kap:.4f}   "
        f"L^2/Q={Lcv * Lcv / Qcv:.7f}"
    )
    print(f"  dMSE(k=1) = Q - 2L = {Qcv - 2 * Lcv:+.7f}  (sicak blok ici)")
    print(f"  TEST olceginde L^2/Q ~ {Lcv * Lcv / Qcv * SICAK_PAY:.7f}")

    print()
    print("=" * 96)
    print("3) BUTCE")
    print("=" * 96)
    Q = float(rap["Q"])
    L0 = kap * Q  # CV'nin ongordugu tam gerceklesme
    print(f"  Q(v108-v102)   = {Q:.7f}")
    print(
        f"  q_perp         = {float(rap['q_perp']):.7f}   "
        f"(span disi pay %{100 * float(rap['q_perp']) / Q:.1f})"
    )
    print(
        f"  q_yeni         = {float(rap['q_yeni']):.7f}   "
        f"(envanter disi pay %{100 * float(rap['q_yeni']) / Q:.1f})"
    )
    print(f"  CV'nin ongordugu L0 (f=1) = {L0:.7f}")
    print(f"  k=1 basa bas f = Q/(2 L0) = {Q / (2 * L0):.4f}")
    print()
    print(f"{'f':>8}{'L=f*L0':>12}{'L^2/Q':>12}{'yeni MSE':>12}{'RMSLE':>10}{'acigin %':>11}")
    print("-" * 96)
    for f in (1.0, 0.60, 0.4115, 0.338, 0.20, 0.10):
        L = f * L0
        kaz = L * L / Q
        yeni = MSE102 - kaz
        print(
            f"{f:>8.4f}{L:>12.7f}{kaz:>12.7f}{yeni:>12.6f}{np.sqrt(yeni):>10.5f}"
            f"{100 * kaz / ACIK:>10.1f}%"
        )

    json.dump(
        {
            "geo": {
                k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                for k, v in rap.items()
            },
            "Q_cv": Qcv,
            "L_cv": Lcv,
            "kappa_cv": kap,
            "gain_cv": Lcv * Lcv / Qcv,
            "L0": L0,
            "Q": Q,
        },
        open(BURA / "07_geometri.json", "w"),
        indent=2,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
