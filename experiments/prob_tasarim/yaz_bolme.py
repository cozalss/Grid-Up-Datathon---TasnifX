"""PROB TASARIMI -- adim 5: MEVSIM IKIZI ICINDE trafo-ayrik dogrulama.

karar.py mevsimler arasi isaret donmesini gosterdi (sicak seviye deseni
yaz25|kis26 rho = -0.787). Test penceresi 2026-04..07 -> mevsim ikizi yaz25.
Bu yuzden desenler YAZ25'ten alinacak. Ama yaz25'in kendi deseni gurultu
olabilir; bu betik onu TRAFO-AYRIK bolmeyle sinar:

  yaz25 trafolari A/B'ye ayrilir, desen A'da ogrenilir, kazanc B'de olculur.
  Ayni trafo iki tarafta olmadigi icin ezber yok.

Ek olarak ZAMAN-AYRIK bolme (Nis-May -> Haz-Tem) da olculur: testte desen
GECMISTEN gelecek, ileri dogru tasinmasi gerekiyor.
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

from tavan import SIC, SICAK_PAY, SOGUK_PAY, _soguk_modul, desil, kova_kva  # noqa: E402

TOHUM = 20260828


def kazanc(e_h: np.ndarray, g_h: pd.Series, desen: pd.Series) -> dict:
    d = g_h.map(desen).fillna(0.0).to_numpy(dtype="float64")
    d = d - d.mean()
    q = float(d @ d) / len(d)
    if q <= 0:
        return {"Q": 0.0, "L": 0.0, "kappa": 0.0, "kazanc": 0.0}
    L = float(d @ (e_h - e_h.mean())) / len(d)
    return {"Q": q, "L": L, "kappa": L / q, "kazanc": -(L * L) / q}


def kos(ad, e, g, tanim, tarih, pay, rejim, cikti):
    g = pd.Series(np.asarray(g).astype(str))
    e = np.asarray(e, dtype="float64")

    # --- trafo-ayrik bolme ---
    rng = np.random.default_rng(TOHUM)
    trafolar = pd.unique(pd.Series(tanim))
    a_set = set(rng.choice(trafolar, size=len(trafolar) // 2, replace=False).tolist())
    a = pd.Series(tanim).isin(a_set).to_numpy()
    dA = pd.Series(e[a] - e[a].mean()).groupby(g[a].reset_index(drop=True)).mean()
    dB = pd.Series(e[~a] - e[~a].mean()).groupby(g[~a].reset_index(drop=True)).mean()
    rA = kazanc(e[~a], g[~a].reset_index(drop=True), dA)
    rB = kazanc(e[a], g[a].reset_index(drop=True), dB)
    trafo_kaz = (rA["kazanc"] * (~a).sum() + rB["kazanc"] * a.sum()) / len(e)

    # --- zaman-ayrik bolme: Nis-May -> Haz-Tem ---
    ay = pd.to_datetime(pd.Series(tarih)).dt.month.to_numpy()
    ilk = np.isin(ay, [4, 5])
    son = ~ilk
    zk = float("nan")
    zkappa = float("nan")
    if ilk.sum() > 100 and son.sum() > 100:
        d1 = pd.Series(e[ilk] - e[ilk].mean()).groupby(g[ilk].reset_index(drop=True)).mean()
        rz = kazanc(e[son], g[son].reset_index(drop=True), d1)
        zk = rz["kazanc"]
        zkappa = rz["kappa"]

    cikti.append(
        {
            "rejim": rejim,
            "yon": ad,
            "trafo_ayrik_kazanc_rejim": trafo_kaz,
            "trafo_ayrik_kazanc_toplam": trafo_kaz * pay,
            "kappa_A": rA["kappa"],
            "kappa_B": rB["kappa"],
            "zaman_ayrik_kazanc_rejim": zk,
            "zaman_ayrik_kazanc_toplam": zk * pay if zk == zk else float("nan"),
            "zaman_kappa": zkappa,
            "n": int(len(e)),
        }
    )
    print(
        f"  {rejim.upper():5s} {ad:18s} trafo-ayrik={trafo_kaz * pay:+.6f}"
        f" (kappa {rA['kappa']:+.2f}/{rB['kappa']:+.2f})   "
        f"zaman-ayrik={zk * pay if zk == zk else float('nan'):+.6f}"
        f" (kappa {zkappa:+.2f})"
    )


def main() -> None:
    cikti: list[dict] = []
    print("YAZ25 (2025-04-01..07-31) -- testin mevsim ikizi\n")

    bl = SIC.bloklari_kur()
    b = bl["yaz25"]
    r0 = SIC.taban_r(b)
    lgp = np.maximum(r0 + b.lgc, 0.0)
    e = b.lgy - lgp
    tanim = b.cerceve["tanim"].to_numpy()
    tarih = b.cerceve["tarih"].to_numpy()
    print(f"SICAK yaz25: {len(e):,} satir, {len(pd.unique(pd.Series(tanim))):,} trafo")
    for ad, g in {
        "ilce": b.cerceve["ilce"].to_numpy(),
        "seviye_desili10": desil(lgp, 10),
        "kva_kovasi": b.cerceve["kova"].to_numpy(),
        "ilce_x_kova": (
            b.cerceve["ilce"].astype(str) + "|" + b.cerceve["kova"].astype(str)
        ).to_numpy(),
        "ay": b.cerceve["ay"].to_numpy(),
    }.items():
        kos(ad, e, g, tanim, tarih, SICAK_PAY, "sicak", cikti)
    del bl

    SOG = _soguk_modul()
    sb = SOG.tum_bloklar()
    b = sb["yaz25"]
    r0 = SOG.taban_r(b)
    lgp = r0 + b.lgc
    e = b.lgy - lgp
    print(f"\nSOGUK yaz25: {len(e):,} satir, {len(pd.unique(pd.Series(b.tanim))):,} trafo")
    for ad, g in {
        "ilce": b.ilce,
        "seviye_desili10": desil(lgp, 10),
        "kva_kovasi": kova_kva(b.guc),
        "ay": pd.to_datetime(b.tarih).month.to_numpy(),
    }.items():
        kos(ad, e, g, b.tanim, b.tarih, SOGUK_PAY, "soguk", cikti)

    (BURA / "yaz_bolme.json").write_text(
        json.dumps(cikti, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nyazildi: yaz_bolme.json")


if __name__ == "__main__":
    main()
