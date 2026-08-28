"""TASINABILIRLIK PROBU -- kis26'da olculen buzulme TEST modelinde de var mi?

Kaygi: kis26 CV modeli 334 gunluk ozet + 5 kokenle egitildi; TEST modeli
455 gun + 9 koken gorur. Olculen "model az yayiliyor" yanliligi VERI HACMI
artefakti ise TEST modelinde kuculur ve duzeltme fazla gelir.

SINAV: kis26 icinde, modelin trafo basina gordugu veri miktarina gore
ayristir (``gecmis_gun`` = trafonun egitim gecmisi uzunlugu). Her dilimde
optimum kuresel genlik ``lam*`` ve optimum seviye-desili ofsetinin genligi
olculur.

  * lam* veri hacmiyle 1'e YAKINSIYORSA -> artefakt, TEST'te kuculur.
  * lam* dilimden bagimsizsa            -> yapisal, TEST'e tasinir.

Ikinci sinav: ofset profili monoton mu (saf de-buzme) yoksa duzensiz mi?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BURA = Path(__file__).resolve().parent
sys.path.insert(0, str(BURA))
sys.path.insert(0, str(BURA.parent / "sicak_kaldirac"))
from olcut import delta_coz, grup_ofseti, hazirla, mse_alt, zincir  # noqa: E402
from ortak import bloklari_kur  # noqa: E402


def lam_opt(b, r, m, izgara):
    en = None
    mm = float(r[m].mean())
    for lam in izgara:
        rr = mm + lam * (r - mm)
        v = mse_alt(b, rr + delta_coz(b, rr, m), m)
        if en is None or v < en[0]:
            en = (v, lam)
    return en[1]


def main() -> int:
    bl = bloklari_kur()
    b = bl["kis26"]
    hazirla(b)
    r = zincir(b)
    izgara = np.round(np.arange(0.90, 1.301, 0.01), 3)

    print("=" * 96)
    print("1) OPTIMUM GENLIK lam* -- trafonun EGITIM GECMISI uzunluguna gore")
    print("=" * 96)
    gg = b.cerceve["gecmis_gun"].to_numpy()
    print(f"{'gecmis_gun dilimi':26}{'n satir':>10}{'n trafo':>9}{'lam*':>8}{'MSE':>10}")
    print("-" * 96)
    kenarlar = [(0, 7), (7, 31), (31, 91), (91, 181), (181, 275), (275, 340)]
    for a, z in kenarlar:
        m = (gg >= a) & (gg < z)
        if m.sum() < 500:
            continue
        L = lam_opt(b, r, m, izgara)
        print(
            f"{f'[{a},{z})':26}{int(m.sum()):>10,}"
            f"{b.cerceve['tanim'].to_numpy()[m].__len__() and len(set(b.cerceve['tanim'].to_numpy()[m])):>9,}"
            f"{L:>8.2f}{mse_alt(b, r + delta_coz(b, r, m), m):>10.4f}"
        )
    m_all = np.ones(b.n, dtype=bool)
    print(
        f"{'TUMU':26}{b.n:>10,}{len(set(b.cerceve['tanim'])):>9,}"
        f"{lam_opt(b, r, m_all, izgara):>8.2f}{mse_alt(b, r + delta_coz(b, r, m_all), m_all):>10.4f}"
    )

    print()
    print("=" * 96)
    print("2) OPTIMUM GENLIK lam* -- trafonun EGITIM SATIR SAYISINA gore (t_gun_sayisi)")
    print("=" * 96)
    tg = b.cerceve["t_gun_sayisi"].fillna(0).to_numpy()
    print(f"{'t_gun_sayisi dilimi':26}{'n satir':>10}{'lam*':>8}")
    print("-" * 96)
    for a, z in [(0, 30), (30, 90), (90, 180), (180, 270), (270, 400)]:
        m = (tg >= a) & (tg < z)
        if m.sum() < 500:
            continue
        print(f"{f'[{a},{z})':26}{int(m.sum()):>10,}{lam_opt(b, r, m, izgara):>8.2f}")

    print()
    print("=" * 96)
    print("3) OPTIMUM GENLIK -- kis26 ICINDE zamanla degisiyor mu? (ay bazli)")
    print("=" * 96)
    ay = pd.to_datetime(b.cerceve["tarih"]).dt.to_period("M").astype(str).to_numpy()
    for a in sorted(set(ay)):
        m = ay == a
        print(f"  {a}  n={int(m.sum()):>7,}  lam*={lam_opt(b, r, m, izgara):.2f}")

    print()
    print("=" * 96)
    print("4) SEVIYE DESILI OFSET PROFILI (tum kis26'dan ogrenilmis, n0=200)")
    print("=" * 96)
    r0 = r + delta_coz(b, r, m_all)
    h = grup_ofseti(b, r0, m_all, "seviye_d10", 200.0)
    hj = grup_ofseti(
        b,
        r0,
        pd.to_datetime(b.cerceve["tarih"]).to_numpy() < np.datetime64("2026-02-01"),
        "seviye_d10",
        200.0,
    )
    hk = grup_ofseti(
        b,
        r0,
        pd.to_datetime(b.cerceve["tarih"]).to_numpy() >= np.datetime64("2026-02-01"),
        "seviye_d10",
        200.0,
    )
    sd = b.cerceve["seviye_d10"].to_numpy()
    print(
        f"{'desil':>7}{'n':>10}{'ort t_log_ort':>15}{'ofset TUM':>12}{'12-01..01-31':>14}"
        f"{'02-01..03-31':>14}"
    )
    print("-" * 96)
    for d in sorted(h):
        m = sd == d
        print(
            f"{d:>7}{int(m.sum()):>10,}{np.nanmean(b.cerceve['t_log_ort'].to_numpy()[m]):>15.3f}"
            f"{h[d]:>+12.4f}{hj.get(d, np.nan):>+14.4f}{hk.get(d, np.nan):>+14.4f}"
        )
    x = pd.Series(hj).sort_index()
    y = pd.Series(hk).sort_index()
    ort = pd.concat([x, y], axis=1, join="inner").dropna()
    print(
        f"\n  iki yarim arasinda korelasyon: {ort.iloc[:, 0].corr(ort.iloc[:, 1]):+.4f}  "
        f"egim (gec~erken): {np.polyfit(ort.iloc[:, 0], ort.iloc[:, 1], 1)[0]:+.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
