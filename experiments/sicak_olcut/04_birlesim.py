"""BIRLESIM ve SAGLAMLIK -- kapiyi gecen yonler birlikte ne yapiyor, ve
bolme noktasi degisince kapi ayakta kaliyor mu?

03'te tek tek gecen yonlerin cogu AYNI eksende (modelin yayilimi az; dusuk
tahminler daha da asagi). Ortusme ciftte olcumsuz kalmasin diye burada
BIRLIKTE olculuyor: her yon bir onceki yonun artiginda yeniden ogreniliyor.

Ayrica uc bolme noktasi denenir:
    B1  OGREN 12-01..01-31  SINA 02-01..03-31   (03'teki)
    B2  OGREN 12-01..12-31  SINA 01-01..03-31
    B3  OGREN 12-01..02-28  SINA 03-01..03-31
Bir yon ancak UC bolmede de ayni isaretteyse tasinabilir sayilir.

OFS tanim (trafo bazli ofset) BILEREK DISARIDA -- gerekcesi raporda.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BURA = Path(__file__).resolve().parent
sys.path.insert(0, str(BURA))
sys.path.insert(0, str(BURA.parent / "sicak_kaldirac"))
from olcut import bootstrap, delta_coz, grup_ofseti, hazirla, mse_alt, zincir  # noqa: E402
from ortak import SICAK_PAY, bloklari_kur  # noqa: E402

BOLMELER = {
    "B1 12-01..01-31 -> 02-01..03-31": "2026-02-01",
    "B2 12-01..12-31 -> 01-01..03-31": "2026-01-01",
    "B3 12-01..02-28 -> 03-01..03-31": "2026-03-01",
}
#: sirayla uygulanan yonler (her biri onceki artikta yeniden ogrenilir)
ZINCIR = ["seviye_d10", "sifir_k", "hg", "kova", "ilce"]


def dusuk_hedge_uygula(b, r, kappa, esik):
    tah = np.expm1(np.maximum(r + b.lgc, 0.0))
    out = r.copy()
    out[tah <= esik] -= kappa
    return out


def main() -> int:
    bl = bloklari_kur()
    b = bl["kis26"]
    hazirla(b)
    tar = pd.to_datetime(b.cerceve["tarih"]).to_numpy()
    r_ham = zincir(b)

    kayit = []
    for etiket, kes_s in BOLMELER.items():
        kes = np.datetime64(kes_s)
        m_og, m_si = tar < kes, tar >= kes
        r0_og = r_ham + delta_coz(b, r_ham, m_og)
        taban_si = mse_alt(b, r_ham + delta_coz(b, r_ham, m_si), m_si)
        print()
        print("=" * 108)
        print(
            f"{etiket}   OGREN {int(m_og.sum()):,}  SINA {int(m_si.sum()):,}  "
            f"taban SINA MSE {taban_si:.6f}"
        )
        print("=" * 108)
        print(
            f"{'adim':44}{'dMSE_SINA':>12}{'GAalt':>10}{'GAust':>10}{'kazanan':>9}"
            f"{'testdMSE':>10}  karar"
        )
        print("-" * 108)

        # --- zincirli grup ofsetleri
        r_og = r0_og.copy()
        r_si = r_ham.copy()
        for anah in ZINCIR:
            h = grup_ofseti(b, r_og, m_og, anah, 200.0)
            d = pd.Series(b.cerceve[anah].to_numpy()).map(h).fillna(0.0).to_numpy("float64")
            r_og_yeni = r_og + d
            r_og = r_og_yeni + delta_coz(b, r_og_yeni, m_og)
            r_si_yeni = r_si + d
            r0 = r_si + delta_coz(b, r_si, m_si)
            r1 = r_si_yeni + delta_coz(b, r_si_yeni, m_si)
            dm, lo, hi, kaz, nt = bootstrap(b, r0, r1, m_si, B=1000)
            karar = (
                "GECTI"
                if (dm < 0 and hi < 0 and kaz >= 0.60)
                else (
                    "RED(zararli)"
                    if dm >= 0
                    else ("red(GA sifir)" if hi >= 0 else f"red(kazanan %{100 * kaz:.0f})")
                )
            )
            print(
                f"{'+OFS ' + anah:44}{dm:>+12.5f}{lo:>+10.5f}{hi:>+10.5f}"
                f"{100 * kaz:>8.1f}%{dm * SICAK_PAY:>+10.5f}  {karar}"
            )
            kayit.append(
                {
                    "bolme": etiket,
                    "adim": f"+OFS {anah}",
                    "dMSE": dm,
                    "GA_alt": lo,
                    "GA_ust": hi,
                    "kazanan": kaz,
                }
            )
            r_si = r_si_yeni  # zinciri surdur

        # --- zincirin toplami
        r0 = r_ham + delta_coz(b, r_ham, m_si)
        r1 = r_si + delta_coz(b, r_si, m_si)
        dm, lo, hi, kaz, nt = bootstrap(b, r0, r1, m_si, B=1000)
        print(
            f"{'== ZINCIR TOPLAMI':44}{dm:>+12.5f}{lo:>+10.5f}{hi:>+10.5f}"
            f"{100 * kaz:>8.1f}%{dm * SICAK_PAY:>+10.5f}"
        )
        kayit.append(
            {
                "bolme": etiket,
                "adim": "ZINCIR TOPLAMI",
                "dMSE": dm,
                "GA_alt": lo,
                "GA_ust": hi,
                "kazanan": kaz,
            }
        )

        # --- yalniz seviye_d10 (en guclu tekil)
        h = grup_ofseti(b, r0_og, m_og, "seviye_d10", 200.0)
        d = pd.Series(b.cerceve["seviye_d10"].to_numpy()).map(h).fillna(0.0).to_numpy("float64")
        for kat in (1.0, 0.75, 0.5):
            rr = r_ham + kat * d
            r1 = rr + delta_coz(b, rr, m_si)
            dm, lo, hi, kaz, _ = bootstrap(b, r0, r1, m_si, B=1000)
            karar = "GECTI" if (dm < 0 and hi < 0 and kaz >= 0.60) else "red"
            print(
                f"{f'TEK seviye_d10 kat={kat}':44}{dm:>+12.5f}{lo:>+10.5f}{hi:>+10.5f}"
                f"{100 * kaz:>8.1f}%{dm * SICAK_PAY:>+10.5f}  {karar}"
            )
            kayit.append(
                {
                    "bolme": etiket,
                    "adim": f"TEK seviye_d10 kat={kat}",
                    "dMSE": dm,
                    "GA_alt": lo,
                    "GA_ust": hi,
                    "kazanan": kaz,
                }
            )

        # --- dusuk hedge, OGREN'de kappa taranmis
        en = None
        for kappa in (0.10, 0.20, 0.30, 0.50, 0.70):
            for esik in (20.0, 50.0, 100.0):
                rr = dusuk_hedge_uygula(b, r_ham, kappa, esik)
                v = mse_alt(b, rr + delta_coz(b, rr, m_og), m_og)
                if en is None or v < en[0]:
                    en = (v, kappa, esik)
        _, kap, esk = en
        rr = dusuk_hedge_uygula(b, r_ham, kap, esk)
        r1 = rr + delta_coz(b, rr, m_si)
        dm, lo, hi, kaz, _ = bootstrap(b, r0, r1, m_si, B=1000)
        karar = "GECTI" if (dm < 0 and hi < 0 and kaz >= 0.60) else "red"
        print(
            f"{f'DUSUK HEDGE k={kap} T={esk:.0f} (OGREN opt)':44}{dm:>+12.5f}{lo:>+10.5f}"
            f"{hi:>+10.5f}{100 * kaz:>8.1f}%{dm * SICAK_PAY:>+10.5f}  {karar}"
        )
        kayit.append(
            {
                "bolme": etiket,
                "adim": f"DUSUK HEDGE k={kap} T={esk}",
                "dMSE": dm,
                "GA_alt": lo,
                "GA_ust": hi,
                "kazanan": kaz,
            }
        )

        # --- seviye_d10 + dusuk hedge birlikte
        rr = dusuk_hedge_uygula(b, r_ham + d, kap, esk)
        r1 = rr + delta_coz(b, rr, m_si)
        dm, lo, hi, kaz, _ = bootstrap(b, r0, r1, m_si, B=1000)
        karar = "GECTI" if (dm < 0 and hi < 0 and kaz >= 0.60) else "red"
        print(
            f"{'seviye_d10 + DUSUK HEDGE':44}{dm:>+12.5f}{lo:>+10.5f}{hi:>+10.5f}"
            f"{100 * kaz:>8.1f}%{dm * SICAK_PAY:>+10.5f}  {karar}"
        )
        kayit.append(
            {
                "bolme": etiket,
                "adim": "seviye_d10+dusuk hedge",
                "dMSE": dm,
                "GA_alt": lo,
                "GA_ust": hi,
                "kazanan": kaz,
            }
        )

    yol = BURA / "04_birlesim.jsonl"
    with yol.open("w", encoding="utf-8") as f:
        for s in kayit:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nyazildi: {yol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
