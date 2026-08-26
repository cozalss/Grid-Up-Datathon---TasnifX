# ruff: noqa
"""EKSEN 4 (b)/(c) ON TESHIS -- egitim ve test DAGILIMLARININ karsilastirmasi.

(b) icin: onem agirliklandirmanin tasiyacagi kayma ne kadar buyuk, ESS ne
    olur, kapsanmayan tabaka payi nedir.
(c) icin: ufuk dagilimi. Uretim 122 gunluk bloklar kullaniyor; ek kokenler
    de dahil egitimin ufuk dagilimi testinkine benziyor mu?
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
import olcut as ol  # noqa: E402
import tuketim_model as tm  # noqa: E402


def main() -> int:
    from gridup.reporting import satir_tamponlu_cikti

    satir_tamponlu_cikti()
    egitim, test = d.cerceveleri_kur()
    te_s = test[test["soguk_mu"] != 1]
    guc_kenar = ol.guc_kenarlari(test)

    ek = d._ek_kokenler_kur(False)
    ek = ek[ek["_blok"].isin([a for a, _, _ in tm.EK_KOKENLER])]
    ortak = [k for k in egitim.columns if k in ek.columns]
    genis = pd.concat([egitim[ortak], ek[ortak]], ignore_index=True)
    gs = genis[genis["soguk_mu"] != 1]

    print("=" * 96)
    print("(c) UFUK DAGILIMI -- egitim (ek kokenli, sicak) vs TEST (sicak)")
    print("=" * 96)
    ken = [0, 31, 61, 91, 122, 10**6]
    ad = ["1-31", "32-61", "62-91", "92-122", ">122"]
    ke = pd.cut(gs["ufuk_gun"], ken, labels=ad).value_counts(normalize=True).reindex(ad)
    kt = pd.cut(te_s["ufuk_gun"], ken, labels=ad).value_counts(normalize=True).reindex(ad)
    print(f"  {'kova':>10}{'EGITIM%':>10}{'TEST%':>10}{'p_t/p_e':>10}")
    for a in ad:
        e_, t_ = float(ke.get(a, 0) or 0), float(kt.get(a, 0) or 0)
        print(f"  {a:>10}{e_ * 100:10.2f}{t_ * 100:10.2f}{(t_ / e_ if e_ else np.nan):10.2f}")
    print(
        f"  egitim ufuk: min {gs['ufuk_gun'].min():.0f} max {gs['ufuk_gun'].max():.0f} "
        f"ort {gs['ufuk_gun'].mean():.1f}"
    )
    print(
        f"  test   ufuk: min {te_s['ufuk_gun'].min():.0f} max {te_s['ufuk_gun'].max():.0f} "
        f"ort {te_s['ufuk_gun'].mean():.1f}"
    )

    print("\n" + "=" * 96)
    print("(b) ONEM AGIRLIKLANDIRMA -- EGITIM setini teste tasimanin bedeli")
    print("=" * 96)
    print(f"  {'eksenler':>28}{'ESS':>8}{'kirpilan':>10}{'kapsanmayan':>13}{'tabaka':>8}")
    for eks in [
        ("bayatlik",),
        ("guc",),
        ("ufuk",),
        ("bayatlik", "guc"),
        ("bayatlik", "ufuk"),
        ("bayatlik", "guc", "ufuk"),
    ]:
        w, tani = ol.test_agirliklari(gs, te_s, guc_kenar, eksenler=eks)
        print(
            f"  {'x'.join(eks):>28}{tani['ess_orani']:8.3f}{tani['kirpilan']:10.4f}"
            f"{tani['kapsanmayan']:13.4f}{tani['tabaka']:8d}"
        )

    print("\n  BAYATLIK kovalari (t_son_kayit_yasi) -- pay%")
    ken2 = ol.BAYATLIK_KENARLARI
    lab = ["0", "1-6", "7-29", "30-89", "90+"]
    ke2 = (
        pd.cut(gs["t_son_kayit_yasi"], ken2, right=False, labels=lab)
        .value_counts(normalize=True)
        .reindex(lab)
    )
    kt2 = (
        pd.cut(te_s["t_son_kayit_yasi"], ken2, right=False, labels=lab)
        .value_counts(normalize=True)
        .reindex(lab)
    )
    print(f"  {'kova':>10}{'EGITIM%':>10}{'TEST%':>10}{'p_t/p_e':>10}")
    for a in lab:
        e_, t_ = float(ke2.get(a, 0) or 0), float(kt2.get(a, 0) or 0)
        print(f"  {a:>10}{e_ * 100:10.2f}{t_ * 100:10.2f}{(t_ / e_ if e_ else np.nan):10.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
