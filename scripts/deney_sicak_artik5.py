# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""SICAK ARTIK -- ADIM D: trafo duzeyindeki YAVAS sapma ONCEDEN BILINEBILIR MI?

Ham veriyle (model yok): trafo x ay sapmasinin
  (a) yildan yila tekrarlanabilirligi  (2025 Oca-Mar  vs  2026 Oca-Mar)
  (b) aydan aya kaliciligi             (lag 1..5 ay)
olculur. Ikisi de sifirsa sicak artik ONCEDEN BILINEMEZ.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))
import tuketim_model as tm  # noqa: E402


def main() -> int:
    tr, te = tm.yukle()
    tr["tarih"] = pd.to_datetime(tr["tarih"])
    print("  ham egitim:", tr.shape, tr["tarih"].min().date(), "->", tr["tarih"].max().date())
    tr["ay"] = tr["tarih"].dt.strftime("%Y%m")
    tr["v"] = np.log1p(tr["tuketim"].clip(lower=0)) - np.log1p(tr["guc"])
    # trafo ve ay sabit etkileri cikarilir (dengesiz panel, iteratif)
    s = tr["v"] - tr["v"].mean()
    for _ in range(8):
        s = s - s.groupby(tr["tanim"]).transform("mean")
        s = s - s.groupby(tr["ay"]).transform("mean")
    tr["d"] = s
    hucre = tr.groupby(["tanim", "ay"]).agg(d=("d", "mean"), n=("d", "size")).reset_index()
    hucre = hucre[hucre["n"] >= 10]
    piv = hucre.pivot(index="tanim", columns="ay", values="d")
    ns = hucre.pivot(index="tanim", columns="ay", values="n")
    print("  aylar:", list(piv.columns))
    print(f"  trafo x ay hucresi: {int(hucre['n'].size):,}   hucre ici gun >= 10")

    print("\n  (a) YILDAN YILA  (2025 ay X  ->  2026 ay X)")
    for a in ("01", "02", "03"):
        k1, k2 = f"2025{a}", f"2026{a}"
        if k1 not in piv or k2 not in piv:
            continue
        x = piv[[k1, k2]].dropna()
        print(
            f"    ay {a}: n={len(x):,}  kor={x[k1].corr(x[k2]):+.3f}  "
            f"OLS egimi={np.polyfit(x[k1], x[k2], 1)[0]:+.3f}  "
            f"std(2025)={x[k1].std():.3f} std(2026)={x[k2].std():.3f}"
        )

    print("\n  (b) AYDAN AYA KALICILIK  (butun ardisik ciftler havuzlanmis)")
    aylar = sorted(piv.columns)
    for lag in (1, 2, 3, 4, 5):
        rs, ns_ = [], 0
        for i in range(len(aylar) - lag):
            x = piv[[aylar[i], aylar[i + lag]]].dropna()
            if len(x) < 200:
                continue
            rs.append(x.iloc[:, 0].corr(x.iloc[:, 1]))
            ns_ += len(x)
        if rs:
            print(
                f"    lag {lag} ay: cift={len(rs):2d}  n={ns_:,}  ort kor={np.mean(rs):+.3f}"
                f"  [{min(rs):+.3f}, {max(rs):+.3f}]"
            )

    print("\n  (c) TEST PENCERESI ICIN GECERLI SORU: son ozet penceresinden 1..4 ay sonrasi")
    # 2025-03 (son gozlem oncesi) -> 2025-04..07 gibi; her baslangic icin
    for bas in ("202503", "202511", "202603"):
        if bas not in piv:
            continue
        for lag in (1, 2, 3, 4):
            i = aylar.index(bas)
            if i + lag >= len(aylar):
                continue
            x = piv[[bas, aylar[i + lag]]].dropna()
            print(
                f"    {bas} -> {aylar[i + lag]}  n={len(x):,}  kor={x.iloc[:, 0].corr(x.iloc[:, 1]):+.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
