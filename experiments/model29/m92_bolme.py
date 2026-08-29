"""m4 yonunu DIK parcalara bol: her parca ayri kappa ile optimize edilirse tavan nedir?
Bilinen: L_toplam = 0.022319, Q_toplam = 0.121396 (olculdu, S=1.04300)."""

import os

import numpy as np
import pandas as pd
from m30_ozellik import KOK

te = pd.read_csv(
    os.path.join(KOK, "data/raw/test.csv"), parse_dates=["tarih"], dtype={"tanim": str}
)
tr = pd.read_csv(
    os.path.join(KOK, "data/raw/train.csv"), parse_dates=["tarih"], dtype={"tanim": str}
)
a = np.log1p(
    pd.read_csv(os.path.join(KOK, "submissions/tuketim_v102_kappa_optimum.csv")).tuketim.values
)
b = np.log1p(
    pd.read_csv(os.path.join(KOK, "submissions/tuketim_m4_hava_capali.csv")).tuketim.values
)
d = b - a
N = len(d)
L_TOP = 0.022319
Q_TOP = float((d**2).mean())
m0 = 1.00553**2
print(
    f"Q_toplam {Q_TOP:.6f}  L_toplam {L_TOP:.6f}  tek-kappa optimum {np.sqrt(m0 - L_TOP**2 / Q_TOP):.5f}"
)

sicak_kume = set(tr.tanim)
soguk = (~te.tanim.isin(sicak_kume)).values
ilk = tr.groupby("tanim").tarih.min()
kuyruk = (~soguk) & (te.tanim.map(ilk) >= pd.Timestamp("2026-03-26")).values
cekirdek = ~soguk & ~kuyruk
ilk_te = te.groupby("tanim").tarih.min()
dalga = te.tanim.map(ilk_te).eq(pd.Timestamp("2026-05-11")).values

print("\nDIK BOLMELER (her parcanin Q'su; L'ler bilinmiyor ama toplamlari 0.022319)")
for ad, parcalar in [
    ("soguk | sicak", [("soguk", soguk), ("sicak", ~soguk)]),
    ("soguk | kuyruk | cekirdek", [("soguk", soguk), ("kuyruk", kuyruk), ("cekirdek", cekirdek)]),
    ("dalga | dalga-disi", [("dalga", dalga), ("disi", ~dalga)]),
    (
        "ay 4-5 | ay 6-7",
        [
            ("nis-may", te.tarih.dt.month.isin([4, 5]).values),
            ("haz-tem", te.tarih.dt.month.isin([6, 7]).values),
        ],
    ),
]:
    print(f"  {ad}")
    for p_ad, m in parcalar:
        Qp = float((d[m] ** 2).sum() / N)
        print(
            f"     {p_ad:10s} satir {m.sum():7,d} (%{100 * m.mean():4.1f})  Q_parca {Qp:.6f} (toplam Q'nun %{100 * Qp / Q_TOP:.1f}'i)"
        )

print("\nTAVAN HESABI -- soguk|sicak bolmesi, L_sicak degerine gore:")
Qs = float((d[soguk] ** 2).sum() / N)
Qw = float((d[~soguk] ** 2).sum() / N)
print(f"  Q_soguk {Qs:.6f}   Q_sicak {Qw:.6f}")
print(f"  {'L_sicak':>9s} {'L_soguk':>9s} {'k_sicak':>8s} {'k_soguk':>8s} {'2-yon optimum':>14s}")
for Lw in [0.0, 0.010, 0.0223, 0.030, 0.040, 0.050, 0.060]:
    Ls = L_TOP - Lw
    mse = m0 - Lw**2 / Qw - Ls**2 / Qs
    print(f"  {Lw:9.4f} {Ls:9.4f} {Lw / Qw:+8.3f} {Ls / Qs:+8.3f} {np.sqrt(max(mse, 0)):14.5f}")
print(
    "\n  (L_sicak = L_toplam ise soguk yonu tamamen notr; ayrisma ne kadar buyukse kazanc o kadar buyuk)"
)
print(f"  2. sira icin gereken: 1.00040  -> MSE {1.00040**2:.6f}")
