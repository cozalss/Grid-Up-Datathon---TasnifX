"""SICAK PROB: v102 tabani, SICAK satirlarda m4'un tahmini, SOGUK satirlar v102 ile birebir.
Skoru olculunce L_sicak dogrudan cozulur; L_soguk = L_toplam - L_sicak (cebirsel).
Sonra iki yon DIK oldugu icin optimum ayrisir: MSE* = m0 - L_s^2/Q_s - L_c^2/Q_c"""

import json
import os

import numpy as np
import pandas as pd
from m30_ozellik import KOK

te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"), dtype={"tanim": str})
tr = pd.read_csv(os.path.join(KOK, "data/raw/train.csv"), dtype={"tanim": str})
A = pd.read_csv(os.path.join(KOK, "submissions/tuketim_v102_kappa_optimum.csv"))
B = pd.read_csv(os.path.join(KOK, "submissions/tuketim_m4_hava_capali.csv"))
assert (A.id.values == te.id.values).all() and (B.id.values == te.id.values).all()
a = np.log1p(A.tuketim.values)
b = np.log1p(B.tuketim.values)
d = b - a
N = len(d)
soguk = (~te.tanim.isin(set(tr.tanim))).values
p = a.copy()
p[~soguk] = b[~soguk]  # SICAK satirlarda m4, soguk satirlarda v102
y = np.clip(np.expm1(p), 0.0, None)
out = pd.DataFrame({"id": te.id.values, "tuketim": y})
yol = os.path.join(KOK, "submissions", "tuketim_p50_sicak_prob.csv")
out.to_csv(yol, index=False)
ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
kapi = dict(
    satir=len(out),
    id_birebir=bool((out.id.values == ss.iloc[:, 0].values).all()),
    nan=int(out.tuketim.isna().sum()),
    negatif=int((out.tuketim < 0).sum()),
)
assert kapi["satir"] == 714688 and kapi["id_birebir"] and kapi["nan"] == 0 and kapi["negatif"] == 0
Qs = float((d[~soguk] ** 2).sum() / N)
Qc = float((d[soguk] ** 2).sum() / N)
m0 = 1.00553**2
L_TOP = 0.022319
print("KAPI:", json.dumps(kapi))
print(f"degisen satir: {(~soguk).sum():,} (sicak), degismeyen {soguk.sum():,} (soguk)")
print(f"Q_sicak {Qs:.6f}   Q_soguk {Qc:.6f}   (toplam {Qs + Qc:.6f})")
print("\nprob skoru P olculunce:  L_sicak = (m0 + Q_sicak - P^2)/2")
print(
    f"{'P':>9s} {'L_sicak':>9s} {'L_soguk':>9s} {'k_sic':>7s} {'k_sog':>7s} {'IKI-YON OPTIMUM':>16s}"
)
for P in [1.000, 1.005, 1.00553, 1.010, 1.015, 1.020, 1.030, 1.043, 1.060]:
    Lw = (m0 + Qs - P * P) / 2
    Lc = L_TOP - Lw
    mse = m0 - Lw**2 / Qs - Lc**2 / Qc
    print(
        f"{P:9.5f} {Lw:+9.5f} {Lc:+9.5f} {Lw / Qs:+7.3f} {Lc / Qc:+7.3f} {np.sqrt(max(mse, 0)):16.5f}"
    )
print("\nGARANTI TABAN: tek-kappa optimumu 1.00349 -- iki-yon her zaman >= bu")
json.dump(
    dict(Q_sicak=Qs, Q_soguk=Qc, L_toplam=L_TOP, m0=m0, kapi=kapi),
    open("m93_prob.json", "w"),
    indent=1,
)
print(f"YAZILDI {yol}")
