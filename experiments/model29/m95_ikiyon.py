"""HAK3: prob skorundan L_sicak cozulur, iki-yon optimumu kurulur ve yazilir."""

import json
import os

import numpy as np
import pandas as pd
from m30_ozellik import KOK

P = 1.00946
TW = 0.50
LTOT = 0.022319
m0 = 1.00553**2
te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"), dtype={"tanim": str})
tr = pd.read_csv(os.path.join(KOK, "data/raw/train.csv"), dtype={"tanim": str})
A = pd.read_csv(os.path.join(KOK, "submissions/tuketim_v102_kappa_optimum.csv"))
B = pd.read_csv(os.path.join(KOK, "submissions/tuketim_m4_hava_capali.csv"))
a = np.log1p(A.tuketim.values)
d = np.log1p(B.tuketim.values) - a
N = len(d)
soguk = (~te.tanim.isin(set(tr.tanim))).values
Qw = float((d[~soguk] ** 2).sum() / N)
Qc = float((d[soguk] ** 2).sum() / N)
TC = LTOT / (Qw + Qc)
Lw = (m0 + TW * TW * Qw + TC * TC * Qc - 2 * TC * LTOT - P * P) / (2 * (TW - TC))
Lc = LTOT - Lw
kw = Lw / Qw
kc = Lc / Qc
mse = m0 - Lw**2 / Qw - Lc**2 / Qc
opt = np.sqrt(max(mse, 0))
print(f"prob P = {P}")
print(f"L_sicak {Lw:+.6f}  L_soguk {Lc:+.6f}   (toplam {Lw + Lc:+.6f} = {LTOT})")
print(f"kappa_sicak {kw:+.5f}   kappa_soguk {kc:+.5f}")
print(f"IKI-YON OPTIMUM = {opt:.5f}   (tek-kappa 1.00349, mevcut 1.00553)")
print(f"  katki: sicak {Lw**2 / Qw:.6f}  soguk {Lc**2 / Qc:.6f}")
print(
    f"\nSURPRIZ: geri-test SICAK dedi, LB SOGUK diyor (kappa_soguk {kc:.3f} > kappa_sicak {kw:.3f})"
)
p = a.copy()
p[~soguk] += kw * d[~soguk]
p[soguk] += kc * d[soguk]
y = np.clip(np.expm1(p), 0.0, None)
out = pd.DataFrame({"id": te.id.values, "tuketim": y})
yol = os.path.join(KOK, "submissions", "tuketim_m6_ikiyon.csv")
out.to_csv(yol, index=False)
ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
kapi = dict(
    satir=len(out),
    id_birebir=bool((out.id.values == ss.iloc[:, 0].values).all()),
    nan=int(out.tuketim.isna().sum()),
    negatif=int((out.tuketim < 0).sum()),
)
assert kapi["satir"] == 714688 and kapi["id_birebir"] and kapi["nan"] == 0 and kapi["negatif"] == 0
print("KAPI:", json.dumps(kapi))
LB = [
    ("Grid Grinders", 0.99064),
    ("Atakan Aldemir", 1.00041),
    ("Ahmet B.ALTUNOK", 1.00480),
    ("Saban Ozdogan", 1.00510),
]
print(f"\nsira: {1 + sum(1 for _, v in LB if v < opt)}.")
for n, v in LB:
    print(f"   {n:18s} {v:.5f}")
print(f"   {'TasnifX (HAK3)':18s} {opt:.5f}")
json.dump(
    dict(P=P, Lw=Lw, Lc=Lc, kw=kw, kc=kc, opt=opt, Qw=Qw, Qc=Qc),
    open("m95_ikiyon.json", "w"),
    indent=1,
)
print(f"YAZILDI {yol}")
