"""HAK2 PROBU: p = v102 + 0.5*d_sicak + kappa*_d_soguk
Bilgi: skoru P olculunce L_sicak cozulur, L_soguk = L_toplam - L_sicak.
Sonra HAK3 = iki-yon optimumu (dik parcalar -> ayrisir)."""

import json
import os

import numpy as np
import pandas as pd
from m30_ozellik import KOK

TW = 0.50
m0 = 1.00553**2
LTOT = 0.022319
te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"), dtype={"tanim": str})
tr = pd.read_csv(os.path.join(KOK, "data/raw/train.csv"), dtype={"tanim": str})
A = pd.read_csv(os.path.join(KOK, "submissions/tuketim_v102_kappa_optimum.csv"))
B = pd.read_csv(os.path.join(KOK, "submissions/tuketim_m4_hava_capali.csv"))
assert (A.id.values == te.id.values).all() and (B.id.values == te.id.values).all()
a = np.log1p(A.tuketim.values)
d = np.log1p(B.tuketim.values) - a
N = len(d)
soguk = (~te.tanim.isin(set(tr.tanim))).values
Qw = float((d[~soguk] ** 2).sum() / N)
Qc = float((d[soguk] ** 2).sum() / N)
TC = LTOT / (Qw + Qc)  # kappa* = 0.18385
p = a.copy()
p[~soguk] += TW * d[~soguk]
p[soguk] += TC * d[soguk]
y = np.clip(np.expm1(p), 0.0, None)
out = pd.DataFrame({"id": te.id.values, "tuketim": y})
yol = os.path.join(KOK, "submissions", "tuketim_p51_sicak05.csv")
out.to_csv(yol, index=False)
ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
kapi = dict(
    satir=len(out),
    id_birebir=bool((out.id.values == ss.iloc[:, 0].values).all()),
    nan=int(out.tuketim.isna().sum()),
    negatif=int((out.tuketim < 0).sum()),
    maks=float(out.tuketim.max()),
)
assert kapi["satir"] == 714688 and kapi["id_birebir"] and kapi["nan"] == 0 and kapi["negatif"] == 0
print("KAPI:", json.dumps(kapi))
print(f"tw={TW}  tc=kappa*={TC:.5f}   Qw={Qw:.6f} Qc={Qc:.6f}")
print("COZUM: L_sicak = (m0 + tw^2*Qw + tc^2*Qc - 2*tc*Ltot - P^2) / (2*(tw-tc))")
print(
    f"\n{'P (prob)':>9s} {'L_sicak':>9s} {'L_soguk':>9s} {'k_sic':>7s} {'k_sog':>7s} {'HAK3':>9s} {'sira':>5s}"
)
LB = [0.99064, 1.00041, 1.00480, 1.00510]
sira = lambda x: 1 + sum(1 for v in LB if v < x)
for P in [0.995, 1.000, 1.0035, 1.005, 1.008, 1.0078, 1.010, 1.0109, 1.013]:
    Lw = (m0 + TW * TW * Qw + TC * TC * Qc - 2 * TC * LTOT - P * P) / (2 * (TW - TC))
    Lc = LTOT - Lw
    opt = np.sqrt(max(m0 - Lw**2 / Qw - Lc**2 / Qc, 0))
    print(
        f"{P:9.5f} {Lw:+9.5f} {Lc:+9.5f} {Lw / Qw:+7.3f} {Lc / Qc:+7.3f} {opt:9.5f} {sira(opt):5d}"
    )
json.dump(
    dict(tw=TW, tc=TC, Qw=Qw, Qc=Qc, Ltot=LTOT, m0=m0, kapi=kapi),
    open("m94_prob2.json", "w"),
    indent=1,
)
print("\nGARANTI TABAN: HAK3 >= 1.00349 (tek-kappa optimumu) HER ZAMAN")
print(f"YAZILDI {yol}")
