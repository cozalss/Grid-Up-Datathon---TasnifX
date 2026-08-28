"""Yeni modelin seviyesini v102'nin LB-KALIBRE seviyesine capala (rejim bazli).
Gerekce: docs/52 s9 -- v102'nin sicak cekirdekte ORTALAMA ARTIGI TAM SIFIR (LB ile olculdu)
ve 'global seviye' yonu elekte %99,07 kapsamla BOS. Yani v102'nin seviyesi LB-optimum.
Geri-testte olculdu: modelin seviye yanliligi +-0.15 ve ISARETI KESIMDEN KESIME DONUYOR
(-0.1535 / +0.1735) -> ongorulemez gurultu -> capalamak serbest kazanc (~0.012 RMSLE)."""

import json
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
    pd.read_csv(os.path.join(KOK, "submissions/tuketim_m1_ileri_huber.csv")).tuketim.values
)

# v83'un uc rejimi (yarin_coz.py:_rejim_maskeleri ile ayni tanim)
sicak_kume = set(tr.tanim)
ilk = tr.groupby("tanim").tarih.min()
soguk = (~te.tanim.isin(sicak_kume)).values
ilk_te = te.tanim.map(ilk)
kuyruk = (~soguk) & (ilk_te >= pd.Timestamp("2026-03-26")).values
cekirdek = ~soguk & ~kuyruk
print(f"rejimler: soguk {soguk.sum():,}  kuyruk {kuyruk.sum():,}  cekirdek {cekirdek.sum():,}")

b2 = b.copy()
rap = {}
for ad, m in [("soguk", soguk), ("kuyruk", kuyruk), ("cekirdek", cekirdek)]:
    if m.sum() == 0:
        continue
    d = a[m].mean() - b[m].mean()
    b2[m] = b[m] + d
    rap[ad] = dict(
        satir=int(m.sum()), v102=float(a[m].mean()), yeni=float(b[m].mean()), kaydirma=float(d)
    )
    print(f"  {ad:9s} v102 {a[m].mean():.4f}  yeni {b[m].mean():.4f}  kaydirma {d:+.4f}")

y = np.clip(np.expm1(b2), 0.0, None)
out = pd.DataFrame({"id": te.id.values, "tuketim": y})
yol = os.path.join(KOK, "submissions", "tuketim_m2_capali.csv")
out.to_csv(yol, index=False)
ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
kapi = dict(
    satir=len(out),
    id_birebir=bool((out.id.values == ss.iloc[:, 0].values).all()),
    nan=int(out.tuketim.isna().sum()),
    negatif=int((out.tuketim < 0).sum()),
    maks=float(out.tuketim.max()),
    log_ort=float(np.log1p(out.tuketim).mean()),
)
print("KAPI:", json.dumps(kapi))
assert kapi["id_birebir"] and kapi["nan"] == 0 and kapi["negatif"] == 0 and kapi["satir"] == 714688

d = b2 - a
Q = float((d**2).mean())
m0 = 1.00553**2
print(f"\nYENI YON: Q = {Q:.6f}  (capalanmadan once 0.138731)")
print(f"  basabas: yeni model {np.sqrt(m0 + Q):.5f}'ten iyi ise HARMAN KAZANDIRIR")
print(f"  {'S':>9s} {'kappa*':>8s} {'optimum RMSLE':>14s}")
for S in [0.95, 0.97, 0.99, 1.00553, 1.02, 1.04, 1.06]:
    L = (m0 + Q - S * S) / 2
    print(f"  {S:9.5f} {L / Q:+8.4f} {np.sqrt(max(m0 - L * L / Q, 0)):14.5f}")
json.dump(dict(rejim=rap, Q=Q, kapi=kapi), open("m44_capa.json", "w"), indent=1)
print(f"\nYAZILDI {yol}")
