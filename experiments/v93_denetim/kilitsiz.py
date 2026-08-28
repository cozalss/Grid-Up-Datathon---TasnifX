"""v93 denetimi -- adim 11: sha kilidi OLMAYAN 9 dosyayi topluca cikar."""

from __future__ import annotations

import coz as C
import numpy as np
from duyarlilik2 import coz_alt

KILITLI = {"v18", "v27", "v30", "v44", "v46", "v47", "v50", "v73", "v81", "v83"}
KILITSIZ = [e for e in C.OLCULENLER if e not in KILITLI]

p0 = C.yukle(C.TABAN)
d93 = C.yukle("v93") - p0
print("kilidi olmayan:", KILITSIZ)
tam = coz_alt(C.OLCULENLER, d93)
print(
    f"TAM havuz (19)          : rank={tam['rank']} RMSLE={tam['rmsle']:.6f} "
    f"|w|1={tam['w_l1']:.2f} artik={tam['artik']:.2%}"
)
alt = [e for e in C.OLCULENLER if e in KILITLI]
r = coz_alt(alt, d93)
print(
    f"YALNIZ kilitli 10 dosya : rank={r['rank']} RMSLE={r['rmsle']:.6f} "
    f"|w|1={r['w_l1']:.2f} artik={r['artik']:.2%}"
)
print(f"  -> sapma {r['rmsle'] - tam['rmsle']:+.6f}")
print(
    "\nNOT: d93 TAM havuzla kuruldu; alt havuz onun %{:.0f}'ini goremez,".format(r["artik"] * 100)
)
print("     bu yuzden fark 'kararsizlik' degil 'gorulemeyen pay'dir.")

# kilitli havuzun kendi optimumu ne olurdu?
w = np.array(list(r["w"].values()))
print(
    f"\nKilitli-10 havuzunun KENDI optimumu: RMSLE = "
    f"{np.sqrt(C.ENV['v83']['skor'] ** 2 - sum(np.array(r['b']) * w)):.6f}"
)
