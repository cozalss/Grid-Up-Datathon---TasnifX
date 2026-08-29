# ruff: noqa: F821
# KANIT KAYDI -- bu dosya TEK BASINA calismaz. Karar analizi ajani bunu
# s2_karar.py ile PAYLASILAN ad alaninda (exec) kosturdu; ADAYLAR, RMAT,
# LMAT, NMC, GEREK, strateji, prob_gonderim, birlesim, kazanc oradan gelir.
# Kostugu HALIYLE saklaniyor -- yeniden yazmak ne olctugunu degistirebilir.
import itertools
import json
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.argv = ["x"]
exec(
    open("s2_karar.py", encoding="utf-8")
    .read()
    .split("# ------------------------------------------------------- 1. GUN IKILI")[0]
)

# 1) uclu sonda taramasi (gun1 = 3 sonda; gun2 = birlesim + 2 sonda)
res = {}
for T in itertools.combinations(ADAYLAR, 3):
    g1 = strateji([[("sonda", t) for t in T]])[0]
    res[T] = g1
sir = sorted(res.items(), key=lambda kv: -kv[1][0])
print("=== 1. GUN 3 SONDA (birlesim YOK) -- en iyi 10 ===")
for T, (p, m) in sir[:10]:
    print(f"  {'+'.join(T):<16} P={p:.3f}  medyan dMSE={-m:.6f}")
print("  ... en kotu:", "+".join(sir[-1][0]), f"P={sir[-1][1][0]:.3f}")

# 2) en iyi ikili (2 sonda + birlesim) ile karsilastirma
best2 = max(
    ((A, B) for A, B in itertools.combinations(ADAYLAR, 2)),
    key=lambda ab: strateji([[("sonda", ab[0]), ("sonda", ab[1]), ("birlesim",)]])[0][0],
)
p2 = strateji([[("sonda", best2[0]), ("sonda", best2[1]), ("birlesim",)]])[0]
print(f"\nen iyi 2sonda+birlesim: {best2[0]}+{best2[1]} P={p2[0]:.3f} med={-p2[1]:.6f}")

# 3) y40 isaret kosullu
mask = RMAT[:, IX["y40"]] > 0
for ad, T in [
    ("z2+y40 (2s+b)", [("sonda", "z2"), ("sonda", "y40"), ("birlesim",)]),
    ("y40+z2+y46 (3s)", [("sonda", "y40"), ("sonda", "z2"), ("sonda", "y46")]),
]:
    en = np.zeros(NMC)
    olc = []
    for e in T:
        if e[0] == "sonda":
            idx, cm = prob_gonderim(e[1])
            olc.append(e[1])
        else:
            idx, cm = birlesim(olc)
        en = np.maximum(en, kazanc(idx, cm, LMAT))
    print(
        f"{ad}: P|r_y40>0 = {(en[mask] >= GEREK).mean():.3f}   P|r_y40<0 = {(en[~mask] >= GEREK).mean():.3f}"
    )

# 4) uc gunluk tam plan, en iyi uclu ile
T = sir[0][0]
kalan = [a for a in ADAYLAR if a not in T]
planB = [
    [("sonda", t) for t in T],
    [("birlesim",), ("sonda", kalan[0]), ("sonda", kalan[1])],
    [("birlesim",), ("sonda", kalan[2]), ("sonda", kalan[3])],
]
planA = [
    [("sonda", T[0]), ("sonda", T[1]), ("birlesim",)],
    [("sonda", T[2]), ("sonda", kalan[0]), ("birlesim",)],
    [("sonda", kalan[1]), ("sonda", kalan[2]), ("birlesim",)],
]
for ad, pl in [
    ("B: 3sonda -> birlesim+2sonda -> birlesim+2sonda", planB),
    ("A: 2sonda+birlesim her gun", planA),
]:
    print(
        f"\n{ad}\n   "
        + "  ".join(f"gun{i + 1} P={p:.3f} med={-m:.6f}" for i, (p, m) in enumerate(strateji(pl)))
    )
print("\nsira onerisi (kalite/ortogonallik):", T, "->", kalan)
json.dump(
    {
        "en_iyi_uclu": list(sir[0][0]),
        "uclu_P2": {"+".join(k): v[0] for k, v in sir},
        "en_iyi_ikili_2s1b": [best2[0], best2[1], p2[0]],
    },
    open("s2_ucler.json", "w", encoding="utf-8"),
    indent=1,
)
