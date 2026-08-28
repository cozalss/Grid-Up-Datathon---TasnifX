"""B (bayat) icin en iyi kestirimci; E icin pencere agirliklarinin EK KARELER cozumu."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m10_ortak import *
from m17_lab import grupla, ozellik

tr = yukle()
res = {"B": {}, "E_agirlik": {}}
KOL = ["k3", "k7", "k14", "k28", "k91", "ly_all"]
for kesim in KESIMLER:
    gec, hed = hazirla(tr, kesim)
    oz = ozellik(gec, kesim)
    gr = grupla(oz)
    kok = float(gec.ly.mean())
    g = hed.tanim.map(gr).values
    y = hed.ly.values
    # ---- B ----
    B = g == "B_bayat"
    print(f"\n===== {kesim} | B satir {B.sum():,} trafo {hed[B].tanim.nunique()} =====")
    res["B"][kesim] = {}
    adaylar = {
        "ly_all": oz.ly_all,
        "s7": oz.s7,
        "s14": oz.s14,
        "s28": oz.s28,
        "s56": oz.s56,
        "s91": oz.s91,
        "s182": oz.s182,
    }
    for ad, s in adaylar.items():
        p = geri_dolgu(hed, s, oz.ly_all, kok=kok)
        r = float(np.sqrt(((y[B] - p[B]) ** 2).mean()))
        res["B"][kesim][ad] = r
        print(f"   B {ad:8s} {r:.4f}")
    for a in [0.3, 0.5, 0.7]:
        p = a * geri_dolgu(hed, oz.s28, oz.ly_all, kok=kok) + (1 - a) * geri_dolgu(
            hed, oz.ly_all, kok=kok
        )
        r = float(np.sqrt(((y[B] - p[B]) ** 2).mean()))
        res["B"][kesim][f"s28*{a}+all"] = r
        print(f"   B s28*{a}+all*{1 - a:.1f} {r:.4f}")
    # ---- E: kisitli EK (agirliklar toplami 1, negatif serbest degil) ----
    E = g == "E_normal"
    X = np.column_stack([geri_dolgu(hed, oz[c], oz.ly_all, kok=kok) for c in KOL])[E]
    ye = y[E]
    # sum w =1 kisiti: son kolonu referans al
    D = X[:, :-1] - X[:, [-1]]
    w = np.linalg.lstsq(D, ye - X[:, -1], rcond=None)[0]
    w = np.append(w, 1 - w.sum())
    r = float(np.sqrt(((ye - X @ w) ** 2).mean()))
    base = 0.8 * X[:, 1] + 0.2 * X[:, 5]
    r0 = float(np.sqrt(((ye - base) ** 2).mean()))
    print(
        "   E EK agirliklari: "
        + " ".join(f"{c}={v:+.3f}" for c, v in zip(KOL, w))
        + f"  -> {r:.4f} (0.8k7+0.2all: {r0:.4f})"
    )
    res["E_agirlik"][kesim] = {"w": dict(zip(KOL, map(float, w))), "rmsle": r, "taban": r0}
json_yaz("eksen_B_E", res)
print("\n=== B: 4 kesim ortalamasi ===")
ad = list(res["B"][KESIMLER[0]].keys())
for a in ad:
    print(f"  {a:14s} {np.mean([res['B'][k][a] for k in KESIMLER]):.4f}")
print("\n=== E agirliklari kesimler arasi ===")
for c in KOL:
    print(f"  {c:8s} " + " ".join(f"{res['E_agirlik'][k]['w'][c]:+.3f}" for k in KESIMLER))
