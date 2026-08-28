"""Hedef 'tuketim' bir kafes uzerinde mi yasiyor?

Elektrik sayaclarinda tuketim = okuma * carpan olur; carpan trafoya ozgudur.
Eger oyleyse her trafonun degerleri trafoya ozgu bir sabitin tam katlaridir
ve tahminleri o kafese oturtmak RMSLE kazandirir.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

tr = pd.read_csv("data/raw/train.csv", usecols=["tanim", "tuketim"], dtype={"tanim": str})
print(f"satir {len(tr):,}  trafo {tr.tanim.nunique():,}")

t = tr.tuketim.to_numpy(dtype=float)
print(f"min {t.min()}  max {t.max()}  sifir {int((t == 0).sum()):,}")

# --- 1. Global ondalik yapisi ---------------------------------------------
# Degeri 100 ile carpip tam sayiya ne kadar yakin oldugunu olc.
s = t * 100.0
kalinti = np.abs(s - np.round(s))
print("\n=== 1. GLOBAL ONDALIK ===")
print(f"  x100 tamsayidan sapma: max {kalinti.max():.3e}  ort {kalinti.mean():.3e}")
print(f"  x100 tamsayi olan oran: {(kalinti < 1e-6).mean():.6f}")

si = np.round(s).astype(np.int64)
# Son iki hane dagilimi -> kurus kismi gercekten serbest mi?
son2 = si % 100
c = Counter(son2.tolist())
print(f"  farkli 'kurus' degeri: {len(c)} / 100")
print(f"  en sik 10: {c.most_common(10)}")

# --- 2. Global GCD ---------------------------------------------------------
nz = si[si != 0]
g_global = 0
for v in nz[:200000]:
    g_global = math.gcd(g_global, int(v))
    if g_global == 1:
        break
print("\n=== 2. GLOBAL GCD (x100) ===")
print(f"  gcd = {g_global}  ->  adim {g_global / 100.0} kWh")

# --- 3. Trafo bazli GCD ----------------------------------------------------
print("\n=== 3. TRAFO BAZLI GCD ===")
tr["si"] = si
adimlar: dict[int, tuple[int, int, int]] = {}
for tanim, grp in tr.groupby("tanim", sort=False):
    v = grp.si.to_numpy()
    v = v[v != 0]
    if len(v) < 10:
        continue
    g = 0
    for x in v:
        g = math.gcd(g, int(x))
        if g == 1:
            break
    adimlar[int(tanim)] = (g, len(v), len(np.unique(v)))

gs = np.array([a[0] for a in adimlar.values()])
print(f"  degerlendirilen trafo: {len(adimlar):,}")
print(f"  gcd == 1 (kafes YOK)      : {(gs == 1).sum():,}  ({(gs == 1).mean():.1%})")
print(f"  gcd  > 1 (kafes VAR ADAYI): {(gs > 1).sum():,}  ({(gs > 1).mean():.1%})")
if (gs > 1).any():
    cc = Counter(gs[gs > 1].tolist())
    print(f"  en sik adimlar (x100): {cc.most_common(15)}")

# --- 4. Trafo basina farkli deger sayisi ------------------------------------
uniq = np.array([a[2] for a in adimlar.values()])
n = np.array([a[1] for a in adimlar.values()])
print("\n=== 4. TEKRAR YAPISI ===")
print(f"  farkli/toplam oran ortancasi: {np.median(uniq / n):.4f}")
print(f"  bu oran < 0.5 olan trafo: {(uniq / n < 0.5).sum():,}")
