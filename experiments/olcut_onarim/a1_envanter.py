"""A1: kimlik-tasiyici kolon envanteri + soguk ezberlenebilirlik olcumu."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))
import tuketim_model as tm  # noqa

ONB = KOK / "data" / "interim" / "deney"
egitim = pd.read_parquet(ONB / "egitim.parquet")
test = pd.read_parquet(ONB / "test.parquet")
print(f"egitim {egitim.shape}  test {test.shape}")
print("egitim bloklar:", egitim["_blok"].value_counts().to_dict())

tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
uretim = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
print(f"\ntum {len(tum)} | uretim {len(uretim)} kolon")
print("YALIN_CIKARILAN onekleri:", tm.YALIN_CIKARILAN)
print("\nURETIM KOLONLARI:")
for i in range(0, len(uretim), 6):
    print("   ", ", ".join(uretim[i : i + 6]))

# ---- kimlik tasiyiciligi: kolonun trafo icinde SABIT mi + trafoyu ne kadar ayirt ediyor
ana = egitim[egitim["_blok"].isin([b.ad for b in tm.BLOKLAR])]
g = ana.groupby(tm.GRUP, observed=True)
n_trafo = ana[tm.GRUP].nunique()
print(f"\nana bloklarda {n_trafo:,} trafo, {len(ana):,} satir")
print(f"\n{'kolon':28}{'trafo-ici sabit%':>18}{'tekil deger':>13}{'ayirt gucu':>12}")
rapor = []
for k in uretim:
    s = ana[k]
    if str(s.dtype) == "category":
        s = s.cat.codes.replace(-1, np.nan)
    s = pd.to_numeric(s, errors="coerce")
    tmp = pd.DataFrame({"t": ana[tm.GRUP].to_numpy(), "v": s.to_numpy()})
    nun = tmp.groupby("t", observed=True)["v"].nunique(dropna=False)
    sabit = float((nun <= 1).mean())
    tekil = int(s.nunique(dropna=True))
    # ayirt gucu: trafo-sabit kolonun aldigi tekil deger sayisi / trafo sayisi
    if sabit > 0.99:
        trafo_deger = tmp.groupby("t", observed=True)["v"].first()
        ayirt = float(trafo_deger.nunique(dropna=False) / n_trafo)
    else:
        ayirt = float("nan")
    rapor.append((k, sabit, tekil, ayirt))
rapor.sort(key=lambda r: (-(r[3] if r[3] == r[3] else -1), -r[1]))
for k, sabit, tekil, ayirt in rapor:
    a = f"{ayirt:>12.4f}" if ayirt == ayirt else f"{'-':>12}"
    print(f"{k:28}{100 * sabit:>17.1f}%{tekil:>13,}{a}")
