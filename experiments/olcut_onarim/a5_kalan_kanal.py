"""A2 devami: tanim_* cikinca KALAN kimlik kanali nedir?"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))
import tuketim_model as tm  # noqa

egitim = pd.read_parquet(KOK / "data/interim/deney/egitim.parquet")
test = pd.read_parquet(KOK / "data/interim/deney/test.parquet")
G = tm.GRUP
tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
uretim = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
KIMLIK = {"tanim_num", "tanim_uzunluk", "tanim_on2", "tanim_on3", "tanim_on4", "tanim_on5"}
# trafo-ici SABIT olan, kimlik-disi uretim kolonlari
ana = egitim
g = ana.groupby(G, observed=True)
sabit = []
for k in uretim:
    if k in KIMLIK:
        continue
    s = ana[k]
    if str(s.dtype) == "category":
        s = s.cat.codes
    s = pd.to_numeric(s, errors="coerce")
    if (
        pd.DataFrame({"t": ana[G].to_numpy(), "v": s.to_numpy()})
        .groupby("t", observed=True)["v"]
        .nunique(dropna=False)
        .le(1)
        .mean()
        > 0.999
    ):
        sabit.append(k)
print(f"kimlik-disi TRAFO-SABIT kolon sayisi: {len(sabit)}")
print("  ", ", ".join(sabit))
prof = ana.drop_duplicates(G).set_index(G)[sabit]
for k in sabit:
    if str(prof[k].dtype) == "category":
        prof[k] = prof[k].astype(str)
anahtar = prof.astype(str).agg("|".join, axis=1)
print(
    f"\nbu {len(sabit)} kolonun urettigi TEKIL profil sayisi: {anahtar.nunique():,} "
    f"/ {len(anahtar):,} trafo  (ayirt gucu {anahtar.nunique() / len(anahtar):.4f})"
)
print("\nBir SOGUK trafo, egitim parcasinda kac trafoyla AYNI profili paylasiyor?")
print(f"{'blok':8}{'medyan':>9}{'ort':>9}{'tek eslesme %':>16}{'<=3 eslesme %':>16}")
for b in tm.BLOKLAR:
    dog = egitim[egitim["_blok"] == b.ad]
    kalan = egitim[egitim["_blok"] != b.ad]
    st = dog.loc[dog["soguk_mu"] == 1, G].unique()
    kt = set(kalan[G].unique()) - set(st)  # ONARIM sonrasi egitimdeki trafolar
    say = anahtar.loc[list(kt)].value_counts()
    e = np.array([int(say.get(anahtar.get(t, "?"), 0)) for t in st])
    print(
        f"{b.ad:8}{np.median(e):>9.0f}{e.mean():>9.1f}{100 * np.mean(e == 1):>15.1f}%"
        f"{100 * np.mean((e >= 1) & (e <= 3)):>15.1f}%"
    )
ts = test.loc[test["soguk_mu"] == 1, G].unique()
kt = set(egitim[G].unique())
say = anahtar.loc[list(kt & set(anahtar.index))].value_counts()
prof_t = test.drop_duplicates(G).set_index(G)[sabit]
for k in sabit:
    if str(prof_t[k].dtype) == "category":
        prof_t[k] = prof_t[k].astype(str)
at = prof_t.astype(str).agg("|".join, axis=1)
e = np.array([int(say.get(at.get(t, "?"), 0)) for t in ts])
print(
    f"{'TEST':8}{np.median(e):>9.0f}{e.mean():>9.1f}{100 * np.mean(e == 1):>15.1f}%"
    f"{100 * np.mean((e >= 1) & (e <= 3)):>15.1f}%"
)
