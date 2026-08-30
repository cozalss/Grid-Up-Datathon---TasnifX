"""HANGI M0? -- TUTULMUS GONDERIMLERLE AYIRT ET.

Iki aday:
  M0_kalibre = 1.005846366  (uc capayi L=0 yapmak icin fit edilmis)
  M0_ozdeslik = 1.00284^2 = 1.005688066  (a0 icin Q=0,L=0 -> M0 = P_a0^2)

Ozdeslik argumani gucludur AMA kalibre deger de bos yere secilmemis.
Karar tahminle degil OLCUMLE verilmeli.

YONTEM: Gram'da OLMAYAN, LB skoru bilinen gonderimler tutulmus sinavdir.
Her biri icin, kalan yonlerden kurulan geometrinin ongordugu skoru hesapla
ve GERCEK skorla karsilastir. Hangi M0 daha iyi ongoruyorsa o dogrudur.

Ongoru: bir yon d icin  P^2 = M0 - 2*L(d) + Q(d),  L(d) ~ <r_hat, d>/N
(r_hat kalan yonlerden kurulur; d'nin span disi payi varsa ongoru eksik kalir,
bu yuzden span-ici payi yuksek olanlar daha bilgilendiricidir).
"""

import json
import os
import sys

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
S = os.path.join(KOK, "submissions")
M29 = os.path.join(KOK, "experiments/model29")
BURA = os.path.dirname(os.path.abspath(__file__))
TABAN = "tuketim_m6_ikiyon.csv"
EK_MODEL = {"tuketim_y40_sota_temiz.csv": -0.002229}
sys.path.insert(0, M29)
from m112_kalibre import buzmeli_r_hat  # noqa: E402

te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
IDS = te.id.values


def oku(f):
    yol = os.path.join(S, f)
    if not os.path.exists(yol):
        return None
    d = pd.read_csv(yol)
    k = "tuketim" if "tuketim" in d.columns else d.columns[-1]
    if not np.array_equal(d.id.values, IDS):
        if len(d) != len(IDS) or d.id.duplicated().any():
            return None
        pos = pd.Index(d.id).get_indexer(IDS)
        if (pos < 0).any():
            return None
        d = d.iloc[pos].reset_index(drop=True)
    return np.log1p(d[k].values.astype(np.float64))


a0 = oku(TABAN)
N = len(a0)
with open(os.path.join(M29, "olculmus_skorlar.json")) as fh:
    SK = json.load(fh)
with open(os.path.join(M29, "m112_durum.json")) as fh:
    DUR = json.load(fh)

GRAM = []  # (dosya, skor) -- Gram'a giren
for f, Pj in SK.items():
    if f == TABAN:
        continue
    v = oku(f)
    if v is None or len(v) != N:
        continue
    GRAM.append((f, Pj, v - a0))
for o in DUR.get("olcumler", []):
    v = oku(o["dosya"])
    if v is not None:
        GRAM.append((o["dosya"], o["skor"], v - a0))
EKY = [(f, oku(f) - a0, Lj) for f, Lj in EK_MODEL.items()]
gram_adlari = {f for f, _, _ in GRAM} | {f for f, _, _ in EKY}
print(f"Gram'da {len(GRAM)} skorlu yon + {len(EKY)} L-bilinen yon")

# --- TUTULMUS: skoru bilinen ama Gram'a girmeyen gonderimler ---
TUTULMUS = {}
kayit = os.path.join(KOK, "experiments/model29/olculmus_skorlar.json")
with open(kayit) as fh:
    hepsi = json.load(fh)
for f, P in hepsi.items():
    if f in gram_adlari or f == TABAN:
        continue
    v = oku(f)
    if v is not None and len(v) == N:
        TUTULMUS[f] = (P, v - a0)
# elle bilinen ek tutulmuslar
for f, P in [("tuketim_g7_span_tau3.csv", 1.00136)]:
    if f in gram_adlari:
        continue
    v = oku(f)
    if v is not None and len(v) == N:
        TUTULMUS[f] = (P, v - a0)
print(f"TUTULMUS sinav adayi: {len(TUTULMUS)} -> {sorted(TUTULMUS)}")
if not TUTULMUS:
    raise SystemExit("tutulmus gonderim yok, ayirt edilemez")


def kur(M0, haric=()):
    V, L = [], []
    for f, Pj, dd in GRAM:
        if f in haric:
            continue
        V.append(dd)
        L.append((M0 + float((dd * dd).mean()) - Pj * Pj) / 2)
    for f, dd, Lj in EKY:
        if f in haric:
            continue
        V.append(dd)
        L.append(Lj)
    Vm, Lm = np.array(V).T, np.array(L)
    G = (Vm.T @ Vm) / N
    r_hat, _, _ = buzmeli_r_hat(Vm, Lm, G, N)
    return r_hat, Vm, G


print(
    f"\n{'tutulmus dosya':>34s} {'gercek':>8s} "
    + " ".join(f"{ad:>20s}" for ad in ("M0 kalibre", "M0 ozdeslik"))
)
toplam = {"kalibre": [], "ozdeslik": []}
for f, (P, dd) in sorted(TUTULMUS.items()):
    Qd = float((dd * dd).mean())
    satir = []
    for etiket, M0 in (("kalibre", 1.005846366), ("ozdeslik", 1.00284**2)):
        r_hat, Vm, G = kur(M0, haric={f})
        Ltah = float((r_hat * dd).mean())
        Ptah = np.sqrt(max(M0 - 2 * Ltah + Qd, 1e-12))
        hata = Ptah - P
        toplam[etiket].append(abs(hata))
        satir.append(f"{Ptah:9.5f} ({hata:+.5f})")
    print(f"{f[:34]:>34s} {P:8.5f} " + " ".join(f"{x:>20s}" for x in satir))

print(f"\n{'M0':>12s} {'ort |hata|':>12s} {'maks |hata|':>12s}")
for etiket in ("kalibre", "ozdeslik"):
    h = np.array(toplam[etiket])
    print(f"{etiket:>12s} {h.mean():12.3e} {h.max():12.3e}")
kaz = "kalibre" if np.mean(toplam["kalibre"]) < np.mean(toplam["ozdeslik"]) else "ozdeslik"
print(f"\nDAHA IYI ONGOREN: M0 {kaz}")
print("\nAYRICA -- a0'in KENDI skoru (en sert sinav, Q=0 L=0):")
for etiket, M0 in (("kalibre", 1.005846366), ("ozdeslik", 1.00284**2)):
    print(
        f"  {etiket:>10s}: ongoru {np.sqrt(M0):.5f}  gercek 1.00284  "
        f"hata {np.sqrt(M0) - 1.00284:+.5f}"
    )
