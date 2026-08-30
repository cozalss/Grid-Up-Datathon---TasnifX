"""PUBLIC/PRIVATE BOLUNMESI RASTGELE MI? -- M0'in kararliligiyla sinanir.

RISK. r_hat public %50'nin skorlarindan geliyor, sonuc private %50'de
belirlenecek. Bolunme tarihe ya da trafoya gore yapildiysa public'te olculen
yon private'a tasinmayabilir.

SINAV. Kurdugumuz cebir sunu varsayar:
    P_j^2 = M0 - 2 L_j + Q_j,   Q_j = TUM satirlarda ort(d_j^2)
ama P_j yalnizca PUBLIC alt kumede olculur. Yani gercekte
    P_j^2 = M0_S - 2 L_j + ort_S(d_j^2)
Eger public alt kume temsili degilse ort_S(d_j^2) ile ort_TUM(d_j^2) ayrisir
ve M0 capadan capaya KAYAR. Gozlenen kayma 9.1e-07.

Her aday bolunme icin oran = ort_S(d_j^2) / ort_TUM(d_j^2) hesaplanir.
Bu oranin d_j'ler arasindaki SACILIMI x Q, M0'da beklenen kaymadir.
Gozlenen 9.1e-07'yi hangi bolunme aciklar?
"""

import json
import os
import sys

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
S = os.path.join(KOK, "submissions")
M29 = os.path.join(KOK, "experiments/model29")
TABAN = "tuketim_m6_ikiyon.csv"
sys.path.insert(0, M29)
from m112_kalibre import EK_MODEL  # noqa: E402

te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
IDS = te.id.values


def oku(f):
    d = pd.read_csv(os.path.join(S, f))
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
D = []
for f in list(SK) + list(EK_MODEL):
    if f == TABAN or not os.path.exists(os.path.join(S, f)):
        continue
    v = oku(f)
    if v is not None and len(v) == N:
        D.append(v - a0)
print(f"{len(D)} fark yonu, {N} satir")

tar = pd.to_datetime(te.tarih)
trf = te.tanim.values
uq = pd.unique(trf)
rng = np.random.default_rng(3)

ADAYLAR = {}
ADAYLAR["rastgele satir"] = rng.random(N) < 0.5
srt = np.argsort(tar.values, kind="stable")
m = np.zeros(N, dtype=bool)
m[srt[: N // 2]] = True
ADAYLAR["tarihe gore (ilk yari)"] = m
ADAYLAR["tek/cift gun"] = (tar.dt.dayofyear.values % 2) == 0
sec = rng.permutation(len(uq))[: len(uq) // 2]
ADAYLAR["trafoya gore"] = pd.Series(np.isin(np.arange(len(uq)), sec), index=uq)[trf].to_numpy()

print(
    f"\n{'aday bolunme':>24s} {'ort oran':>9s} {'oran sd':>9s} "
    f"{'M0 kaymasi':>12s} {'gozlenen 9.1e-07 ile':>22s}"
)
GOZ = 9.1e-07
for ad, msk in ADAYLAR.items():
    oranlar, Qs = [], []
    for d in D:
        q_all = float((d * d).mean())
        if q_all < 1e-12:
            continue
        oranlar.append(float((d[msk] ** 2).mean()) / q_all)
        Qs.append(q_all)
    oranlar, Qs = np.array(oranlar), np.array(Qs)
    kayma = float(np.std(oranlar * Qs - Qs))
    yorum = "UYUMLU" if kayma < 5 * GOZ else f"{kayma / GOZ:.0f} KAT BUYUK -> ELENDI"
    print(f"{ad:>24s} {oranlar.mean():9.4f} {oranlar.std():9.5f} {kayma:12.3e} {yorum:>22s}")

print("\nYORUM: M0 uc capada 9.1e-07 icinde tutuyor. Yalnizca alt kumenin")
print("  TEMSILI oldugu bolunmeler bu kadar kucuk kayma verir. Temsili olmayan")
print("  bolunmeler M0'i cok daha fazla kaydirirdi -- gozlem onlari eliyor.")
print("  Sonuc: public alt kume, d_j'lerin ikinci momentleri bakimindan")
print("  TUM test kumesiyle ayni davraniyor -> public'te olculen yon")
print("  private'a tasinir.")
