"""TUM ADAYLARI olculmus span'a DIK bilesenlerine gore tara.

Mantik: bir yon zaten olculmus 24 yonun span'indaysa L'si BILINIYOR -> yeni bilgi YOK,
sonda hakki bosa gider. Yeni bilgi yalnizca span'a DIK bilesende.
Ortogonallestirilmis bir yonun kazanci rho_dik^2 -- Q'dan bagimsiz.
Ama OLCULEBILMESI icin Q_dik yeterince buyuk olmali (gurultu 1.6e-4).

Cikti: her aday icin Q_dik, dik oran, olcum SNR'i.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
S = os.path.join(KOK, "submissions")
SK = json.load(open(os.path.join(KOK, "experiments/model29/olculmus_skorlar.json")))
TABAN = "tuketim_m6_ikiyon.csv"
GUR = 1.6e-4  # sonda ile L olcum hatasi


def oku(f):
    df = pd.read_csv(os.path.join(S, f))
    kol = "tuketim" if "tuketim" in df.columns else df.columns[-1]
    return np.log1p(df[kol].values.astype(np.float64))


a0 = oku(TABAN)
N = len(a0)

# --- olculmus span (24 yon) + g7 ---
span_ad = []
V = []
for f in list(SK) + ["tuketim_g7_span_tau3.csv"]:
    if f == TABAN or not os.path.exists(os.path.join(S, f)):
        continue
    v = oku(f)
    if len(v) != N:
        continue
    span_ad.append(f)
    V.append(v - a0)
V = np.array(V).T  # N x K
print(f"span yonu: {len(span_ad)}")

# ortonormal taban (rank tespitli)
U, sv, _ = np.linalg.svd(V, full_matrices=False)
rank = int((sv > sv[0] * 1e-7).sum())
B = np.ascontiguousarray(U[:, :rank])  # N x rank, ortonormal
print(f"span rank = {rank} (tekil degerler {sv[0]:.3e} .. {sv[rank - 1]:.3e})")
del U, V


def dik(d):
    return d - B @ (B.T @ d)


aday = sorted(p.name for p in Path(S).glob("tuketim_*.csv"))
atla = set(span_ad) | {TABAN} | {f for f in aday if f.startswith("tuketim_s2")}
sonuc = []
for i, f in enumerate(aday):
    if f in atla:
        continue
    try:
        v = oku(f)
    except Exception:
        continue
    if len(v) != N:
        continue
    d = v - a0
    Q = float((d * d).mean())
    if Q < 1e-8:
        continue
    dp = dik(d)
    Qd = float((dp * dp).mean())
    sonuc.append(
        dict(
            dosya=f,
            Q=Q,
            Q_dik=Qd,
            oran=Qd / Q,
            snr=0.025 * np.sqrt(Qd) / GUR,
            kurtoz=float(pd.Series(dp).kurt()),
        )
    )
    if i % 25 == 0:
        print(f"  ... {i}/{len(aday)}", flush=True)

df = pd.DataFrame(sonuc).sort_values("Q_dik", ascending=False)
df.to_csv("z4_dik_tarama.csv", index=False)
print(f"\ntoplam {len(df)} aday tarandi\n")
print("EN BUYUK DIK BILESENLI 30 ADAY")
print(
    df.head(30).to_string(
        index=False, float_format=lambda x: f"{x:.5f}" if abs(x) < 1000 else f"{x:.0f}"
    )
)
print("\nMEVCUT PLANDAKILER")
plan = [
    "tuketim_y40_sota_temiz.csv",
    "tuketim_z2_analog.csv",
    "tuketim_t1_sulama.csv",
    "tuketim_y46_amnezik_kirpik.csv",
    "tuketim_y45_mevsimsel_kirpik.csv",
    "tuketim_q1c_kapasite_siki.csv",
    "tuketim_t3_turizm.csv",
    "tuketim_h1_isil.csv",
    "tuketim_t2_bayram.csv",
    "tuketim_k5_kesinti.csv",
]
d2 = df[df.dosya.isin(plan)].copy()
d2["sira"] = [int(np.where(df.dosya.values == f)[0][0]) + 1 for f in d2.dosya]
print(d2.to_string(index=False, float_format=lambda x: f"{x:.5f}" if abs(x) < 1000 else f"{x:.0f}"))
