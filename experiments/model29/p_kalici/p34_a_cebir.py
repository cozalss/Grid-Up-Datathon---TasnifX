"""p34-a: SPAN cebrini 31 olcumle yeniden coz (H1 ve H2 dahil).

p33_a_cebir.py'nin birebir kopyasi + iki YENI LB olcumu:
  H1 = tuketim_H1_span0_kuyruk8.csv  1.00299  (span k=0 + kuyruk kapagi k=8)
  H2 = tuketim_H2_harman311.csv      1.00715  (soguk harman 3/1/1 + olu x0.5)

Once 28 yonle p33'u YENIDEN URETIR (TABAN_MSE = 1.00202690323433 beklenir),
sonra 30 yonle yeniden cozer. Cikti: p34_*.npy + p34_a_cebir.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
M29 = os.path.join(KOK, "experiments/model29")
S = os.path.join(KOK, "submissions")
GEC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, M29)
from m112_kalibre import EK_MODEL, M0, buzmeli_r_hat  # noqa: E402

TABAN = "tuketim_m6_ikiyon.csv"
YENI_OLCUMLER = {
    "tuketim_H1_span0_kuyruk8.csv": 1.00299,
    "tuketim_H2_harman311.csv": 1.00715,
}

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
with open(os.path.join(M29, "olculmus_skorlar.json"), encoding="utf-8") as fh:
    SK = json.load(fh)
with open(os.path.join(M29, "m112_durum.json"), encoding="utf-8") as fh:
    DUR = json.load(fh)

V, L, AD, TUR = [], [], [], []
for f, Pj in SK.items():
    if f == TABAN or not os.path.exists(os.path.join(S, f)):
        continue
    v = oku(f)
    if v is None or len(v) != N:
        continue
    dd = v - a0
    V.append(dd)
    L.append((M0 + float((dd * dd).mean()) - Pj * Pj) / 2)
    AD.append(f)
    TUR.append("LB_olculmus")
for f, Lj in EK_MODEL.items():
    V.append(oku(f) - a0)
    L.append(Lj)
    AD.append(f)
    TUR.append("TURETILMIS_L")
for o in DUR.get("olcumler", []):
    dd = oku(o["dosya"]) - a0
    V.append(dd)
    L.append((M0 + float((dd * dd).mean()) - o["skor"] ** 2) / 2)
    AD.append(o["dosya"])
    TUR.append("LB_olculmus")

V28 = np.array(V).T
L28 = np.array(L)
AD28 = list(AD)
assert V28.shape[1] == 28, f"28 yon bekleniyordu, {V28.shape[1]} geldi"

G28 = (V28.T @ V28) / N
r28, kaz28, kL28 = buzmeli_r_hat(V28, L28, G28, N)
nrm28 = float((r28 * r28).mean())
TM28 = float(M0 - 2 * kL28 + nrm28)
print("=== 28 YON (p33 yeniden uretimi) ===")
print(f"  kL          = {kL28:.9f}")
print(f"  ||r_hat||^2 = {nrm28:.9f}")
print(f"  kazanc      = {kaz28:.9f}")
print(f"  TABAN_MSE   = {TM28:.9f}   (p33: 1.002026903)")
print(f"  saf span    = {np.sqrt(TM28):.7f}  durust {np.sqrt(M0 - kaz28):.7f}")
UYUM = abs(TM28 - 1.00202690323433) < 1e-9
print(f"  p33 ile UYUM: {UYUM}")

# --- H1/H2'nin ESKI cozume gore ONGORUSU vs OLCUM -------------------------
ONG = {}
for f, Pj in YENI_OLCUMLER.items():
    v = oku(f)
    assert v is not None and len(v) == N, f
    dd = v - a0
    Q = float((dd * dd).mean())
    L_olculmus = (M0 + Q - Pj * Pj) / 2
    # eski cozumun ongordugu L: <r_hat, dd>/N (r ~ r_hat varsayimi)
    L_ongoru = float((r28 * dd).mean())
    P_ongoru = float(np.sqrt(max(M0 - 2 * L_ongoru + Q, 1e-12)))
    ONG[f] = {
        "LB": Pj,
        "Q": Q,
        "L_olculmus": L_olculmus,
        "L_ongoru_28yon": L_ongoru,
        "P_ongoru_28yon": P_ongoru,
        "P_hata": Pj - P_ongoru,
        "L_hata": L_olculmus - L_ongoru,
    }
    print(f"\n  {f}: LB={Pj}  Q={Q:.6f}")
    print(f"    L olculmus {L_olculmus:+.6f}  vs 28-yon ongoru {L_ongoru:+.6f}"
          f"  (hata {L_olculmus - L_ongoru:+.6f})")
    print(f"    P ongoru {P_ongoru:.5f} -> gercek {Pj}  (hata {Pj - P_ongoru:+.5f})")

# --- 31 olcum (30 yon) ile yeniden coz ------------------------------------
Vl = [V28[:, j] for j in range(V28.shape[1])]
Ll = list(L28)
ADl = list(AD28)
for f, Pj in YENI_OLCUMLER.items():
    dd = oku(f) - a0
    Vl.append(dd)
    Ll.append((M0 + float((dd * dd).mean()) - Pj * Pj) / 2)
    ADl.append(f)

V30 = np.array(Vl).T
L30 = np.array(Ll)
G30 = (V30.T @ V30) / N
r30, kaz30, kL30 = buzmeli_r_hat(V30, L30, G30, N)
nrm30 = float((r30 * r30).mean())
TM30 = float(M0 - 2 * kL30 + nrm30)
print("\n=== 30 YON (H1 + H2 eklendi) ===")
print(f"  kL          = {kL30:.9f}   (28: {kL28:.9f})")
print(f"  ||r_hat||^2 = {nrm30:.9f}   (28: {nrm28:.9f})")
print(f"  kazanc      = {kaz30:.9f}   (28: {kaz28:.9f})")
print(f"  TABAN_MSE   = {TM30:.9f}   (28: {TM28:.9f})")
print(f"  saf span cebirsel = {np.sqrt(TM30):.7f}   (28: {np.sqrt(TM28):.7f})")
print(f"  saf span durust   = {np.sqrt(M0 - kaz30):.7f}   (28: {np.sqrt(M0 - kaz28):.7f})")
print(f"  r_hat degisimi: ||r30-r28||/||r28|| = "
      f"{np.sqrt(float(((r30 - r28) ** 2).mean()) / nrm28):.4f}")

# 30-yon cozumu H1/H2'yi ne kadar aciklıyor
print("\n  30-yon cozumun kendi olcumlerine uyumu:")
GERI = {}
for j, f in enumerate(ADl):
    if f not in YENI_OLCUMLER and f not in ("tuketim_YP_seviye.csv",):
        continue
    dd = V30[:, j]
    Q = float((dd * dd).mean())
    Lf = float((r30 * dd).mean())
    Pf = float(np.sqrt(max(M0 - 2 * Lf + Q, 1e-12)))
    hedef = YENI_OLCUMLER.get(f, 1.00115)
    GERI[f] = {"P_model_30yon": Pf, "P_LB": hedef, "artik": hedef - Pf}
    print(f"    {f:34s} model {Pf:.5f}  LB {hedef:.5f}  artik {hedef - Pf:+.5f}")

np.save(os.path.join(GEC, "p34_a0.npy"), a0)
np.save(os.path.join(GEC, "p34_V30.npy"), V30)
np.save(os.path.join(GEC, "p34_r30.npy"), r30)
np.save(os.path.join(GEC, "p34_r28.npy"), r28)
np.save(os.path.join(GEC, "p34_L30.npy"), L30)

CIK = {
    "00_KURAL": "Kaggle gonderimi YOK, submissions/ yazilmadi, commit yok",
    "N": N,
    "M0": M0,
    "adlar_30": ADl,
    "c28_p33_yeniden_uretim": {
        "kL": float(kL28), "r_hat_norm2": nrm28, "beklenen_kazanc": float(kaz28),
        "TABAN_MSE": TM28, "saf_span_cebirsel": float(np.sqrt(TM28)),
        "saf_span_durust": float(np.sqrt(M0 - kaz28)),
        "p33_ile_uyum": bool(UYUM),
    },
    "c30_yeni": {
        "kL": float(kL30), "r_hat_norm2": nrm30, "beklenen_kazanc": float(kaz30),
        "TABAN_MSE": TM30, "saf_span_cebirsel": float(np.sqrt(TM30)),
        "saf_span_durust": float(np.sqrt(M0 - kaz30)),
        "r_hat_goreli_degisim": float(
            np.sqrt(float(((r30 - r28) ** 2).mean()) / nrm28)),
    },
    "H_ongoru_vs_olcum": ONG,
    "geri_uyum": GERI,
}
with open(os.path.join(GEC, "p34_a_cebir.json"), "w", encoding="utf-8") as fh:
    json.dump(CIK, fh, ensure_ascii=False, indent=1)
print("\n-> p34_a_cebir.json")
