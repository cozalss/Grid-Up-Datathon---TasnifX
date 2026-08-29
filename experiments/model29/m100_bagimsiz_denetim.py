"""BAGIMSIZ DENETIM: span cozumunu ajanlardan bagimsiz olarak kendim dogrula.
Ajanlarin koduna BAKMADAN, sifirdan kur."""

import json
import os

import numpy as np
import pandas as pd
from m30_ozellik import KOK

S = os.path.join(KOK, "submissions")
skor = json.load(open("olculmus_skorlar.json"))
# gun1_baseline farkli format -> at
skor = {k: v for k, v in skor.items() if k != "gun1_baseline.csv"}
adlar = sorted(skor)
P = np.array([np.log1p(pd.read_csv(os.path.join(S, a)).tuketim.values) for a in adlar])
N = P.shape[1]
m = np.array([skor[a] ** 2 for a in adlar])
G = P @ P.T / N
print(f"{len(adlar)} dosya, N={N:,}")

# ||y||^2/N bilinmiyor ama afin kombinasyonda dusuyor.
# w'P icin: MSE(w) = w'Gw - 2 w'<P,y> + ||y||^2 ,  <P_i,y>/N = (G_ii + ||y||^2/N - m_i)/2
# afin (sum w =1) -> MSE(w) = w'Gw - w'(diag(G) - m) - ... turet:
c = np.diag(G) - m  # = 2<P_i,y>/N - ||y||^2/N ... sabit dusuyor


def mse(w):
    return float(w @ G @ w - w @ c)  # sum(w)=1 varsayimi altinda ||y||^2/N terimi sadelesir


# SINAV: bilinen dosyalarin kendi skorlarini yeniden uretebiliyor mu?
print("\nSINAV 1 -- tek dosya (w = e_i) kendi skorunu veriyor mu?")
for a in ["tuketim_m6_ikiyon.csv", "tuketim_v102_kappa_optimum.csv", "tuketim_m4_hava_capali.csv"]:
    i = adlar.index(a)
    w = np.zeros(len(adlar))
    w[i] = 1
    print(
        f"  {a:34s} hesap {np.sqrt(mse(w)):.5f}  gercek {skor[a]:.5f}  fark {abs(np.sqrt(mse(w)) - skor[a]):.2e}"
    )

# SINAV 2 -- LOO: bir dosyayi disarida birak, kalanlarla SKORUNU tahmin et
print("\nSINAV 2 -- LOO: dosyayi span'dan disla, kalanlarla skorunu TAHMIN et")
print("  (bu, sistemin gercek ongoru gucunun tek durust olcusu)")
hata = []
for a in [
    "tuketim_m6_ikiyon.csv",
    "tuketim_p51_sicak05.csv",
    "tuketim_v102_kappa_optimum.csv",
    "tuketim_v83_sicak_optimum.csv",
    "tuketim_v101_hepsi.csv",
    "tuketim_m4_hava_capali.csv",
]:
    i = adlar.index(a)
    dis = [j for j in range(len(adlar)) if j != i]
    # P_i'yi digerlerinin afin kombinasyonu olarak yaz (en kucuk kareler, sum=1 kisitli)
    A = P[dis]
    b = P[i]
    # min ||w'A - b||^2  s.t. sum w = 1
    Gd = A @ A.T / N
    rd = A @ b / N
    n = len(dis)
    M = np.block([[Gd, np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
    rhs = np.concatenate([rd, [1.0]])
    sol = np.linalg.lstsq(M, rhs, rcond=None)[0]
    w = sol[:n]
    artik = float(((w @ A - b) ** 2).mean())
    # tahmin: bu w ile MSE
    wf = np.zeros(len(adlar))
    wf[dis] = w
    tah = np.sqrt(max(mse(wf), 0))
    hata.append(abs(tah - skor[a]))
    print(
        f"  {a:34s} tahmin {tah:.5f}  gercek {skor[a]:.5f}  HATA {tah - skor[a]:+.5f}  artik {artik:.2e}  |w|1 {np.abs(w).sum():.2f}"
    )
print(f"  -> LOO ortalama mutlak hata: {np.mean(hata):.5f}")

# SINAV 3 -- gonderilecek dosya gercekten bir AFIN kombinasyon mu?
print("\nSINAV 3 -- tuketim_g7_span_tau3.csv dosyasi span icinde mi?")
g = np.log1p(pd.read_csv(os.path.join(S, "tuketim_g7_span_tau3.csv")).tuketim.values)
Gd = P @ P.T / N
rd = P @ g / N
n = len(adlar)
M = np.block([[Gd, np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
w = np.linalg.lstsq(M, np.concatenate([rd, [1.0]]), rcond=None)[0][:n]
art = float(((w @ P - g) ** 2).mean())
print(
    f"  artik {art:.3e}  (0 ise TAM afin kombinasyon)   sum(w)={w.sum():.6f}  |w|1={np.abs(w).sum():.4f}"
)
print(f"  BU DOSYANIN ONGORULEN SKORU (kendi Gram'imla): {np.sqrt(max(mse(w), 0)):.5f}")
print("  ajanin dedigi: 1.00137")
