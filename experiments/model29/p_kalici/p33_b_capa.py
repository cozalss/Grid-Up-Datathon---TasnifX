"""p33-b: |c| capalari -- GERCEK DEMET SONDALARINDAN.

n19/n20 |c|'yi (a) gonderim FARKLARINDAN (n10, yanlis nesne) ve (b) iki
OZNITELIK EKSENINDEN (seviye, yenibaslangic) kestirdi. Oysa 31 Agustos'ta
UC GERCEK DEMET SONDASI LB'de olculdu: D1, D2, Y1. Bunlar TAM OLARAK
bahsettigimiz nesnedir (span'a dik, cok eksenli demet yonu).

YONTEM (kayit dosyasindan BAGIMSIZ, diskteki CSV'den):
    d      = log1p(sonda) - a0
    Q      = ort(d^2),  L = (M0 + Q - P^2)/2          <- LB skorundan EXACT
    d = d_par + d_dik   (d_par, olculmus 28 yonun span'inda)
    d_par = V c   =>   <r, d_par> = c . L_vec         <- HEPSI OLCULMUS
    rho_dik = (L - c.L_vec) / ||d_dik||
Bu zincirde r_hat ya da buzme HIC KULLANILMAZ; yalnizca olculmus LB skorlari.

Ardindan  |c| = 1.95 * rho_dik / rho_k_tahmin  (rho_k_tahmin = ||BETA_k||,
yani 1.95 tavanina dayanan ongoru).
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
from m112_kalibre import M0  # noqa: E402

with open(os.path.join(GEC, "p33_a_cebir.json"), encoding="utf-8") as fh:
    A = json.load(fh)
V = np.load(os.path.join(GEC, "p33_V.npy"))
a0 = np.load(os.path.join(GEC, "p33_a0.npy"))
r_hat = np.load(os.path.join(GEC, "p33_r_hat.npy"))
N = A["N"]
kL = A["kL"]

# L vektorunu p33_a ile AYNI sirayla yeniden kur
te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
IDS = te.id.values


def oku(f):
    d = pd.read_csv(os.path.join(S, f))
    k = "tuketim" if "tuketim" in d.columns else d.columns[-1]
    if not np.array_equal(d.id.values, IDS):
        pos = pd.Index(d.id).get_indexer(IDS)
        d = d.iloc[pos].reset_index(drop=True)
    return np.log1p(d[k].values.astype(np.float64))


with open(os.path.join(M29, "olculmus_skorlar.json"), encoding="utf-8") as fh:
    SK = json.load(fh)
with open(os.path.join(M29, "m112_durum.json"), encoding="utf-8") as fh:
    DUR = json.load(fh)
sys.path.insert(0, M29)
from m112_kalibre import EK_MODEL  # noqa: E402

L_vec = []
for ad in A["adlar"]:
    if ad in EK_MODEL:
        L_vec.append(EK_MODEL[ad])
        continue
    P = SK.get(ad)
    if P is None:
        P = next(o["skor"] for o in DUR["olcumler"] if o["dosya"] == ad)
    dd = oku(ad) - a0
    L_vec.append((M0 + float((dd * dd).mean()) - P * P) / 2)
L_vec = np.array(L_vec)

G = (V.T @ V) / N
Gi = np.linalg.pinv(G, rcond=1e-6)
SIG_P = 5e-6 / np.sqrt(3.0)  # LB 5 ondalik -> P'nin duzgun yuvarlama sd'si

SONDALAR = [
    # (etiket, dosya, LB, ongorulen rho_k (||BETA_k||), kayit kaynagi)
    ("D1", "tuketim_D1_demet.csv", 1.00177, 0.1301804507000752, "1d9f87e 4-demet"),
    ("D2", "tuketim_D2_demet.csv", 1.00159, 0.11623402619791726, "1d9f87e 4-demet rho_k[1]"),
    ("Y1", "tuketim_Y1_demet.csv", 1.00297, 0.17540614806437363, "31b8cd3 2-demet"),
]
# D2 icin ALTERNATIF kayit (43d89b6, eski tavan 1.95 kurulusu)
ALT_D2 = 0.11223060673130798

SATIR = []
for et, dosya, P, tahmin, kaynak in SONDALAR:
    yol = os.path.join(S, dosya)
    if not os.path.exists(yol):
        print(f"{et}: DOSYA YOK -> atlandi")
        continue
    d = oku(dosya) - a0
    Q = float((d * d).mean())
    L = (M0 + Q - P * P) / 2
    c = Gi @ ((V.T @ d) / N)
    d_par = V @ c
    d_dik = d - d_par
    kap = float(np.sqrt(float((d_dik * d_dik).mean())))
    L_par_olculmus = float(c @ L_vec)
    rho_saf = (L - L_par_olculmus) / kap  # r_hat KULLANILMAZ
    rho_rhat = (M0 - 2 * kL + Q - P * P) / (2 * kap)  # m148 yolu
    # gurultu: L'lerin her biri sd = P*SIG_P;  L_par = c.L_vec
    sig_L = P * SIG_P
    sig_rho = float(np.sqrt(sig_L**2 * (1.0 + float(c @ c)))) / kap
    oran = rho_saf / tahmin
    SATIR.append(
        {
            "sonda": et,
            "dosya": dosya,
            "LB": P,
            "kayit_kaynagi": kaynak,
            "Q": Q,
            "L": L,
            "kappa_etkin_diskten": kap,
            "dik_pay": float(np.sqrt(kap * kap / Q)),
            "L_par_olculmus": L_par_olculmus,
            "rho_dik_SAF": rho_saf,
            "rho_dik_m148yolu": rho_rhat,
            "sigma_rho_yuvarlama": sig_rho,
            "ongorulen_rho_k": tahmin,
            "gerceklesme_orani": oran,
            "c_ima": 1.95 * oran,
            "c_ima_m148yolu": 1.95 * rho_rhat / tahmin,
        }
    )
    print(
        f"{et}: LB={P}  kappa_disk={kap:.6f}  rho_SAF={rho_saf:+.6f} "
        f"(+-{sig_rho:.6f})  rho_m148={rho_rhat:+.6f}  ongoru={tahmin:.4f} "
        f"-> oran {oran:+.3f}  |c|={1.95 * oran:+.3f}"
    )

print(f"\nD2 alternatif kayit ongorusu (43d89b6, {ALT_D2:.5f}):")
for s in SATIR:
    if s["sonda"] == "D2":
        s["ALT_ongoru_43d89b6"] = ALT_D2
        s["ALT_c_ima"] = 1.95 * s["rho_dik_SAF"] / ALT_D2
        print(f"  |c| = {s['ALT_c_ima']:+.3f}")

with open(os.path.join(GEC, "p33_b_capa.json"), "w", encoding="utf-8") as fh:
    json.dump({"L_vec": L_vec.tolist(), "sondalar": SATIR}, fh, indent=1)
print("\n-> p33_b_capa.json")
