"""p34-b: capalar + H1/H2'nin DIK bilesenleri (r_hat KULLANILMADAN).

Yontem p33_b ile ayni: her dosya icin
    d = log1p(dosya) - a0,  Q = ort(d^2),  L = (M0 + Q - P^2)/2
    d = V c + d_dik ;  <r, Vc> = c . L_vec   (hepsi OLCULMUS)
    rho_dik = (L - c.L_vec) / ||d_dik||
Buzme/r_hat zincire GIRMEZ.

Iki span'a gore ayri ayri:
  S28 = p33'un yon kumesi (H1/H2 HARIC)  -> H1/H2'nin GETIRDIGI yeni bilgi
  S30 = H1/H2 dahil                      -> D1/D2/Y1'in KALAN dik bilgisi
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

with open(os.path.join(GEC, "p34_a_cebir.json"), encoding="utf-8") as fh:
    A = json.load(fh)
a0 = np.load(os.path.join(GEC, "p34_a0.npy"))
V30 = np.load(os.path.join(GEC, "p34_V30.npy"))
L30 = np.load(os.path.join(GEC, "p34_L30.npy"))
N = A["N"]
V28, L28 = V30[:, :28], L30[:28]

te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
IDS = te.id.values


def oku(f, kok=S):
    d = pd.read_csv(os.path.join(kok, f))
    k = "tuketim" if "tuketim" in d.columns else d.columns[-1]
    if not np.array_equal(d.id.values, IDS):
        pos = pd.Index(d.id).get_indexer(IDS)
        assert (pos >= 0).all(), f
        d = d.iloc[pos].reset_index(drop=True)
    return np.log1p(d[k].values.astype(np.float64))


SIG_P = 5e-6 / np.sqrt(3.0)  # LB 5 ondalik -> duzgun yuvarlama sd


def coz(d, P, V, L_vec, etiket):
    G = (V.T @ V) / N
    Gi = np.linalg.pinv(G, rcond=1e-6)
    Q = float((d * d).mean())
    L = (M0 + Q - P * P) / 2
    c = Gi @ ((V.T @ d) / N)
    d_dik = d - V @ c
    kap = float(np.sqrt(float((d_dik * d_dik).mean())))
    L_par = float(c @ L_vec)
    rho = (L - L_par) / kap
    sig_L = P * SIG_P
    sig_rho = float(np.sqrt(sig_L**2 * (1.0 + float(c @ c)))) / kap
    return {
        "etiket": etiket, "LB": P, "Q": Q, "L": L,
        "kappa_dik_normu": kap, "dik_pay": float(np.sqrt(kap * kap / Q)),
        "L_par_olculmus": L_par, "rho_dik": rho, "sigma_rho": sig_rho,
        "kazanc_rho2": rho * rho,
    }


SONDALAR = [
    ("D1", "tuketim_D1_demet.csv", 1.00177, 0.1301804507000752),
    ("D2", "tuketim_D2_demet.csv", 1.00159, 0.11623402619791726),
    ("Y1", "tuketim_Y1_demet.csv", 1.00297, 0.17540614806437363),
]
YENI = [
    ("H1", "tuketim_H1_span0_kuyruk8.csv", 1.00299, None),
    ("H2", "tuketim_H2_harman311.csv", 1.00715, None),
]

print("=== S28 SPANINA GORE (H1/H2 span'a GIRMEDEN) ===")
R28 = []
for et, f, P, tah in SONDALAR + YENI:
    if not os.path.exists(os.path.join(S, f)):
        print(f"  {et}: dosya yok")
        continue
    r = coz(oku(f) - a0, P, V28, L28, et)
    r["ongorulen_rho_k"] = tah
    if tah:
        r["gerceklesme_orani"] = r["rho_dik"] / tah
    R28.append(r)
    ek = f"  oran {r['rho_dik'] / tah:+.3f}" if tah else ""
    print(f"  {et:3s} LB={P}  ||d_dik||={r['kappa_dik_normu']:.5f}  dik_pay="
          f"{r['dik_pay']:.3f}  rho_dik={r['rho_dik']:+.6f} (+-{r['sigma_rho']:.6f})"
          f"  kazanc rho^2={r['rho_dik']**2:.6f}{ek}")

print("\n=== S30 SPANINA GORE (H1/H2 span'da) ===")
R30 = []
for et, f, P, tah in SONDALAR:
    r = coz(oku(f) - a0, P, V30, L30, et)
    r["ongorulen_rho_k"] = tah
    r["gerceklesme_orani"] = r["rho_dik"] / tah
    R30.append(r)
    print(f"  {et:3s} ||d_dik||={r['kappa_dik_normu']:.5f}  dik_pay={r['dik_pay']:.3f}"
          f"  rho_dik={r['rho_dik']:+.6f} (+-{r['sigma_rho']:.6f})"
          f"  oran {r['gerceklesme_orani']:+.3f}")

# --- D1/D2/Y1'in S30'a ve BIRBIRINE dik ORTONORMAL kumesi -----------------
# Ardisik Gram-Schmidt; her adimda rho o adimin BIRIM yonu icin yeniden cozulur.
G30 = (V30.T @ V30) / N
Gi30 = np.linalg.pinv(G30, rcond=1e-6)
baz, rho_baz, ad_baz = [], [], []
for et, f, P, tah in SONDALAR:
    d = oku(f) - a0
    c = Gi30 @ ((V30.T @ d) / N)
    L = (M0 + float((d * d).mean()) - P * P) / 2
    L_kalan = L - float(c @ L30)
    x = d - V30 @ c
    for u_o, r_o in zip(baz, rho_baz):
        pr = float((x * u_o).mean())
        x = x - pr * u_o
        L_kalan -= pr * r_o
    nx = float(np.sqrt(float((x * x).mean())))
    if nx < 1e-6:
        print(f"  {et}: dik bilesen tukendi, atlandi")
        continue
    u = x / nx
    rho_u = L_kalan / nx
    baz.append(u)
    rho_baz.append(rho_u)
    ad_baz.append(et)

TOP = float(sum(r * r for r in rho_baz))
print("\n=== ORTONORMAL OLCULMUS DIK BAZ (S30'a dik) ===")
for et, r in zip(ad_baz, rho_baz):
    print(f"  {et}: rho = {r:+.6f}   kazanc {r * r:.7f}")
print(f"  TOPLAM ek kazanc (MSE) = {TOP:.7f}")

np.save(os.path.join(GEC, "p34_dik_baz.npy"), np.array(baz))
with open(os.path.join(GEC, "p34_b_capa.json"), "w", encoding="utf-8") as fh:
    json.dump({"S28": R28, "S30": R30,
               "ortonormal_baz": {"adlar": ad_baz, "rho": rho_baz,
                                  "toplam_kazanc": TOP}},
              fh, ensure_ascii=False, indent=1)
print("\n-> p34_b_capa.json")
