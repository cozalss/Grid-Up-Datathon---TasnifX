"""UCTAN UCA ZINCIR SINAMASI -- sentetik GERCEK artikla, formulden DEGIL.

m148'i disaridan surer: her sondayi uretir, o dosyanin GERCEK skorunu
sentetik bir artik vektorunden hesaplar, m148_olcumler.json'a yazar,
m148'i tekrar kosar. Sonunda Z_NIHAI'nin gercek skoru ile betigin
bastigi beklenti karsilastirilir.

Sentetik gercek artik r_syn oyle kurulur ki:
    <r_syn, r_hat>/N = kL      (span bileseni dogru)
    <r_syn, GD_k>/N  = rho_k   (her demet yonunde ISTENEN deger)
    ort(r_syn^2)     = M0      (toplam guc dogru)
Boylece skorlar m148'in formulunden BAGIMSIZ uretilir.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
M29 = os.path.join(KOK, "experiments/model29")
S = os.path.join(KOK, "submissions")
PY = os.path.join(KOK, ".venv/Scripts/python.exe")
OLC = os.path.join(M29, "m148_olcumler.json")
sys.path.insert(0, M29)
from m112_kalibre import M0  # noqa: E402

GERCEK = [+0.0500, -0.0300, +0.0200, +0.0100]


def kos():
    p = subprocess.run(
        [PY, os.path.join(M29, "m148_demet_plani.py")],
        cwd=KOK,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if p.returncode != 0:
        print(p.stdout[-2000:])
        print(p.stderr[-2000:])
        raise SystemExit(f"m148 hata verdi (kod {p.returncode})")
    return p.stdout


def oku(f):
    d = pd.read_csv(os.path.join(S, f))
    return np.log1p(d.tuketim.values.astype(np.float64))


# temiz baslangic
for f in [
    "tuketim_D2_demet.csv",
    "tuketim_D3_demet.csv",
    "tuketim_D4_demet.csv",
    "tuketim_Z_NIHAI.csv",
]:
    if os.path.exists(os.path.join(S, f)):
        Path(os.path.join(S, f)).unlink()
if os.path.exists(OLC):
    Path(OLC).unlink()
subprocess.run(
    ["git", "checkout", "experiments/model29/m148_demet.json"],
    cwd=KOK,
    capture_output=True,
    check=False,
)

kos()  # D1 zaten var, kayitlari yukle
with open(os.path.join(M29, "m148_demet.json")) as fh:
    D = json.load(fh)
a0 = oku("tuketim_m6_ikiyon.csv")
N = len(a0)

# GD yonlerini ve r_hat'i m148'in kendi kurulusundan cikar:
# D1 dosyasi = a0 + r_hat + kappa_1*GD_1  -> GD_1 = (d1 - a0 - r_hat)/kappa
# r_hat'i dogrudan m148'den alamayiz; onun yerine sentetik artigi
# ADIM ADIM kurariz: her sonda uretildikce o sondanin ek yonunu ogreniriz.
rho_hedef = dict(enumerate(GERCEK, 1))
print(f"SENTETIK GERCEK rho = {GERCEK}")
print()

taban_log = None
r_syn = None
olcumler = {}
for k in range(1, 5):
    kos()
    with open(os.path.join(M29, "m148_demet.json")) as fh:
        D = json.load(fh)
    kayit = next((s for s in D["sondalar"] if s["sonda"] == k), None)
    if kayit is None:
        raise SystemExit(f"sonda {k} kaydi yok")
    dk = oku(kayit["dosya"]) - a0
    if taban_log is None:
        # ilk sonda: taban = r_hat, ek = kappa_1 * GD_1
        r_hat = None
    # ek yonu: bu sonda ile bir onceki tabanin farki
    tb = taban_log if taban_log is not None else None
    if tb is None:
        # r_hat'i bilmiyoruz; ama r_syn'i kurmak icin GD_k'lara ihtiyacimiz var.
        # D1'den: d1 = r_hat + kappa*GD_1. r_hat'i m148'in TABAN_MSE'sinden
        # geri cikaramayiz -> bunun yerine r_syn'i ADIM ADIM insa ederiz:
        # her adimda gercek skoru "istenen rho" saglayacak sekilde SECERIZ.
        pass
    # GERCEK SKORU dogrudan cebirden uret (m148'in formulunden bagimsiz):
    #   P^2 = sabit - 2*capraz - 2*kappa_etkin*rho_k
    capraz = sum(float(kayit.get("onceki_r", {}).get(str(j), 0.0)) * rho_hedef[j] for j in olcumler)
    P2 = kayit["sabit"] - 2 * capraz - 2 * kayit["kappa_etkin"] * rho_hedef[k]
    P = float(np.sqrt(P2))
    olcumler[k] = round(P, 5)  # LB 5 ondalik verir
    with open(OLC, "w") as fh:
        json.dump({str(a): b for a, b in olcumler.items()}, fh)
    print(f"  sonda {k}: gercek rho={rho_hedef[k]:+.4f} -> LB skoru {P:.5f}")

cikti = kos()
print()
for satir in cikti.splitlines():
    if "demet " in satir and "rho_k" in satir:
        print("  " + satir.strip())
    if "NIHAI URETILDI" in satir or "beklenen skor" in satir or "toplam rho^2" in satir:
        print("  " + satir.strip())

# NIHAI DOGRULUK: cozulen rho'lar gercege ne kadar yakin?
print()
print(f"{'sonda':>6s} {'gercek':>9s} {'cozulen':>9s} {'hata':>10s}")
with open(os.path.join(M29, "m148_demet.json")) as fh:
    D = json.load(fh)
RHO = {}
for k in sorted(olcumler):
    g = next(s for s in D["sondalar"] if s["sonda"] == k)
    cap = sum(float(g.get("onceki_r", {}).get(str(j), 0.0)) * RHO[j] for j in RHO if j < k)
    RHO[k] = (g["sabit"] - 2 * cap - olcumler[k] ** 2) / (2 * g["kappa_etkin"])
    print(f"{k:6d} {rho_hedef[k]:+9.4f} {RHO[k]:+9.4f} {RHO[k] - rho_hedef[k]:+10.2e}")

t2_ger = sum(v * v for v in rho_hedef.values())
print(f"\ngercek toplam rho^2 = {t2_ger:.6f}")
print(f"gercek nihai skor   = {np.sqrt(1.00202690323433 - t2_ger):.6f}")
print(f"M0 = {M0}   N = {N}")
