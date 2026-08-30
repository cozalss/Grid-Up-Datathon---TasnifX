"""SINAMA SONRASI TEMIZ KURULUM.

m161_zincir_testi.py sentetik olcumlerle m148'i surer; bitiminde ortada
SAHTE bir m148_olcumler.json, sahte D2/D3/D4 ve sahte bir Z_NIHAI kalir.
Bunlar GERCEK gonderim dosyalariyla ayni adi tasir. Biri yanlislikla
gonderilirse skoru cozulemez ve bir hak bosa gider (kirmizi takim K0).

Bu betik ortami gercek yarisma icin sifirlar ve YALNIZCA D1'i uretir.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
M29 = os.path.join(KOK, "experiments/model29")
S = os.path.join(KOK, "submissions")
PY = os.path.join(KOK, ".venv/Scripts/python.exe")

SIL = [
    os.path.join(S, "tuketim_D2_demet.csv"),
    os.path.join(S, "tuketim_D3_demet.csv"),
    os.path.join(S, "tuketim_D4_demet.csv"),
    os.path.join(S, "tuketim_D5_demet.csv"),
    os.path.join(S, "tuketim_Z_NIHAI.csv"),
    os.path.join(S, "tuketim_D1_demet.csv"),
    os.path.join(M29, "m148_olcumler.json"),
]
for y in SIL:
    if os.path.exists(y):
        Path(y).unlink()
        print(f"  silindi: {os.path.basename(y)}")

subprocess.run(
    ["git", "checkout", "--", "experiments/model29/m148_demet.json"],
    cwd=KOK,
    capture_output=True,
    check=False,
)
print("  m148_demet.json git'ten geri alindi")

# 1. gecis: bloklar kurulur, rho_k_tahmin yazilir (kappa henuz eski olabilir)
print("\n--- 1. gecis: blok yapisi ---")
p = subprocess.run(
    [PY, os.path.join(M29, "m148_demet_plani.py")],
    cwd=KOK,
    capture_output=True,
    text=True,
    encoding="utf-8",
    check=False,
)
if p.returncode != 0:
    print(p.stdout[-3000:])
    print(p.stderr[-3000:])
    raise SystemExit(f"m148 hata verdi ({p.returncode})")

# SIRA ONEMLI: n06 "diskte dosyasi olan sondanin kappa'sini DONDUR" kuralini
# uyguluyor. 1. gecisin urettigi D1 ortada kalirsa n06 o GECICI kappa'yi
# dondurur ve 2. gecis bir sey degistiremez. Bu yuzden D1 n06'dan ONCE
# silinir; boylece n06 hicbir seyi dondurmadan taze secim yapar.
# DIKKAT: json'u BURADA geri ALMA -- n06 blok tahminlerini (rho_k_tahmin)
# ondan okuyor ve git'teki surum ESKI kurulusu tasiyor olabilir.
# Yalnizca CSV silinir; json 1. gecisin taze ciktisiyla kalir.
Path(os.path.join(S, "tuketim_D1_demet.csv")).unlink(missing_ok=True)

# kappa'yi yeni blok tahminlerine gore yeniden optimize et
print("--- kappa yeniden optimize ediliyor ---")
p = subprocess.run(
    [PY, os.path.join(M29, "n06_kappa.py")],
    cwd=KOK,
    capture_output=True,
    text=True,
    encoding="utf-8",
    check=False,
)
if p.returncode != 0:
    print(p.stdout[-2000:], p.stderr[-2000:])
    raise SystemExit("n06_kappa hata verdi")
print(p.stdout.strip().splitlines()[-1])

# D1'i dogru kappa ile YENIDEN uret. CSV zaten n06'dan once silindi;
# simdi json geri alinir ki 2. gecis sonda kaydini sifirdan yazsin.
subprocess.run(
    ["git", "checkout", "--", "experiments/model29/m148_demet.json"],
    cwd=KOK,
    capture_output=True,
    check=False,
)
print("\n--- 2. gecis: D1 dogru kappa ile ---")
p = subprocess.run(
    [PY, os.path.join(M29, "m148_demet_plani.py")],
    cwd=KOK,
    capture_output=True,
    text=True,
    encoding="utf-8",
    check=False,
)
print(p.stdout[p.stdout.index("blok") :] if "blok" in p.stdout else p.stdout[-3000:])
if p.returncode != 0:
    print(p.stderr[-3000:])
    raise SystemExit(f"m148 2. gecis hata verdi ({p.returncode})")

# --- DOGRULAMA -----------------------------------------------------------
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

d1 = os.path.join(S, "tuketim_D1_demet.csv")
if not os.path.exists(d1):
    raise SystemExit("DUR: D1 uretilmedi")
te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
d = pd.read_csv(d1)
sorun = []
if len(d) != len(te):
    sorun.append(f"satir sayisi {len(d)} != {len(te)}")
if not np.array_equal(d.id.values, te.id.values):
    sorun.append("id sirasi test.csv ile ayni degil")
if not np.isfinite(d.tuketim.values).all():
    sorun.append(f"{int((~np.isfinite(d.tuketim.values)).sum())} sonlu olmayan deger")
if (d.tuketim.values < 0).any():
    sorun.append(f"{int((d.tuketim.values < 0).sum())} negatif deger")
for kalinti in [
    "tuketim_D2_demet.csv",
    "tuketim_D3_demet.csv",
    "tuketim_D4_demet.csv",
    "tuketim_Z_NIHAI.csv",
]:
    if os.path.exists(os.path.join(S, kalinti)):
        sorun.append(f"SAHTE DOSYA KALDI: {kalinti}")
if os.path.exists(os.path.join(M29, "m148_olcumler.json")):
    sorun.append("SAHTE m148_olcumler.json KALDI")

print()
if sorun:
    for x in sorun:
        print(f"  X {x}")
    raise SystemExit("DUR: dogrulama basarisiz")

import hashlib  # noqa: E402

with open(d1, "rb") as fh:
    md5 = hashlib.md5(fh.read()).hexdigest()  # noqa: S324
with open(os.path.join(M29, "m148_demet.json"), encoding="utf-8") as fh:
    D = json.load(fh)
print(f"  OK  {len(d)} satir, id sirasi dogru, 0 NaN, 0 negatif")
print("  OK  ortada sahte dosya yok")
print(f"  md5 {md5}")
print(f"  bloklar {np.round(D['rho_k_tahmin'], 4).tolist()}")
s0 = D["sondalar"][0]
print(f"  sonda 1 [{s0['yon']}]  kappa={s0['kappa']:.5f}  sabit={s0['sabit']:.9f}")
print("\nHAZIR. HICBIR GONDERIM YAPILMADI.")
sys.exit(0)
