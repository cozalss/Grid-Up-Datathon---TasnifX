"""BAGIMSIZ UCTAN UCA ZINCIR SINAMASI. m161'in yerini alir.

m161 NEDEN GECERSIZDI (kirmizi takim n12, bulgu K8): docstring'i
"skorlar m148'in formulunden BAGIMSIZ uretilir" diyor ve bir r_syn
kurulumu anlatiyordu, ama kod bunu YAPMIYORDU -- `r_syn = None` ve ilgili
dal bos `pass` idi. Gercek skoru dogrudan

    P^2 = kayit["sabit"] - 2*capraz - 2*kayit["kappa_etkin"]*rho

ile uretiyordu; yani m148'in TERS CEVIRDIGI formulun ta kendisiyle.
Yanlis bir `sabit`, yanlis bir `kappa_etkin` ya da kirpma yanliligi
ASLA yakalanamazdi -- test her zaman hata ~1e-16 verirdi.

BU BETIK NE YAPAR:
  1. m148'i DOKUM=1 ile kosar; r_hat, GD, a0 ve sabitler diske doker.
  2. SENTETIK GERCEK ARTIK r_syn kurar. Istenen ic carpimlar:
         <r_syn, r_hat>/N = kL
         <r_syn, GD_k>/N  = rho_k        (bizim SECTIGIMIZ gercek degerler)
         <r_syn, r_syn>/N = M0
     Ilk iki kosul {r_hat, GD_1..GD_B} span'inda en kucuk normlu cozumle
     saglanir; ucuncusu span'a DIK rastgele bir bilesenle tamamlanir.
  3. GERCEK LOG HEDEF  t = a0 + r_syn.
  4. Her sonda dosyasi icin skoru DOGRUDAN hesaplar:
         P = sqrt(ort((log1p(CSV) - t)^2))
     Bu, m148'in hicbir formulunu KULLANMAZ. Diskteki dosyayi okur, yani
     CSV yuvarlamasi ve KIRPMA da olcume DAHILDIR.
  5. Skorlari m148'e verir, cozulen rho'lari GERCEK rho'larla karsilastirir.
  6. Nihai dosyanin GERCEK skorunu yine dogrudan hesaplar ve betigin
     bildirdigi beklentiyle karsilastirir.

Gercek dosyalara DOKUNMAZ: olcumler kendi dosyasina yazilir (OLCUM_DOSYA),
uretilen CSV'ler sonunda silinir, m148_demet.json git'ten geri alinir.
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
GEC = os.environ.get(
    "GECICI",
    r"C:/Users/Cem/AppData/Local/Temp/claude/"
    r"c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/"
    r"e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad/m162",
)
os.makedirs(GEC, exist_ok=True)
OLC = os.path.join(GEC, "olcumler_SENTETIK.json")

ORTAM = dict(os.environ)
ORTAM["OLCUM_DOSYA"] = OLC
ORTAM["DOKUM"] = GEC

URETILEN = [f"tuketim_D{i}_demet.csv" for i in range(1, 7)] + ["tuketim_Z_NIHAI.csv"]


def temizle():
    for f in URETILEN:
        y = os.path.join(S, f)
        if os.path.exists(y):
            Path(y).unlink()
    if os.path.exists(OLC):
        Path(OLC).unlink()
    subprocess.run(
        ["git", "checkout", "--", "experiments/model29/m148_demet.json"],
        cwd=KOK,
        capture_output=True,
        check=False,
    )


def kos(ek=None):
    o = dict(ORTAM)
    if ek:
        o.update(ek)
    p = subprocess.run(
        [PY, os.path.join(M29, "m148_demet_plani.py")],
        cwd=KOK,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        env=o,
    )
    if p.returncode != 0:
        print(p.stdout[-2500:])
        print(p.stderr[-2500:])
        raise SystemExit(f"m148 hata verdi (kod {p.returncode})")
    return p.stdout


def oku_log(f):
    d = pd.read_csv(os.path.join(S, f))
    return np.log1p(d.tuketim.values.astype(np.float64))


try:
    temizle()
    print("--- 1. kosu: yapi kuruluyor, dokum aliniyor ---")
    kos()
    r_hat = np.load(os.path.join(GEC, "r_hat.npy"))
    GD = np.load(os.path.join(GEC, "GD.npy"))
    a0 = np.load(os.path.join(GEC, "a0.npy"))
    with open(os.path.join(GEC, "sabitler.json"), encoding="utf-8") as fh:
        SB = json.load(fh)
    kL, M0, N = SB["kL"], SB["M0"], SB["N"]
    B = len(GD)
    print(f"  {B} blok, N={N}, kL={kL:.9f}, M0={M0:.9f}")

    # --- SENTETIK GERCEK ARTIK -------------------------------------------
    rng = np.random.default_rng(2026)
    GERCEK = np.array([0.09, -0.05, 0.04, 0.02, 0.01, -0.008])[:B]
    A = np.vstack([r_hat[None, :], GD])  # (B+1, N)
    hedef = np.concatenate([[kL], GERCEK])  # istenen ic carpimlar
    Gm = (A @ A.T) / N
    c = np.linalg.solve(Gm, hedef)
    r_par = c @ A
    guc_par = float((r_par * r_par).mean())
    if guc_par > M0:
        raise SystemExit(f"DUR: span bileseni M0'i asiyor ({guc_par:.6f} > {M0:.6f})")
    # span'a DIK rastgele bilesen, kalan gucu tasisin
    w = rng.standard_normal(N)
    w -= A.T @ np.linalg.solve(Gm, (A @ w) / N)
    w /= np.sqrt(float((w * w).mean()))
    r_syn = r_par + np.sqrt(M0 - guc_par) * w

    # kurulusu DOGRULA
    kn = float((r_syn * r_hat).mean())
    print(f"\n  r_syn kuruldu: <r_syn,r_hat>/N = {kn:.9f} (istenen {kL:.9f})")
    assert abs(kn - kL) < 1e-9, "r_syn: kL tutmadi"
    for k in range(B):
        v = float((r_syn * GD[k]).mean())
        assert abs(v - GERCEK[k]) < 1e-9, f"r_syn: rho_{k + 1} tutmadi ({v})"
    guc = float((r_syn * r_syn).mean())
    print(f"  ort(r_syn^2) = {guc:.9f} (istenen M0 {M0:.9f})")
    assert abs(guc - M0) < 1e-9, "r_syn: M0 tutmadi"
    print(f"  SENTETIK GERCEK rho = {GERCEK.tolist()}")

    t = a0 + r_syn  # GERCEK log hedef

    # --- SONDA ZINCIRI ---------------------------------------------------
    print(f"\n{'sonda':>6s} {'gercek rho':>11s} {'DOGRUDAN skor':>14s}")
    olcumler = {}
    for k in range(1, B + 1):
        kos()
        dosya = f"tuketim_D{k}_demet.csv"
        if not os.path.exists(os.path.join(S, dosya)):
            raise SystemExit(f"DUR: {dosya} uretilmedi")
        # SKOR: m148'in HICBIR formulu kullanilmadan, diskteki dosyadan
        d = oku_log(dosya) - t
        P = float(np.sqrt(float((d * d).mean())))
        olcumler[k] = round(P, 5)  # LB 5 ondalik verir
        with open(OLC, "w", encoding="utf-8") as fh:
            json.dump({str(a): b for a, b in olcumler.items()}, fh)
        print(f"{k:6d} {GERCEK[k - 1]:+11.4f} {P:14.5f}")

    # --- NIHAI -----------------------------------------------------------
    cikti = kos({"NIHAI": "1"})
    znl = os.path.join(S, "tuketim_Z_NIHAI.csv")
    if not os.path.exists(znl):
        print(cikti[-2000:])
        raise SystemExit("DUR: Z_NIHAI uretilmedi")
    d = oku_log("tuketim_Z_NIHAI.csv") - t
    P_ger = float(np.sqrt(float((d * d).mean())))
    bek = None
    for satir in cikti.splitlines():
        if "beklenen skor" in satir:
            bek = float(satir.split("beklenen skor")[1].split()[0])
        if "demet " in satir and "rho_k" in satir:
            print("  " + satir.strip())

    with open(os.path.join(M29, "m148_demet.json"), encoding="utf-8") as fh:
        D = json.load(fh)
    print(f"\n{'sonda':>6s} {'gercek':>10s} {'cozulen':>10s} {'hata':>11s}")
    RHO = {}
    enb = 0.0
    for k in sorted(olcumler):
        g = next(s for s in D["sondalar"] if s["sonda"] == k)
        cap = sum(float(g.get("onceki_r", {}).get(str(j), 0.0)) * RHO[j] for j in RHO if j < k)
        RHO[k] = (g["sabit"] - 2 * cap - olcumler[k] ** 2) / (2 * g["kappa_etkin"])
        h = RHO[k] - GERCEK[k - 1]
        enb = max(enb, abs(h))
        print(f"{k:6d} {GERCEK[k - 1]:+10.4f} {RHO[k]:+10.4f} {h:+11.2e}")

    print(f"\nNIHAI DOSYANIN GERCEK SKORU  = {P_ger:.6f}   (dogrudan hesaplandi)")
    if bek is not None:
        print(f"BETIGIN BILDIRDIGI BEKLENTI  = {bek:.6f}   (fark {P_ger - bek:+.2e})")
    tam = float(np.sqrt(max(SB["TABAN_MSE"] - float((GERCEK**2).sum()), 1e-9)))
    print(f"KUSURSUZ OLCUMLE ULASILABILIR = {tam:.6f}")

    print("\n--- HUKUM ---")
    ok = True
    if enb > 5e-3:
        print(f"  X cozulen rho hatasi buyuk: {enb:.2e}")
        ok = False
    else:
        print(f"  OK cozulen rho hatasi en fazla {enb:.2e}")
    if bek is not None and abs(P_ger - bek) > 2e-4:
        print(f"  X bildirilen beklenti gercekten {P_ger - bek:+.2e} sapiyor")
        ok = False
    elif bek is not None:
        print(f"  OK bildirilen beklenti gercege {P_ger - bek:+.2e} yakin")
    if P_ger > tam + 5e-4:
        print(f"  X nihai skor kusursuz olcumden {P_ger - tam:+.2e} kotu")
        ok = False
    else:
        print(f"  OK nihai skor kusursuz olcume {P_ger - tam:+.2e} yakin")
    print("\n" + ("ZINCIR SAGLAM." if ok else "ZINCIRDE SORUN VAR."))
    sys.exit(0 if ok else 1)
finally:
    temizle()
    print("(sentetik dosyalar temizlendi)")
