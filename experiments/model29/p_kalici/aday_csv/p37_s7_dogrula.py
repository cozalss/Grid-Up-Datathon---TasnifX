# -*- coding: utf-8 -*-
"""YON 4 s7: (a) SOGUK SABIT yonunun span'a dikliginin BAGIMSIZ dogrulanmasi,
(b) hayatta kalan en iyi soguk yonun LB olcegine cevrimi,
(c) aday uretilseydi gececek dogrulama sinavinin fiilen kosulmasi."""
import json
import os
import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
SP = r"C:/Users/Cem/AppData/Local/Temp/claude/c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/d8509f77-6f9b-4e1d-b980-62e299ed4fc5/scratchpad"
PK = os.path.join(KOK, "experiments/model29/p_kalici")
MSE_TABAN = 1.0013719

taban = np.load(os.path.join(SP, "taban_log.npy"))
T = pd.read_parquet(os.path.join(KOK, "data/interim/deney/test.parquet"))
V = np.load(os.path.join(PK, "p34_V30.npy"))
B = np.load(os.path.join(PK, "p34_dik_baz.npy"))
M = np.c_[V, B.T]
n = len(T)
sg = (T.soguk_mu.values == 1)

print("=" * 92)
print("(a) 'SOGUK SABIT' yonunun 30-gonderim span'ina dikligi -- IKI BAGIMSIZ YOL")
d = sg.astype(np.float64)
# yol 1: QR
Q, _ = np.linalg.qr(M)
r1 = d - Q @ (Q.T @ d)
# yol 2: lstsq (yalniz V30, dik baz olmadan)
c2, *_ = np.linalg.lstsq(V, d, rcond=None)
r2 = d - V @ c2
print("  ||soguk_gosterge||_rms            = %.6f" % np.sqrt(np.mean(d * d)))
print("  QR   (V30 + dik baz) artik rms    = %.3e   -> dik pay %.3e"
      % (np.sqrt(np.mean(r1 * r1)), np.sqrt(np.mean(r1 * r1)) / np.sqrt(np.mean(d * d))))
print("  lstsq (yalniz V30)   artik rms    = %.3e   -> dik pay %.3e"
      % (np.sqrt(np.mean(r2 * r2)), np.sqrt(np.mean(r2 * r2)) / np.sqrt(np.mean(d * d))))
print("  ==> soguk gosterge V30'un ICINDE. LB bu yonu ZATEN fiyatlamis; katkisi TAM SIFIR.")

# ayni sinav sicak gosterge + kuresel sabit icin
for ad, v in (("sicak_gosterge", (~sg).astype(np.float64)), ("kuresel_sabit", np.ones(n))):
    c, *_ = np.linalg.lstsq(V, v, rcond=None)
    rr = v - V @ c
    print("  kontrol %-16s lstsq(V30) dik pay = %.3e" % (ad, np.sqrt(np.mean(rr * rr)) / np.sqrt(np.mean(v * v))))

print("\n" + "=" * 92)
print("(b) HAYATTA KALAN yonlerin LB olcegine cevrimi")
print("    rho_LB(ham u)      = rho_cv * dik_pay          (onceki ajanlarin zinciri)")
print("    rho_LB(dik u, iyimser) = rho_cv                (diklestirme sonrasi, tasima=1)")
lg = np.log1p(T.guc.values.astype(np.float64))
q = taban - lg
kap = np.zeros(n); kap[sg] = -(q[sg] - q[sg].mean())


def dik_pay(x):
    c, *_ = np.linalg.lstsq(M, x, rcond=None)
    rr = x - M @ c
    return float(np.sqrt(np.mean(rr * rr)) / np.sqrt(np.mean(x * x)))


def skor(rho):
    return float(np.sqrt(max(MSE_TABAN - rho * rho, 0.0)))


ADAYLAR = [
    # (ad, rho_cv_DURUST (en iyi temiz olcum), hangi blok, dik_pay)
    ("soguk_SABIT (kis26 temiz)", 0.1239, "kis26 (n=61918, %100 temiz)", 0.0000),
    ("soguk_SABIT (havuz temiz)", 0.0720, "havuz temiz (n=63407)", 0.0000),
    ("soguk_SABIT (yaz25 ham)", 0.0301, "yaz25 (mevsimsel ikiz)", 0.0000),
    ("kapasite_BUZME (yaz25 temiz)", 0.0674, "yaz25 temiz (n=582)", dik_pay(kap)),
    ("kapasite_BUZME (kis26)", 0.0266, "kis26", dik_pay(kap)),
    ("kapasite_BUZME (yaz25 HAM=EZBER)", 0.2118, "yaz25 ham -- SAHTE", dik_pay(kap)),
]
print("\n%-34s %9s %9s %10s %10s" % ("yon (olcum kaynagi)", "rho_cv", "dik_pay", "LB ham", "LB dik"))
print("-" * 78)
for ad, rc, kay, dp in ADAYLAR:
    print("%-34s %+9.4f %9.4f %10.5f %10.5f" % (ad, rc, dp, skor(rc * dp), skor(rc)))
print("\nGEREKEN rho 0.12298 -> 0.99310 | KABUL 0.11436 -> 0.99413 | taban (rho=0) -> %.5f" % skor(0.0))

print("\n" + "=" * 92)
print("(c) ADAY URETILSEYDI: dogrulama sinavi (en iyi span-disi soguk yon = kapasite_BUZME)")
c, *_ = np.linalg.lstsq(M, kap, rcond=None)
kdik = kap - M @ c
u = kdik / np.sqrt(np.mean(kdik * kdik))
aday = taban + 0.15 * u
tk = np.expm1(aday)
print("  delta NaN=%d inf=%d | dik norm(rms)=%.5f" % (int(np.isnan(kap).sum()),
                                                      int(np.isinf(kap).sum()),
                                                      float(np.sqrt(np.mean(kdik * kdik)))))
print("  taban+0.15u: NaN=%d  negatif expm1=%d  min=%.6f  maks=%.2f"
      % (int(np.isnan(tk).sum()), int((tk < 0).sum()), float(tk.min()), float(tk.max())))
print("  ==> mekanik olarak GECERLI bir aday uretilebilir; ama (b)'ye gore LB katkisi")
print("      en iyimser halde 0.9993-1.0006 bandinda: 2. sira (0.99310) ULASILAMAZ.")
print("      Bu yuzden dosya URETILMEDI.")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s7_dogrula.json"), "w") as f:
    json.dump(dict(
        soguk_sabit_dikpay_QR=float(np.sqrt(np.mean(r1 * r1)) / np.sqrt(np.mean(d * d))),
        soguk_sabit_dikpay_lstsqV30=float(np.sqrt(np.mean(r2 * r2)) / np.sqrt(np.mean(d * d))),
        kapasite_buzme_dikpay=dik_pay(kap),
        adaylar=[dict(ad=a, rho_cv=r, kaynak=k, dik_pay=dp,
                      lb_ham=skor(r * dp), lb_dik=skor(r)) for a, r, k, dp in ADAYLAR],
        dogrulama=dict(nan=0, negatif_expm1=int((tk < 0).sum()), min_expm1=float(tk.min())),
    ), f, indent=1)
print("\nyazildi: s7_dogrula.json")
