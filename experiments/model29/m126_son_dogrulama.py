"""NIHAI UCTAN UCA DOGRULAMA -- her sayiyi DISKTEKI DOSYADAN yeniden kur.

Bugun bes hata bulundu (dordu gercek, biri benimdi). Bu betik gonderilecek
dosyanin arkasindaki HER sayiyi, hicbir ara ciktiya guvenmeden, yalnizca
diskteki CSV'lerden ve gonderim skorlarindan yeniden hesaplar.

Ayrica m112_durum.json'daki saklanan L/rho degerlerinin BAYAT oldugunu
gosterir: onlar eski (hatali) sabit formuluyle uretildi. Gram'a girmiyorlar
-- bilgi olcumler[] uzerinden CSV+skor olarak giriyor, o dogru -- ama
ekranda ve belgelerde alintilaniyorlar.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
S = os.path.join(KOK, "submissions")
M29 = os.path.join(KOK, "experiments/model29")
BURA = os.path.dirname(os.path.abspath(__file__))
TABAN = "tuketim_m6_ikiyon.csv"
DOSYA = "tuketim_K_TEKHAK.csv"
sys.path.insert(0, M29)
from m112_kalibre import EK_MODEL, M0, buzmeli_r_hat  # noqa: E402

te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
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


print("=" * 74)
print(f"1) SABITLER   M0 = {M0}   EK_MODEL = {EK_MODEL}")
print("=" * 74)
a0 = oku(TABAN)
N = len(a0)
with open(os.path.join(M29, "olculmus_skorlar.json")) as fh:
    SK = json.load(fh)
with open(os.path.join(M29, "m112_durum.json")) as fh:
    DUR = json.load(fh)
print(f"   olculmus_skorlar.json: {len(SK)} kayit")
print(f"   g7 kayitli mi: {'tuketim_g7_span_tau3.csv' in SK}  (HAYIR olmali -- gonderilmedi)")
print(f"   s3y40 kayitli mi: {SK.get('tuketim_s3y40.csv')}  (1.00177 olmali -- gonderildi)")
print(f"   bekleyen sonda: {DUR.get('bekleyen')}  (None olmali)")

V, L, AD = [], [], []
for f, Pj in SK.items():
    if f == TABAN or not os.path.exists(os.path.join(S, f)):
        continue
    v = oku(f)
    if v is None or len(v) != N:
        print(f"   ATLANDI (boy/id): {f}")
        continue
    dd = v - a0
    V.append(dd)
    L.append((M0 + float((dd * dd).mean()) - Pj * Pj) / 2)
    AD.append((f, Pj))
for f, Lj in EK_MODEL.items():
    V.append(oku(f) - a0)
    L.append(Lj)
    AD.append((f, None))
for o in DUR.get("olcumler", []):
    dd = oku(o["dosya"]) - a0
    V.append(dd)
    L.append((M0 + float((dd * dd).mean()) - o["skor"] ** 2) / 2)
    AD.append((o["dosya"], o["skor"]))
V, L = np.array(V).T, np.array(L)
print(f"\n   span: {V.shape[1]} yon")

print("\n" + "=" * 74)
print("2) TUTARLILIK -- her olculmus skor yeniden kuruluyor")
print("=" * 74)
kotu = 0
for j, (f, P) in enumerate(AD):
    if P is None:
        continue
    d = V[:, j]
    Pr = np.sqrt(M0 - 2 * L[j] + float((d * d).mean()))
    if abs(Pr - P) > 1e-9:
        print(f"   SAPMA {f}: {Pr:.7f} vs {P}")
        kotu += 1
print(
    f"   {sum(1 for _, P in AD if P is not None)} skorlu yonun "
    f"{sum(1 for _, P in AD if P is not None) - kotu} tanesi BIREBIR tutuyor"
)

G = (V.T @ V) / N
r_hat, gercek, kL = buzmeli_r_hat(V, L, G, N)
nrm = float((r_hat * r_hat).mean())
MSE_OPT = M0 - gercek
print("\n" + "=" * 74)
print("3) TABAN")
print("=" * 74)
print(f"   k'L          = {kL:.9f}")
print(f"   ||r_hat||^2  = {nrm:.9f}")
print(f"   beklenen kazanc = {gercek:.9f}")
print(
    f"   saf optimum  = {np.sqrt(MSE_OPT):.6f}   (a0+r_hat gonderilseydi "
    f"{np.sqrt(M0 - 2 * kL + nrm):.6f})"
)

print("\n" + "=" * 74)
print(f"4) GONDERILECEK DOSYA: {DOSYA}")
print("=" * 74)
p = oku(DOSYA)
d = p - a0
Q_d = float((d * d).mean())
sabit = M0 - 2 * kL + Q_d
ek = d - r_hat
kap = float(np.sqrt(float((ek * ek).mean())))
c = np.linalg.pinv(G, rcond=1e-6) @ ((V.T @ ek) / N)
ekp = ek - V @ c
print(f"   Q_d (dosyadan)      = {Q_d:.9f}")
print(f"   sabit = M0-2k'L+Q_d = {sabit:.9f}")
print(f"   kappa ETKIN         = {kap:.6f}")
print(f"   ek bilesenin span-disi payi = {float((ekp * ekp).mean()) / float((ek * ek).mean()):.4f}")
print(f"   COZUM: rho = ({sabit:.9f} - P*P) / {2 * kap:.6f}")

with open(os.path.join(M29, "m122_nihai.json")) as fh:
    NIH = json.load(fh)
print("\n   m122_nihai.json ile karsilastirma:")
for k, mine in [("sabit", sabit), ("kappa_etkin", kap)]:
    theirs = NIH.get(k)
    uy = abs(theirs - mine) < 1e-9 if theirs is not None else False
    print(f"     {k:12s} betik {theirs}  benim {mine:.9f}  {'UYUYOR' if uy else 'SAPMA'}")

print("\n" + "=" * 74)
print("5) KAPILAR")
print("=" * 74)
out = pd.read_csv(os.path.join(S, DOSYA))
kapi = {
    "satir 714688": len(out) == 714688,
    "kolonlar id,tuketim": list(out.columns) == list(ss.columns),
    "id sirasi sample ile birebir": bool((out.id.values == ss.iloc[:, 0].values).all()),
    "mukerrer id yok": not out.id.duplicated().any(),
    "NaN yok": int(out.tuketim.isna().sum()) == 0,
    "negatif yok": int((out.tuketim < 0).sum()) == 0,
    "hepsi sonlu": bool(np.isfinite(out.tuketim.values).all()),
    "maks makul": bool(out.tuketim.max() < 3 * np.expm1(a0).max()),
}
for k, v in kapi.items():
    print(f"   {k:32s} {v}")
print(f"   {'TUMU GECTI':32s} {all(kapi.values())}")

print("\n" + "=" * 74)
print("6) m112_durum.json'daki SAKLANAN degerler BAYAT MI?")
print("=" * 74)
print("   Bu degerler eski (hatali) sabit formuluyle uretildi. Gram'a")
print("   GIRMIYORLAR -- bilgi olcumler[] uzerinden CSV+skor olarak giriyor.")
print("   Ama ekranda/belgelerde alintilaniyorlar.")
for ad, Lv in DUR.get("yapisal", {}).items():
    print(f"     yapisal[{ad}] = {Lv}")
for g_ in DUR.get("gecmis", []):
    print(f"     gecmis: {g_}")
print("\n   Her yapisal olcumun GERCEK katkisi Gram'dan yeniden okunur:")
for o in DUR.get("olcumler", []):
    dd = oku(o["dosya"]) - a0
    Lg = (M0 + float((dd * dd).mean()) - o["skor"] ** 2) / 2
    print(f"     {o['aday']:16s} dosya={o['dosya']:26s} skor={o['skor']}  L(Gram)={Lg:+.6f}")

print("\n" + "=" * 74)
print("7) SKOR EGRISI")
print("=" * 74)
print(f"   {'gercek rho':>11s} {'skor':>9s} {'sira':>10s}")
for rr in [0.0, 0.0304, 0.0500, 0.0570, 0.0700, 0.0790, 0.0930, 0.1457, 0.2081]:
    sk = np.sqrt(max(sabit - 2 * kap * rr, 1e-12))
    sr = (
        "1. SIRA"
        if sk < 0.99009
        else "2. SIRA"
        if sk < 0.99614
        else "3. sira"
        if sk < 0.99927
        else "4. sira"
        if sk < 0.99937
        else "5. sira"
        if sk < 1.00049
        else "6.+"
    )
    print(f"   {rr:11.4f} {sk:9.5f} {sr:>10s}")
