"""x2 -- A) g7 span iddiasini BAGIMSIZ kir."""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from x1_yukle import KOK, matris, oku

dos, X, s = matris()
N = X.shape[1]
adlar = [d.replace("tuketim_", "").replace(".csv", "") for d in dos]
g7 = oku(KOK / "submissions/tuketim_g7_span_tau3.csv")
m6 = oku(KOK / "submissions/tuketim_m6_ikiyon.csv")

# ---- 1. serbest (kisitsiz) EKK ile w bul
A = X @ X.T / N
b = X @ g7 / N
w, *_ = np.linalg.lstsq(X.T, g7, rcond=None)
art = g7 - X.T @ w
print("== A1 SERBEST EKK ==")
print(f"  agirlik toplami sum(w) = {w.sum():.15f}   (1'den sapma {w.sum() - 1:+.3e})")
print(f"  |w|_1 = {np.abs(w).sum():.6f}")
print(f"  artik RMS = {np.sqrt((art**2).mean()):.6e}   maks|artik| = {np.abs(art).max():.6e}")
print(f"  artik enerjisi ||r||^2/N = {(art**2).mean():.6e}")
print(f"  X Gram kosul sayisi = {np.linalg.cond(A):.4e}")

# ---- 2. beyan edilen agirliklarla karsilastir
bey = json.loads((Path(__file__).parent / "g3_aday.json").read_text(encoding="utf-8"))["tau=3"]["w"]
wb = np.zeros(len(adlar))
for k, v in bey.items():
    wb[adlar.index(k)] = v
print("\n== A2 BEYAN EDILEN AGIRLIKLAR ==")
print(f"  beyan sum(w) = {wb.sum():.15f}   (1'den sapma {wb.sum() - 1:+.3e})")
print(f"  beyan |w|_1 = {np.abs(wb).sum():.15f}")
pb = X.T @ wb
print(f"  beyan kombinasyonu vs g7 dosyasi: RMS fark = {np.sqrt(((pb - g7) ** 2).mean()):.6e}")
print(
    f"     maks fark = {np.abs(pb - g7).max():.6e}, farkli satir(>1e-9) = {(np.abs(pb - g7) > 1e-9).sum()}"
)
print(f"  serbest w ile beyan w farki (maks) = {np.abs(w - wb).max():.3e}")

# ---- 3. HATA GRAM'i -> bagimsiz ongoru
m = s**2
D = np.empty((25, 25))
for i in range(25):
    D[i] = ((X - X[i]) ** 2).mean(1)
E = (m[:, None] + m[None, :] - D) / 2
lam = np.linalg.eigvalsh(E)
print("\n== A3 BAGIMSIZ ONGORU (hata-Gram) ==")
print(
    f"  E ozdegerleri min={lam.min():.6e} maks={lam.max():.6e} kosul={lam.max() / max(lam.min(), 1e-300):.3e}"
)
print(
    f"  E POZITIF YARI-TANIMLI MI? min ozdeger {lam.min():+.3e} -> {'EVET' if lam.min() > -1e-9 else 'HAYIR (TUTARSIZ SKORLAR!)'}"
)
for ad, ww in [("beyan w", wb), ("serbest w", w)]:
    mse = ww @ E @ ww
    print(f"  {ad:10s}: ongorulen MSE={mse:.8f}  RMSLE={np.sqrt(mse):.7f}")
# sum(w)!=1 hatasi icin duyarlilik
t2 = float((X[adlar.index("m6_ikiyon")] ** 2).mean())
print(f"  ||t||^2/N yaklasik (m6 ile) = {t2:.4f}")
for eps in [1e-9, 1e-6, 1e-4, 1e-3]:
    print(
        f"    sum(w)=1+{eps:.0e} olsaydi MSE kaymasi ~ {2 * eps * np.sqrt(1.0057 * t2) + eps**2 * t2:+.3e}"
    )

# ---- 4. KIRPMA
p_ham = X.T @ wb
neg = p_ham < 0
print("\n== A4 KIRPMA ==")
print(f"  ham kombinasyonda log1p<0 olan satir = {neg.sum()}")
delta = g7 - p_ham
print(f"  dosya-ham fark: sifirdan farkli satir = {(np.abs(delta) > 1e-12).sum()}")
print(f"  ||delta||^2/N = {(delta**2).mean():.6e}   |delta|max = {np.abs(delta).max():.6e}")
mse_ham = wb @ E @ wb
sinir = 2 * np.sqrt((delta**2).mean()) * np.sqrt(mse_ham)
print(
    f"  kirpmanin MSE'ye etkisi: |2<d,e>/N| <= {sinir:.3e}, + ||d||^2/N = {(delta**2).mean():.3e}"
)
print(f"  -> MSE araligi [{mse_ham - sinir:.8f}, {mse_ham + sinir + (delta**2).mean():.8f}]")
print(
    f"  -> RMSLE araligi [{np.sqrt(mse_ham - sinir):.7f}, {np.sqrt(mse_ham + sinir + (delta**2).mean()):.7f}]"
)

# ---- 5. KIRPMA SONRASI gercek span artigi
w2, *_ = np.linalg.lstsq(X.T, g7, rcond=None)
r2 = g7 - X.T @ w2
print("\n== A5 KIRPMA SONRASI DOSYANIN SPAN ARTIGI ==")
print(f"  ||r||^2/N = {(r2**2).mean():.6e}  (RMS {np.sqrt((r2**2).mean()):.3e})")
print(
    f"  kirpma delta enerjisiyle ayni mi? oran = {(r2**2).mean() / max((delta**2).mean(), 1e-300):.4f}"
)

# ---- 6. yuvarlama: dosyadaki ondalik
import pandas as pd

raw = pd.read_csv(KOK / "submissions/tuketim_g7_span_tau3.csv", dtype={"tuketim": str}).tuketim
ond = raw.str.split(".").str[-1].str.len()
print(f"\n== A6 YUVARLAMA ==  ondalik basamak dagilimi: {ond.value_counts().head(5).to_dict()}")
json.dump(
    {
        "sum_w_serbest": float(w.sum()),
        "sum_w_beyan": float(wb.sum()),
        "artik_rms": float(np.sqrt((art**2).mean())),
        "ongoru_mse_beyan": float(wb @ E @ wb),
        "E_min_ozdeger": float(lam.min()),
        "kirpma_satir": int(neg.sum()),
        "kirpma_delta2": float((delta**2).mean()),
    },
    open(os.environ["XCACHE"] + "/a.json", "w"),
)
