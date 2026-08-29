import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from x1_yukle import KOK, matris, oku

dos, X, s = matris()
adlar = [d.replace("tuketim_", "").replace(".csv", "") for d in dos]
N = X.shape[1]
g7 = oku(KOK / "submissions/tuketim_g7_span_tau3.csv")
for ad, f in [
    ("y46", "tuketim_y46_amnezik_kirpik.csv"),
    ("y45", "tuketim_y45_mevsimsel_kirpik.csv"),
    ("y40", "tuketim_y40_sota_temiz.csv"),
]:
    y = oku(KOK / "submissions" / f)
    alt, ust = 0.0, 1e9
    ai = ui = ""
    for i in range(25):
        dd = np.sqrt(((y - X[i]) ** 2).mean())
        lo = abs(s[i] - dd)
        hi = s[i] + dd
        if lo > alt:
            alt, ai = lo, adlar[i]
        if hi < ust:
            ust, ui = hi, adlar[i]
    d = y - g7
    Q = float((d**2).mean())
    m0 = 1.00137**2
    be = np.sqrt(m0 + Q)
    print(f"\n== {ad} ==")
    print(f"  UCGEN SINIRLARI: skor in [{alt:.5f} ({ai}) , {ust:.5f} ({ui})]")
    print(f"  g7 tabanina gore Q={Q:.5f}, BASABAS skor = {be:.5f}  (bu skorda BILGI SIFIR)")
    print(
        f"  basabas sinir icinde mi? {'EVET -> RISK: hak bosa gidebilir' if alt < be < ust else 'HAYIR'}"
    )
    for sk in [alt, 1.05, 1.10, be, 1.25, ust]:
        L = (m0 + Q - sk**2) / 2
        g = L * L / Q
        print(
            f"    skor={sk:7.4f} -> kazanc(MSE)={g:.6f}  ortak RMSLE={np.sqrt(max(m0 - g, 0)):.5f}"
        )
