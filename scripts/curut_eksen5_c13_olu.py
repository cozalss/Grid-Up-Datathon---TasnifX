"""CURUTME -- receteyi YAZILDIGI GIBI kosmanin bedeli: 56.024 fazla satirin ayrimi."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path("C:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX")
tr = pd.read_csv(
    KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
)
te = pd.read_csv(
    KOK / "data/raw/test.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
)
v55 = pd.read_csv(KOK / "submissions/tuketim_v55_gunolcek.csv", encoding="utf-8")
te = te.merge(v55, on="id", validate="one_to_one")
te["lg"] = np.log1p(te["tuketim"])

SICAK = set(tr["tanim"].unique())
q1 = tr[(tr["tarih"] >= "2026-01-01") & (tr["tarih"] <= "2026-03-31")]
var_q1 = set(q1["tanim"].unique())
canli = set(q1.loc[q1["tuketim"] > 0, "tanim"].unique())
OLU_A = var_q1 - canli  # Q1'de kaydi VAR ama hepsi sifir -> GERCEKTEN OLU
OLU_B = SICAK - var_q1  # Q1'de HIC kaydi yok -> kayip kapsama

for ad, kume in (
    ("A: Q1'de kayit var, hepsi SIFIR (olu)", OLU_A),
    ("B: Q1'de HIC kayit yok (kapsama bosluğu)", OLU_B),
):
    m = te["tanim"].isin(kume)
    print(
        f"{ad}: {len(kume & set(te['tanim'])):>5} trafo, {int(m.sum()):>7,} test satiri"
        f"  v55 ort log1p {te.loc[m, 'lg'].mean():.4f}  medyan {te.loc[m, 'lg'].median():.4f}"
    )

# IKIZ olcum: 2025 Q1'de kaydi olup hepsi sifir olanlarin 2025 Nis-Tem gerceklesmesi
q0 = tr[(tr["tarih"] >= "2025-01-01") & (tr["tarih"] <= "2025-03-31")]
var0 = set(q0["tanim"].unique())
canli0 = set(q0.loc[q0["tuketim"] > 0, "tanim"].unique())
olu0 = var0 - canli0
il = tr[(tr["tarih"] >= "2025-04-01") & (tr["tarih"] <= "2025-07-31") & tr["tanim"].isin(olu0)]
print(
    f"\nIKIZ A: 2025 Q1'de olu {len(olu0)} trafo -> Nis-Tem {len(il):,} satir,"
    f" gerceklesen ort log1p {np.log1p(il['tuketim']).mean():.4f}, sifir payi %{100 * (il['tuketim'] <= 0).mean():.1f}"
)
mA = te["tanim"].isin(OLU_A)
bA = float(np.log1p(il["tuketim"]).mean() - te.loc[mA, "lg"].mean())
print(f"   => b_A = {bA:+.4f}")

# B grubu: gecmisi var ama Q1 kapsamasi yok -- son 60 gunluk kaydina bak
son = tr[tr["tanim"].isin(OLU_B)].sort_values("tarih").groupby("tanim").tail(60)
print(
    f"\nB grubu son 60 gun: ort log1p {np.log1p(son['tuketim']).mean():.4f}"
    f"  sifir payi %{100 * (son['tuketim'] <= 0).mean():.1f}"
    f"  son kayit tarihi medyan {son.groupby('tanim')['tarih'].max().median().date()}"
)
mB = te["tanim"].isin(OLU_B)
print(f"   v55 B grubunda ort log1p {te.loc[mB, 'lg'].mean():.4f}")

# dMSE: recete yazildigi gibi
pA, pB, pC = mA.mean(), mB.mean(), 500295 / len(te)
print(f"\npaylar: hedef {pC:.4f}  A {pA:.4f}  B {pB:.4f}  toplam {pC + pA + pB:.4f}")
print("\ndMSE -- recete YAZILDIGI GIBI (SICAK=%.4f) vs DOGRU hedef (%.4f)" % (pC + pA + pB, pC))
for delta in (0.06, 0.10):
    for b in (0.047, 0.065, 0.084):
        d_dogru = pC * (delta**2 - 2 * delta * b)
        # B grubunda yanliligi hedefle ayni varsay (iyimser), A'da olculen bA
        d_yaz = d_dogru + pA * (delta**2 - 2 * delta * bA) + pB * (delta**2 - 2 * delta * b)
        print(
            f"  delta {delta:.2f}  b {b:+.3f}:  DOGRU {d_dogru:+.5f}   YAZILDIGI GIBI {d_yaz:+.5f}"
            f"   (A grubunun bedeli {pA * (delta**2 - 2 * delta * bA):+.5f})"
        )
