"""CURUTME FINAL -- b'nin belirsizligi, OLU satirlarin maliyeti, dMSE tablosu."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path("C:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX")
A = pd.read_csv(KOK / "reports/_c10_b.csv", index_col=0)
A.index = A.index.astype(str)


def ag(v, w):
    return float((v * w).sum() / w.sum())


HAVA = {
    "gunluk (-0.0485)": -0.0485,
    "7g yumusatma (-0.0546)": -0.0546,
    "14g yumusatma (-0.0574)": -0.0574,
    "aylik ust sinir (-0.0776)": -0.0776,
}
ATIF_YANLILIK = -0.0076  # c11 capraz dogrulama: atif hafif YUKARI sapiyor
print("=== b (SATIR AGIRLIKLI, tum 500.295 hedef satir) ===")
ham = ag(A["S25"] - A["kald26"], A["w"])
print(f"  hava duzeltmesi ONCESI ham b = {ham:+.5f}")
tahmin = {}
for ad, hv in HAVA.items():
    b = ham + hv + ATIF_YANLILIK
    tahmin[ad] = b
    print(f"  {ad:>26}: b = {b:+.5f}")

rng = np.random.default_rng(11)
idx = np.arange(len(A))
bs = []
for _ in range(500):
    s = rng.choice(idx, len(idx), replace=True)
    q = A.iloc[s]
    bs.append(ag(q["S25"] - q["kald26"], q["w"]) - 0.0546 + ATIF_YANLILIK)
bs = np.array(bs)
print(
    f"\n  bootstrap (500, trafo blogu, hava=-0.0546): ort {bs.mean():+.5f}"
    f"  SH {bs.std():.5f}  %95 [{np.quantile(bs, 0.025):+.5f}, {np.quantile(bs, 0.975):+.5f}]"
)

# ---- OLU satirlar: recete YAZILDIGI GIBI kosulursa
tr = pd.read_csv(
    KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
)
te = pd.read_csv(
    KOK / "data/raw/test.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
)
v55 = pd.read_csv(KOK / "submissions/tuketim_v55_gunolcek.csv", encoding="utf-8")
te = te.merge(v55, on="id", validate="one_to_one")
SICAK = set(tr["tanim"].unique())
q126 = tr[(tr["tarih"] >= "2026-01-01") & (tr["tarih"] <= "2026-03-31")]
CANLI = set(q126.loc[q126["tuketim"] > 0, "tanim"].unique())
OLU = SICAK - CANLI
m_olu = te["tanim"].isin(OLU)
print(
    f"\n=== SICAK ama 2026 Q1'de OLU: {int(m_olu.sum()):,} satir "
    f"({100 * m_olu.mean():.2f}%), {len(OLU & set(te['tanim'])):,} trafo ==="
)
print(f"  v55'in oralardaki ort log1p = {np.log1p(te.loc[m_olu, 'tuketim']).mean():.4f}")

# ILERI PENCERE IKIZI: 2025 Q1'de olu olanlarin 2025 Nis-Tem gerceklesmesi
q125 = tr[(tr["tarih"] >= "2025-01-01") & (tr["tarih"] <= "2025-03-31")]
canli25 = set(q125.loc[q125["tuketim"] > 0, "tanim"].unique())
olu25 = set(q125["tanim"].unique()) - canli25
ileri = tr[(tr["tarih"] >= "2025-04-01") & (tr["tarih"] <= "2025-07-31") & tr["tanim"].isin(olu25)]
print(f"  IKIZ: 2025 Q1'de olu {len(olu25):,} trafo -> 2025 Nis-Tem'de {len(ileri):,} satir")
print(
    f"        gerceklesen ort log1p = {np.log1p(ileri['tuketim']).mean():.4f}"
    f"  (sifir payi %{100 * (ileri['tuketim'] <= 0).mean():.1f})"
)
b_olu = np.log1p(ileri["tuketim"]).mean() - np.log1p(te.loc[m_olu, "tuketim"]).mean()
print(f"  => OLU satirlarda tahmini yanlilik b_olu = {b_olu:+.4f}  (NEGATIF = model zaten YUKSEK)")

# ---- dMSE TABLOSU
p_canli, p_olu = 500295 / 714688, int(m_olu.sum()) / 714688
print(f"\n=== dMSE TABLOSU  (p_canli {p_canli:.4f}, p_olu {p_olu:.4f}, taban MSE 1.03207) ===")
print(f"{'delta':>7} | " + " | ".join(f"b={b:+.3f}" for b in (0.030, 0.047, 0.065, 0.084, 0.118)))
for delta in (0.03, 0.04, 0.05, 0.06, 0.08, 0.10):
    hue = []
    for b in (0.030, 0.047, 0.065, 0.084, 0.118):
        dm = p_canli * (delta**2 - 2 * delta * b)
        hue.append(f"{dm:+.5f}")
    print(f"{delta:>7.3f} | " + " | ".join(hue))
print("\n  ayni tablo RMSLE olarak (1.01591 tabanindan):")
print(f"{'delta':>7} | " + " | ".join(f"b={b:+.3f}" for b in (0.030, 0.047, 0.065, 0.084, 0.118)))
for delta in (0.03, 0.04, 0.05, 0.06, 0.08, 0.10):
    hue = []
    for b in (0.030, 0.047, 0.065, 0.084, 0.118):
        dm = p_canli * (delta**2 - 2 * delta * b)
        hue.append(f"{np.sqrt(1.03207 + dm) - 1.01591:+.5f}")
    print(f"{delta:>7.3f} | " + " | ".join(hue))

print("\n=== RECETE YAZILDIGI GIBI KOSULURSA (556.319 SICAK satir, olular DAHIL) ===")
for delta in (0.06, 0.10):
    for b in (0.047, 0.065, 0.084):
        dm = p_canli * (delta**2 - 2 * delta * b) + p_olu * (delta**2 - 2 * delta * float(b_olu))
        dm_dogru = p_canli * (delta**2 - 2 * delta * b)
        print(
            f"  delta {delta:.2f} b {b:+.3f}: DOGRU hedef dMSE {dm_dogru:+.5f}"
            f"  | recete komutu dMSE {dm:+.5f}  (fark {dm - dm_dogru:+.5f})"
        )
