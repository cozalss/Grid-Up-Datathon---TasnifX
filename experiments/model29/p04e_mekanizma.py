"""p04e: en iyi fikrin (ilce x et0 egimi) MEKANIZMASI. Hangi ilceler,
hangi isaret, alan bilgisiyle tutarli mi? Katsayi blok DISINDA olculur,
yaz25 sadece dogrulama.
"""

import json
import os

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN, AO = os.path.join(KOK, "data/interim/deney"), os.path.join(KOK, "data/interim/aile_onbellek")
BURA = os.path.dirname(os.path.abspath(__file__))
HEDEF_SOGUK = 0.222
e = pd.read_parquet(
    os.path.join(DN, "egitim.parquet"),
    columns=[
        "tanim",
        "tarih",
        "tuketim",
        "ilce_key",
        "soguk_mu",
        "_blok",
        "et0_toplam",
        "tarim_orani",
        "cdd24",
    ],
)


def ba(ad):
    blk = e[e._blok == ad]
    sic, sog = blk[blk.soguk_mu == 0], blk[blk.soguk_mu == 1]
    P = [
        np.load(os.path.join(AO, f"{ad}_{t}_{aa}_uretim.npy")).astype(np.float64)
        for t in (1000, 1001, 1002)
        for aa in ("cat", "xgb", "lgbm")
        if os.path.exists(os.path.join(AO, f"{ad}_{t}_{aa}_uretim.npy"))
    ]
    z = np.load(os.path.join(DN, f"soguk_tahmin_{ad}.npz"))
    idx = np.concatenate([sic.index.values, sog.index.values])
    pb = np.concatenate([np.mean(P, axis=0), np.mean([z[q] for q in z.files], axis=0)])
    b = e.loc[idx].copy()
    b["r"] = np.log1p(b.tuketim.values.astype(np.float64)) - pb
    s = b.soguk_mu.values.astype(np.float64)
    w = np.where(s == 1, HEDEF_SOGUK / s.mean(), (1 - HEDEF_SOGUK) / (1 - s.mean()))
    b["w"] = w / w.mean()
    b["tarih"] = pd.to_datetime(b.tarih)
    b["_h"] = b._blok.astype(str) + "_" + b.tarih.dt.month.astype(str)
    return b.reset_index(drop=True)


B = {a: ba(a) for a in ("yaz25", "guz25", "kis26")}
DIS = pd.concat([B["guz25"], B["kis26"]], ignore_index=True)
Y = B["yaz25"]
m0 = float((Y.w * Y.r**2).mean())


def ilce_egim(df):
    """Her ilce icin: blok x ay merkezli artigin et0'a egimi."""
    x = df.et0_toplam.values.astype(np.float64)
    xc = x - pd.Series(x).groupby(df._h.values).transform("mean").values
    rc = df.r.values - pd.Series(df.r.values).groupby(df._h.values).transform("mean").values
    w = df.w.values
    d = pd.DataFrame({"i": df.ilce_key.values, "xr": w * xc * rc, "xx": w * xc * xc})
    g = d.groupby("i").sum()
    return (g.xr / g.xx.clip(lower=1e-9)).rename("beta"), d.groupby("i").size()


bd, nd = ilce_egim(DIS)
by, ny = ilce_egim(Y)
ort = bd.index.intersection(by.index)
kor = float(np.corrcoef(bd.loc[ort], by.loc[ort])[0, 1])
print(f"ilce et0-egimi kor(blok_disi, yaz25) = {kor:+.4f}  n={len(ort)}", flush=True)

tab = pd.DataFrame({"beta_dis": bd, "beta_yaz25": by, "n_dis": nd}).loc[ort]
tab = tab.sort_values("beta_dis")
print("\nEN NEGATIF 10 (et0 artinca model FAZLA tahmin ediyor):", flush=True)
print(tab.head(10).round(4).to_string(), flush=True)
print("\nEN POZITIF 10 (et0 artinca model AZ tahmin ediyor):", flush=True)
print(tab.tail(10).round(4).to_string(), flush=True)

# ILCE bazli tam serbest duzeltme, katsayi BLOK DISINDAN
x = Y.et0_toplam.values.astype(np.float64)
xc = x - pd.Series(x).groupby(Y._h.values).transform("mean").values
d = Y.ilce_key.map(bd).fillna(0.0).values * xc
d = np.clip(d, -0.6, 0.6)
g = float(np.sqrt(m0) - np.sqrt(float((Y.w * (Y.r - d) ** 2).mean())))
print(f"\nF6 ILCE bazli et0 egimi (46 katsayi, blok disi) kazanc = {g:+.6f}", flush=True)
# buzmeli surum (asiri uydurmaya karsi)
R = {"kor_ilce_egimi": kor, "F6_ilce_et0_tam": g, "buzme": {}}
for lam in (0.25, 0.5, 0.75):
    dd = np.clip(lam * Y.ilce_key.map(bd).fillna(0.0).values * xc, -0.6, 0.6)
    gg = float(np.sqrt(m0) - np.sqrt(float((Y.w * (Y.r - dd) ** 2).mean())))
    R["buzme"][lam] = gg
    print(f"  buzme lambda={lam}: kazanc={gg:+.6f}", flush=True)
R["ilce_beta_dis"] = {k: round(float(v), 5) for k, v in bd.items()}
with open(os.path.join(BURA, "p04e_mekanizma.json"), "w", encoding="utf-8") as fh:
    json.dump(R, fh, ensure_ascii=False, indent=1)
print("yazildi p04e_mekanizma.json", flush=True)

# --- EK: web arastirmasinin ortaya cikardigi anomali ---
# 2025 Ramazan Bayrami 29 Mart-1 Nisan + 2-4 Nisan IDARI IZIN.
# Yani yaz25'in ILK DORT GUNU (1-4 Nisan 2025) 9 gunluk tatil blogunun kuyrugu.
print("\n=== EK: 2025-04-01..04 (Ramazan idari izin kuyrugu) ===", flush=True)
mm = (Y.tarih >= "2025-04-01") & (Y.tarih <= "2025-04-04")
tot = float((Y.w * Y.r**2).sum())
print(
    f"  satir%={mm.mean():.4f} SSE%={(Y.w[mm] * Y.r[mm] ** 2).sum() / tot:.4f} "
    f"ort_artik={np.average(Y.r[mm], weights=Y.w[mm]):+.4f} "
    f"(4-11 Nisan {np.average(Y.r[(Y.tarih >= '2025-04-05') & (Y.tarih <= '2025-04-11')], weights=Y.w[(Y.tarih >= '2025-04-05') & (Y.tarih <= '2025-04-11')]):+.4f})",
    flush=True,
)
gn = (
    Y[Y.tarih <= "2025-04-14"]
    .groupby("tarih")
    .apply(lambda g: float(np.average(g.r, weights=g.w)), include_groups=False)
)
print(gn.round(4).to_string(), flush=True)
