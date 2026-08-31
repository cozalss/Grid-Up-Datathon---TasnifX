"""p04c: p04/p04b'de POZITIF cikan iki fikrin DAYANIKLILIK sinavi + sifir kanali.

Sinav 1  KARARLILIK: et0 egiminin kiyi/tarim FARKI her blokta ayni isarette mi?
         (guz25 ve kis26 AYRI AYRI katsayi verir; ikisi de yaz25'te olculur.)
Sinav 2  BIRLESTIRME: bayram ilce katsayisi + et0 etkilesimi birlikte.
Sinav 3  SIFIR KANALI: yaz25 SSE'sinin %40'i hedef=0 satirlarindan. Alan
         bilgisi (mevsimlik abone: sulama kis kapali, turizm kis kapali)
         yaz25 sifirlarini aciklayabiliyor mu?

SIZINTI: her katsayi yalniz blok DISINDA kestirilir.
"""

import json
import os

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = os.path.join(KOK, "data/interim/deney")
AO = os.path.join(KOK, "data/interim/aile_onbellek")
BURA = os.path.dirname(os.path.abspath(__file__))
HEDEF_SOGUK = 0.222
KOL = ["tanim", "tarih", "tuketim", "ilce_key", "soguk_mu", "_blok", "guc",
       "tarim_orani", "et0_toplam", "cdd24", "t_sifir_orani", "t_log_ort",
       "t_son_kayit_yasi", "t_olu_mu", "yerlesim_orani"]
e = pd.read_parquet(os.path.join(DN, "egitim.parquet"), columns=KOL)


def blok_artik(ad):
    blk = e[e._blok == ad]
    sic, sog = blk[blk.soguk_mu == 0], blk[blk.soguk_mu == 1]
    P = [np.load(os.path.join(AO, f"{ad}_{t}_{aa}_uretim.npy")).astype(np.float64)
         for t in (1000, 1001, 1002) for aa in ("cat", "xgb", "lgbm")
         if os.path.exists(os.path.join(AO, f"{ad}_{t}_{aa}_uretim.npy"))]
    z = np.load(os.path.join(DN, f"soguk_tahmin_{ad}.npz"))
    idx = np.concatenate([sic.index.values, sog.index.values])
    pb = np.concatenate([np.mean(P, axis=0), np.mean([z[q] for q in z.files], axis=0)])
    bf = e.loc[idx].copy()
    bf["p"] = pb
    bf["r"] = np.log1p(bf.tuketim.values.astype(np.float64)) - pb
    s = bf.soguk_mu.values.astype(np.float64)
    w = np.where(s == 1, HEDEF_SOGUK / s.mean(), (1 - HEDEF_SOGUK) / (1 - s.mean()))
    bf["w"] = w / w.mean()
    bf["tarih"] = pd.to_datetime(bf.tarih)
    bf["_hucre"] = bf._blok.astype(str) + "_" + bf.tarih.dt.month.astype(str)
    return bf.reset_index(drop=True)


B = {a: blok_artik(a) for a in ("yaz25", "guz25", "kis26")}
Y = B["yaz25"]
m0 = float((Y.w * Y.r**2).mean())
R = {"taban_rmsle": float(np.sqrt(m0))}
print(f"taban RMSLE={np.sqrt(m0):.6f}", flush=True)

KIYI = {"cesme", "karaburun", "urla", "seferihisar", "foca", "dikili", "selcuk",
        "guzelbahce", "menderes", "aliaga"}


def mrk(df, x):
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x), x, np.nanmean(x))
    return x - pd.Series(x).groupby(df._hucre.values).transform("mean").values


def x_kiyi_et0(df):
    return df.ilce_key.isin(KIYI).astype(float).values * df.et0_toplam.values


def beta(df, xf):
    xc, w = mrk(df, xf(df)), df.w.values
    rc = df.r.values - pd.Series(df.r.values).groupby(df._hucre.values).transform("mean").values
    return float((w * xc * rc).sum() / max((w * xc * xc).sum(), 1e-12))


def olc(d, ad):
    d = np.clip(np.where(np.isfinite(d), d, 0.0), -0.6, 0.6)
    g = float(np.sqrt(m0) - np.sqrt(float((Y.w * (Y.r - d) ** 2).mean())))
    print(f"  {ad:46s} kazanc={g:+.6f}", flush=True)
    return g


print("\n=== SINAV 1: kiyi x et0 egimi blok basina ===", flush=True)
bl = {a: beta(df, x_kiyi_et0) for a, df in B.items()}
print("  " + "  ".join(f"{a}={v:+.5f}" for a, v in bl.items()), flush=True)
R["kiyi_et0_beta_bloklar"] = bl
R["kiyi_et0_guz25_uygula"] = olc(bl["guz25"] * mrk(Y, x_kiyi_et0(Y)), "yalniz guz25 katsayisi")
R["kiyi_et0_kis26_uygula"] = olc(bl["kis26"] * mrk(Y, x_kiyi_et0(Y)), "yalniz kis26 katsayisi")
bd = 0.5 * (bl["guz25"] + bl["kis26"])
d_et0 = bd * mrk(Y, x_kiyi_et0(Y))
R["kiyi_et0_ortalama"] = olc(d_et0, "guz25+kis26 ortalamasi (BLOK DISI)")

print("\n=== SINAV 2: bayram ilce katsayisi (blok disi) + et0 ===", flush=True)
KURBAN25 = pd.to_datetime(["2025-06-05", "2025-06-06", "2025-06-07", "2025-06-08",
                           "2025-06-09"])
DISB = pd.concat([B["guz25"], B["kis26"]], ignore_index=True)
RAM26 = pd.to_datetime(["2026-03-19", "2026-03-20", "2026-03-21", "2026-03-22"])
TATIL_HEPSI = pd.to_datetime(list(RAM26) + ["2025-08-30", "2025-10-29", "2026-01-01",
                                            "2025-12-31", "2025-10-28"])
ILCE_OF = e.drop_duplicates("tanim").set_index("tanim").ilce_key
par = []
for g in RAM26:
    pen = DISB[(DISB.tarih >= g - pd.Timedelta(days=10)) & (DISB.tarih <= g + pd.Timedelta(days=10))
               & (DISB.tarih.dt.dayofweek == g.dayofweek) & (~DISB.tarih.isin(TATIL_HEPSI))]
    tb = pen.groupby("tanim").r.agg(["mean", "size"])
    tb = tb[tb["size"] >= 2]["mean"]
    gn = DISB[DISB.tarih == g].groupby("tanim").r.mean()
    o = gn.index.intersection(tb.index)
    par.append((gn.loc[o] - tb.loc[o]).rename(str(g.date())))
sp = pd.concat(par, axis=1).mean(axis=1)
glob = float(sp.median())
kats = sp.groupby(sp.index.map(ILCE_OF)).median()
kats = kats - kats.median()
km = Y.tarih.isin(KURBAN25).values
d_bay = np.zeros(len(Y))
d_bay[km] = (glob + Y.ilce_key.map(kats).fillna(0.0).values)[km]
R["bayram"] = olc(d_bay, "F3 bayram ilce (Ramazan26 -> Kurban25)")
R["birlesik"] = olc(d_bay + d_et0, "BIRLESIK: bayram + kiyi_x_et0")

print("\n=== SINAV 3: SIFIR KANALI ===", flush=True)
sf = Y.tuketim.values == 0
tot = float((Y.w * Y.r**2).sum())
print(f"  yaz25 sifir: satir%={sf.mean():.4f} SSE%={(Y.w[sf]*Y.r[sf]**2).sum()/tot:.4f} "
      f"ort_tahmin(log1p)={Y.p.values[sf].mean():.4f}", flush=True)
# mevsimlik abone tezi: sifir orani ilce tipine gore aya bagli mi?
Y["_ay"] = Y.tarih.dt.month
Y["_kiyi"] = Y.ilce_key.isin(KIYI)
tab = Y.assign(sf=sf).pivot_table(index="_ay", columns="_kiyi", values="sf", aggfunc="mean")
print("  yaz25 sifir orani (ay x kiyi):\n", tab.round(4).to_string(), flush=True)
R["sifir_ay_kiyi"] = {str(k): {str(c): float(v) for c, v in row.items()}
                      for k, row in tab.iterrows()}
for a, df in B.items():
    s2 = df.tuketim.values == 0
    print(f"  {a}: sifir orani={s2.mean():.4f}  kiyi={s2[df.ilce_key.isin(KIYI).values].mean():.4f}"
          f"  tarim={s2[~df.ilce_key.isin(KIYI).values].mean():.4f}", flush=True)
# tahmin edilen sifir satirlari icin en iyi SABIT (blok disinda secilir)
for a in ("guz25", "kis26"):
    df = B[a]
    m = df.p.values < np.log1p(0.05)
    if m.sum() > 100:
        opt = float(np.average(np.log1p(df.tuketim.values[m]), weights=df.w.values[m]))
        print(f"  {a}: p<log1p(0.05) satir={m.sum()} optimal sabit(log1p)={opt:.4f}", flush=True)
        R[f"sifir_optimal_{a}"] = opt
my = Y.p.values < np.log1p(0.05)
if my.sum() > 100 and "sifir_optimal_guz25" in R:
    sab = 0.5 * (R.get("sifir_optimal_guz25", 0) + R.get("sifir_optimal_kis26", 0))
    d = np.zeros(len(Y))
    d[my] = sab - Y.p.values[my]
    print(f"  yaz25 dokunan satir={my.sum()} uygulanan sabit={sab:.4f}", flush=True)
    R["sifir_hedge"] = olc(d, "F5 dusuk-tahmin satirlarina blok disi sabit")

with open(os.path.join(BURA, "p04c_dogrula.json"), "w", encoding="utf-8") as fh:
    json.dump(R, fh, ensure_ascii=False, indent=1)
print("\nyazildi p04c_dogrula.json", flush=True)
