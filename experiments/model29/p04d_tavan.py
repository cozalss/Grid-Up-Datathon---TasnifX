"""p04d: iki pozitif fikrin TAVANI. Katsayi yaz25'in KENDISINDEN (oracle)
secilirse ne kadar kazanilirdi? Blok disi tahminle arasindaki fark, fikrin
"kotu kestirim" mi yoksa "kucuk etki" mi oldugunu soyler.

ORACLE SAYI DEGILDIR -- yalniz TAVAN raporlanir, gonderime girmez.
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
KOL = ["tanim", "tarih", "tuketim", "ilce_key", "soguk_mu", "_blok", "et0_toplam",
       "tatil_mi", "tatil_agirligi"]
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
    bf["r"] = np.log1p(bf.tuketim.values.astype(np.float64)) - pb
    s = bf.soguk_mu.values.astype(np.float64)
    w = np.where(s == 1, HEDEF_SOGUK / s.mean(), (1 - HEDEF_SOGUK) / (1 - s.mean()))
    bf["w"] = w / w.mean()
    bf["tarih"] = pd.to_datetime(bf.tarih)
    bf["_hucre"] = bf._blok.astype(str) + "_" + bf.tarih.dt.month.astype(str)
    return bf.reset_index(drop=True)


Y = blok_artik("yaz25")
m0 = float((Y.w * Y.r**2).mean())
KIYI = {"cesme", "karaburun", "urla", "seferihisar", "foca", "dikili", "selcuk",
        "guzelbahce", "menderes", "aliaga"}
KURBAN25 = pd.to_datetime(["2025-06-05", "2025-06-06", "2025-06-07", "2025-06-08",
                           "2025-06-09"])


def mrk(df, x):
    x = np.asarray(x, dtype=np.float64)
    return x - pd.Series(x).groupby(df._hucre.values).transform("mean").values


def g(d):
    return float(np.sqrt(m0) - np.sqrt(float((Y.w * (Y.r - np.clip(d, -0.6, 0.6)) ** 2).mean())))


R = {}
# ORACLE tek katsayi: her yon icin yaz25'te en iyi olcek
x_et0 = mrk(Y, Y.ilce_key.isin(KIYI).astype(float).values * Y.et0_toplam.values)
km = Y.tarih.isin(KURBAN25).values
x_bay = np.zeros(len(Y))
x_bay[km] = 1.0
x_bay = mrk(Y, x_bay)
# ORACLE ilce x Kurban tam serbestlik (46 parametre)
ilce_kod = Y.ilce_key.astype("category").cat.codes.values
x_ilce = np.zeros((len(Y), ilce_kod.max() + 1))
x_ilce[np.arange(len(Y)), ilce_kod] = km.astype(float)

for ad, X in (("et0 x kiyi (1 par)", x_et0[:, None]),
              ("Kurban global (1 par)", x_bay[:, None]),
              ("Kurban x ILCE (46 par)", x_ilce),
              ("ikisi birlikte", np.column_stack([x_et0, x_bay]))):
    w = Y.w.values
    A = X.T @ (X * w[:, None])
    b = np.linalg.pinv(A, rcond=1e-8) @ (X.T @ (w * Y.r.values))
    R[ad] = g(X @ b)
    print(f"  TAVAN {ad:26s} = {R[ad]:+.6f}", flush=True)

# Kurban gununu tam serbest tahmin: her (ilce, gun) hucresi icin oracle ortalama
grp = Y[km].groupby(["ilce_key", Y.tarih[km].dt.date])
ort = grp.apply(lambda gg: np.average(gg.r, weights=gg.w), include_groups=False)
d = np.zeros(len(Y))
key = list(zip(Y.ilce_key.values, Y.tarih.dt.date.values))
mp = ort.to_dict()
d[km] = [mp.get(k, 0.0) for k, m in zip(key, km) if m]
R["Kurban x ilce x GUN (oracle tam)"] = g(d)
print(f"  TAVAN Kurban x ilce x GUN (oracle tam) = {R['Kurban x ilce x GUN (oracle tam)']:+.6f}",
      flush=True)

with open(os.path.join(BURA, "p04d_tavan.json"), "w", encoding="utf-8") as fh:
    json.dump(R, fh, ensure_ascii=False, indent=1)
print("yazildi p04d_tavan.json", flush=True)
