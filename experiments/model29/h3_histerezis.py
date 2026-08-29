"""H3 -- HISTEREZIS ekseninin BAGIMSIZ dogrulamasi ve MODELSIZ TAVANI.

Tez: ayni sicaklikta SONBAHAR ilkbahardan yuksek. Test tumuyle YUKSELEN
tarafta (Nis->Tem), egitim iki tarafi da iceriyor -> model ortaliyor.

Burada:
  1) Iddia dogrudan etiketlerden yeniden olculur (cdd22 kovalarina gore
     eslestirilmis, trafo-ici sapma).
  2) Yillik buyume kontrolu: ayni kontrast, global dogrusal zaman trendi
     cikarildiktan sonra.
  3) DURUST TAVAN: trafolar ikiye bolunur, (kova x yon) tablosu A yarisinda
     kestirilir, B yarisinda degerlendirilir. dMSE = cov(f,r)^2 / var(f).
     Aday YONUN Q'su bu degeri asamaz.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
KOK = os.path.dirname(os.path.dirname(BURA))
sys.path.insert(0, os.path.join(KOK, "src"))
from gridup.turkish import join_key  # noqa: E402

tr = pd.read_csv(
    os.path.join(KOK, "data/raw/train.csv"),
    parse_dates=["tarih"],
    dtype={"tanim": str},
    usecols=["tanim", "tarih", "tuketim", "lokasyon"],
)
tr["ilce_key"] = tr.lokasyon.str.split(">").str[-1].str.strip().map(join_key).astype(object)
tr.drop(columns=["lokasyon"], inplace=True)
tr["ly"] = np.log1p(tr.tuketim.clip(lower=0.0))

g = pd.read_parquet(
    os.path.join(KOK, "data/external/hava_gunluk.parquet"),
    columns=["ilce_key", "tarih", "sicaklik_ort"],
)
g["ilce_key"] = g.ilce_key.astype(object)
g["tarih"] = pd.to_datetime(g.tarih).dt.normalize()
g = g.drop_duplicates(["ilce_key", "tarih"]).sort_values(["ilce_key", "tarih"])
g["cdd22"] = (g.sicaklik_ort - 22.0).clip(lower=0.0)
gr = g.groupby("ilce_key", observed=True)
for p in (7, 14, 60):
    g[f"cdd22_ort{p}"] = gr.cdd22.transform(lambda s, _p=p: s.rolling(_p, min_periods=5).mean())
    g[f"sic_ort{p}"] = gr.sicaklik_ort.transform(
        lambda s, _p=p: s.rolling(_p, min_periods=5).mean()
    )
g["yon"] = g.sic_ort14 - g.sic_ort60
G = g.set_index(["ilce_key", "tarih"])[["cdd22_ort7", "sic_ort60", "yon"]]

idx = pd.MultiIndex.from_arrays([tr.ilce_key.to_numpy(), tr.tarih.to_numpy()])
d = G.reindex(idx)
tr["cdd7"] = d.cdd22_ort7.to_numpy()
tr["yon"] = d.yon.to_numpy()
tr = tr.dropna(subset=["cdd7", "yon"])
tr["r"] = tr.ly - tr.groupby("tanim").ly.transform("mean")

# yillik buyume kontrolu: global dogrusal zaman trendi
x = (tr.tarih - tr.tarih.min()).dt.days.to_numpy().astype(float)
b = float(np.cov(x, tr.r.to_numpy(), bias=True)[0, 1] / x.var())
tr["r_trendsiz"] = tr.r - b * (x - x.mean())
print(f"global dogrusal trend: {b * 365:+.4f} log/yil  (yillik buyume vekili)")

KOVA = [(0.5, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 9.0), (9.0, 99.0)]
print("\n=== cdd22_ort7 kovasi x MEVSIM YONU: trafo-ici sapma ortalamasi ===")
print(
    f"{'kova':>12} {'n_ilk':>8} {'n_son':>8} {'ilkbahar':>9} {'sonbahar':>9} {'FARK':>8} {'trendsiz':>9}"
)
tab = {}
for a, bb in KOVA:
    m = (tr.cdd7 >= a) & (tr.cdd7 < bb)
    ilk = m & (tr.yon > 0.5)
    son = m & (tr.yon < -0.5)
    if ilk.sum() < 500 or son.sum() < 500:
        continue
    f1, f2 = tr.r[ilk].mean(), tr.r[son].mean()
    t1, t2 = tr.r_trendsiz[ilk].mean(), tr.r_trendsiz[son].mean()
    tab[f"{a}-{bb}"] = dict(
        n_ilkbahar=int(ilk.sum()),
        n_sonbahar=int(son.sum()),
        ilkbahar=float(f1),
        sonbahar=float(f2),
        fark=float(f1 - f2),
        fark_trendsiz=float(t1 - t2),
    )
    print(
        f"{a:5.1f}-{bb:5.1f} {ilk.sum():8,} {son.sum():8,} "
        f"{f1:+9.4f} {f2:+9.4f} {f1 - f2:+8.4f} {t1 - t2:+9.4f}"
    )

# ---------------- DURUST TAVAN: (kova x yon) tablosu, trafo yarisi capraz
kod = pd.factorize(tr.tanim.to_numpy())[0]
A = (kod % 2) == 0
kv = np.digitize(tr.cdd7.to_numpy(), [0.5, 2.0, 4.0, 6.0, 9.0])
yn = np.digitize(tr.yon.to_numpy(), [-0.5, 0.5])
hucre = kv * 3 + yn
nh = hucre.max() + 1
for ad, hedef in (("ham", "r"), ("trendsiz", "r_trendsiz")):
    rr = tr[hedef].to_numpy()
    say = np.bincount(hucre[A], minlength=nh)
    top = np.bincount(hucre[A], weights=rr[A], minlength=nh)
    prof = top / np.maximum(say, 1)
    f = prof[hucre[~A]]
    f = f - f.mean()
    r2 = rr[~A] - rr[~A].mean()
    vf = float((f**2).mean())
    c = float((f * r2).mean())
    k = c * c / vf if vf > 0 else 0.0
    print(f"\nDURUST TAVAN ({ad}): dMSE={k:.6f}  std={np.sqrt(k):.4f}  lambda*={c / vf:+.3f}")
    tab[f"tavan_{ad}"] = dict(dMSE=k, std=float(np.sqrt(k)), lam=float(c / vf))

print("\nKAPI: aday YONUN Q'su bu dMSE'yi ASAMAZ. Gereken esik Q >= 0,01000")
json.dump(tab, open(os.path.join(BURA, "h3_histerezis.json"), "w"), indent=1)
