"""p02: SOGUK (gecmissiz) trafolarda hangi onceligin gercekten bilgi tasidigi.
Her onceligi EN IYI sabit kaymayla duzelterek olcuyorum (seviye yanliligini ayirmak icin).
yaz25 hedefi yalnizca OLCUM icin, hicbir seyi uydurmak icin kullanilmiyor."""
import numpy as np
import pandas as pd

K = "c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
tr = pd.read_csv(f"{K}/data/raw/train.csv", parse_dates=["tarih"])
tr["tanim"] = tr.tanim.astype(str)
tr["y"] = np.log1p(tr.tuketim.clip(lower=0))
kes = pd.Timestamp("2025-04-01")
h = tr[tr.tarih < kes]
Y = tr[(tr.tarih >= kes) & (tr.tarih <= "2025-07-31")].copy()
Y["soguk"] = ~Y.tanim.isin(set(h.tanim))
s = Y[Y.soguk].copy()
print("soguk satir", len(s), "soguk trafo", s.tanim.nunique())

sp = tr.lokasyon.fillna(">").str.split(">", expand=True)
tr["ilce"] = sp[2].fillna("YOK")
mp = tr.drop_duplicates("tanim").set_index("tanim")[["ilce", "guc"]]
s = s.join(mp[["ilce"]], on="tanim", rsuffix="_x")
s["gk"] = np.log2(s.guc.clip(lower=1)).round().astype(int)
s["logg"] = np.log(s.guc.clip(lower=1))

ty = h.groupby("tanim").y.mean()
hm = ty.to_frame("ty").join(mp)
hm["gk"] = np.log2(hm.guc.clip(lower=1)).round().astype(int)


def olc(ad, v):
    v = pd.Series(np.asarray(v, dtype=float), index=s.index)
    ok = v.notna()
    if ok.sum() < 100:
        print(f"{ad:26s} kapsam cok dusuk ({ok.sum()})"); return
    r = s.y[ok] - v[ok]
    kay = r.mean()
    rm = float(np.sqrt(((r - kay) ** 2).mean()))
    # kapsanmayanlar icin genel ortalama
    tam = v.fillna(v[ok].mean() if ok.any() else 0) + kay
    rt = float(np.sqrt(((s.y - tam) ** 2).mean()))
    print(f"{ad:26s} kapsam={ok.mean():.3f}  kaymasiz_RMSE={rm:.4f}  tam_RMSLE={rt:.4f}")


print("\n-- referans --")
print(f"{'en iyi tek sabit':26s} RMSE={float(np.sqrt(((s.y-s.y.mean())**2).mean())):.4f}")
print(f"{'ORACLE trafo ortalamasi':26s} "
      f"RMSE={float(np.sqrt(((s.y-s.groupby(s.tanim).y.transform('mean'))**2).mean())):.4f}")
print("\n-- oncelikler (gecmisten) --")
olc("guc (log, dogrusal)", np.nan)  # yer tutucu
olc("guc kovasi ort", s.gk.map(hm.groupby("gk").ty.mean()))
olc("guc (tam deger) ort", s.guc.map(hm.groupby("guc").ty.mean()))
olc("ilce ort", s.ilce.map(hm.groupby("ilce").ty.mean()))
ix = pd.MultiIndex.from_arrays([s.ilce, s.gk])
olc("ilce x guc kovasi",
    hm.groupby(["ilce", "gk"]).ty.mean().reindex(ix).to_numpy())
for n in (2, 3, 4, 5, 6, 7):
    pr = ty.groupby(ty.index.str[:n]).mean()
    olc(f"kimlik onek {n}", s.tanim.str[:n].map(pr))
for n in (4, 5, 6):
    pr = ty.groupby([ty.index.str[:n]]).mean()
    v = s.tanim.str[:n].map(pr)
    v2 = s.guc.map(hm.groupby("guc").ty.mean())
    olc(f"onek{n} + guc ort/2", 0.5 * (v.fillna(v2) + v2))
# guc ile dogrusal regresyon
X = np.c_[np.ones(len(hm)), np.log(hm.guc.clip(lower=1))]
b = np.linalg.lstsq(X, hm.ty.to_numpy(), rcond=None)[0]
olc("log(guc) dogrusal", b[0] + b[1] * s.logg)
