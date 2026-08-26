"""H4 -- 2026-05-11 neden sira etkisini tek basina domine ediyor?

Bulgu adayi: gun ekseni genligi (c=1,335) OLAY GUNU ARTIKLARIYLA KIRLI bir
b_gun kestirimine uygulaniyor. Kismi-gun kayitlari o gunun ortalamasini
asagi cekiyor, gunolcek bunu "gun etkisi" sanip 1,335 kat BUYUTUYOR.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
S = Path(os.environ["S"])

te = pd.read_csv(KOK / "data/raw/test.csv", dtype={"tanim": str}, parse_dates=["tarih"])
tr = pd.read_csv(
    KOK / "data/raw/train.csv",
    usecols=["tanim", "tarih"],
    dtype={"tanim": str},
    parse_dates=["tarih"],
)
tr_son = tr.groupby("tanim")["tarih"].max()
sicak = te["tanim"].isin(set(tr["tanim"])).to_numpy()

v50 = pd.read_csv(KOK / "submissions/tuketim_v50_nihai30.csv")["tuketim"].to_numpy("float64")
b1 = pd.read_csv(S / "B1_olay.csv")["tuketim"].to_numpy("float64")  # olay(v50)

m = te.copy()
m["p"] = v50
m["po"] = b1
m["sicak"] = sicak
m = m.sort_values(["tanim", "tarih"], kind="mergesort")
g = m.groupby("tanim", observed=True)
m["ilk"] = g["tarih"].transform("min")
m["bosluk"] = ((m["tarih"] - g["tarih"].shift(1)).dt.days - 1.0).fillna(-1.0)
m["tr_son"] = m["tanim"].map(tr_son)
m["yeni"] = m["tr_son"].isna()
m["gb"] = (m["tarih"] - m["tr_son"]).dt.days - 1.0
m["dokunuldu"] = np.abs(np.log1p(m["po"]) - np.log1p(m["p"])) > 1e-12

print("=" * 78)
print("A. OLAY DUZELTMESI GUNLERE NASIL DAGILMIS? (yalniz SICAK satirlar)")
print("=" * 78)
s = m[m["sicak"]]
gd = s.groupby("tarih").agg(
    n=("p", "size"),
    dok=("dokunuldu", "sum"),
)
gd["oran"] = gd["dok"] / gd["n"]
print(f"toplam sicak satirda dokunulan = {int(gd['dok'].sum())}")
print("\nen yogun 10 gun:")
print(gd.sort_values("dok", ascending=False).head(10).to_string())

print()
print("=" * 78)
print("B. 2026-05-11 NE OLDU?")
print("=" * 78)
d = m[(m["tarih"] == "2026-05-11")]
print(f"toplam satir = {len(d)}, sicak = {int(d['sicak'].sum())}")
dd = d[d["sicak"] & d["dokunuldu"]]
print(f"olay duzeltmesi alan SICAK satir = {len(dd)}")
if len(dd):
    dd = dd.copy()
    dd["kayma"] = np.log1p(dd["po"]) - np.log1p(dd["p"])
    print(f"  kayma ort = {dd['kayma'].mean():+.4f}  toplam = {dd['kayma'].sum():+.2f}")
    print(
        f"  bunlarin {int((dd['bosluk'] > 0).sum())} tanesi IC BOSLUK donusu, "
        f"{int((dd['tarih'] == dd['ilk']).sum())} tanesi test-ilk-gunu"
    )
    print(f"  gb (train sonundan bu yana bosluk) ozeti: {dd['gb'].describe().to_dict()}")
# ic bosluk gunlerinin dagilimi
ib = m[m["sicak"] & (m["bosluk"] > 0)]
print(f"\nSICAK ic-bosluk donus satiri = {len(ib)}")
print("en yogun 8 gun:")
print(ib["tarih"].value_counts().head(8).to_string())

print()
print("=" * 78)
print("C. b_gun KESTIRIMI OLAY GUNLERINDEN NE KADAR KIRLENIYOR?")
print("=" * 78)


def gun_etkisi(tanim, gun, r):
    x = pd.DataFrame({"t": tanim, "g": gun, "r": r})
    x["c"] = x["r"] - x.groupby("t")["r"].transform("mean")
    b = x.groupby("g")["c"].mean()
    return b - b.mean()


log_guc = np.log1p(m["guc"].to_numpy("float64"))
r_ham = np.log1p(m["p"].to_numpy("float64")) - log_guc
sm = m["sicak"].to_numpy()
tanim = m["tanim"].to_numpy()
gun = m["tarih"].to_numpy()

b_ham = gun_etkisi(tanim[sm], gun[sm], r_ham[sm])
# olay satirlarini TAMAMEN DISLAYARAK
dok = m["dokunuldu"].to_numpy()
sec = sm & ~dok
b_temiz = gun_etkisi(tanim[sec], gun[sec], r_ham[sec])
# olay DUZELTILMIS (B1) ile
r_duz = np.log1p(m["po"].to_numpy("float64")) - log_guc
b_duz = gun_etkisi(tanim[sm], gun[sm], r_duz[sm])

ort = pd.DataFrame({"ham": b_ham, "olaysiz": b_temiz, "duzeltilmis": b_duz})
ort["ham-olaysiz"] = ort["ham"] - ort["olaysiz"]
print(
    f"b_gun std:  ham={b_ham.std():.6f}  olaysiz={b_temiz.std():.6f}  duzeltilmis={b_duz.std():.6f}"
)
print(f"ham vs olaysiz kor = {np.corrcoef(ort['ham'], ort['olaysiz'])[0, 1]:.6f}")
print(
    f"|ham - olaysiz| : ort={ort['ham-olaysiz'].abs().mean():.6f} "
    f"max={ort['ham-olaysiz'].abs().max():.6f}"
)
print("\nen buyuk 8 sapma:")
print(ort.reindex(ort["ham-olaysiz"].abs().sort_values(ascending=False).index).head(8).to_string())

# c formulu her varyantta
g25 = pd.read_csv(KOK / "data/raw/train.csv", dtype={"tanim": str}, parse_dates=["tarih"])
g25 = g25[(g25["tarih"] >= "2025-04-01") & (g25["tarih"] <= "2025-07-31") & (g25["tuketim"] > 0)]
rg = np.log1p(g25["tuketim"].to_numpy("float64")) - np.log1p(g25["guc"].to_numpy("float64"))
xg = pd.DataFrame({"t": g25["tanim"].to_numpy(), "gg": g25["tarih"].to_numpy()})
tamg = xg.groupby("t")["gg"].nunique()
tamg = set(tamg[tamg >= 0.9 * xg["gg"].nunique()].index)
secg = np.isin(g25["tanim"].to_numpy(), list(tamg))
b_gecen = gun_etkisi(g25["tanim"].to_numpy()[secg], g25["tarih"].to_numpy()[secg], rg[secg])

print()
for ad, b in [
    ("ham (URETIM)", b_ham),
    ("olay satirlari DISLANMIS", b_temiz),
    ("olay DUZELTILMIS", b_duz),
]:
    oran = float(b_gecen.std() / b.std())
    ia = pd.Series(b_gecen.values, index=pd.to_datetime(b_gecen.index).dayofyear)
    ib2 = pd.Series(b.values, index=pd.to_datetime(b.index).dayofyear)
    o = ia.index.intersection(ib2.index)
    kor = float(np.corrcoef(ia[o], ib2[o])[0, 1])
    print(
        f"  {ad:26s} std={b.std():.6f} oran={oran:6.3f} kor={kor:.4f} "
        f"c_formul={kor * oran:6.4f}  c_lb(x0.893)={1 + 0.893 * (kor * oran - 1):.4f}"
    )

print()
print("=" * 78)
print("D. URETIMDE UYGULANAN EKSTRA KAYMA (kirli b_gun x (c-1))")
print("=" * 78)
fark = (b_ham - b_temiz) * 0.335
print(
    f"gun basina ekstra kayma (c-1=0.335): ort={fark.mean():+.6f} "
    f"std={fark.std():.6f} max|.|={fark.abs().max():.6f}"
)
# satirlara yayilinca
say = pd.Series(gun[sm]).value_counts().sort_index()
agir = (fark.reindex(say.index) ** 2 * say).sum() / len(m)
print(f"satir-agirlikli E[kayma^2] (tum test uzerinden) = {agir:.3e}")
print(
    f"  -> UST SINIR |dMSE| ~ 2*sqrt({agir:.3e})*0.065 = "
    f"{2 * np.sqrt(agir) * 0.065:.3e}  (0.065 = gun bazli artik yanlilik olcegi)"
)
