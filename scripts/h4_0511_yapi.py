"""H4 son adim -- 2026-05-11 donus partisi: IDARI KESINTI mi, kismi gun mu?

Ayirt edici (ETIKETSIZ): eger ~1000 trafo AYNI GUN durup AYNI GUN dondulerse
bu bir veri-hatti kesintisidir ve donus gunu TAM'dir. Dagilmis duruslar ise
bireysel kesintilerin toplu geri-dolgusudur.

Ayrica: train'in SON gunlerinde trafo sayisi dusuyor mu? (kesinti train'in
kuyruguna da vurmus olabilir -- "son pencere" ozniteliklerini kirletir)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

KOK = Path(__file__).resolve().parents[1]

tr = pd.read_csv(KOK / "data/raw/train.csv", dtype={"tanim": str}, parse_dates=["tarih"])
te = pd.read_csv(KOK / "data/raw/test.csv", dtype={"tanim": str}, parse_dates=["tarih"])
tr_son = tr.groupby("tanim")["tarih"].max()

m = te.sort_values(["tanim", "tarih"], kind="mergesort").copy()
g = m.groupby("tanim", observed=True)
m["ilk"] = g["tarih"].transform("min")
m["yas"] = (m["tarih"] - m["ilk"]).dt.days
m["bosluk"] = ((m["tarih"] - g["tarih"].shift(1)).dt.days - 1.0).fillna(-1.0)
m["tr_son"] = m["tanim"].map(tr_son)
m["yeni"] = m["tr_son"].isna()

d5 = m[(m["tarih"] == "2026-05-11")]
don = d5[(d5["yas"] == 0) & (~d5["yeni"])]  # ESKI gun-0 = train'den sonra donen
print("=" * 90)
print("1) 2026-05-11'de DONEN SICAK TRAFOLARIN DURUS GUNU (train'deki son gun)")
print("=" * 90)
print(f"donen trafo = {len(don)}")
vc = don["tr_son"].value_counts().sort_values(ascending=False)
print(f"essiz durus gunu = {len(vc)}")
print("en yogun 10 durus gunu:")
print(vc.head(10).to_string())
print(f"\nen buyuk tek durus gunu payi = {vc.iloc[0] / len(don):.1%}")
ust3 = vc.head(3).sum() / len(don)
print(f"ilk 3 durus gunu payi         = {ust3:.1%}")

print()
print("=" * 90)
print("2) TRAIN'IN SON 20 GUNU -- trafo sayisi dusuyor mu?")
print("=" * 90)
gs = tr.groupby("tarih")["tanim"].nunique()
print(gs.tail(20).to_string())
tepe = gs.iloc[-120:-10].max()
print(f"\nson 120 gunun tepe trafo sayisi = {tepe}")
print(f"son gun (2026-03-31) = {gs.iloc[-1]}  -> tepeye gore {gs.iloc[-1] / tepe - 1:+.1%}")
for k in (1, 2, 3, 4, 5, 7):
    print(f"  son-{k} gun: {gs.iloc[-k]}  ({gs.iloc[-k] / tepe - 1:+.1%})")

print()
print("=" * 90)
print("3) AYNI KESINTI TEST'IN BASINDA DA VAR MI? (gun basina trafo sayisi)")
print("=" * 90)
ts = te.groupby("tarih")["tanim"].nunique()
print(f"test gun basina trafo: ilk 5 = {ts.head(5).to_dict()}")
print("2026-05-09..05-13:")
print(ts.loc["2026-05-09":"2026-05-13"].to_string())
print(f"son 3 = {ts.tail(3).to_dict()}")
print(
    f"\n05-10 -> 05-11 sicrama = {int(ts.loc['2026-05-11'] - ts.loc['2026-05-10'])} trafo "
    f"({ts.loc['2026-05-11'] / ts.loc['2026-05-10'] - 1:+.1%})"
)

print()
print("=" * 90)
print("4) BU TRAFOLAR TEST'TE 05-11'DEN SONRA KESINTISIZ MI?")
print("=" * 90)
t5 = set(don["tanim"])
alt = m[m["tanim"].isin(t5)]
kesik = alt[alt["bosluk"] > 0]
print(f"05-11'de donen {len(t5)} trafonun test icindeki ic-bosluk sayisi = {len(kesik)}")
gun_say = alt.groupby("tanim")["tarih"].nunique()
bekl = (pd.Timestamp("2026-07-31") - pd.Timestamp("2026-05-11")).days + 1
print(f"beklenen gun (05-11..07-31) = {bekl}")
print(
    f"tam olan trafo = {int((gun_say == bekl).sum())}/{len(gun_say)} "
    f"({(gun_say == bekl).mean():.1%})"
)

print()
print("=" * 90)
print("5) dMSE BANDI -- 2026-05-11 donus partisi icin uc senaryo")
print("=" * 90)
N = len(te)
gr = {
    "ESKI gun-0 bosluk 1-60g": (
        (m["yas"] == 0) & ~m["yeni"] & ((m["tarih"] - m["tr_son"]).dt.days - 1).between(1, 60),
        -0.5259,
        -0.0205,
    ),
    "ESKI gun-0 bosluk 60+g": (
        (m["yas"] == 0) & ~m["yeni"] & (((m["tarih"] - m["tr_son"]).dt.days - 1) > 60),
        -0.5576,
        -0.0020,
    ),
    "IC BOSLUK donusu": (m["bosluk"] > 0, -0.5289, -0.0011),
}
sen = {
    "URETIM (parti-kor, D=-0,529)": None,  # b = dg - dv
    "TOPLU KATILIM (D=-0,106)": -0.1060,
    "TOPLU DONUS ikiz parti20+ (D=-0,896)": -0.8956,
    "TOPLU DONUS train parti100+ (D=-0,946)": -0.9458,
}
print(f"{'senaryo':<40} {'n':>6} {'dMSE(mevcut k)':>15} {'dMSE(optimal k)':>16} {'kazanc':>10}")
for ad, dger in sen.items():
    tot_m, tot_o, nn = 0.0, 0.0, 0
    for gad, (msk, dg, dv) in gr.items():
        n5 = int((msk & (m["tarih"] == "2026-05-11")).sum())
        k = 0.6 * (dg - dv)  # uretimde uygulanan
        b = (dg - dv) if dger is None else (dger - dv)
        kopt = 0.6 * b  # ayni buzme ile optimal
        tot_m += (n5 / N) * (k**2 - 2 * k * b)
        tot_o += (n5 / N) * (kopt**2 - 2 * kopt * b)
        nn += n5
    print(f"{ad:<40} {nn:>6,} {tot_m:>+15.6f} {tot_o:>+16.6f} {tot_o - tot_m:>+10.6f}")
print("\nNOT: 'dMSE(mevcut k)' = uretimin su an bankaya yazdigi; 'dMSE(optimal k)' =")
print("o senaryo dogruysa ayni s=0,6 buzmesiyle alinabilecek. 'kazanc' ikisinin farki.")
