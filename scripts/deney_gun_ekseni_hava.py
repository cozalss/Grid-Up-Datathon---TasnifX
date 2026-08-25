# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""TEST PENCERESINDE c_hava'nin SAGLAMLIGI + LB etkisi."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(r"C:\Users\cemmo\Documents\Datahon")
sys.path.insert(0, str(KOK / "src"))
from gridup.turkish import join_key  # noqa: E402

DOLULUK = 0.90
T25 = ("2025-04-02", "2025-08-01")
T26 = ("2026-04-01", "2026-07-31")


def panel_tanim(df):
    n = df["tarih"].nunique()
    s = df.groupby("tanim", observed=True)["tarih"].nunique()
    return set(s[s >= DOLULUK * n].index)


def gun_ort(df, deger, tut):
    q = df[df["tanim"].isin(tut)].copy()
    q[deger] = q[deger] - q.groupby("tanim", observed=True)[deger].transform("mean")
    return q.groupby("tarih")[deger].mean()


def ols(x, y):
    o = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    xc, yc = o["x"] - o["x"].mean(), o["y"] - o["y"].mean()
    b = float((xc * yc).sum() / (xc * xc).sum())
    n = len(o)
    res = yc - b * xc
    return (
        b,
        float(np.corrcoef(xc, yc)[0, 1]),
        float(np.sqrt((res**2).sum() / (n - 2) / (xc**2).sum())),
        n,
    )


tr = pd.read_csv(KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str})
tr["tarih"] = pd.to_datetime(tr["tarih"])
tr["a"] = np.log1p(tr["tuketim"].clip(lower=0.0)) - np.log1p(tr["guc"])
te = pd.read_csv(KOK / "data/raw/test.csv", encoding="utf-8", dtype={"tanim": str})
te["tarih"] = pd.to_datetime(te["tarih"])
te = te.merge(
    pd.read_csv(KOK / "submissions/tuketim_v50_ham30.csv", encoding="utf-8"),
    on="id",
    validate="one_to_one",
)
te["soguk"] = ~te["tanim"].isin(set(tr["tanim"].unique()))
ts = te[~te["soguk"]].copy()
ts["r"] = np.log1p(ts["tuketim"].clip(lower=0.0)) - np.log1p(ts["guc"])
for f in (tr, ts):
    f["ilce_key"] = f["lokasyon"].str.split(">").str[-1].str.strip().map(join_key)

hava = pd.read_parquet(
    KOK / "data/external/hava_gunluk.parquet",
    columns=["ilce_key", "tarih", "sicaklik_ort", "sicaklik_max", "sicaklik_min"],
).drop_duplicates(["ilce_key", "tarih"])
hava["tarih"] = pd.to_datetime(hava["tarih"])
for b in (18, 20, 22, 24):
    hava[f"cdd{b}"] = (hava["sicaklik_ort"] - b).clip(lower=0.0)
hava["hdd18"] = (18.0 - hava["sicaklik_ort"]).clip(lower=0.0)
hava["cddmax26"] = (hava["sicaklik_max"] - 26.0).clip(lower=0.0)

ref25 = tr[(tr["tarih"] >= T25[0]) & (tr["tarih"] <= T25[1])]
tut = panel_tanim(ts) & panel_tanim(ref25)
print(f"ORTAK panel {len(tut):,} trafo")
m_d = gun_ort(ts, "r", tut)
a25 = gun_ort(ref25, "a", tut)
a25.index = a25.index + pd.Timedelta(days=364)
agir = (
    tr[tr["tanim"].isin(tut)]
    .drop_duplicates("tanim")
    .set_index("tanim")["ilce_key"]
    .value_counts(normalize=True)
)

KOL = [
    "sicaklik_ort",
    "sicaklik_max",
    "sicaklik_min",
    "cdd18",
    "cdd20",
    "cdd22",
    "cdd24",
    "hdd18",
    "cddmax26",
]


def hs(bas, son):
    h = hava[(hava["tarih"] >= bas) & (hava["tarih"] <= son)].merge(
        agir.rename("w"), left_on="ilce_key", right_index=True
    )
    g = h.groupby("tarih").apply(
        lambda q: pd.Series({k: float(np.average(q[k], weights=q["w"])) for k in KOL}),
        include_groups=False,
    )
    return g


w25 = hs(*T25)
w25.index = w25.index + pd.Timedelta(days=364)
w26 = hs(*T26)


def tas(kols, hg=True):
    def X(w):
        x = pd.DataFrame(index=w.index)
        for k in kols:
            x[k] = w[k]
            if k.startswith("cdd"):
                x[k + "_2"] = w[k] ** 2
        if hg:
            g = pd.Series(w.index.dayofweek, index=w.index)
            for j in range(1, 7):
                x[f"hg{j}"] = (g == j).astype(float)
        return x

    Xf, Xa = X(w25), X(w26)
    o = pd.concat([Xf, a25.rename("y")], axis=1).dropna()
    A = np.c_[np.ones(len(o)), o[Xf.columns].to_numpy()]
    beta, *_ = np.linalg.lstsq(A, o["y"].to_numpy(), rcond=None)
    fit = pd.Series(A @ beta, index=o.index)
    r2 = 1 - np.var(o["y"] - fit) / np.var(o["y"])
    return pd.Series(np.c_[np.ones(len(Xa)), Xa.to_numpy()] @ beta, index=Xa.index), fit, r2


print("\n" + "=" * 100)
print("TEST PENCERESI -- c kestirimleri")
print("=" * 100)
print(f"  std(model) {m_d.std():.4f}   std(2025 gerceklesen) {a25.std():.4f}")
cr, kr, shr, n = ols(m_d, a25)
print(f"  c_ref (364-gun NAIF)              {cr:+.3f}  SH {shr:.3f}  kor {kr:+.3f}  n={n}")
print(
    f"\n  {'tasarim':>34}{'R2_2025':>9}{'std_fit25':>11}{'std_kes26':>11}{'oran':>7}{'c_hava':>9}{'SH':>7}"
)
sonuc = []
for ad, kols, hg in (
    ("cdd22+kare+hdd+T+haftagunu", ["cdd22", "hdd18", "sicaklik_ort"], True),
    ("cdd22+kare+hdd+T (haftagunu YOK)", ["cdd22", "hdd18", "sicaklik_ort"], False),
    ("YALNIZ cdd22 dogrusal", ["cdd22"], False),
    ("cdd24+kare+hdd+T+hg", ["cdd24", "hdd18", "sicaklik_ort"], True),
    ("cdd20+cdd24+hdd+T+hg", ["cdd20", "cdd24", "hdd18", "sicaklik_ort"], True),
    ("sicaklik_max tabanli (cddmax26)", ["cddmax26", "hdd18", "sicaklik_max"], True),
    ("T + T^2 (cdd yok)", ["sicaklik_ort"], True),
):
    kes, fit, r2 = tas(kols, hg)
    if "sicaklik_ort" in kols and len(kols) == 1 and ad.startswith("T +"):
        pass
    c, k, sh, _ = ols(m_d, kes)
    sonuc.append(c)
    print(
        f"  {ad:>34}{r2:9.3f}{fit.std():11.4f}{kes.std():11.4f}{kes.std() / fit.std():7.3f}{c:+9.3f}{sh:7.3f}"
    )
print(f"\n  c_hava medyani {np.median(sonuc):+.3f}  aralik [{min(sonuc):+.3f}, {max(sonuc):+.3f}]")

# ---- gun kirpma ------------------------------------------------------------
kes, fit, _ = tas(["cdd22", "hdd18", "sicaklik_ort"], True)
print("\n  GUN KIRPMA (modelden en uzak K gun atilir)")
o = pd.concat([m_d.rename("m"), a25.rename("a"), kes.rename("h")], axis=1).dropna()
res = o["a"] - o["m"] * ols(o["m"], o["a"])[0]
sira = res.abs().sort_values(ascending=False).index
print(f"  {'K':>4}{'c_ref':>9}{'c_hava':>9}")
for K in (0, 1, 5, 10, 25, 50):
    q = o.drop(index=sira[:K])
    print(f"  {K:>4}{ols(q['m'], q['a'])[0]:+9.3f}{ols(q['m'], q['h'])[0]:+9.3f}")

# ---- LB etkisi -------------------------------------------------------------
print("\n" + "=" * 100)
print("LB ETKISI  MSLE(c) = MSLE(c*) + A*(c-c*)^2,  A = pay_sicak * sigma_model^2")
print("=" * 100)
SIG, PAY, MSLE0 = 0.1675, 0.7784, 1.034493
A = PAY * SIG**2
RM0 = MSLE0**0.5
print(f"  A {A:.6f}   v50 MSLE {MSLE0:.6f} -> RMSLE {RM0:.5f}")
print(f"  {'c*':>7}{'kirilma c':>11}{'v55 c=1,49':>13}{'v57 c=1,75':>13}{'en iyi c':>10}")
for cs in (1.161, 1.24, 1.30, 1.49, 1.60, 1.636):
    d55 = A * ((1 - cs) ** 2 - (1.49 - cs) ** 2)
    d57 = A * ((1 - cs) ** 2 - (1.75 - cs) ** 2)
    r55 = (MSLE0 - d55) ** 0.5 - RM0
    r57 = (MSLE0 - d57) ** 0.5 - RM0
    print(
        f"  {cs:7.3f}{2 * cs - 1:11.3f}{r55:+13.5f}{r57:+13.5f}{(MSLE0 - A * (1 - cs) ** 2) ** 0.5 - RM0:+10.5f}"
    )

print("\n" + "=" * 100)
print("AYLIK -- iddianin tablosuna hava-tasimali 2026 kestirimi eklenir")
print("=" * 100)
kes, fit, _ = tas(["cdd22", "hdd18", "sicaklik_ort"], True)
tab = pd.DataFrame({"model": m_d, "ref2025": a25, "hava26": kes}).dropna()
tab = tab - tab.mean()
print(
    f"  {'ay':>4}{'model 2026':>12}{'2025 gercek':>13}{'HAVA-2026':>11}{'model-hava':>12}{'ref-hava':>10}"
)
for a_, q in tab.groupby(tab.index.month):
    print(
        f"  {a_:>4}{q['model'].mean():+12.3f}{q['ref2025'].mean():+13.3f}{q['hava26'].mean():+11.3f}"
        f"{q['model'].mean() - q['hava26'].mean():+12.3f}{q['ref2025'].mean() - q['hava26'].mean():+10.3f}"
    )
