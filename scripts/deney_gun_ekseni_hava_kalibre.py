# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""KALIBRE: kis26'nin 2026 parcasinda c_true vs c_ref(364-gun) vs c_hava."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(r"C:\Users\cemmo\Documents\Datahon")
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))
import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402

from gridup.turkish import join_key  # noqa: E402

DIZIN = KOK / "data/interim/aile_onbellek"
TOHUMLAR = (1000, 1001, 1002)
AILELER = ("cat", "xgb", "lgbm", "sinir_agi")
AGIRLIK = (3.0, 1.0, 1.0, 1.4)
DOLULUK = 0.90


def blend(blok, tohum):
    return sum(
        w * np.load(DIZIN / f"{blok}_{tohum}_{a}_uretim.npy").astype("float64")
        for a, w in zip(AILELER, AGIRLIK, strict=True)
    ) / sum(AGIRLIK)


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
    sh = float(np.sqrt((res**2).sum() / (n - 2) / (xc**2).sum()))
    return b, float(np.corrcoef(xc, yc)[0, 1]), sh, n


hava = pd.read_parquet(
    KOK / "data/external/hava_gunluk.parquet", columns=["ilce_key", "tarih", "sicaklik_ort"]
).drop_duplicates(["ilce_key", "tarih"])
hava["tarih"] = pd.to_datetime(hava["tarih"])
hava["cdd22"] = (hava["sicaklik_ort"] - 22.0).clip(lower=0.0)
hava["hdd18"] = (18.0 - hava["sicaklik_ort"]).clip(lower=0.0)


def hava_serisi(bas, son, agirlik):
    h = hava[(hava["tarih"] >= bas) & (hava["tarih"] <= son)].merge(
        agirlik.rename("w"), left_on="ilce_key", right_index=True, how="inner"
    )
    return h.groupby("tarih").apply(
        lambda q: pd.Series(
            {
                "T": float(np.average(q["sicaklik_ort"], weights=q["w"])),
                "cdd22": float(np.average(q["cdd22"], weights=q["w"])),
                "hdd18": float(np.average(q["hdd18"], weights=q["w"])),
            }
        ),
        include_groups=False,
    )


def tasarim(w):
    x = pd.DataFrame(index=w.index)
    x["cdd22"] = w["cdd22"]
    x["cdd22_2"] = w["cdd22"] ** 2
    x["hdd18"] = w["hdd18"]
    x["hdd18_2"] = w["hdd18"] ** 2
    x["T"] = w["T"]
    hg = pd.Series(w.index.dayofweek, index=w.index)
    for k in range(1, 7):
        x[f"hg{k}"] = (hg == k).astype(float)
    return x


def tasi(w_fit, y_fit, w_app):
    X, Xa = tasarim(w_fit), tasarim(w_app)
    o = pd.concat([X, y_fit.rename("y")], axis=1).dropna()
    A = np.c_[np.ones(len(o)), o[X.columns].to_numpy()]
    beta, *_ = np.linalg.lstsq(A, o["y"].to_numpy(), rcond=None)
    return pd.Series(np.c_[np.ones(len(Xa)), Xa.to_numpy()] @ beta, index=Xa.index)


t0 = time.time()
egitim, test = d.cerceveleri_kur()
print(f"  cerceveler {time.time() - t0:.0f} sn")
_, dog, gercek, soguk = di.blok_parcalari(egitim, "kis26")
dg = dog[~soguk].copy()
dg["tarih"] = pd.to_datetime(dg["tarih"])
lg = np.log1p(dg["guc"].to_numpy(dtype="float64"))
dg["m"] = np.mean([blend("kis26", t) for t in TOHUMLAR], axis=0) - lg
dg["a"] = np.log1p(np.clip(gercek[~soguk], 0.0, None)) - lg
dg["tanim"] = dg["tanim"].astype(str)
p = dg["lokasyon"].str.split(">") if "lokasyon" in dg.columns else None
print(
    f"  kis26 sicak {len(dg):,} satir  {dg['tarih'].min():%Y-%m-%d}..{dg['tarih'].max():%Y-%m-%d}"
)

tr = pd.read_csv(KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str})
tr["tarih"] = pd.to_datetime(tr["tarih"])
tr["a"] = np.log1p(tr["tuketim"].clip(lower=0.0)) - np.log1p(tr["guc"])
pp = tr["lokasyon"].str.split(">")
tr["ilce_key"] = pp.str[-1].str.strip().map(join_key)
ilce_map = tr.drop_duplicates("tanim").set_index("tanim")["ilce_key"]

print("\n" + "=" * 104)
print("KALIBRE -- 2026 gunleri; c_true (etiketli) vs c_ref (364-gun) vs c_hava (hava tasimasi)")
print("=" * 104)
print(
    f"{'altkume':>22}{'gun':>5}{'c_true':>9}{'SH':>7}{'c_ref':>9}{'SH':>7}{'c_hava':>9}{'SH':>7}"
    f"{'|ref-true|':>11}{'|hava-true|':>12}"
)

for ad, bas, son in (
    ("2026-01..03 TAMAMI", "2026-01-01", "2026-03-31"),
    ("2026-02..03 AY GORULDU", "2026-02-01", "2026-03-31"),
    ("2026-01 AY YOK", "2026-01-01", "2026-01-31"),
):
    k = dg[(dg["tarih"] >= bas) & (dg["tarih"] <= son)]
    tut_m = panel_tanim(k)
    rb = (pd.Timestamp(bas) - pd.Timedelta(days=364)).strftime("%Y-%m-%d")
    rs = (pd.Timestamp(son) - pd.Timedelta(days=364)).strftime("%Y-%m-%d")
    ref = tr[(tr["tarih"] >= rb) & (tr["tarih"] <= rs)]
    tut = tut_m & panel_tanim(ref)
    m_d = gun_ort(k, "m", tut)
    a_d = gun_ort(k, "a", tut)
    r_d = gun_ort(ref, "a", tut)
    r_d.index = r_d.index + pd.Timedelta(days=364)
    agir = ilce_map.reindex(sorted(tut)).value_counts(normalize=True)
    w_ref = hava_serisi(rb, rs, agir)
    w_ref.index = w_ref.index + pd.Timedelta(days=364)
    w_cur = hava_serisi(bas, son, agir)
    y_ref = gun_ort(ref, "a", tut)
    y_ref.index = y_ref.index + pd.Timedelta(days=364)
    h_d = tasi(w_ref, y_ref, w_cur)
    ct, _, sht, ng = ols(m_d, a_d)
    cr, _, shr, _ = ols(m_d, r_d)
    ch, _, shh, _ = ols(m_d, h_d)
    print(
        f"{ad:>22}{ng:5d}{ct:+9.3f}{sht:7.3f}{cr:+9.3f}{shr:7.3f}{ch:+9.3f}{shh:7.3f}"
        f"{abs(cr - ct):11.3f}{abs(ch - ct):12.3f}"
    )
    print(
        f"{'':>22}  panel {len(tut):,} trafo | std model {m_d.std():.4f}"
        f" gercek {a_d.std():.4f} ref2025 {r_d.std():.4f} hava {h_d.std():.4f}"
    )
