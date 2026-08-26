"""EKSEN 2b -- CAPRAZ DOGRULAMA: capa makinesi kis26'nin OLCULEN yanliligini
yeniden uretebiliyor mu?

Fikir: iki olcum de ayni nesneyi soruyor -- "modelin ima ettigi YoY, gercek
YoY'un ne kadar altinda?". kis26'da GERCEK YoY biliniyor (Sub-Mar 2026 vs
Sub-Mar 2025), modelin ima ettigi de onbellekten okunabiliyor. Fark, dogrudan
olculen yanliligA esit olmali. Esitse capa makinesi TESTTE de gecerlidir.

Ayrica HAVA MODELINDEN BAGIMSIZ "sicaklik-eslesmis YoY" hesaplanir: gunler
bolgesel ortalama sicakliga gore kovalanir, 2026 gunleri ayni sicaklik
kovasindaki 2025 gunleriyle karsilastirilir. Dogrusal HDD/CDD varsayimi YOK.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))
import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import tuketim_model as tm  # noqa: E402

from gridup.turkish import join_key  # noqa: E402

AILELER = ("cat", "xgb", "lgbm")
AGIRLIK = (3.0, 1.0, 1.0)
DOLULUK = 0.90
SUB = KOK / "submissions/tuketim_v55_gunolcek.csv"


def harman(z, on: str) -> np.ndarray:
    pay = sum(AGIRLIK)
    return np.mean(
        [
            sum(AGIRLIK[i] * z[f"{on}{t}_{a}"] for i, a in enumerate(AILELER)) / pay
            for t in di.TOHUMLAR
        ],
        axis=0,
    )


def main() -> int:
    print("=" * 100)
    print("EKSEN 2b -- CAPRAZ DOGRULAMA + SICAKLIK-ESLESMIS YoY")
    print("=" * 100)

    tr = pd.read_csv(KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str})
    tr["tarih"] = pd.to_datetime(tr["tarih"])
    tr["r"] = np.log1p(tr["tuketim"].clip(lower=0.0)) - np.log1p(tr["guc"])
    te = pd.read_csv(KOK / "data/raw/test.csv", encoding="utf-8", dtype={"tanim": str})
    te["tarih"] = pd.to_datetime(te["tarih"])
    te = te.merge(pd.read_csv(SUB, encoding="utf-8"), on="id", validate="one_to_one")
    te["r"] = np.log1p(te["tuketim"].clip(lower=0.0)) - np.log1p(te["guc"])
    for f in (tr, te):
        f["ilce_key"] = f["lokasyon"].str.split(">").str[-1].str.strip().map(join_key)

    ngun = tr["tarih"].nunique()
    say = tr.groupby("tanim", observed=True)["tarih"].nunique()
    panel = set(say[say >= DOLULUK * ngun].index)
    fe = tr[tr["tanim"].isin(panel)].groupby("tanim", observed=True)["r"].mean()

    def gunluk(df, sut="r", kume=None):
        k = panel if kume is None else kume
        q = df[df["tanim"].isin(k)]
        v = q[sut].to_numpy() - q["tanim"].map(fe).to_numpy()
        return pd.Series(v, index=q["tarih"].to_numpy()).groupby(level=0).mean()

    g_tr = gunluk(tr)
    g_te = gunluk(te)

    # --- bolgesel gunluk sicaklik (panel trafo sayisiyla agirlikli)
    hava = pd.read_parquet(
        KOK / "data/external/hava_gunluk.parquet",
        columns=["ilce_key", "tarih", "sicaklik_ort"],
    ).drop_duplicates(["ilce_key", "tarih"])
    hava["tarih"] = pd.to_datetime(hava["tarih"])
    ag = tr[tr["tanim"].isin(panel)].drop_duplicates("tanim").groupby("ilce_key").size()
    hv = hava[hava["ilce_key"].isin(ag.index)].copy()
    hv["w"] = hv["ilce_key"].map(ag).to_numpy()
    T = hv.groupby("tarih").apply(
        lambda x: float(np.average(x["sicaklik_ort"], weights=x["w"])), include_groups=False
    )

    print("\n### AY AY PANEL SEVIYESI (trafo etkisi cikarilmis)")
    aylik = pd.concat([g_tr, g_te]).groupby(lambda x: pd.Timestamp(x).to_period("M")).mean()
    Ta = T.groupby(lambda x: pd.Timestamp(x).to_period("M")).mean()
    print(f"  {'ay':10}{'seviye':>9}{'T':>7}   |  {'ay':10}{'seviye':>9}{'T':>7}{'YoY':>9}")
    a25 = [a for a in aylik.index if a.year == 2025]
    for a in a25:
        b_ = a + 12
        sag = (
            (f"{str(b_):10}{aylik[b_]:+9.4f}{Ta.get(b_, np.nan):7.2f}{aylik[b_] - aylik[a]:+9.4f}")
            if b_ in aylik.index
            else ""
        )
        print(f"  {str(a):10}{aylik[a]:+9.4f}{Ta.get(a, np.nan):7.2f}   |  {sag}")

    # --- SICAKLIK-ESLESMIS YoY (hava modeli YOK)
    print("\n### SICAKLIK-ESLESMIS YoY  (2 C'lik kovalar, 2026 gun sayisiyla agirlikli)")

    def eslesmis(a1, b1, a2, b2, sr2=None):
        s2 = g_te if sr2 is not None else g_tr
        m1 = (g_tr.index >= pd.Timestamp(a1)) & (g_tr.index <= pd.Timestamp(b1))
        m2 = (s2.index >= pd.Timestamp(a2)) & (s2.index <= pd.Timestamp(b2))
        x1, x2 = g_tr[m1], s2[m2]
        k1 = np.floor(T.reindex(x1.index).to_numpy() / 2.0)
        k2 = np.floor(T.reindex(x2.index).to_numpy() / 2.0)
        d1 = pd.Series(x1.to_numpy(), index=k1).groupby(level=0).mean()
        d2 = pd.Series(x2.to_numpy(), index=k2).groupby(level=0).mean()
        n2 = pd.Series(1, index=k2).groupby(level=0).sum()
        ort = d1.index.intersection(d2.index)
        w = n2.reindex(ort).to_numpy(dtype=float)
        kap = float(n2.reindex(ort).sum() / n2.sum())
        ham = float(x2.mean() - x1.mean())
        esl = float(np.average((d2.reindex(ort) - d1.reindex(ort)).to_numpy(), weights=w))
        return ham, esl, kap, len(ort)

    for et, a1, b1, a2, b2, ist in (
        ("Oca-Mar  2025->2026", "2025-01-01", "2025-03-31", "2026-01-01", "2026-03-31", None),
        ("Sub-Mar  2025->2026", "2025-02-01", "2025-03-31", "2026-02-01", "2026-03-31", None),
        ("Nis-Tem  2025->v55 2026", "2025-04-01", "2025-07-31", "2026-04-01", "2026-07-31", 1),
    ):
        ham, esl, kap, nk = eslesmis(a1, b1, a2, b2, ist)
        print(
            f"  {et:26} ham {ham:+.4f}   SICAKLIK-ESLESMIS {esl:+.4f}"
            f"   ({nk} kova, kapsam {kap:.2f})"
        )

    # --- kis26 MODELININ ima ettigi YoY  (capa makinesi vs dogrudan olcum)
    print("\n### CAPRAZ DOGRULAMA -- kis26")
    egitim, test = d.cerceveleri_kur()
    tm.kategorik_kodla(egitim, test)
    z = np.load(KOK / "data/interim/deney/sicak_tahmin.npz")
    _, dog, gercek, soguk = di.blok_parcalari(egitim, "kis26")
    dg = dog[~soguk].reset_index(drop=True)
    lg = np.log1p(dg["guc"].to_numpy(dtype="float64"))
    r_mod = harman(z, "kis26_") - lg
    kis = pd.DataFrame(
        {
            "tanim": dg["tanim"].to_numpy(),
            "tarih": pd.to_datetime(dg["tarih"]).to_numpy(),
            "r": r_mod,
            "g": np.log1p(gercek[~soguk]) - lg,
        }
    )
    kis_p = kis[kis["tanim"].isin(panel)]
    print(
        f"  kis26 sicak satirlarin {len(kis_p):,}'i sabit panelde"
        f" ({kis_p['tanim'].nunique():,} trafo)"
    )
    for et, a2, b2, a1, b1 in (
        ("Sub-Mar", "2026-02-01", "2026-03-31", "2025-02-01", "2025-03-31"),
        ("Oca-Mar", "2026-01-01", "2026-03-31", "2025-01-01", "2025-03-31"),
    ):
        m = (kis_p["tarih"] >= pd.Timestamp(a2)) & (kis_p["tarih"] <= pd.Timestamp(b2))
        alt = kis_p[m]
        fe_v = alt["tanim"].map(fe).to_numpy()
        mod = float((alt["r"].to_numpy() - fe_v).mean())
        ger = float((alt["g"].to_numpy() - fe_v).mean())
        ge25 = float(
            g_tr[(g_tr.index >= pd.Timestamp(a1)) & (g_tr.index <= pd.Timestamp(b1))].mean()
        )
        print(
            f"  {et}: 2025 gercek {ge25:+.4f} | 2026 gercek {ger:+.4f} (YoY {ger - ge25:+.4f})"
            f" | kis26 MODEL {mod:+.4f} (ima YoY {mod - ge25:+.4f})"
            f" -> yanlilik {ger - mod:+.4f}"
        )
    print(
        "  (yanlilik = gercek YoY - ima YoY  <=> dogrudan olculen b; ESITSE capa makinesi saglam)"
    )

    # --- v55 ima YoY, panelde ve tum sicakta
    print("\n### v55 IMA YoY")
    for ad, kume in (
        ("sabit panel", panel),
        ("tum ortak sicak", set(tr["tanim"]) & set(te["tanim"])),
    ):
        fe2 = tr[tr["tanim"].isin(kume)].groupby("tanim", observed=True)["r"].mean()
        q1 = tr[
            tr["tanim"].isin(kume) & (tr["tarih"] >= "2025-04-01") & (tr["tarih"] <= "2025-07-31")
        ]
        q2 = te[te["tanim"].isin(kume)]
        v1 = float((q1["r"] - q1["tanim"].map(fe2).to_numpy()).mean())
        v2 = float((q2["r"] - q2["tanim"].map(fe2).to_numpy()).mean())
        print(
            f"  {ad:18} 2025 {v1:+.4f}  v55 2026 {v2:+.4f}  ima YoY {v2 - v1:+.4f}"
            f"  ({len(kume):,} trafo, 2025 satir {len(q1):,} / 2026 {len(q2):,})"
        )

    # --- ULUSAL, hava-arindirilmis
    print("\n### ULUSAL YUK -- bolgesel hava ile arindirilmis")
    ul = pd.read_parquet(KOK / "data/external/epias/tuketim_saatlik.parquet")
    ul["tarih"] = pd.to_datetime(ul["zaman"]).dt.normalize()
    ug = np.log(ul.groupby("tarih")["consumption"].sum())
    ug = ug[(ug.index >= pd.Timestamp("2024-01-01")) & (ug.index <= pd.Timestamp("2026-07-31"))]
    tt = T.reindex(ug.index)
    ok = tt.notna().to_numpy()
    ug, tt = ug[ok], tt[ok]
    hdd = (18.0 - tt).clip(lower=0.0).to_numpy()
    cdd = (tt - 22.0).clip(lower=0.0).to_numpy()
    hg = pd.get_dummies(ug.index.dayofweek, prefix="h", drop_first=True).to_numpy(dtype=float)
    ay_d = pd.get_dummies(ug.index.month, prefix="a", drop_first=True).to_numpy(dtype=float)
    yil = (ug.index >= pd.Timestamp("2025-08-01")).to_numpy(dtype=float)  # yer tutucu
    X = np.column_stack([np.ones(len(ug)), hdd, cdd, hg, ay_d])
    ko = np.linalg.lstsq(X, ug.to_numpy(), rcond=None)[0]
    art = pd.Series(ug.to_numpy() - X @ ko, index=ug.index)
    print(f"  n={len(ug)}  HDD {ko[1]:+.5f}  CDD {ko[2]:+.5f}  (ay+haftagunu kuklalari ile)")

    def pen(sr, a, b_):
        return float(sr[(sr.index >= pd.Timestamp(a)) & (sr.index <= pd.Timestamp(b_))].mean())

    for et, a, b_ in (("Oca-Mar", "01-01", "03-31"), ("Nis-Tem", "04-01", "07-31")):
        h1, h2 = pen(ug, f"2025-{a}", f"2025-{b_}"), pen(ug, f"2026-{a}", f"2026-{b_}")
        r1, r2 = pen(art, f"2025-{a}", f"2025-{b_}"), pen(art, f"2026-{a}", f"2026-{b_}")
        print(f"  {et}: ham YoY {h2 - h1:+.4f}   HAVA-ARINDIRILMIS YoY {r2 - r1:+.4f}")
    _ = yil
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
