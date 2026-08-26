"""EKSEN 2b -- SAGLAMLIK: v55'in IMA ETTIGI HAVA-NOTR BUYUME ne kadar saglam?

Kritik iddia:
    b_test = g_gercek(Nis-Tem)  -  g_v55(Nis-Tem)
    g_v55 = v55'in SICAKLIK-ESLESMIS ima ettigi buyume.

Kestirici:  g = ort_gun[ L_v55(gun) - f2025(T(gun)) ]
f2025, 2025'in AYNI penceresindeki sicaklik-seviye tepkisi. Bu kestirici
v55'in hava tepkisinin DOGRU olmasini gerektirmez; yalnizca f2025'in
2026 karsi-olgusunu temsil etmesini gerektirir.

Burada kestirici her yonden sarsiliyor: kova genisligi, 2025 havuzu,
ay ay, panel tanimi, agirliklandirma. Ayrica ayni makine kis26'ya
uygulanip DOGRUDAN olculen yanlilikla karsilastiriliyor.
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
SUB = KOK / "submissions/tuketim_v55_gunolcek.csv"
N25 = (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-31"))


def main() -> int:
    print("=" * 100)
    print("EKSEN 2b -- v55'IN IMA ETTIGI BUYUMENIN SAGLAMLIGI")
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

    hava = pd.read_parquet(
        KOK / "data/external/hava_gunluk.parquet",
        columns=["ilce_key", "tarih", "sicaklik_ort"],
    ).drop_duplicates(["ilce_key", "tarih"])
    hava["tarih"] = pd.to_datetime(hava["tarih"])

    # --- PANEL TANIMLARI ---
    ngun = tr["tarih"].nunique()
    say = tr.groupby("tanim", observed=True)["tarih"].nunique()
    n25 = (
        tr[(tr["tarih"] >= N25[0]) & (tr["tarih"] <= N25[1])]
        .groupby("tanim", observed=True)["tarih"]
        .nunique()
    )
    n26 = te.groupby("tanim", observed=True)["tarih"].nunique()
    paneller = {
        "P15 (15 ay >=%90)": set(say[say >= 0.90 * ngun].index) & set(te["tanim"]),
        "PESLI (2025N-T>=100 & test>=100)": set(n25[n25 >= 100].index) & set(n26[n26 >= 100].index),
        "PESLI-gevsek (>=60 & >=60)": set(n25[n25 >= 60].index) & set(n26[n26 >= 60].index),
    }
    for ad, k in paneller.items():
        print(f"  {ad:34} {len(k):,} trafo")
    tum_sicak = set(tr["tanim"]) & set(te["tanim"])
    print(
        f"  {'TUM ortak sicak':34} {len(tum_sicak):,} trafo"
        f"  (test sicak satirlarin payi: "
        f"{te['tanim'].isin(tum_sicak).mean():.3f})"
    )

    def seriler(kume):
        fe = tr[tr["tanim"].isin(kume)].groupby("tanim", observed=True)["r"].mean()
        q1 = tr[tr["tanim"].isin(kume)]
        s1 = (
            pd.Series(
                (q1["r"] - q1["tanim"].map(fe).to_numpy()).to_numpy(), index=q1["tarih"].to_numpy()
            )
            .groupby(level=0)
            .mean()
        )
        q2 = te[te["tanim"].isin(kume)]
        s2 = (
            pd.Series(
                (q2["r"] - q2["tanim"].map(fe).to_numpy()).to_numpy(), index=q2["tarih"].to_numpy()
            )
            .groupby(level=0)
            .mean()
        )
        ag = q1.drop_duplicates("tanim").groupby("ilce_key").size()
        hv = hava[hava["ilce_key"].isin(ag.index)].copy()
        hv["w"] = hv["ilce_key"].map(ag).to_numpy()
        T = hv.groupby("tarih").apply(
            lambda x: float(np.average(x["sicaklik_ort"], weights=x["w"])), include_groups=False
        )
        return s1, s2, T

    def esles(s_hedef, s_kaynak, T, gen, hedef_pen, kaynak_pen):
        m2 = (s_hedef.index >= hedef_pen[0]) & (s_hedef.index <= hedef_pen[1])
        m1 = (s_kaynak.index >= kaynak_pen[0]) & (s_kaynak.index <= kaynak_pen[1])
        x2, x1 = s_hedef[m2], s_kaynak[m1]
        k2 = np.floor(T.reindex(x2.index).to_numpy() / gen)
        k1 = np.floor(T.reindex(x1.index).to_numpy() / gen)
        d1 = pd.Series(x1.to_numpy(), index=k1).groupby(level=0).mean()
        d2 = pd.Series(x2.to_numpy(), index=k2).groupby(level=0).mean()
        n2 = pd.Series(1, index=k2).groupby(level=0).size()
        ort = d1.index.intersection(d2.index)
        if len(ort) == 0:
            return np.nan, 0.0
        w = n2.reindex(ort).to_numpy(dtype=float)
        return float(np.average((d2.reindex(ort) - d1.reindex(ort)).to_numpy(), weights=w)), float(
            n2.reindex(ort).sum() / n2.sum()
        )

    print("\n### A) g_v55 (Nis-Tem 2026) -- SARSMA TABLOSU")
    print(f"  {'panel':34}{'2025 havuzu':16}{'kova':>6}{'g_v55':>9}{'kapsam':>8}")
    kayit = {}
    for ad, kume in paneller.items():
        s1, s2, T = seriler(kume)
        for hv_ad, pen in (
            ("Nis-Tem 2025", N25),
            ("Nis-Kas 2025", (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-11-30"))),
            ("Mar-Agu 2025", (pd.Timestamp("2025-03-01"), pd.Timestamp("2025-08-31"))),
        ):
            for gen in (1.0, 2.0, 3.0):
                g, kap = esles(
                    s2, s1, T, gen, (pd.Timestamp("2026-04-01"), pd.Timestamp("2026-07-31")), pen
                )
                print(f"  {ad:34}{hv_ad:16}{gen:6.0f}{g:+9.4f}{kap:8.2f}")
                kayit[(ad, hv_ad, gen)] = g
        # ay ay (2 C kova, havuz Nis-Tem 2025)
        s = []
        for a in ("04", "05", "06", "07"):
            son = {"04": 30, "05": 31, "06": 30, "07": 31}[a]
            g, _ = esles(
                s2, s1, T, 2.0, (pd.Timestamp(f"2026-{a}-01"), pd.Timestamp(f"2026-{a}-{son}")), N25
            )
            s.append(f"{a}:{g:+.4f}")
        print("      ay ay -> " + "  ".join(s))

    print("\n### B) AYNI MAKINE kis26'da -- DOGRUDAN OLCUMLE KARSILASTIRMA")
    egitim, test = d.cerceveleri_kur()
    tm.kategorik_kodla(egitim, test)
    z = np.load(KOK / "data/interim/deney/sicak_tahmin.npz")
    _, dog, gercek, soguk = di.blok_parcalari(egitim, "kis26")
    dg = dog[~soguk].reset_index(drop=True)
    lg = np.log1p(dg["guc"].to_numpy(dtype="float64"))
    pay = sum(AGIRLIK)
    r_mod = (
        np.mean(
            [
                sum(AGIRLIK[i] * z[f"kis26_{t}_{a}"] for i, a in enumerate(AILELER)) / pay
                for t in di.TOHUMLAR
            ],
            axis=0,
        )
        - lg
    )
    kis = pd.DataFrame(
        {
            "tanim": dg["tanim"].to_numpy(),
            "tarih": pd.to_datetime(dg["tarih"]).to_numpy(),
            "r": r_mod,
            "g": np.log1p(gercek[~soguk]) - lg,
        }
    )
    for ad, kume in paneller.items():
        alt = kis[kis["tanim"].isin(kume)]
        if len(alt) < 1000:
            continue
        s1, _, T = seriler(kume)
        fe = tr[tr["tanim"].isin(kume)].groupby("tanim", observed=True)["r"].mean()
        fev = alt["tanim"].map(fe).to_numpy()
        s_mod = (
            pd.Series(alt["r"].to_numpy() - fev, index=alt["tarih"].to_numpy())
            .groupby(level=0)
            .mean()
        )
        s_ger = (
            pd.Series(alt["g"].to_numpy() - fev, index=alt["tarih"].to_numpy())
            .groupby(level=0)
            .mean()
        )
        for et, p0, p1, kp in (
            (
                "Sub-Mar",
                "2026-02-01",
                "2026-03-31",
                (pd.Timestamp("2025-02-01"), pd.Timestamp("2025-03-31")),
            ),
            (
                "Ara-Mar",
                "2025-12-01",
                "2026-03-31",
                (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-31")),
            ),
        ):
            hp = (pd.Timestamp(p0), pd.Timestamp(p1))
            g_mod, _ = esles(s_mod, s1, T, 2.0, hp, kp)
            g_ger, _ = esles(s_ger, s1, T, 2.0, hp, kp)
            m = (alt["tarih"] >= hp[0]) & (alt["tarih"] <= hp[1])
            dogrudan = float((alt.loc[m, "g"] - alt.loc[m, "r"]).mean())
            print(
                f"  {ad:34}{et:9} g_gercek {g_ger:+.4f}  g_model {g_mod:+.4f}"
                f"  fark {g_ger - g_mod:+.4f}  | DOGRUDAN b {dogrudan:+.4f}"
            )

    print("\n### C) g_gercek(Nis-Tem 2026) icin CAPALAR")
    s1, s2, T = seriler(paneller["P15 (15 ay >=%90)"])
    for et, hp, kp in (
        ("Oca-Mar 26 vs Oca-Mar 25", ("2026-01-01", "2026-03-31"), ("2025-01-01", "2025-03-31")),
        ("Sub-Mar 26 vs Sub-Mar 25", ("2026-02-01", "2026-03-31"), ("2025-02-01", "2025-03-31")),
        ("Mar 26 vs Mar 25", ("2026-03-01", "2026-03-31"), ("2025-03-01", "2025-03-31")),
        ("Oca 26 vs Oca 25", ("2026-01-01", "2026-01-31"), ("2025-01-01", "2025-01-31")),
    ):
        g, kap = esles(
            s1,
            s1,
            T,
            2.0,
            (pd.Timestamp(hp[0]), pd.Timestamp(hp[1])),
            (pd.Timestamp(kp[0]), pd.Timestamp(kp[1])),
        )
        print(f"  {et:28} sicaklik-eslesmis YoY {g:+.4f}  (kapsam {kap:.2f})")

    ul = pd.read_parquet(KOK / "data/external/epias/tuketim_saatlik.parquet")
    ul["tarih"] = pd.to_datetime(ul["zaman"]).dt.normalize()
    ug = np.log(ul.groupby("tarih")["consumption"].sum())
    su = pd.Series(ug.to_numpy(), index=ug.index)
    gu, _ = esles(
        su,
        su,
        T,
        2.0,
        (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31")),
        (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-31")),
    )
    gn, _ = esles(
        su,
        su,
        T,
        2.0,
        (pd.Timestamp("2026-04-01"), pd.Timestamp("2026-07-31")),
        (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-31")),
    )
    print(
        f"  ULUSAL sicaklik-eslesmis YoY: Oca-Mar {gu:+.4f}  Nis-Tem {gn:+.4f}"
        f"  MEVSIM KAYMASI {gn - gu:+.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
