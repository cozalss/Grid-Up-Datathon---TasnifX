"""EKSEN 2b -- ETIKETSIZ CAPALAR: b'yi kis26'dan BAGIMSIZ olcmenin yollari.

(a4) sabit panel YoY  (a5) ulusal yuk YoY  (a6) hava CIFTE SAYIM denetimi
+    v55'in IMA ETTIGI Nis-Tem YoY (delta = hedef - ima edilen)

Butun seviyeler ``r = log1p(tuketim) - log1p(guc)`` uzerinde, TRAFO ETKISI
CIKARILMIS (her trafonun 2025-01..2026-03 ortalamasi dusulur). Boylece
karisim degisimi seviyeye karismaz.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
from gridup.turkish import join_key  # noqa: E402

DOLULUK = 0.90
TR0, TR1 = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-03-31")
SUB = KOK / "submissions/tuketim_v55_gunolcek.csv"


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def main() -> int:
    print("=" * 100)
    print("EKSEN 2b -- ETIKETSIZ CAPALAR")
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

    # ---- SABIT PANEL: 15 ayin (tum gunlerin) >=%90'inda kaydi olan trafolar
    ngun = tr["tarih"].nunique()
    say = tr.groupby("tanim", observed=True)["tarih"].nunique()
    panel = set(say[say >= DOLULUK * ngun].index)
    ortak = panel & set(te["tanim"].unique())
    print(f"  sabit panel {len(panel):,} trafo ({ngun} gun, >=%{DOLULUK * 100:.0f} doluluk)")
    print(f"  bunlardan testte de olan {len(ortak):,}")

    p = tr[tr["tanim"].isin(ortak)].copy()
    fe = p.groupby("tanim", observed=True)["r"].mean()
    p["d"] = p["r"] - p["tanim"].map(fe).to_numpy()
    gun = p.groupby("tarih")["d"].mean()

    q = te[te["tanim"].isin(ortak)].copy()
    q["d"] = q["r"] - q["tanim"].map(fe).to_numpy()
    gun_te = q.groupby("tarih")["d"].mean()

    # ---- HAVA: panel trafo sayisiyla agirlikli bolgesel gunluk sicaklik
    hava = pd.read_parquet(
        KOK / "data/external/hava_gunluk.parquet",
        columns=["ilce_key", "tarih", "sicaklik_ort", "sicaklik_max"],
    ).drop_duplicates(["ilce_key", "tarih"])
    hava["tarih"] = pd.to_datetime(hava["tarih"])
    agir = p.drop_duplicates("tanim").groupby("ilce_key").size().rename("w")
    hv = hava.merge(agir, left_on="ilce_key", right_index=True, how="inner")
    hv["hdd18"] = (18.0 - hv["sicaklik_ort"]).clip(lower=0.0)
    hv["cdd22"] = (hv["sicaklik_ort"] - 22.0).clip(lower=0.0)
    hv["cddx26"] = (hv["sicaklik_max"] - 26.0).clip(lower=0.0)
    W = hv.groupby("tarih").apply(
        lambda x: pd.Series(
            {
                k: float(np.average(x[k], weights=x["w"]))
                for k in ("sicaklik_ort", "hdd18", "cdd22", "cddx26")
            }
        ),
        include_groups=False,
    )
    print(f"  hava kapsami {W.index.min().date()} .. {W.index.max().date()}")

    def pen(a, b_):
        return (W.index >= pd.Timestamp(a)) & (W.index <= pd.Timestamp(b_))

    for et, a, b_ in (
        ("2025 Nis-Tem", "2025-04-01", "2025-07-31"),
        ("2026 Nis-Tem", "2026-04-01", "2026-07-31"),
        ("2025 Oca-Mar", "2025-01-01", "2025-03-31"),
        ("2026 Oca-Mar", "2026-01-01", "2026-03-31"),
    ):
        s = W[pen(a, b_)]
        print(
            f"    {et}: T {s['sicaklik_ort'].mean():5.2f}  HDD18 {s['hdd18'].mean():5.2f}"
            f"  CDD22 {s['cdd22'].mean():5.2f}  CDDmax26 {s['cddx26'].mean():5.2f}"
        )

    # ---- GUNLUK REGRESYON: seviye ~ trend + hava + haftagunu (2025-01..2026-03)
    dfr = pd.DataFrame({"y": gun}).join(W, how="inner")
    dfr = dfr[(dfr.index >= TR0) & (dfr.index <= TR1)]
    t_yil = (dfr.index - TR0).days.to_numpy() / 365.25
    hg = pd.get_dummies(dfr.index.dayofweek, prefix="h", drop_first=True).to_numpy(dtype=float)
    Xh = dfr[["hdd18", "cdd22", "cddx26"]].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(dfr)), t_yil, Xh, hg])
    bco = ols(X, dfr["y"].to_numpy(dtype=float))
    art = dfr["y"].to_numpy() - X @ bco
    print("\n### (a4) SABIT PANEL gunluk regresyon (2025-01-01..2026-03-31)")
    print(f"  n={len(dfr)}  R2={1 - art.var() / dfr['y'].var():.3f}")
    print(f"  YILLIK TREND g = {bco[1]:+.4f} / yil   (hava-arindirilmis)")
    print(f"  HDD18 {bco[2]:+.5f}  CDD22 {bco[3]:+.5f}  CDDmax26 {bco[4]:+.5f}")
    # trend katsayisinin SH'si (Newey-West'siz, naif -> alt sinir)
    XtXi = np.linalg.pinv(X.T @ X)
    s2 = float(art @ art) / (len(dfr) - X.shape[1])
    print(f"  naif SH(g) = {np.sqrt(s2 * XtXi[1, 1]):.4f}  (otokorelasyon yok sayildi)")

    def hava_etkisi(a1, b1, a2, b2):
        """pencere2 - pencere1 icin regresyonun ongordugu hava katkisi."""
        s1 = W[pen(a1, b1)][["hdd18", "cdd22", "cddx26"]].mean().to_numpy()
        s2_ = W[pen(a2, b2)][["hdd18", "cdd22", "cddx26"]].mean().to_numpy()
        return float((s2_ - s1) @ bco[2:5])

    # ---- Oca-Mar YoY (ham + hava-arindirilmis)
    def ort(sr, a, b_):
        m = (sr.index >= pd.Timestamp(a)) & (sr.index <= pd.Timestamp(b_))
        return float(sr[m].mean())

    om25, om26 = ort(gun, "2025-01-01", "2025-03-31"), ort(gun, "2026-01-01", "2026-03-31")
    hav_om = hava_etkisi("2025-01-01", "2025-03-31", "2026-01-01", "2026-03-31")
    print(f"\n  Oca-Mar seviye  2025 {om25:+.4f} -> 2026 {om26:+.4f}   HAM YoY {om26 - om25:+.4f}")
    print(f"    hava katkisi {hav_om:+.4f}  ->  HAVA-ARINDIRILMIS YoY {om26 - om25 - hav_om:+.4f}")
    for a in ("01", "02", "03"):
        v25 = ort(gun, f"2025-{a}-01", f"2025-{a}-28")
        v26 = ort(gun, f"2026-{a}-01", f"2026-{a}-28")
        hv_ = hava_etkisi(f"2025-{a}-01", f"2025-{a}-28", f"2026-{a}-01", f"2026-{a}-28")
        print(
            f"      ay {a}: ham {v26 - v25:+.4f}  hava {hv_:+.4f}  arindirilmis {v26 - v25 - hv_:+.4f}"
        )

    # ---- (a5) ULUSAL YUK
    print("\n### (a5) ULUSAL YUK (EPIAS)")
    ul = pd.read_parquet(KOK / "data/external/epias/tuketim_saatlik.parquet")
    ul["tarih"] = pd.to_datetime(ul["zaman"]).dt.normalize()
    ug = ul.groupby("tarih")["consumption"].sum()
    lg = np.log(ug)
    print(f"  kapsam {ug.index.min().date()} .. {ug.index.max().date()}")

    def uyoy(a, b_):
        x2 = ort(lg, f"2026-{a}", f"2026-{b_}")
        x1 = ort(lg, f"2025-{a}", f"2025-{b_}")
        return x2 - x1

    uom = uyoy("01-01", "03-31")
    unt = uyoy("04-01", "07-31")
    print(f"  ULUSAL log-YoY  Oca-Mar {uom:+.4f}   Nis-Tem {unt:+.4f}   fark {unt - uom:+.4f}")
    for a in ("01", "02", "03", "04", "05", "06", "07"):
        print(f"    ay {a}: {uyoy(f'{a}-01', f'{a}-28'):+.4f}")

    # yerel ~ ulusal orani (Oca/Sub/Mar 3 nokta, hava-arindirilmis yerel)
    ay_y, ay_u = [], []
    for a in ("01", "02", "03"):
        v = ort(gun, f"2026-{a}-01", f"2026-{a}-28") - ort(gun, f"2025-{a}-01", f"2025-{a}-28")
        v -= hava_etkisi(f"2025-{a}-01", f"2025-{a}-28", f"2026-{a}-01", f"2026-{a}-28")
        ay_y.append(v)
        ay_u.append(uyoy(f"{a}-01", f"{a}-28"))
    A = np.column_stack([np.ones(3), ay_u])
    kk = ols(A, np.array(ay_y))
    print(f"  yerel_YoY = {kk[0]:+.4f} {kk[1]:+.3f} * ulusal_YoY   (3 nokta -- ASIRI UYDURMA)")
    print(f"    -> ulusal Nis-Tem {unt:+.4f} girilirse yerel {kk[0] + kk[1] * unt:+.4f}")
    print(
        f"    -> SADE kaydirma (yerel_OcaMar + (ulusal_NisTem - ulusal_OcaMar)):"
        f" {om26 - om25 - hav_om + (unt - uom):+.4f}"
    )

    # ---- (a6) HAVA + v55'in IMA ETTIGI YoY
    print("\n### (a6) v55'in IMA ETTIGI Nis-Tem YoY  ve CIFTE SAYIM denetimi")
    nt25 = ort(gun, "2025-04-01", "2025-07-31")
    nt26 = float(gun_te[(gun_te.index >= pd.Timestamp("2026-04-01"))].mean())
    hav_nt = hava_etkisi("2025-04-01", "2025-07-31", "2026-04-01", "2026-07-31")
    print(f"  panel seviye  2025 Nis-Tem {nt25:+.4f}   v55 2026 Nis-Tem {nt26:+.4f}")
    print(f"  IMA EDILEN YoY {nt26 - nt25:+.4f}")
    print(
        f"  HAVA katkisi (regresyon) {hav_nt:+.4f}  -> v55'in ima ettigi BUYUME"
        f" {nt26 - nt25 - hav_nt:+.4f}"
    )
    print("  ay ay v55 ima:")
    for a in ("04", "05", "06", "07"):
        v25 = ort(gun, f"2025-{a}-01", f"2025-{a}-28")
        v26 = float(
            gun_te[
                (gun_te.index >= pd.Timestamp(f"2026-{a}-01"))
                & (gun_te.index <= pd.Timestamp(f"2026-{a}-28"))
            ].mean()
        )
        hv_ = hava_etkisi(f"2025-{a}-01", f"2025-{a}-28", f"2026-{a}-01", f"2026-{a}-28")
        print(f"    {a}: ima {v26 - v25:+.4f}  hava {hv_:+.4f}  ima-buyume {v26 - v25 - hv_:+.4f}")

    # ---- HEDEF SEVIYE ve delta
    print("\n### HEDEF Nis-Tem 2026 SEVIYESI -> delta = hedef - v55")
    buyume_adaylari = {
        "trend g (yillik, hava-arindirilmis)": float(bco[1]),
        "Oca-Mar YoY (hava-arindirilmis)": om26 - om25 - hav_om,
        "Oca-Mar YoY + ulusal mevsim kaydirmasi": om26 - om25 - hav_om + (unt - uom),
        "yerel~ulusal regresyonu (3 nokta)": float(kk[0] + kk[1] * unt),
    }
    print(f"  {'buyume kaynagi':44}{'buyume':>9}{'+hava':>9}{'hedef':>9}{'v55':>9}{'delta':>9}")
    for ad, g in buyume_adaylari.items():
        hedef = nt25 + g + hav_nt
        print(f"  {ad:44}{g:+9.4f}{hav_nt:+9.4f}{hedef:+9.4f}{nt26:+9.4f}{hedef - nt26:+9.4f}")
    print("\n  NOT: 'hedef' hava katkisini EKLER; v55 zaten 2026 havasini kullandigi icin")
    print("       fark (delta) hava farkindan ARINMISTIR -- cifte sayim yok.")
    print(f"  hava-arindirilmamis naif capa olsaydi delta {hav_nt:+.4f} kadar FAZLA cikardi.")

    # panel disi kontrol: TUM sicak trafolar (panel secim yanliligi denetimi)
    hep = set(tr["tanim"].unique()) & set(te["tanim"].unique())
    p2 = tr[tr["tanim"].isin(hep)]
    fe2 = p2.groupby("tanim", observed=True)["r"].mean()
    g2 = p2["r"] - p2["tanim"].map(fe2).to_numpy()
    s2s = pd.Series(g2.to_numpy(), index=p2["tarih"].to_numpy()).groupby(level=0).mean()
    q2 = te[te["tanim"].isin(hep)]
    t2 = (
        pd.Series(
            (q2["r"] - q2["tanim"].map(fe2).to_numpy()).to_numpy(), index=q2["tarih"].to_numpy()
        )
        .groupby(level=0)
        .mean()
    )
    a25 = float(
        s2s[
            (s2s.index >= pd.Timestamp("2025-04-01")) & (s2s.index <= pd.Timestamp("2025-07-31"))
        ].mean()
    )
    print(
        f"\n  KONTROL -- TUM ortak sicak trafolar ({len(hep):,}): 2025 Nis-Tem {a25:+.4f}"
        f"  v55 2026 {float(t2.mean()):+.4f}  ima YoY {float(t2.mean()) - a25:+.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
