# ruff: noqa
"""EKSEN 3 (a)+(b): kis26 yanliliginin GUNLUK anatomisi -- HAVA mi SURUKLENME mi?

Onbellekten okur (data/interim/deney/sicak_tahmin.npz). FIT YOK.

    python scripts/eksen3_a_hava_vs_suruklenme.py
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

AILELER = ("cat", "xgb", "lgbm")
AGIRLIK = (3.0, 1.0, 1.0)
ONBELLEK = KOK / "data" / "interim" / "deney" / "sicak_tahmin.npz"
CIKTI = KOK / "reports" / "_eksen3_gunluk.csv"


def blok_verisi(egitim: pd.DataFrame, blok: str, z) -> dict:
    _, dogrulama, gercek, soguk = di.blok_parcalari(egitim, blok)
    pay = sum(AGIRLIK)
    loglar = [
        sum(AGIRLIK[i] * z[f"{blok}_{t}_{a}"] for i, a in enumerate(AILELER)) / pay
        for t in di.TOHUMLAR
    ]
    log_t = np.mean(loglar, axis=0)
    dg = dogrulama[~soguk].reset_index(drop=True)
    lg = np.log1p(dg["guc"].to_numpy(dtype="float64"))
    r = log_t - lg
    g = np.log1p(np.clip(gercek[~soguk], 0, None)) - lg
    return {"dg": dg, "r": r, "g": g, "e": g - r}


def iki_yonlu(e: np.ndarray, trafo: np.ndarray, gun: np.ndarray, tur: int = 60):
    """e ~ mu + a_i + d_t  (dengesiz panel, ardisik izdusum)."""
    s = pd.Series(e)
    ti = pd.Series(trafo)
    gi = pd.Series(gun)
    mu = float(s.mean())
    a = pd.Series(0.0, index=s.index)
    dd = pd.Series(0.0, index=s.index)
    for _ in range(tur):
        a = (s - mu - dd).groupby(ti).transform("mean")
        a = a - a.mean()
        dd = (s - mu - a).groupby(gi).transform("mean")
        dd = dd - dd.mean()
    return mu, a.to_numpy(), dd.to_numpy()


def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    ss_t = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((res**2).sum()) / ss_t if ss_t > 0 else np.nan
    return beta, r2


def nw_se(y: np.ndarray, X: np.ndarray, beta: np.ndarray, gecikme: int = 14) -> np.ndarray:
    """Newey-West standart hatalari (gunluk seri otokorelasyonlu)."""
    n, k = X.shape
    u = y - X @ beta
    XtX_inv = np.linalg.pinv(X.T @ X)
    S = (X * u[:, None]).T @ (X * u[:, None])
    for L in range(1, gecikme + 1):
        w = 1.0 - L / (gecikme + 1.0)
        A = (X[L:] * u[L:, None]).T @ (X[:-L] * u[:-L, None])
        S += w * (A + A.T)
    V = XtX_inv @ S @ XtX_inv * n / max(n - k, 1)
    return np.sqrt(np.clip(np.diag(V), 0, None))


def hava_serisi(ilce_agirlik: pd.Series) -> pd.DataFrame:
    """Ilce-satir-agirlikli gunluk IZMIR hava serisi + iklim normali sapmasi."""
    h = pd.read_parquet(
        KOK / "data" / "external" / "hava_gunluk.parquet",
        columns=["ilce_key", "tarih", "sicaklik_ort", "sicaklik_min", "sicaklik_max"],
    )
    h["tarih"] = pd.to_datetime(h["tarih"]).dt.normalize()
    h = h[h["ilce_key"].isin(ilce_agirlik.index)].copy()
    h["w"] = h["ilce_key"].map(ilce_agirlik).astype("float64")
    for k in ("sicaklik_ort", "sicaklik_min", "sicaklik_max"):
        h[k + "_w"] = h[k] * h["w"]
    g = h.groupby("tarih", as_index=False).agg(
        {"w": "sum", **{k + "_w": "sum" for k in ("sicaklik_ort", "sicaklik_min", "sicaklik_max")}}
    )
    for k in ("sicaklik_ort", "sicaklik_min", "sicaklik_max"):
        g[k] = g[k + "_w"] / g["w"]
    g = g[["tarih", "sicaklik_ort", "sicaklik_min", "sicaklik_max"]]
    g["hdd18"] = (18.0 - g["sicaklik_ort"]).clip(lower=0)
    g["hdd15"] = (15.0 - g["sicaklik_ort"]).clip(lower=0)
    g["cdd22"] = (g["sicaklik_ort"] - 22.0).clip(lower=0)
    # iklim normali: 2020-2024, yilin gunu bazinda 15 gunluk kayan pencere
    g["yg"] = g["tarih"].dt.dayofyear
    ref = g[g["tarih"].dt.year.between(2020, 2024)]
    norm = ref.groupby("yg")[["sicaklik_ort", "hdd18", "cdd22"]].mean()
    norm = pd.concat([norm, norm, norm]).rolling(15, center=True, min_periods=1).mean()
    norm = norm.iloc[len(norm) // 3 : 2 * (len(norm) // 3)]
    norm.index = sorted(g["yg"].unique())[: len(norm)] if False else norm.index
    g = g.merge(
        norm.rename(columns=lambda c: c + "_norm"), left_on="yg", right_index=True, how="left"
    )
    g["sicaklik_sapma"] = g["sicaklik_ort"] - g["sicaklik_ort_norm"]
    g["hdd_sapma"] = g["hdd18"] - g["hdd18_norm"]
    return g


def main() -> int:
    z = np.load(ONBELLEK)
    egitim, _ = d.cerceveleri_kur()
    V = {b.ad: blok_verisi(egitim, b.ad, z) for b in tm.BLOKLAR}

    kis = V["kis26"]
    dgk = kis["dg"]
    agirlik = dgk["ilce_key"].value_counts(normalize=True)
    hava = hava_serisi(agirlik)

    print("=" * 92)
    print("EKSEN 3(b)  IZMIR (satir-agirlikli) HDD/SICAKLIK -- 2025 vs 2026 ayni aylar")
    print("=" * 92)
    hh = hava[hava["tarih"].dt.year.isin([2025, 2026])].copy()
    hh["ay"] = hh["tarih"].dt.month
    hh["yil"] = hh["tarih"].dt.year
    tab = hh.pivot_table(index="ay", columns="yil", values=["sicaklik_ort", "hdd18", "cdd22"])
    print(f"  {'ay':>3} {'T2025':>8}{'T2026':>8}{'dT':>8} | {'HDD25':>8}{'HDD26':>8}{'dHDD':>8}")
    for ay in range(1, 8):
        try:
            t25 = tab[("sicaklik_ort", 2025)][ay]
            t26 = tab[("sicaklik_ort", 2026)][ay]
            h25 = tab[("hdd18", 2025)][ay]
            h26 = tab[("hdd18", 2026)][ay]
        except KeyError:
            continue
        if np.isnan(t26):
            continue
        print(
            f"  {ay:>3} {t25:>8.2f}{t26:>8.2f}{t26 - t25:>+8.2f} |"
            f" {h25:>8.2f}{h26:>8.2f}{h26 - h25:>+8.2f}"
        )
    q1 = hh[hh["ay"].isin([1, 2, 3])].groupby("yil")[["sicaklik_ort", "hdd18"]].mean()
    print(f"\n  Q1 ortalamasi:\n{q1.to_string()}")
    print(
        f"  Q1 2026 - 2025:  dT {q1.loc[2026, 'sicaklik_ort'] - q1.loc[2025, 'sicaklik_ort']:+.3f} C"
        f"   dHDD {q1.loc[2026, 'hdd18'] - q1.loc[2025, 'hdd18']:+.3f}"
    )
    ara = hava[(hava["tarih"] >= "2025-12-01") & (hava["tarih"] <= "2025-12-31")]
    print(
        f"  Aralik 2025: T {ara['sicaklik_ort'].mean():.2f}  HDD {ara['hdd18'].mean():.2f}"
        f"   normalden sapma {ara['sicaklik_sapma'].mean():+.3f} C"
    )
    kd = hava[(hava["tarih"] >= "2025-12-01") & (hava["tarih"] <= "2026-03-31")]
    print(
        f"  kis26 penceresi tumu: T sapma {kd['sicaklik_sapma'].mean():+.3f} C"
        f"   HDD sapma {kd['hdd_sapma'].mean():+.3f}"
    )

    print()
    print("=" * 92)
    print("EKSEN 3(a)  GUNLUK YANLILIK  b_t = mu + d_t   (TRAFO ETKISI CIKARILMIS -- kural 6)")
    print("=" * 92)

    satirlar = []
    for ad in ("kis26", "guz25", "yaz25"):
        v = V[ad]
        dg = v["dg"]
        trafo = dg["tanim"].to_numpy()
        gun = pd.to_datetime(dg["tarih"]).to_numpy()
        mu, a, dd = iki_yonlu(v["e"], trafo, gun)
        gs = pd.DataFrame({"tarih": gun, "b_ham": v["e"], "d": dd, "r": v["r"], "g": v["g"]})
        gunluk = gs.groupby("tarih").agg(
            b_ham=("b_ham", "mean"), d=("d", "mean"), n=("b_ham", "size")
        )
        gunluk["b"] = mu + gunluk["d"]
        gunluk = gunluk.reset_index()
        gunluk["blok"] = ad
        gunluk["mu"] = mu
        gunluk["ufuk"] = (
            gunluk["tarih"] - pd.Timestamp(next(b for b in tm.BLOKLAR if b.ad == ad).etiket_basi)
        ).dt.days
        gunluk = gunluk.merge(
            hava[
                [
                    "tarih",
                    "sicaklik_ort",
                    "hdd18",
                    "hdd15",
                    "cdd22",
                    "sicaklik_sapma",
                    "hdd_sapma",
                ]
            ],
            on="tarih",
            how="left",
        )
        satirlar.append(gunluk)
        print(f"\n--- {ad}: n={len(dg):,}  agirlikli ort e = {float(v['e'].mean()):+.4f}")
        gunluk["ay"] = gunluk["tarih"].dt.to_period("M").astype(str)
        ay = gunluk.groupby("ay").agg(
            b_ham=("b_ham", "mean"),
            b=("b", "mean"),
            T=("sicaklik_ort", "mean"),
            HDD=("hdd18", "mean"),
            Tsap=("sicaklik_sapma", "mean"),
            n=("n", "sum"),
        )
        print(
            f"    {'ay':>8}{'b_ham':>9}{'b(trafo-ar)':>13}{'T':>7}{'HDD':>7}{'T sapma':>9}{'n':>9}"
        )
        for i, r_ in ay.iterrows():
            print(
                f"    {i:>8}{r_['b_ham']:>+9.4f}{r_['b']:>+13.4f}{r_['T']:>7.1f}"
                f"{r_['HDD']:>7.2f}{r_['Tsap']:>+9.2f}{int(r_['n']):>9,}"
            )

    G = pd.concat(satirlar, ignore_index=True)
    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    G.to_csv(CIKTI, index=False)

    print()
    print("=" * 92)
    print("REGRESYON: gunluk b (trafo-arindirilmis) ~ SURUKLENME (ufuk) + HAVA")
    print("=" * 92)
    for ad in ("kis26", "guz25", "yaz25"):
        gg = G[G["blok"] == ad].dropna(subset=["hdd18"]).reset_index(drop=True)
        y = gg["b"].to_numpy()
        n = len(y)
        u = (gg["ufuk"].to_numpy() - gg["ufuk"].mean()) / gg["ufuk"].std()
        hdd = (gg["hdd18"].to_numpy() - gg["hdd18"].mean()) / max(gg["hdd18"].std(), 1e-9)
        cdd = (gg["cdd22"].to_numpy() - gg["cdd22"].mean()) / max(gg["cdd22"].std(), 1e-9)
        sap = (gg["sicaklik_sapma"].to_numpy() - gg["sicaklik_sapma"].mean()) / gg[
            "sicaklik_sapma"
        ].std()
        one = np.ones(n)
        modeller = {
            "yalniz sabit": np.c_[one],
            "yalniz UFUK": np.c_[one, u],
            "yalniz HDD+CDD": np.c_[one, hdd, cdd],
            "yalniz T-SAPMA": np.c_[one, sap],
            "UFUK+HDD+CDD": np.c_[one, u, hdd, cdd],
            "UFUK+T-SAPMA": np.c_[one, u, sap],
        }
        print(f"\n--- {ad}  (n gun={n}, b std={y.std():.4f})")
        for isim, X in modeller.items():
            if X.shape[1] == 1:
                continue
            beta, r2 = ols(y, X)
            se = nw_se(y, X, beta)
            ad_ler = ["sabit"] + {
                "yalniz UFUK": ["ufuk"],
                "yalniz HDD+CDD": ["hdd", "cdd"],
                "yalniz T-SAPMA": ["Tsap"],
                "UFUK+HDD+CDD": ["ufuk", "hdd", "cdd"],
                "UFUK+T-SAPMA": ["ufuk", "Tsap"],
            }[isim]
            kat = "  ".join(
                f"{k}={b_:+.4f}(NW t={b_ / s if s > 0 else np.nan:+.1f})"
                for k, b_, s in zip(ad_ler[1:], beta[1:], se[1:])
            )
            print(f"    {isim:<16} R2={r2:>6.3f}   {kat}")
        # kismi R2 (Type III) tam modelde
        X = np.c_[one, u, hdd, cdd]
        _, r2_tam = ols(y, X)
        for isim, sut in (("ufuk", [0, 2, 3]), ("hava", [0, 1])):
            _, r2_eksik = ols(y, X[:, sut])
            print(
                f"    KISMI R2 [{isim:>5} cikarilinca]: {r2_tam:.3f} -> {r2_eksik:.3f}"
                f"  (dusus {r2_tam - r2_eksik:+.3f})"
            )

    print()
    print("=" * 92)
    print("MODEL HAVAYI KACIRIYOR MU?  gunluk ort tahmin/gercek ~ HDD")
    print("=" * 92)
    for ad in ("kis26", "guz25", "yaz25"):
        v = V[ad]
        dg = v["dg"]
        gun = pd.to_datetime(dg["tarih"]).to_numpy()
        trafo = dg["tanim"].to_numpy()
        # trafo etkisini HEM tahminden HEM gercekten cikar
        _, _, dr = iki_yonlu(v["r"], trafo, gun)
        _, _, dgg = iki_yonlu(v["g"], trafo, gun)
        t = pd.DataFrame({"tarih": gun, "dr": dr, "dg": dgg}).groupby("tarih").mean()
        t = t.merge(hava[["tarih", "hdd18", "cdd22", "sicaklik_ort"]], on="tarih", how="left")
        for k in ("hdd18", "cdd22"):
            if float(t[k].std()) < 1e-6:
                print(f"  {ad:<7} {k}: pencerede sabit (std~0) -- olculemez")
                continue
            print(
                f"  {ad:<7} kor(gun_gercek,{k})={t['dg'].corr(t[k]):+.3f}"
                f"   kor(gun_tahmin,{k})={t['dr'].corr(t[k]):+.3f}"
                f"   egim gercek={np.polyfit(t[k], t['dg'], 1)[0]:+.5f}"
                f"  tahmin={np.polyfit(t[k], t['dr'], 1)[0]:+.5f}"
                f"   ACIK={np.polyfit(t[k], t['dg'] - t['dr'], 1)[0]:+.5f}"
            )

    print()
    print("=" * 92)
    print("TESTE CEVIRI: kis26 yanliligi HDD ile aciklanirsa, TEST penceresinde HDD ~ 0")
    print("=" * 92)
    gk = G[G["blok"] == "kis26"].dropna(subset=["hdd18"])
    y = gk["b"].to_numpy()
    X = np.c_[np.ones(len(gk)), gk["hdd18"].to_numpy(), gk["ufuk"].to_numpy()]
    beta, r2 = ols(y, X)
    print(f"  kis26:  b = {beta[0]:+.4f} {beta[1]:+.5f}*HDD {beta[2]:+.6f}*ufuk   R2={r2:.3f}")
    test_hdd = hava[(hava["tarih"] >= "2026-04-01") & (hava["tarih"] <= "2026-07-31")]
    thdd = float(test_hdd["hdd18"].mean())
    tufuk = float(
        np.mean((pd.date_range("2026-04-01", "2026-07-31") - pd.Timestamp("2026-04-01")).days)
    )
    print(f"  TEST penceresi ort HDD = {thdd:.2f}  (kis26 ort {gk['hdd18'].mean():.2f})")
    print(
        f"  kis26 katsayilari TESTE tasinirsa beklenen b = "
        f"{beta[0] + beta[1] * thdd + beta[2] * tufuk:+.4f}"
        f"   (HDD payi {beta[1] * thdd:+.4f}, kis26'da {beta[1] * float(gk['hdd18'].mean()):+.4f})"
    )
    hdd_payi = beta[1] * float(gk["hdd18"].mean())
    print(
        f"  >> kis26 ort b = {y.mean():+.4f};  bunun {hdd_payi:+.4f}'i ({hdd_payi / y.mean() * 100:.0f}%)"
        f" saf HDD SEVIYESINDEN geliyor -- testte HDD~{thdd:.1f} oldugu icin BU KISIM TASINMAZ."
    )
    print(f"\n  gunluk tablo yazildi: {CIKTI}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
