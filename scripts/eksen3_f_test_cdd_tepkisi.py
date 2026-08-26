# ruff: noqa
"""EKSEN 3 (belirleyici): TEST MODELI CDD'ye ne kadar tepki veriyor?

yaz25 folduna gore model CDD'yi 6x ALTINDAN tepkiliyor. Ama yaz25 foldu
KENDI blogunu gormeden egitiliyor -- uretim modeli Nis-Tem 2025 etiketlerini
GORUYOR. Dolayisiyla folddaki eksik tepki bir LEAVE-ONE-OUT yapaylik olabilir.

Ayrim ETIKETSIZ yapilabilir (kural 5 ihlali yok):
    gercek CDD duyarliligi   <- 2025 Nis-Tem (train icinde, ETIKETLI)
    model CDD duyarliligi    <- 2026 Nis-Tem GONDERIM (etiket kullanilmaz)

    python scripts/eksen3_f_test_cdd_tepkisi.py
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
GONDERIMLER = {
    "v55 (gun olcekli, LB 1,01591)": "tuketim_v55_gunolcek.csv",
    "v50 (taban, LB 1,01686)": "tuketim_v50_nihai30.csv",
}


def iki_yonlu(e, trafo, gun, tur: int = 60):
    s = pd.Series(np.asarray(e, dtype="float64"))
    ti, gi = pd.Series(np.asarray(trafo)), pd.Series(np.asarray(gun))
    mu = float(s.mean())
    a = pd.Series(0.0, index=s.index)
    dd = pd.Series(0.0, index=s.index)
    for _ in range(tur):
        a = (s - mu - dd).groupby(ti).transform("mean")
        a = a - a.mean()
        dd = (s - mu - a).groupby(gi).transform("mean")
        dd = dd - dd.mean()
    return mu, dd.to_numpy()


def hava_serisi(ilce_agirlik: pd.Series) -> pd.DataFrame:
    h = pd.read_parquet(
        KOK / "data" / "external" / "hava_gunluk.parquet",
        columns=["ilce_key", "tarih", "sicaklik_ort"],
    )
    h["tarih"] = pd.to_datetime(h["tarih"]).dt.normalize()
    h = h[h["ilce_key"].isin(ilce_agirlik.index)].copy()
    h["w"] = h["ilce_key"].map(ilce_agirlik).astype("float64")
    h["sw"] = h["sicaklik_ort"] * h["w"]
    g = h.groupby("tarih", as_index=False)[["w", "sw"]].sum()
    g["sicaklik_ort"] = g["sw"] / g["w"]
    g["cdd22"] = (g["sicaklik_ort"] - 22.0).clip(lower=0)
    g["hdd18"] = (18.0 - g["sicaklik_ort"]).clip(lower=0)
    return g[["tarih", "sicaklik_ort", "cdd22", "hdd18"]]


def gun_ekseni(frame: pd.DataFrame, deger: np.ndarray) -> pd.DataFrame:
    mu, dd = iki_yonlu(deger, frame["tanim"].to_numpy(), frame["tarih"].to_numpy())
    t = (
        pd.DataFrame({"tarih": frame["tarih"].to_numpy(), "d": dd})
        .groupby("tarih", as_index=False)
        .mean()
    )
    t["mu"] = mu
    return t


def egim(t: pd.DataFrame, hava: pd.DataFrame, k: str = "cdd22") -> tuple[float, float, float]:
    x = t.merge(hava, on="tarih", how="left")
    b = float(np.polyfit(x[k], x["d"], 1)[0])
    return b, float(x["d"].corr(x[k])), float(x["d"].std())


def main() -> int:
    z = np.load(ONBELLEK)
    egitim, test = d.cerceveleri_kur()

    # ---------- 1) GERCEK CDD duyarliligi: 2025 Nis-Tem (etiketli, train icinde)
    _, dogrulama, gercek, soguk = di.blok_parcalari(egitim, "yaz25")
    dg = dogrulama[~soguk].reset_index(drop=True)
    lg = np.log1p(dg["guc"].to_numpy(dtype="float64"))
    g_ofs = np.log1p(np.clip(gercek[~soguk], 0, None)) - lg
    pay = sum(AGIRLIK)
    r_ofs = (
        np.mean(
            [
                sum(AGIRLIK[i] * z[f"yaz25_{t}_{a}"] for i, a in enumerate(AILELER)) / pay
                for t in di.TOHUMLAR
            ],
            axis=0,
        )
        - lg
    )
    dgy = pd.DataFrame(
        {
            "tanim": dg["tanim"].to_numpy(),
            "tarih": pd.to_datetime(dg["tarih"]).to_numpy(),
            "ilce_key": dg["ilce_key"].to_numpy(),
        }
    )
    hava25 = hava_serisi(dgy["ilce_key"].value_counts(normalize=True))
    tg = gun_ekseni(dgy, g_ofs)
    tr = gun_ekseni(dgy, r_ofs)
    bg, cg, sg = egim(tg, hava25)
    br, cr, sr = egim(tr, hava25)

    print("=" * 100)
    print("GUN EKSENI CDD DUYARLILIGI  (trafo etkisi cikarilmis, ofs = log1p(tuketim)-log1p(guc))")
    print("=" * 100)
    print(f"  {'seri':<46}{'egim/CDD':>11}{'kor':>8}{'gun std':>10}")
    print(f"  {'2025 Nis-Tem GERCEK (etiketli)':<46}{bg:>+11.5f}{cg:>+8.3f}{sg:>10.4f}")
    print(f"  {'2025 Nis-Tem yaz25 FOLD tahmini':<46}{br:>+11.5f}{cr:>+8.3f}{sr:>10.4f}")

    # ---------- 2) guz25 gercek duyarliligi (ikinci bagimsiz olcum)
    _, dv2, gr2, sk2 = di.blok_parcalari(egitim, "guz25")
    dg2 = dv2[~sk2].reset_index(drop=True)
    lg2 = np.log1p(dg2["guc"].to_numpy(dtype="float64"))
    g2 = np.log1p(np.clip(gr2[~sk2], 0, None)) - lg2
    d2 = pd.DataFrame(
        {"tanim": dg2["tanim"].to_numpy(), "tarih": pd.to_datetime(dg2["tarih"]).to_numpy()}
    )
    hava2 = hava_serisi(dg2["ilce_key"].value_counts(normalize=True))
    b2, c2, s2 = egim(gun_ekseni(d2, g2), hava2)
    print(
        f"  {'2025 Agu-Kas GERCEK (etiketli, ikinci blok)':<46}{b2:>+11.5f}{c2:>+8.3f}{s2:>10.4f}"
    )

    # ---------- 3) TEST MODELI: 2026 Nis-Tem gonderimleri
    te = test[test["soguk_mu"] == 0].reset_index(drop=True) if "soguk_mu" in test.columns else test
    hava26 = hava_serisi(te["ilce_key"].value_counts(normalize=True))
    print()
    for isim, dosya in GONDERIMLER.items():
        yol = KOK / "submissions" / dosya
        if not yol.exists():
            print(f"  {isim}: DOSYA YOK ({dosya})")
            continue
        s = pd.read_csv(yol)
        m = te[["id", "tanim", "tarih", "guc"]].merge(s, on="id", how="left")
        ofs = np.log1p(np.clip(m["tuketim"].to_numpy(dtype="float64"), 0, None)) - np.log1p(
            m["guc"].to_numpy(dtype="float64")
        )
        t = gun_ekseni(m, ofs)
        bb, cc, ss = egim(t, hava26)
        print(f"  {'2026 Nis-Tem ' + isim:<46}{bb:>+11.5f}{cc:>+8.3f}{ss:>10.4f}")

    print()
    print("  OKUMA: GERCEK duyarlilik iki bagimsiz blokta +0,067 / +0,071 -- TASINABILIR.")
    print("  Test modelinin duyarliligi bunlarin ALTINDA kaliyorsa gun ekseninde")
    print("  CDD'ye yonelmis (blanket olcek degil) bir duzeltme icin yer var.")

    # ---------- 4) yaz25'te MSE etkisi: blanket olcek vs CDD-yonelimli duzeltme
    print()
    print("=" * 100)
    print("yaz25 (MEVSIMSEL IKIZ) HOT satirlarda MSE: gun ekseni islemleri")
    print("=" * 100)
    gun = dgy["tarih"].to_numpy()
    dfr = pd.DataFrame({"gun": gun, "r": r_ofs, "g": g_ofs})
    gr_ = dfr.groupby("gun")["r"].transform("mean").to_numpy()
    gg_ = dfr.groupby("gun")["g"].transform("mean").to_numpy()
    ici = r_ofs - gr_
    m0 = float(((g_ofs - r_ofs) ** 2).mean())
    print(f"  {'islem':<52}{'MSE':>10}{'RMSLE':>10}{'dMSE(hot)':>12}")
    print(f"  {'(0) ham harman':<52}{m0:>10.5f}{np.sqrt(m0):>10.5f}{0.0:>+12.5f}")

    cdd_map = hava25.set_index("tarih")["cdd22"]
    cdd_row = pd.Series(gun).map(pd.Series(cdd_map)).to_numpy(dtype="float64")

    for etiket, yeni_gun in (
        ("(1) v55 gun olcegi c=1,604 (ort korunur)", gr_.mean() + 1.604 * (gr_ - gr_.mean())),
        ("(2) c=1,75 (yaz25 etiketli optimum)", gr_.mean() + 1.75 * (gr_ - gr_.mean())),
        (
            "(3) gun ekseni CDD'den: mu_model + 0,0676*(CDD-ort)",
            gr_.mean() + 0.0676 * (cdd_row - np.nanmean(cdd_row)),
        ),
        (
            "(4) (3) + sabit delta (yaz25 ort yanliligi)",
            gr_.mean() + 0.0676 * (cdd_row - np.nanmean(cdd_row)) + float((g_ofs - r_ofs).mean()),
        ),
        ("(5) yalniz sabit delta", gr_ + float((g_ofs - r_ofs).mean())),
        ("(6) TAVAN: gercek gun ekseni", gg_),
    ):
        yeni = yeni_gun + ici
        m1 = float(((g_ofs - yeni) ** 2).mean())
        print(f"  {etiket:<52}{m1:>10.5f}{np.sqrt(m1):>10.5f}{m1 - m0:>+12.5f}")

    print()
    print("  NOT: (3)/(4)'un katsayisi guz25'ten (BASKA BLOK) geliyor -- capraz blok.")
    print("       (2)/(5)/(6) yaz25'in KENDI etiketini kullanir -- TAVAN, tasinabilir degil.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
