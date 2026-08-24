"""364-GUN REFERANSI: gecen yilin GERCEKLESEN gun ekseni bir CAPA mi?

FIKIR
-----
Test penceresi 2026-04-01..07-31. Onun 364 gun oncesi (2025-04-02..08-01)
ham egitim verisinde TAMAMEN var. Yani "gecen yil ayni takvim gunlerinde
gunluk ofset ekseni ne kadar genisti" sorusu ETIKETSIZ (test etiketi
kullanilmadan) yanitlanabilir. docs/39'un dersi: model-disi nicelikten
turetilen kestirim LB'ye TASINDI.

BILESIM TUZAGI
--------------
Ham panel dengesiz: 2025-01'de 64.027 satir, 2026-03'te 120.766. Gun
ortalamasi bu yuzden hem MEVSIMI hem KIMIN OLCULDUGUNU tasiyor. Bu betik
her pencerede DENGELI PANEL kurar: pencerenin gunlerinin en az %90'inda
gorunen trafolar, ve her trafo pencere ici ortalamasindan merkezlenir.

KALIBRE EDILEBILIR MI
---------------------
Referans yalnizca 2026 gunleri icin kurulabilir (veri 2025-01-01'de
basliyor). ``kis26``in 2026-01-01..03-31 parcasi TEK yer ki hem referans
kurulabiliyor hem GERCEK biliniyor. Kalibre orada yapilir.

    python scripts/deney_gun_ekseni_referans.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402

DIZIN = KOK / "data" / "interim" / "aile_onbellek"
TOHUMLAR = (1000, 1001, 1002)
AILELER = ("cat", "xgb", "lgbm", "sinir_agi")
AGIRLIK = (3.0, 1.0, 1.0, 1.4)
DOLULUK = 0.90


def blend(blok: str, tohum: int) -> np.ndarray:
    pay = sum(AGIRLIK)
    return (
        sum(
            w * np.load(DIZIN / f"{blok}_{tohum}_{a}_uretim.npy").astype("float64")
            for a, w in zip(AILELER, AGIRLIK, strict=True)
        )
        / pay
    )


def gun_ekseni(df: pd.DataFrame, deger: str, dengeli: bool = True) -> tuple[pd.Series, int, int]:
    """Gunluk ortalama ofset. ``dengeli`` ise panel dengelenir + merkezlenir."""
    q = df
    n_gun = q["tarih"].nunique()
    if dengeli:
        say = q.groupby("tanim", observed=True)["tarih"].nunique()
        tut = set(say[say >= DOLULUK * n_gun].index)
        q = q[q["tanim"].isin(tut)]
        q = q.assign(
            **{deger: q[deger] - q.groupby("tanim", observed=True)[deger].transform("mean")}
        )
    return q.groupby("tarih")[deger].mean(), q["tanim"].nunique(), len(q)


def egim(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    ort = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    xc, yc = ort["x"] - ort["x"].mean(), ort["y"] - ort["y"].mean()
    return float(np.polyfit(xc, yc, 1)[0]), float(np.corrcoef(xc, yc)[0, 1])


def main() -> int:
    t0 = time.time()
    print("=" * 100)
    print("364-GUN REFERANSI -- gecen yilin gerceklesen gun ekseni bir CAPA mi?")
    print("=" * 100)

    ham = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "tarih", "guc", "tuketim"],
        encoding="utf-8",
        dtype={"tanim": str},
    )
    ham["tarih"] = pd.to_datetime(ham["tarih"])
    ham["a"] = np.log1p(ham["tuketim"].clip(lower=0.0)) - np.log1p(ham["guc"])
    print(
        f"  ham egitim {len(ham):,} satir  "
        f"{ham['tarih'].min():%Y-%m-%d}..{ham['tarih'].max():%Y-%m-%d}"
    )

    egitim, test = d.cerceveleri_kur()
    sub = pd.read_csv(KOK / "submissions/tuketim_v50_ham30.csv", encoding="utf-8")
    t2 = test[["id", "tarih", "guc", "soguk_mu", "tanim"]].merge(sub, on="id", how="left")
    ts = t2[t2["soguk_mu"] != 1].copy()
    ts["r"] = np.log1p(ts["tuketim"].clip(lower=0.0)) - np.log1p(ts["guc"])
    ts["tarih"] = pd.to_datetime(ts["tarih"])

    # ------------------------------------------------------------- kalibre
    print("\n" + "-" * 100)
    print("A) KALIBRE -- kis26'nin 2026 parcasi (referans 2025-01-02..04-01)")
    print("-" * 100)
    _, dog, gercek, soguk = di.blok_parcalari(egitim, "kis26")
    dg = dog[~soguk].copy()
    dg["tarih"] = pd.to_datetime(dg["tarih"])
    lg = np.log1p(dg["guc"].to_numpy(dtype="float64"))
    dg["m"] = np.mean([blend("kis26", t) for t in TOHUMLAR], axis=0) - lg
    dg["a"] = np.log1p(np.clip(gercek[~soguk], 0.0, None)) - lg
    k26 = dg[dg["tarih"] >= "2026-01-01"]
    print(f"  kis26 2026 parcasi: {len(k26):,} satir, {k26['tarih'].nunique()} gun")

    for dengeli in (False, True):
        m_d, n_tr, n_r = gun_ekseni(k26, "m", dengeli)
        a_d, _, _ = gun_ekseni(k26, "a", dengeli)
        ref_pen = ham[(ham["tarih"] >= "2025-01-02") & (ham["tarih"] <= "2025-04-01")]
        r_d, n_tr_r, n_r_r = gun_ekseni(ref_pen, "a", dengeli)
        r_d.index = r_d.index + pd.Timedelta(days=364)
        e_true, k_true = egim(m_d, a_d)
        e_ref, k_ref = egim(m_d, r_d)
        _, k_rt = egim(r_d, a_d)
        etk = "DENGELI PANEL" if dengeli else "HAM"
        print(
            f"\n  {etk}  (model paneli {n_tr:,} trafo / {n_r:,} satir; "
            f"referans {n_tr_r:,} trafo / {n_r_r:,} satir)"
        )
        print(f"    std(model)      {float(m_d.std()):.4f}")
        print(f"    std(GERCEK)     {float(a_d.std()):.4f}")
        print(f"    std(2025 ref)   {float(r_d.std()):.4f}")
        print(f"    c_true = egim(gercek ~ model)   {e_true:+.3f}   kor {k_true:+.3f}")
        print(f"    c_ref  = egim(2025ref ~ model)  {e_ref:+.3f}   kor {k_ref:+.3f}")
        print(f"    kor(2025 ref, GERCEK)           {k_rt:+.3f}")
        print(f"    KALIBRE HATASI  c_ref - c_true = {e_ref - e_true:+.3f}")

    # --------------------------------------------------------------- test
    print("\n" + "-" * 100)
    print("B) TEST -- 2026-04-01..07-31, referans 2025-04-02..08-01")
    print("-" * 100)
    for dengeli in (False, True):
        m_d, n_tr, n_r = gun_ekseni(ts, "r", dengeli)
        ref_pen = ham[(ham["tarih"] >= "2025-04-02") & (ham["tarih"] <= "2025-08-01")]
        r_d, n_tr_r, n_r_r = gun_ekseni(ref_pen, "a", dengeli)
        r_d.index = r_d.index + pd.Timedelta(days=364)
        e_ref, k_ref = egim(m_d, r_d)
        etk = "DENGELI PANEL" if dengeli else "HAM"
        print(
            f"\n  {etk}  (test paneli {n_tr:,} trafo / {n_r:,} satir; "
            f"referans {n_tr_r:,} trafo / {n_r_r:,} satir)"
        )
        print(f"    std(model TEST) {float(m_d.std()):.4f}   std(2025 ref) {float(r_d.std()):.4f}")
        print(f"    c_ref = egim(2025ref ~ model)  {e_ref:+.3f}   kor {k_ref:+.3f}")
        ay = pd.DataFrame({"m": m_d, "r": r_d}).dropna()
        ay["_ay"] = ay.index.month
        print(f"    {'ay':>4}{'model 2026':>13}{'2025 ref':>11}{'fark':>9}")
        for a_, q in ay.groupby("_ay"):
            print(
                f"    {a_:>4}{q['m'].mean() - ay['m'].mean():+13.3f}"
                f"{q['r'].mean() - ay['r'].mean():+11.3f}"
                f"{(q['r'].mean() - ay['r'].mean()) - (q['m'].mean() - ay['m'].mean()):+9.3f}"
            )

    print(f"\nTAMAM  {time.time() - t0:.0f} sn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
