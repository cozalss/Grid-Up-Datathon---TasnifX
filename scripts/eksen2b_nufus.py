"""EKSEN 2b -- NUFUS KOPRUSU: capa PANEL'de olculuyor, delta TUM SICAK satirlara
uygulanacak. Aradaki fark ne kadar?

kis26'da her ikisi de olculebiliyor:
    b_panel  (capa makinesinin ulasabildigi trafolar)
    b_disi   (panel disi sicak trafolar -- bayat/aralikli/yeni)
Fark AYLARA gore SABIT ise yapisal, mevsimden bagimsizdir ve teste tasinir.
Mevsimle birlikte eriyorsa tasinmaz.

Ayrica: soguk yanliligi, test satir paylari, delta egrisi, LB probu.
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
import olcut  # noqa: E402
import tuketim_model as tm  # noqa: E402

AILELER = ("cat", "xgb", "lgbm")
AGIRLIK = (3.0, 1.0, 1.0)
S0 = 1.01591
MSE0 = S0 * S0


def main() -> int:
    print("=" * 100)
    print("EKSEN 2b -- NUFUS KOPRUSU, SOGUK, DELTA EGRISI, LB PROBU")
    print("=" * 100)

    tr = pd.read_csv(KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str})
    tr["tarih"] = pd.to_datetime(tr["tarih"])
    te = pd.read_csv(KOK / "data/raw/test.csv", encoding="utf-8", dtype={"tanim": str})
    ngun = tr["tarih"].nunique()
    say = tr.groupby("tanim", observed=True)["tarih"].nunique()
    panel = set(say[say >= 0.90 * ngun].index)
    sicak_te = te["tanim"].isin(set(tr["tanim"]))
    p_sicak = float(sicak_te.mean())
    p_panel = float(te["tanim"].isin(panel).mean())
    print(f"\n### TEST SATIR PAYLARI  (n={len(te):,})")
    print(f"  SICAK  {p_sicak:.4f}   SOGUK {1 - p_sicak:.4f}")
    print(f"  P15 panelde {p_panel:.4f}  -> sicak satirlarin {p_panel / p_sicak:.4f}'u")
    print(
        f"  sicak ama panel DISI {p_sicak - p_panel:.4f}  (sicagin"
        f" {(p_sicak - p_panel) / p_sicak:.4f}'u)"
    )

    egitim, test = d.cerceveleri_kur()
    tm.kategorik_kodla(egitim, test)
    z = np.load(KOK / "data/interim/deney/sicak_tahmin.npz")
    _, dog, gercek, soguk = di.blok_parcalari(egitim, "kis26")
    pay = sum(AGIRLIK)

    dg = dog[~soguk].reset_index(drop=True)
    lg = np.log1p(dg["guc"].to_numpy(dtype="float64"))
    r = (
        np.mean(
            [
                sum(AGIRLIK[i] * z[f"kis26_{t}_{a}"] for i, a in enumerate(AILELER)) / pay
                for t in di.TOHUMLAR
            ],
            axis=0,
        )
        - lg
    )
    b = np.log1p(gercek[~soguk]) - lg - r
    ay = pd.to_datetime(dg["tarih"]).dt.to_period("M").astype(str).to_numpy()
    ic = dg["tanim"].isin(panel).to_numpy()

    print("\n### A) kis26 SICAK: PANEL ICI vs DISI, ay ay")
    print(f"  {'ay':10}{'n_ic':>9}{'b_ic':>9}{'n_dis':>9}{'b_dis':>9}{'FARK':>9}")
    farklar = []
    for a in sorted(set(ay)):
        m = ay == a
        bi, bd = b[m & ic].mean(), b[m & ~ic].mean()
        farklar.append(bd - bi)
        print(
            f"  {a:10}{int((m & ic).sum()):9,}{bi:+9.4f}{int((m & ~ic).sum()):9,}"
            f"{bd:+9.4f}{bd - bi:+9.4f}"
        )
    print(
        f"  FARKIN ay ay std {np.std(farklar):.4f}  ort {np.mean(farklar):+.4f}"
        f"   -> {'YAPISAL (mevsimden bagimsiz)' if np.std(farklar) < 0.35 * abs(np.mean(farklar)) else 'MEVSIMLE DEGISIYOR'}"
    )
    sm = np.isin(ay, ["2026-02", "2026-03"])
    print(
        f"  Sub-Mar: b_ic {b[sm & ic].mean():+.4f}  b_dis {b[sm & ~ic].mean():+.4f}"
        f"  fark {b[sm & ~ic].mean() - b[sm & ic].mean():+.4f}"
        f"  (panel disi satir payi {float((~ic)[sm].mean()):.3f})"
    )

    # bayatliga gore ayristirma -- testte bayat pay COK daha yuksek
    yas = dg["t_son_kayit_yasi"].to_numpy(dtype="float64")
    print("\n  bayatliga gore (Sub-Mar):")
    print(f"  {'yas kovasi':14}{'n':>9}{'b':>9}   | test sicak payi")
    te_y = test[test["tanim"].isin(set(tr["tanim"]))]["t_son_kayit_yasi"].to_numpy(dtype="float64")
    for et, lo, hi in (
        ("0", -0.5, 0.5),
        ("1-6", 0.5, 6.5),
        ("7-29", 6.5, 29.5),
        ("30-89", 29.5, 89.5),
        (">=90", 89.5, 1e9),
    ):
        m = sm & (yas > lo) & (yas <= hi)
        pt = float(((te_y > lo) & (te_y <= hi)).mean())
        print(f"  {et:14}{int(m.sum()):9,}{b[m].mean() if m.any() else np.nan:+9.4f}   | {pt:.4f}")

    # ---- SOGUK ----
    print("\n### B) SOGUK yanliligi (kis26)")
    zs = np.load(KOK / "data/interim/deney/soguk_tahmin_kis26.npz")
    ds = dog[soguk].reset_index(drop=True)
    lgs = np.log1p(ds["guc"].to_numpy(dtype="float64"))
    rs = (
        np.mean(
            [
                sum(AGIRLIK[i] * zs[f"{t}_{a}"] for i, a in enumerate(AILELER)) / pay
                for t in di.TOHUMLAR
            ],
            axis=0,
        )
        - lgs
    )
    bs = np.log1p(gercek[soguk]) - lgs - rs
    ays = pd.to_datetime(ds["tarih"]).dt.to_period("M").astype(str).to_numpy()
    soguk_te = test[~test["tanim"].isin(set(tr["tanim"]))]
    ke = olcut.guc_kenarlari(test)
    print(f"  {'ay':10}{'n':>9}{'b_satir':>10}{'b_trafo':>10}{'dogum yasi ort':>16}")
    dogum = (
        pd.to_datetime(ds["tarih"])
        - ds.groupby("tanim")["tarih"].transform("min").pipe(pd.to_datetime)
    ).dt.days.to_numpy()
    for a in sorted(set(ays)):
        m = ays == a
        tb = pd.DataFrame({"t": ds["tanim"].to_numpy()[m], "b": bs[m]}).groupby("t")["b"].mean()
        print(
            f"  {a:10}{int(m.sum()):9,}{bs[m].mean():+10.4f}{tb.mean():+10.4f}"
            f"{dogum[m].mean():16.1f}"
        )
    sms = np.isin(ays, ["2026-02", "2026-03"])
    for et, msk in (("tum", np.ones(len(bs), bool)), ("Sub-Mar", sms)):
        w, tani = olcut.test_agirliklari(ds[msk], soguk_te, ke)
        print(
            f"  {et:9} ham {bs[msk].mean():+.4f}  test-agirlikli"
            f" {float(np.dot(w, bs[msk]) / w.sum()):+.4f}  ESS {tani['ess_orani']:.3f}"
            f" kapsanmayan {tani['kapsanmayan']:.3f}"
        )
    # soguk trafolarin dogum zamani: kis26'da blok ICINDE dogmus olanlar mi?
    print(
        f"  soguk trafolarin ilk kaydi blok icinde: "
        f"{float((ds.groupby('tanim')['tarih'].transform('min') >= '2025-12-01').mean()):.3f}"
    )

    # ---- DELTA EGRISI + PROB ----
    print("\n### C) dMSE(delta)  ve  BASABAS")
    print(f"  MSE0 = {MSE0:.5f}   p_sicak = {p_sicak:.4f}")
    print(
        f"  {'delta':>7}"
        + "".join(f"{f'b={x:+.02f}':>11}" for x in (-0.02, 0.00, 0.02, 0.04, 0.06, 0.09, 0.12))
    )
    for de in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10):
        s = f"  {de:7.2f}"
        for x in (-0.02, 0.00, 0.02, 0.04, 0.06, 0.09, 0.12):
            s += f"{np.sqrt(MSE0 + p_sicak * (de**2 - 2 * de * x)):11.5f}"
        print(s)
    print("  BASABAS: delta = 2*b  (delta < 2*b iken kazanc, ustunde kayip)")
    print("  OPTIMAL: delta = E[b];  beklenen kazanc dMSE = -p * E[b]^2")
    for eb in (0.02, 0.03, 0.05, 0.09):
        print(
            f"    E[b]={eb:.2f} -> delta={eb:.2f}, dMSE {-p_sicak * eb * eb:+.5f},"
            f" RMSLE {np.sqrt(MSE0 - p_sicak * eb * eb):.5f}"
        )

    print("\n### D) LB PROBU -- tek gonderimle b'nin TAM cozumu")
    print("  S1^2 = S0^2 + p*(delta^2 - 2*delta*b)")
    print("  =>  b = (delta^2 - (S1^2 - S0^2)/p) / (2*delta)")
    for de in (0.03, 0.05, 0.06):
        duy = p_sicak * de / S0  # |dS1/db|
        print(
            f"  delta={de:.2f}: |dS1/db| = {duy:.4f}  -> LB'nin 1e-5 cozunurlugu"
            f" b'yi {1e-5 / duy:.5f} hassasiyetle verir"
        )
        print(
            f"     beklenen skor: b=0 -> {np.sqrt(MSE0 + p_sicak * de * de):.5f}"
            f" | b=0.03 -> {np.sqrt(MSE0 + p_sicak * (de * de - 2 * de * 0.03)):.5f}"
            f" | b=0.06 -> {np.sqrt(MSE0 + p_sicak * (de * de - 2 * de * 0.06)):.5f}"
            f" | b=0.10 -> {np.sqrt(MSE0 + p_sicak * (de * de - 2 * de * 0.10)):.5f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
