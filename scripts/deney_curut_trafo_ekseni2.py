# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""IDDIA CURUTME -- 2. tur: URETIMDE MEVCUT olan tek protokol.

D adimi (curut_trafo_ekseni.py) blogun ILK yarisinin etiketlerini kullaniyordu;
uretimde ufuk 1..122 gunun HICBIRINDE etiket yok. Gecerli protokol: a_i, TAMAMEN
GECMIS bir blogun kendi ufkundan kestirilir ve SONRAKI bloga uygulanir.

  yaz25 (2025-04..07, koken 2025-03-31)  ->  guz25 (2025-08..11, koken 2025-07-31)
  guz25 (2025-08..11, koken 2025-07-31)  ->  kis26 (2025-12..26-03, koken 2025-11-30)

Uretim karsiligi: kis26 -> TEST (2026-04..07, koken 2026-03-31).

Iddia sahibinin testi lambda*ort(DIGER IKI blok) kullaniyordu; iki kaynagi
ortalamak ZIT isaretli kaynaklari seyreltir. Burada YONLU tek kaynak, OLS
egimi, ampirik-Bayes buzme ve DURUST lambda (hedef disi ciftten kalibre) var.
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
import deney_sicak_artik as sa  # noqa: E402
import olcut  # noqa: E402
import tuketim_model as tm  # noqa: E402

TOHUMLAR = (1000, 1001, 1002)
CIFTLER = [("yaz25", "guz25"), ("guz25", "kis26")]
LAM = (-0.20, 0.00, 0.10, 0.20, 0.30, 0.50)


def wmean(x, w):
    return float(np.dot(w, x) / w.sum())


def main() -> int:
    egitim, test = d.cerceveleri_kur()
    gk = olcut.guc_kenarlari(test)
    tsicak = test[test["soguk_mu"] == 0]

    V = {}
    for b in tm.BLOKLAR:
        v = sa.blok_verisi(egitim, b.ad)
        dg = v["cerceve"]
        v["trafo"] = pd.Series(dg["tanim"].to_numpy())
        v["w"], v["tani"] = olcut.test_agirliklari(dg, tsicak, gk)
        V[b.ad] = v

    # tohum bazinda a_i (kaynak bloklar icin) -- her tohumun kendi artigi
    AI = {}
    for ad, v in V.items():
        for i, t in enumerate(TOHUMLAR):
            r = v["tohum_loglari"][i] - v["lg"]
            e = np.asarray(v["g"] - r, dtype="float64")
            e = e - e.mean()
            s = pd.Series(e)
            g = v["trafo"]
            n_i = s.groupby(g).size()
            m_i = s.groupby(g).mean()
            ici = float(s.groupby(g).transform("mean").sub(s).pow(2).mean())
            arasi = max(float(m_i.var(ddof=1)) - ici / float(n_i.mean()), 1e-6)
            M = ici / arasi
            AI[(ad, t)] = {"ham": m_i, "eb": (n_i / (n_i + M)) * m_i, "M": M}

    print("=" * 78)
    print("E) YONLU ILERI TASIMA  kaynak -> hedef  (tohum esli, agirlikli)")
    tut = {}
    for kay, hed in CIFTLER:
        vh = V[hed]
        w = vh["w"]
        trh = vh["trafo"]
        satir = {la: [] for la in LAM}
        satir["EB"] = []
        ols_list = []
        for i, t in enumerate(TOHUMLAR):
            r = vh["tohum_loglari"][i] - vh["lg"]
            taban = olcut.agirlikli_rmsle(vh["y"], np.expm1(vh["lg"] + r), w)
            ak = AI[(kay, t)]
            ah = AI[(hed, t)]
            ort = pd.concat([ak["ham"], ah["ham"]], axis=1, join="inner").dropna()
            ort.columns = ["k", "h"]
            ols_list.append(float(np.polyfit(ort["k"], ort["h"], 1)[0]))
            for la in LAM:
                dai = trh.map(ak["ham"] * la).fillna(0.0).to_numpy()
                satir[la].append(
                    taban - olcut.agirlikli_rmsle(vh["y"], np.expm1(vh["lg"] + r + dai), w)
                )
            dai = trh.map(ak["eb"]).fillna(0.0).to_numpy()
            satir["EB"].append(
                taban - olcut.agirlikli_rmsle(vh["y"], np.expm1(vh["lg"] + r + dai), w)
            )
        kapsam = float(trh.isin(AI[(kay, 1000)]["ham"].index).mean())
        print(
            f"\n  {kay} -> {hed}   kapsam %{kapsam * 100:.1f}   hedef-ici OLS egimi "
            f"{np.mean(ols_list):+.3f}   M(EB kaynak) {AI[(kay, 1000)]['M']:.1f}"
        )
        print(f"    {'lambda':>8}{'ort kazanc':>12}{'SH':>10}{'t':>8}{'poz':>7}")
        for la in list(LAM) + ["EB"]:
            a = np.array(satir[la])
            sh = a.std(ddof=1) / np.sqrt(len(a))
            print(
                f"    {str(la):>8}{a.mean():+12.5f}{sh:10.5f}"
                f"{(a.mean() / sh if sh > 0 else 0):+8.2f}{int((a > 0).sum()):>4}/{len(a)}"
            )
        tut[(kay, hed)] = satir

    print("\n  DURUST LAMBDA KURALI: hedef DISI ciftten kalibre et, hedefe uygula")
    print("    yaz25->guz25 OLS egimi ile guz25->kis26 uygula (ve tersi)")
    egim = {}
    for kay, hed in CIFTLER:
        vh = V[hed]
        ak = AI[(kay, 1000)]["ham"]
        ah = AI[(hed, 1000)]["ham"]
        x = pd.concat([ak, ah], axis=1, join="inner").dropna()
        x.columns = ["k", "h"]
        egim[(kay, hed)] = float(np.polyfit(x["k"], x["h"], 1)[0])
    print(
        f"    egim(yaz->guz) = {egim[('yaz25', 'guz25')]:+.3f}   "
        f"egim(guz->kis) = {egim[('guz25', 'kis26')]:+.3f}"
    )

    for (kay, hed), la_kaynak in [
        (("guz25", "kis26"), egim[("yaz25", "guz25")]),
        (("yaz25", "guz25"), egim[("guz25", "kis26")]),
    ]:
        vh = V[hed]
        w = vh["w"]
        trh = vh["trafo"]
        farklar = []
        for i, t in enumerate(TOHUMLAR):
            r = vh["tohum_loglari"][i] - vh["lg"]
            taban = olcut.agirlikli_rmsle(vh["y"], np.expm1(vh["lg"] + r), w)
            dai = trh.map(AI[(kay, t)]["ham"] * la_kaynak).fillna(0.0).to_numpy()
            farklar.append(taban - olcut.agirlikli_rmsle(vh["y"], np.expm1(vh["lg"] + r + dai), w))
        a = np.array(farklar)
        sh = a.std(ddof=1) / np.sqrt(len(a))
        print(
            f"    {kay}->{hed}  lambda={la_kaynak:+.3f}  kazanc {a.mean():+.5f}"
            f"  SH {sh:.5f}  t {(a.mean() / sh if sh > 0 else 0):+.2f}  {int((a > 0).sum())}/{len(a)}"
        )

    # --------------------------------------------------------------- MERCEK
    print("\n" + "=" * 78)
    print("MERCEK: guz25->kis26, lambda=+0.20 (tek kazanan yon) TRAFO BAZINDA")
    kay, hed, la = "guz25", "kis26", 0.20
    vh = V[hed]
    w, trh = vh["w"], vh["trafo"]
    dd_top = None
    for i, t in enumerate(TOHUMLAR):
        r = vh["tohum_loglari"][i] - vh["lg"]
        g = np.log1p(np.clip(vh["y"], 0, None))
        dai = trh.map(AI[(kay, t)]["ham"] * la).fillna(0.0).to_numpy()
        e0 = (g - (vh["lg"] + r)) ** 2
        e1 = (g - (vh["lg"] + r + dai)) ** 2
        s = pd.Series((e0 - e1) * w).groupby(trh).sum()
        dd_top = s if dd_top is None else dd_top + s
    dd = (dd_top / 3.0).sort_values(ascending=False)
    p = (dd / dd.sum()).to_numpy()
    print(
        f"  toplam d(MSE) {dd.sum():+.1f}   EN BUYUK %{p[0] * 100:.1f}   ilk5 %{p[:5].sum() * 100:.1f}"
        f"   pozitif trafo orani %{float((dd > 0).mean()) * 100:.1f}   (n={len(dd):,})"
    )
    srt = dd.index.to_numpy()
    print(f"  {'K':>4}{'kalan':>9}{'kazanc':>10}{'SH':>9}{'t':>7}{'tohum':>7}")
    for K in (0, 1, 5, 10, 25, 50):
        msk = ~trh.isin(set(srt[:K])).to_numpy()
        f = []
        for i, t in enumerate(TOHUMLAR):
            r = vh["tohum_loglari"][i] - vh["lg"]
            dai = trh.map(AI[(kay, t)]["ham"] * la).fillna(0.0).to_numpy()
            t0 = olcut.agirlikli_rmsle(vh["y"][msk], np.expm1(vh["lg"][msk] + r[msk]), w[msk])
            t1 = olcut.agirlikli_rmsle(
                vh["y"][msk], np.expm1(vh["lg"][msk] + r[msk] + dai[msk]), w[msk]
            )
            f.append(t0 - t1)
        a = np.array(f)
        sh = a.std(ddof=1) / np.sqrt(len(a))
        print(
            f"  {K:4d}{int(msk.sum()):9,}{a.mean():+10.5f}{sh:9.5f}"
            f"{(a.mean() / sh if sh > 0 else 0):+7.2f}{int((a > 0).sum()):>4}/{len(a)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
