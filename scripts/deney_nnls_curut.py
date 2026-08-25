# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""CURUTME: NNLS/simpleks capraz uydurma iddiasinin BAGIMSIZ yeniden uretimi.

Onbellekten okur, egitim yapmaz. Uretilen sayilar:
  1) uretim taban skoru ve ORAKUL simpleks tavani
  2) trafo bazinda 5 kat capraz simpleks / NNLS
  3) BLOK-DISI simpleks / NNLS
  4) NNLS agirlik toplamlari (blok bazinda) + KOR SKALAR c testi
  5) KIRPILMIS tablo (K trafo atilarak), yogunlasma
  6) orakul agirlik bulutu
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import olcut as ol  # noqa: E402
import tuketim_model as tm  # noqa: E402

DIZIN = KOK / "data" / "interim" / "aile_onbellek"
TOHUMLAR = (1000, 1001, 1002)
AILELER = ("cat", "xgb", "lgbm", "sinir_agi")
URETIM = np.array([3.0, 1.0, 1.0, 1.4])
URETIM_N = URETIM / URETIM.sum()
SICAK_KATSAYI = 0.7784 * 0.74263 / 1.07907
EKSEN = ("bayatlik",)


# ----------------------------------------------------------------- cozuculer
def _wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    s = np.sqrt(w)[:, None]
    return np.linalg.lstsq(X * s, y * np.sqrt(w), rcond=None)[0]


def cozum_nnls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    s = np.sqrt(w)
    b, _ = nnls(X * s[:, None], y * s)
    return b


def cozum_simpleks(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """beta>=0, toplam=1. Tam aktif-kume: butun yuzler taranir (p=4 -> 15 alt kume)."""
    p = X.shape[1]
    s = np.sqrt(w)
    Xs, ys = X * s[:, None], y * s
    en_iyi, en_iyi_j = None, np.inf
    for maske in range(1, 1 << p):
        idx = [k for k in range(p) if maske >> k & 1]
        A = Xs[:, idx]
        if len(idx) == 1:
            b = np.ones(1)
        else:
            # sum(b)=1 -> b_last = 1 - sum(b_rest); indirgenmis serbest problem
            Z = A[:, :-1] - A[:, [-1]]
            r = ys - A[:, -1]
            br = np.linalg.lstsq(Z, r, rcond=None)[0]
            b = np.append(br, 1.0 - br.sum())
        if (b < -1e-9).any():
            continue
        j = float(((A @ b - ys) ** 2).sum())
        if j < en_iyi_j:
            en_iyi_j, en_iyi = j, (idx, np.clip(b, 0.0, None))
    beta = np.zeros(p)
    beta[en_iyi[0]] = en_iyi[1]
    return beta


# ----------------------------------------------------------------- yardimci
def a_mse(e: np.ndarray, w: np.ndarray) -> float:
    return float(np.dot(w, e * e) / w.sum())


def hukum(f: list[float], ad: str) -> tuple[float, float, float, int]:
    fa = np.asarray(f, dtype="float64")
    sh = float(fa.std(ddof=1) / np.sqrt(len(fa)))
    t = float(fa.mean() / sh) if sh > 0 else 0.0
    print(
        f"  {ad:36}{fa.mean():+10.5f}{sh:10.5f}{t:+8.2f}{int((fa > 0).sum()):>5}/{len(fa)}"
        f"{-fa.mean() * SICAK_KATSAYI:+10.5f}"
    )
    return fa.mean(), sh, t, int((fa > 0).sum())


def main() -> int:
    egitim, test = d.cerceveleri_kur()
    tm.kategorik_kodla(egitim, test)
    guc_kenar = ol.guc_kenarlari(test)
    te_s = test[test["soguk_mu"] != 1]

    V = {}
    for b in tm.BLOKLAR:
        _, dogrulama, gercek, soguk = di.blok_parcalari(egitim, b.ad)
        sicak = ~soguk
        dg = dogrulama[sicak]
        w, tani = ol.test_agirliklari(dg, te_s, guc_kenar, eksenler=EKSEN)
        y = np.log1p(np.clip(np.load(DIZIN / f"{b.ad}_gercek.npy").astype("float64"), 0.0, None))
        X = {}
        for t in TOHUMLAR:
            X[t] = np.column_stack(
                [np.load(DIZIN / f"{b.ad}_{t}_{a}_uretim.npy").astype("float64") for a in AILELER]
            )
        assert len(y) == X[TOHUMLAR[0]].shape[0] == len(w) == len(dg)
        V[b.ad] = {
            "y": y,
            "w": w,
            "X": X,
            "trafo": dg["tanim"].to_numpy(),
            "guc": dg["guc"].to_numpy(dtype="float64"),
            "n": len(y),
            "ess": tani["ess_orani"],
        }
        print(
            f"  {b.ad:7} sicak {len(y):>9,}  trafo {pd.unique(dg['tanim']).size:>5}"
            f"  ESS %{100 * tani['ess_orani']:.1f}  kapsanmayan %{100 * tani['kapsanmayan']:.2f}"
        )

    print("\n" + "=" * 104)
    print(
        "URETIM NORMALIZE AGIRLIK: " + "  ".join(f"{a} {v:.4f}" for a, v in zip(AILELER, URETIM_N))
    )
    print("=" * 104)

    # ---------------------------------------------------- 1) taban + protokoller
    def taban_ve(protokol) -> list[float]:  # noqa: ANN001
        f = []
        for b in tm.BLOKLAR:
            v = V[b.ad]
            for t in TOHUMLAR:
                X, y, w = v["X"][t], v["y"], v["w"]
                p0 = X @ URETIM_N
                p1 = protokol(b.ad, t, X, y, w)
                f.append(np.sqrt(a_mse(y - p0, w)) - np.sqrt(a_mse(y - p1, w)))
        return f

    def orakul(coz):  # noqa: ANN001, ANN202
        def _p(bad, t, X, y, w):  # noqa: ANN001, ANN202
            return X @ coz(X, y, w)

        return _p

    def kat_endeksi(trafo: np.ndarray, k: int = 5, tohum: int = 7) -> np.ndarray:
        u = pd.unique(trafo)
        rng = np.random.default_rng(tohum)
        atama = pd.Series(rng.integers(0, k, len(u)), index=u)
        return atama.reindex(trafo).to_numpy()

    KAT = {b.ad: kat_endeksi(V[b.ad]["trafo"]) for b in tm.BLOKLAR}

    def capraz(coz):  # noqa: ANN001, ANN202
        def _p(bad, t, X, y, w):  # noqa: ANN001, ANN202
            kat = KAT[bad]
            p = np.empty_like(y)
            for kk in range(5):
                te = kat == kk
                tr = ~te
                p[te] = X[te] @ coz(X[tr], y[tr], w[tr])
            return p

        return _p

    def blok_disi(coz):  # noqa: ANN001, ANN202
        def _p(bad, t, X, y, w):  # noqa: ANN001, ANN202
            Xs = np.vstack([V[o.ad]["X"][t] for o in tm.BLOKLAR if o.ad != bad])
            ys = np.concatenate([V[o.ad]["y"] for o in tm.BLOKLAR if o.ad != bad])
            ws = np.concatenate([V[o.ad]["w"] for o in tm.BLOKLAR if o.ad != bad])
            return X @ coz(Xs, ys, ws)

        return _p

    print(f"\n  {'PROTOKOL':36}{'fark':>10}{'SH':>10}{'t':>8}{'tohum':>8}{'genele':>10}")
    sonuc = {}
    for ad, prot in (
        ("ORAKUL simpleks (ayni veride)", orakul(cozum_simpleks)),
        ("trafo bazinda 5 kat capraz simpleks", capraz(cozum_simpleks)),
        ("BLOK-DISI simpleks", blok_disi(cozum_simpleks)),
        ("ORAKUL NNLS (ayni veride)", orakul(cozum_nnls)),
        ("trafo capraz NNLS (toplam serbest)", capraz(cozum_nnls)),
        ("BLOK-DISI NNLS", blok_disi(cozum_nnls)),
    ):
        f = taban_ve(prot)
        sonuc[ad] = f
        hukum(f, ad)

    # ---------------------------------------------------- 2) NNLS agirlik toplami
    print("\n" + "-" * 104)
    print("2) NNLS AGIRLIK TOPLAMLARI (orakul, blok x tohum)  +  ORAKUL SIMPLEKS BULUTU")
    print("-" * 104)
    print(f"  {'blok':8}{'tohum':>7}{'NNLS toplam':>13}   " + "".join(f"{a:>11}" for a in AILELER))
    for b in tm.BLOKLAR:
        v = V[b.ad]
        for t in TOHUMLAR:
            bn = cozum_nnls(v["X"][t], v["y"], v["w"])
            print(f"  {b.ad:8}{t:>7}{bn.sum():13.4f}   " + "".join(f"{x:11.4f}" for x in bn))
    print()
    bulut = []
    for b in tm.BLOKLAR:
        v = V[b.ad]
        for t in TOHUMLAR:
            bs = cozum_simpleks(v["X"][t], v["y"], v["w"])
            bulut.append(bs)
            print(
                f"  simpleks {b.ad:8}{t:>7}{bs.sum():13.4f}   " + "".join(f"{x:11.4f}" for x in bs)
            )
    bu = np.array(bulut)
    print("\n  ORAKUL SIMPLEKS ARALIK:")
    for i, a in enumerate(AILELER):
        print(
            f"    {a:10} {bu[:, i].min():.3f} - {bu[:, i].max():.3f}"
            f"   tam sifir {int((bu[:, i] < 1e-9).sum())}/9   uretim {URETIM_N[i]:.4f}"
        )

    # ---------------------------------------------------- 3) KOR SKALAR c
    print("\n" + "-" * 104)
    print("3) KOR SKALAR c TESTI: log tahmini tek bir c ile carp (harman DEGIL, SEVIYE)")
    print("-" * 104)
    print(
        f"  {'blok':8}{'c_opt(kendi)':>14}{'fark':>11}   NNLS-capraz-fark   c_opt(blok-disi)  fark"
    )
    for b in tm.BLOKLAR:
        v = V[b.ad]
        # kendi bloguna uydurulmus tek skalar (tohum ortalamali)
        num = den = 0.0
        for t in TOHUMLAR:
            p0 = v["X"][t] @ URETIM_N
            num += float(np.dot(v["w"], p0 * v["y"]))
            den += float(np.dot(v["w"], p0 * p0))
        c = num / den
        fk = []
        for t in TOHUMLAR:
            p0 = v["X"][t] @ URETIM_N
            fk.append(np.sqrt(a_mse(v["y"] - p0, v["w"])) - np.sqrt(a_mse(v["y"] - c * p0, v["w"])))
        # blok-disi c
        num2 = den2 = 0.0
        for o in tm.BLOKLAR:
            if o.ad == b.ad:
                continue
            vo = V[o.ad]
            for t in TOHUMLAR:
                q = vo["X"][t] @ URETIM_N
                num2 += float(np.dot(vo["w"], q * vo["y"]))
                den2 += float(np.dot(vo["w"], q * q))
        c2 = num2 / den2
        fk2 = []
        for t in TOHUMLAR:
            p0 = v["X"][t] @ URETIM_N
            fk2.append(
                np.sqrt(a_mse(v["y"] - p0, v["w"])) - np.sqrt(a_mse(v["y"] - c2 * p0, v["w"]))
            )
        nn = [
            x
            for i, x in enumerate(sonuc["trafo capraz NNLS (toplam serbest)"])
            if [bb.ad for bb in tm.BLOKLAR for _ in TOHUMLAR][i] == b.ad
        ]
        print(
            f"  {b.ad:8}{c:14.4f}{np.mean(fk):+11.5f}   {np.mean(nn):+16.5f}"
            f"   {c2:16.4f}  {np.mean(fk2):+10.5f}"
        )

    # ---------------------------------------------------- 4) KIRPILMIS
    print("\n" + "-" * 104)
    print("4) KIRPILMIS HUKUM -- trafo capraz SIMPLEKS (K trafo atilarak, EN COK KAZANANLAR)")
    print("-" * 104)

    # kirpma kumesi: tohumlar uzerinde ortalanmis d(MSE)
    KIRP = {}
    for b in tm.BLOKLAR:
        v = V[b.ad]
        dm = np.zeros(v["n"])
        for t in TOHUMLAR:
            X, y, w = v["X"][t], v["y"], v["w"]
            e0 = y - X @ URETIM_N
            e1 = y - capraz(cozum_simpleks)(b.ad, t, X, y, w)
            dm += w * (e0 * e0 - e1 * e1)
        KIRP[b.ad] = pd.Series(dm / len(TOHUMLAR)).groupby(pd.Series(v["trafo"])).sum()

    print(f"  {'K':>4}{'fark':>11}{'SH':>10}{'t':>8}{'tohum':>8}{'genele':>10}")
    for K in (0, 1, 5, 10, 25, 50):
        f = []
        for b in tm.BLOKLAR:
            v = V[b.ad]
            at = set(KIRP[b.ad].nlargest(K).index) if K else set()
            tut = ~pd.Series(v["trafo"]).isin(at).to_numpy()
            for t in TOHUMLAR:
                X, y, w = v["X"][t], v["y"], v["w"]
                p0 = X @ URETIM_N
                p1 = capraz(cozum_simpleks)(b.ad, t, X, y, w)
                f.append(
                    np.sqrt(a_mse((y - p0)[tut], w[tut])) - np.sqrt(a_mse((y - p1)[tut], w[tut]))
                )
        fa = np.array(f)
        sh = float(fa.std(ddof=1) / np.sqrt(len(fa)))
        print(
            f"  {K:>4}{fa.mean():+11.5f}{sh:10.5f}{fa.mean() / sh:+8.2f}"
            f"{int((fa > 0).sum()):>5}/9{-fa.mean() * SICAK_KATSAYI:+10.5f}"
        )

    print("\n  YOGUNLASMA (trafo capraz simpleks, k=3 ortalamali d(MSE))")
    print(f"  {'blok':8}{'trafo':>8}{'toplam d(MSE)':>16}{'EN BUYUK':>11}{'ilk5':>9}")
    for b in tm.BLOKLAR:
        pay = KIRP[b.ad].sort_values(ascending=False)
        top = pay.sum()
        print(
            f"  {b.ad:8}{pay.size:>8}{top:16.2f}"
            f"{100 * pay.iloc[0] / top:10.1f}%{100 * pay.iloc[:5].sum() / top:8.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
