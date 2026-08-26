# ruff: noqa
"""H1 EK-2 -- tek pozitif adayin (blok,tohum) KIRILIMI ve GUN-DUSURME saglamligi.

``eksen6_gun_bant_uretim.py`` B tablosunda tek havuz-negatif aday
``(c_hafta=1,0, c_dusuk=1,335, c_kalan=1,0)`` cikti: havuz -0,00016, t=-2,21.
Havuzlanmis skora GUVENME kurali geregi burada KIRILIM verilir; ayrica
kazancin birkac gune mi dayandigi olculur (gun bazli kirpma).
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
import olcut as ol  # noqa: E402
import tuketim_model as tm  # noqa: E402

DIZIN = KOK / "data" / "interim" / "aile_onbellek"
TOHUMLAR = (1000, 1001, 1002)
AILELER = ("cat", "xgb", "lgbm", "sinir_agi")
AGIRLIK = (3.0, 1.0, 1.0, 1.4)
SICAK_KATSAYI = 0.7784 * 0.74263 / 1.07907
LB_RMSLE = 1.01591
W_MA = 15
C_URETIM = 1.335


def blend(blok, tohum):  # noqa: ANN001
    return sum(
        w * np.load(DIZIN / f"{blok}_{tohum}_{a}_uretim.npy").astype("float64")
        for a, w in zip(AILELER, AGIRLIK, strict=True)
    ) / sum(AGIRLIK)


def amse(e, w):  # noqa: ANN001
    return float(np.dot(w, e * e) / w.sum())


def gun_etkisi_kod(trafo_kod, gun_kod, r, n_gun):  # noqa: ANN001
    nt = np.bincount(trafo_kod)
    mt = np.bincount(trafo_kod, weights=r) / nt
    c = r - mt[trafo_kod]
    nd = np.bincount(gun_kod, minlength=n_gun).astype("float64")
    b = np.bincount(gun_kod, weights=c, minlength=n_gun) / np.maximum(nd, 1.0)
    return b - b.mean()


def uc_bilesen(b, hafta, W=W_MA):  # noqa: ANN001
    h = np.zeros_like(b)
    for k in range(7):
        m = hafta == k
        if m.any():
            h[m] = b[m].mean()
    h = h - h.mean()
    kalan = b - h
    s = pd.Series(kalan).rolling(W, center=True, min_periods=1).mean().to_numpy()
    s = s - s.mean()
    e = kalan - s
    return h, s, e - e.mean()


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    t0 = time.time()
    egitim, test = d.cerceveleri_kur()
    tm.kategorik_kodla(egitim, test)
    guc_kenar = ol.guc_kenarlari(test)
    te_s = test[test["soguk_mu"] != 1]

    V = {}
    for b in tm.BLOKLAR:
        _, dogrulama, gercek, soguk = di.blok_parcalari(egitim, b.ad)
        dg = dogrulama[~soguk]
        w, _ = ol.test_agirliklari(dg, te_s, guc_kenar, eksenler=("bayatlik",))
        lg = np.log1p(dg["guc"].to_numpy(dtype="float64"))
        gk, gunler = pd.factorize(pd.to_datetime(dg["tarih"]), sort=True)
        tk, _ = pd.factorize(dg["tanim"].to_numpy())
        V[b.ad] = {
            "w": w,
            "g": np.log1p(np.clip(gercek[~soguk], 0.0, None)) - lg,
            "r": {t: blend(b.ad, t) - lg for t in TOHUMLAR},
            "kod": gk.astype("int64"),
            "trafo": tk.astype("int64"),
            "gunler": pd.DatetimeIndex(gunler),
        }

    ADAYLAR = {
        "A (1,000 / 1,335 / 1,000)": (1.0, C_URETIM, 1.0),
        "B (1,335 / 1,335 / 1,000)": (C_URETIM, C_URETIM, 1.0),
        "C (0,933 / 1,335 / 1,335)": (0.933, C_URETIM, C_URETIM),
        "D (0,933 / 1,556 / 1,000)": (0.933, 1.556, 1.0),
    }

    print("=" * 104)
    print("(blok,tohum) KIRILIMI -- dRMSLE_sicak, TABAN uniform 1,335 (NEGATIF = kazanc)")
    print("=" * 104)
    print(f"  {'aday':28}", end="")
    for b in tm.BLOKLAR:
        for t in TOHUMLAR:
            print(f"{b.ad[:3] + str(t)[-1]:>9}", end="")
    print(f"{'ORT':>10}{'SH':>9}{'t':>7}")
    for ad, (ch, cs, ce) in ADAYLAR.items():
        f = []
        print(f"  {ad:28}", end="")
        for b in tm.BLOKLAR:
            v = V[b.ad]
            ng = len(v["gunler"])
            for t in TOHUMLAR:
                r = v["r"][t]
                bm = gun_etkisi_kod(v["trafo"], v["kod"], r, ng)
                h, s, e = uc_bilesen(bm, v["gunler"].dayofweek.to_numpy())

                def sat(u):  # noqa: ANN001, ANN202
                    x = u[v["kod"]]
                    return x - x.mean()

                base = r + (C_URETIM - 1) * (sat(h) + sat(s) + sat(e))
                cand = r + (ch - 1) * sat(h) + (cs - 1) * sat(s) + (ce - 1) * sat(e)
                dv = np.sqrt(amse(v["g"] - cand, v["w"])) - np.sqrt(amse(v["g"] - base, v["w"]))
                f.append(dv)
                print(f"{dv:+9.5f}", end="")
        f = np.array(f)
        sh = float(f.std(ddof=1) / np.sqrt(len(f)))
        print(f"{f.mean():+10.5f}{sh:9.5f}{f.mean() / sh:+7.2f}")

    # --------------------------------------------- GUN DUSURME (kirpma benzeri)
    print("\n" + "=" * 104)
    print("GUN DUSURME -- A adayinin blok kazanci, en cok KATKI veren K gun atilarak")
    print("  (kalici kural 1'in gun eksenindeki karsiligi; sicak tarafta trafo degil GUN)")
    print("=" * 104)
    ch, cs, ce = ADAYLAR["A (1,000 / 1,335 / 1,000)"]
    print(f"  {'blok':8}" + "".join(f"{'K=' + str(k):>11}" for k in (0, 1, 5, 10, 25, 50)))
    for b in tm.BLOKLAR:
        v = V[b.ad]
        ng = len(v["gunler"])
        r = v["r"][TOHUMLAR[0]]
        r = np.mean([v["r"][t] for t in TOHUMLAR], axis=0)
        bm = gun_etkisi_kod(v["trafo"], v["kod"], r, ng)
        h, s, e = uc_bilesen(bm, v["gunler"].dayofweek.to_numpy())

        def sat(u):  # noqa: ANN001, ANN202
            x = u[v["kod"]]
            return x - x.mean()

        base = r + (C_URETIM - 1) * (sat(h) + sat(s) + sat(e))
        cand = r + (ch - 1) * sat(h) + (cs - 1) * sat(s) + (ce - 1) * sat(e)
        e0 = (v["g"] - base) ** 2
        e1 = (v["g"] - cand) ** 2
        # gun basina agirlikli katki farki (pozitif = o gun kazandiriyor)
        katki = np.bincount(v["kod"], weights=v["w"] * (e0 - e1), minlength=ng)
        sira = np.argsort(-katki)
        print(f"  {b.ad:8}", end="")
        for K in (0, 1, 5, 10, 25, 50):
            at = np.zeros(ng, dtype=bool)
            at[sira[:K]] = True
            tut = ~at[v["kod"]]
            d0 = np.sqrt(amse((v["g"] - base)[tut], v["w"][tut]))
            d1 = np.sqrt(amse((v["g"] - cand)[tut], v["w"][tut]))
            print(f"{d1 - d0:+11.5f}", end="")
        print()

    print(f"\n  toplam sure {time.time() - t0:.0f} sn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
