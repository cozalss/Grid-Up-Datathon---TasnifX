"""C7: kazanan soguk yapilandirmasini TEST tarafinda uret, v83 uzerine yaz.

Delta yaklasimi (zincir-agnostik):
    r      = lg - log1p(guc)
    zincir = ort(r) + beta*(r - ort(r)) + log1p(guc) + delta
    d_log  = zincir(yeni) - zincir(eski)
           = (1-beta)*(ort r_y - ort r_e) + beta*(r_y - r_e)
``delta`` ve gun olcegi iki tarafta AYNI oldugu icin sadelesir; boylece v83'un
gercek zincirini bilmek gerekmez.

    uv run python experiments/olcut_onarim/uret_v107.py --yeni <ayar> [--tohum 3]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import deney as d, deney_ileri as di, tuketim_model as tm  # noqa
import tezgah as tz  # noqa

BETA = 0.60
TABAN_DOSYA = KOK / "submissions" / "tuketim_v102_kappa_optimum.csv"
BURA = Path(__file__).resolve().parent


def test_tahmin(ayar: str, egitim, test, uretim, tohumlar, soguk_test) -> np.ndarray:
    yol = BURA / "onbellek" / f"TEST_{ayar}.npy"
    if yol.exists():
        return np.load(yol)
    aile, ust = tz.AYARLAR[ayar]
    kols = [k for k in uretim if k not in set(tz.CIKARIM.get(ayar, ()))]
    out = []
    hedef = test[soguk_test]
    for t in tohumlar:
        t0 = time.time()
        maskeli = d.soguk_maskele(egitim, kols, 1.00, t)
        lg = di.egit_tahmin(aile, maskeli, hedef, kols, t, **ust)
        del maskeli
        out.append(lg)
        print(f"    TEST {ayar:16} tohum {t} ({time.time() - t0:.0f}s)", flush=True)
    a = np.asarray(out, dtype="float64")
    np.save(yol, a)
    return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yeni", required=True)
    ap.add_argument("--eski", default="cat_d7")
    ap.add_argument("--tohum", type=int, default=3)
    ap.add_argument("--cikti", default="tuketim_v107_soguk_onarim.csv")
    a = ap.parse_args()
    tohumlar = tuple(1000 + i for i in range(a.tohum))

    egitim, uretim = tz.veri()
    test = pd.read_parquet(KOK / "data/interim/deney/test.parquet")
    tm.kategorik_kodla(egitim, test)
    soguk = (test["soguk_mu"] == 1).to_numpy()
    print(f"  test {len(test):,} satir, soguk {soguk.sum():,}")

    lg_e = test_tahmin(a.eski, egitim, test, uretim, tohumlar, soguk).mean(axis=0)
    lg_y = test_tahmin(a.yeni, egitim, test, uretim, tohumlar, soguk).mean(axis=0)
    lgc = np.log1p(test.loc[soguk, "guc"].to_numpy(dtype="float64"))
    r_e, r_y = lg_e - lgc, lg_y - lgc
    d_log = (1 - BETA) * (r_y.mean() - r_e.mean()) + BETA * (r_y - r_e)
    print(f"  d_log: ort {d_log.mean():+.5f} std {d_log.std():.5f} |max| {np.abs(d_log).max():.4f}")

    taban = pd.read_csv(TABAN_DOSYA)
    hedef_kol = [c for c in taban.columns if c != "id"][0]
    yeni = taban.copy()
    v = np.log1p(taban[hedef_kol].to_numpy(dtype="float64"))
    # test.parquet ve gonderim id hizasi
    idx = pd.Series(np.arange(len(taban)), index=taban["id"].to_numpy())
    konum = idx.reindex(test.loc[soguk, "id"].to_numpy()).to_numpy()
    assert not np.isnan(konum).any(), "id hizasi bozuk"
    konum = konum.astype(int)
    # YALNIZ soguk satirlar yeniden hesaplanir; digerleri BIREBIR taban
    # (log1p/expm1 gidis-donusu 1e-10 mertebesinde sahte fark uretiyordu).
    deger = taban[hedef_kol].to_numpy(dtype="float64").copy()
    deger[konum] = np.clip(np.expm1(v[konum] + d_log), 0.0, None)
    yeni[hedef_kol] = deger
    yol = KOK / "submissions" / a.cikti
    yeni.to_csv(yol, index=False)
    print(f"  YAZILDI {yol}")

    # Q (v102'ye gore)
    for ref_ad in ("tuketim_v102_kappa_optimum.csv", "tuketim_v83_sicak_optimum.csv"):
        ref = pd.read_csv(KOK / "submissions" / ref_ad)
        rk = [c for c in ref.columns if c != "id"][0]
        assert (ref["id"].to_numpy() == yeni["id"].to_numpy()).all()
        dd = np.log1p(yeni[hedef_kol].to_numpy()) - np.log1p(ref[rk].to_numpy())
        print(f"  Q({ref_ad}) = {float(np.mean(dd**2)):.8f}   degisen {int((dd != 0).sum()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
