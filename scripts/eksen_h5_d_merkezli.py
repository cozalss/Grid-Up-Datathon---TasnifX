# ruff: noqa
"""H5 -- capanin YANLILIGI cikarilmis hali. Adim 4'un acigi kapatiliyor.

Ham capa taban'a gore +0,30 log kayik; w>0'un zarari SALT bu kaymadan
geliyor olabilir. Burada capa, tahmin aninda BILINEN taban ortalamasina
yeniden merkezlenir (sizinti yok: gercek y kullanilmaz) ve ayrica
capa'nin ARTIK bilgisi (capa - lok ortalamasi) ayri test edilir.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "scripts"))
import olcut  # noqa: E402

GE = KOK / "data" / "interim" / "gun_ekseni"
CIK = KOK / "reports" / "eksen_h5"
BLOK = {
    "yaz25": (pd.Timestamp("2025-04-01"), (1000, 1001, 1002, 1003, 1004, 1005)),
    "guz25": (pd.Timestamp("2025-08-01"), (1000, 1001, 1002)),
}
WLER = tuple(np.round(np.arange(0.0, 0.61, 0.05), 3))
ALFALAR = (0.0, 0.5, 0.75)
KLER = (0, 1, 5, 10, 25, 50)
P_SOGUK = 0.22159


def main() -> int:
    out = []

    def yaz(s=""):
        print(s)
        out.append(s)

    tr = pd.read_csv(
        KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str}, parse_dates=["tarih"]
    )
    lok_of = tr.groupby("tanim", observed=True)["lokasyon"].first()
    eg = pd.read_parquet(
        KOK / "data/interim/deney/egitim.parquet",
        columns=["_blok", "soguk_mu", "tanim", "tarih", "guc", "t_son_kayit_yasi", "ufuk_gun"],
    )
    te = pd.read_parquet(
        KOK / "data/interim/deney/test.parquet",
        columns=["soguk_mu", "guc", "t_son_kayit_yasi", "ufuk_gun"],
    )
    ts = te[te["soguk_mu"] == 1].reset_index(drop=True)
    gk = olcut.guc_kenarlari(ts)

    for blok, (kesme, tohumlar) in BLOK.items():
        meta = pd.read_parquet(GE / f"{blok}_meta.parquet")
        bit = meta["tarih"].max()
        ly = np.log1p(np.clip(meta["y"].to_numpy("float64"), 0, None))
        tanim = meta["tanim"].to_numpy()
        guc = meta["guc"].to_numpy("float64")
        lok = lok_of.reindex(pd.Index(tanim)).to_numpy()
        d = eg[(eg["_blok"] == blok) & (eg["soguk_mu"] == 1)].reset_index(drop=True)
        w, _ = olcut.test_agirliklari(d, ts, gk)
        tabanlar = [np.load(GE / f"{blok}_{t}_taban.npy").astype("float64") for t in tohumlar]
        tb0 = np.mean(tabanlar, axis=0)

        bas = kesme - pd.Timedelta(days=90)
        p = tr[(tr["tarih"] >= bas) & (tr["tarih"] < kesme)]
        g = p.groupby("tanim", observed=True)
        dd = pd.DataFrame(
            {"ort": g["tuketim"].mean(), "guc": g["guc"].first(), "lok": g["lokasyon"].first()}
        )
        agg = dd.groupby("lok").agg(yuk=("ort", "sum"), kap=("guc", "sum"))
        yog = (agg["yuk"] / agg["kap"].clip(lower=1.0)).reindex(pd.Index(lok)).to_numpy("float64")
        ilk = tr.groupby("tanim", observed=True)["tarih"].min()
        gucs = tr.groupby("tanim", observed=True)["guc"].first()
        loks = tr.groupby("tanim", observed=True)["lokasyon"].first()
        yeni = ilk[(ilk >= kesme) & (ilk <= bit)].index
        yk = (
            pd.DataFrame({"guc": gucs.loc[yeni], "lok": loks.loc[yeni]}).groupby("lok")["guc"].sum()
        )
        payv = (yk / (agg["kap"].reindex(yk.index).fillna(0.0) + yk)).clip(0, 0.9)
        payv = payv.reindex(pd.Index(lok)).fillna(0.0).to_numpy("float64")

        yaz(
            f"\n{'=' * 78}\nBLOK {blok}  taban MSE {float(np.dot(w, (tb0 - ly) ** 2) / w.sum()):.5f}"
        )
        for alfa in ALFALAR:
            capa = np.log1p(yog * guc) + np.log(np.clip(1.0 - alfa * payv, 1e-3, None))
            # tahmin aninda bilinen kayma duzeltmesi: taban ortalamasina merkezle
            kayma = float(np.dot(w, capa - tb0) / w.sum())
            capa_m = capa - kayma
            en = None
            for wv in WLER:
                dl = [
                    float(np.dot(w, ((1 - wv) * t + wv * capa_m - ly) ** 2) / w.sum())
                    - float(np.dot(w, (t - ly) ** 2) / w.sum())
                    for t in tabanlar
                ]
                if en is None or np.mean(dl) < en[1]:
                    en = (wv, float(np.mean(dl)), float(np.std(dl, ddof=1) / np.sqrt(len(dl))))
            yaz(
                f"  alfa={alfa:<5} kayma {kayma:+.4f}  MERKEZLI en iyi w={en[0]:.2f}"
                f"  dMSE_soguk {en[1]:+.5f} (sh {en[2]:.5f})  ~ toplam {en[1] * P_SOGUK:+.6f}"
            )
            if en[1] < -1e-6 and en[0] > 0:
                pr = (1 - en[0]) * tb0 + en[0] * capa_m
                fark = w * ((pr - ly) ** 2 - (tb0 - ly) ** 2)
                s = pd.Series(fark).groupby(pd.Series(tanim)).sum().sort_values()
                for K in KLER:
                    m = ~pd.Series(tanim).isin(set(s.index[:K])).to_numpy()
                    yaz(f"      K={K:<3} dMSE_soguk {float(fark[m].sum() / w.sum()):+.5f}")
    (CIK / "adim4_merkezli.txt").write_text("\n".join(out), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
