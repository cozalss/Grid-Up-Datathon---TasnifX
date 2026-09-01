"""p18c: SOGUK URETIM HARMANI CAT-TEKIL MI, ESIT MI? -- AMPIRIK HUKUM.

Celiski:
  * ``tuketim_model.REJIM_AYARLARI['soguk']['agirlik'] = {"cat": 1.0}``
  * p02/p06/p11/p14 olcumleri ``soguk_tahmin_*.npz``in BUTUN anahtarlarini
    ESIT agirlikla ortaliyor (cat+xgb+lgbm).

Kanit yolu: ``p06_test_soguk_aile.npy`` TEST'in soguk satirlari icin
(cat, xgb, lgbm) log tahminlerini tasiyor (tohum 1000-1002 ortalamasi,
maske 1.00, cat depth 7 -- uretim soguk uzmaninin ayari). Gonderim
dosyalarinin ayni satirlardaki log tahmini bu uc kolona REGRESE edilir:

    log1p(tuketim_gonderim) ~ a + b_cat*cat + b_xgb*xgb + b_lgbm*lgbm

Son islem (buzme/olcek) LOG uzayinda AFFINE oldugu icin katsayilarin
ORANI harmanin oranini korur. Cat-tekil bir uretimde b_xgb ve b_lgbm
sifira yakin cikmali; esit harmanda ucu de yakin olmali.

Kova bazli buzme afin ama HUCRE BAZLI sabitli oldugu icin, ek olarak
kVA kovasi icinde merkezlenmis regresyon da kosulur.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

KOK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BURA = os.path.dirname(os.path.abspath(__file__))
S = os.path.join(KOK, "submissions")
A_YOL = os.path.join(BURA, "p_kalici", "aday_csv", "p06_test_soguk_aile.npy")

ADAYLAR = [
    "tuketim_YP_seviye.csv",
    "tuketim_m6_ikiyon.csv",
    "tuketim_K_yenibas.csv",
    "tuketim_v90_temiz_sota.csv",
    "tuketim_v83_sicak_optimum.csv",
    "tuketim_v80_optimum.csv",
    "tuketim_v27_v18hedge.csv",
    "tuketim_v46_gun.csv",
    "tuketim_v30_buzme.csv",
]


def main() -> None:
    te = pd.read_parquet(os.path.join(KOK, "data/interim/deney/test.parquet"))
    soguk = (te["soguk_mu"] == 1).to_numpy()
    ids = te["id"].to_numpy()
    guc = te["guc"].to_numpy(dtype="float64")[soguk]
    A = np.load(A_YOL).astype("float64")
    assert A.shape == (int(soguk.sum()), 3), A.shape
    ham = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"), usecols=["id"])
    ID_SIRA = ham["id"].to_numpy()

    kenar = [0, 60, 110, 175, 260, 410, 650, 1050, 1e9]
    kova = np.digitize(guc, kenar) - 1

    R: dict = {
        "kaynak_dizi": os.path.relpath(A_YOL, KOK).replace("\\", "/"),
        "n_soguk": int(soguk.sum()),
        "aile_sirasi": ["cat", "xgb", "lgbm"],
        "aile_ortalamasi": {
            a: round(float(A[:, i].mean()), 4) for i, a in enumerate(("cat", "xgb", "lgbm"))
        },
        "aile_korelasyonu": np.round(np.corrcoef(A.T), 4).tolist(),
        "dosyalar": {},
    }

    for f in ADAYLAR:
        yol = os.path.join(S, f)
        if not os.path.exists(yol):
            continue
        d = pd.read_csv(yol)
        k = "tuketim" if "tuketim" in d.columns else d.columns[-1]
        if not np.array_equal(d["id"].to_numpy(), ID_SIRA):
            pos = pd.Index(d["id"]).get_indexer(ID_SIRA)
            if (pos < 0).any():
                continue
            d = d.iloc[pos].reset_index(drop=True)
        # test.parquet sirasi ile hizala
        pos = pd.Index(ID_SIRA).get_indexer(ids)
        L = np.log1p(d[k].to_numpy(dtype="float64"))[pos][soguk]

        cikti: dict = {}
        # 1) ham regresyon
        X = np.c_[np.ones(len(L)), A]
        b, *_ = np.linalg.lstsq(X, L, rcond=None)
        art = L - X @ b
        cikti["regresyon_ham"] = {
            "sabit": round(float(b[0]), 4),
            "b_cat": round(float(b[1]), 4),
            "b_xgb": round(float(b[2]), 4),
            "b_lgbm": round(float(b[3]), 4),
            "R2": round(1 - float(art.var() / L.var()), 5),
        }

        # 2) kVA kovasi icinde merkezlenmis (hucre sabitleri temizlenir)
        def merkez(v: np.ndarray) -> np.ndarray:
            s = pd.Series(v)
            return (s - s.groupby(kova).transform("mean")).to_numpy()

        Xc = np.c_[[merkez(A[:, i]) for i in range(3)]].T
        Lc = merkez(L)
        bc, *_ = np.linalg.lstsq(Xc, Lc, rcond=None)
        artc = Lc - Xc @ bc
        top = float(bc.sum())
        cikti["regresyon_kova_merkezli"] = {
            "b_cat": round(float(bc[0]), 4),
            "b_xgb": round(float(bc[1]), 4),
            "b_lgbm": round(float(bc[2]), 4),
            "toplam": round(top, 4),
            "PAY_cat": round(float(bc[0]) / top, 4) if abs(top) > 1e-9 else None,
            "PAY_xgb": round(float(bc[1]) / top, 4) if abs(top) > 1e-9 else None,
            "PAY_lgbm": round(float(bc[2]) / top, 4) if abs(top) > 1e-9 else None,
            "R2": round(1 - float(artc.var() / Lc.var()), 5),
        }
        # 3) iki hipotezin dogrudan uyumu (olcek serbest, afin uydurma)
        for ad, v in (
            ("CAT_TEKIL", A[:, 0]),
            ("ESIT", A.mean(axis=1)),
            ("URETIM_SICAK_3_1_1", A @ np.array([0.6, 0.2, 0.2])),
        ):
            Xh = np.c_[np.ones(len(L)), v]
            bh, *_ = np.linalg.lstsq(Xh, L, rcond=None)
            arth = L - Xh @ bh
            cikti[f"hipotez_{ad}"] = {
                "R2": round(1 - float(arth.var() / L.var()), 5),
                "kor": round(float(np.corrcoef(L, v)[0, 1]), 5),
                "rms_artik": round(float(np.sqrt((arth * arth).mean())), 5),
            }
        R["dosyalar"][f] = cikti
        rk = cikti["regresyon_kova_merkezli"]
        print(
            f"{f:32} PAY cat={rk['PAY_cat']} xgb={rk['PAY_xgb']} lgbm={rk['PAY_lgbm']}"
            f"  toplam={rk['toplam']}  R2={rk['R2']}"
            f"  | kor(CAT)={cikti['hipotez_CAT_TEKIL']['kor']}"
            f" kor(ESIT)={cikti['hipotez_ESIT']['kor']}"
        )

    yol = os.path.join(BURA, "p_kalici", "p18_harman_hukmu.json")
    with open(yol, "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)
    print(f"\nkayit: {yol}")


if __name__ == "__main__":
    main()
