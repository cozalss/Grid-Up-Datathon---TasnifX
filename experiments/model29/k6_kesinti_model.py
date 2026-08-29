"""K6 -- kesinti ozellikleriyle / ozelliksiz IKI model, AYNI tohum ve turlarla.

Amac kalite degil YON: iki tahminin FARKI, kesinti bilgisinin tek basina
tasidigi bileseni izole eder (base ozellikler ortak oldugu icin sadelesir).

Tezgah m33_durust/m71_nihai_hava ile ayni: sizintisiz ileri-pencere, kesim
2026-03-31, hedef test.csv (2026-04-01..07-31). huber(a=2, l2=20) + l1 harmani,
her biri 3 tohum.

Cikti (npy, .gitignore'da):
    k6_p_taban.npy     base ozellikler
    k6_p_kesinti.npy   base + kesinti ozellikleri
    k6_kesinti_model.json  egitim kunyesi + ozellik onemleri
"""

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
KOK = os.path.dirname(os.path.dirname(BURA))
sys.path.insert(0, BURA)

from k5_kesinti_veri import anahtarla, filo, kesinti_panel  # noqa: E402
from m30_ozellik import kur, yukle_ham  # noqa: E402
from m33_durust import VARSAYILAN as V0  # noqa: E402
from m33_durust import hizala  # noqa: E402

KESIM = "2026-03-31"
# kesinti paneli 2025-05-08'den itibaren KESINTISIZ; oncesi (2025-01..04) YOK.
# Hedef penceresi kapsanan gunlere dusen kesimler secildi.
AY = [
    "2025-07-31",
    "2025-08-31",
    "2025-09-30",
    "2025-10-31",
    "2025-11-30",
    "2025-12-31",
]
TUR_HUB = 185
TUR_L1 = 520
TOHUM = (7, 17, 27)
V = dict(V0)
V["num_threads"] = int(os.environ.get("IPLIK", "14"))
HUB = dict(objective="huber", alpha=2.0, lambda_l2=20.0)
L1 = dict(objective="l1")


def main():
    t0 = time.time()
    tr, te = yukle_ham()
    anahtarla(tr)
    anahtarla(te)
    KP = kesinti_panel(filo(tr, te))
    KKOL = list(KP.columns)
    print(f"kesinti paneli {KP.shape} ({time.time() - t0:.0f}s)", flush=True)

    def ekle(X, ilce, tarih):
        idx = pd.MultiIndex.from_arrays([np.asarray(ilce, dtype=object), np.asarray(tarih)])
        d = KP.reindex(idx)
        return pd.concat(
            [
                X.reset_index(drop=True),
                pd.DataFrame({c: d[c].to_numpy(dtype=np.float32) for c in KKOL}),
            ],
            axis=1,
        )

    Xs, Xks, ys = [], [], []
    tavan = pd.Timestamp(KESIM)
    for k in AY:
        kk = pd.Timestamp(k)
        son = min(kk + pd.DateOffset(months=4), tavan)
        gec = tr[tr.tarih <= kk]
        hed = tr[(tr.tarih > kk) & (tr.tarih <= son)]
        if len(hed) == 0:
            continue
        X = kur(gec, hed, kk, set(gec.tanim))
        Xs.append(X)
        Xks.append(ekle(X, hed.ilce_key.to_numpy(), hed.tarih.to_numpy()))
        ys.append(np.log1p(hed.tuketim.to_numpy()))
        print(f"  kesim {k}: {len(X):,} satir ({time.time() - t0:.0f}s)", flush=True)
    Xtr = pd.concat(Xs, ignore_index=True)
    Xktr = pd.concat(Xks, ignore_index=True)
    ytr = np.concatenate(ys)
    del Xs, Xks, ys

    Xte = kur(tr, te, tavan, set(tr.tanim))
    Xkte = ekle(Xte, te.ilce_key.to_numpy(), te.tarih.to_numpy())
    print(f"  TEST {len(Xte):,} satir ({time.time() - t0:.0f}s)", flush=True)

    Xtr, Xte = hizala(Xtr, Xte)
    Xte = Xte[Xtr.columns]
    Xktr, Xkte = hizala(Xktr, Xkte)
    Xkte = Xkte[Xktr.columns]

    nan_te = {c: float(Xkte[c].isna().mean()) for c in KKOL}
    kunye = dict(
        kesim=KESIM,
        kesimler=AY,
        egitim_satir=int(len(Xtr)),
        test_satir=int(len(Xte)),
        taban_ozellik=int(Xtr.shape[1]),
        kesinti_ozellik=len(KKOL),
        tur=[TUR_HUB, TUR_L1],
        tohum=list(TOHUM),
        kesinti_test_nan={k: round(v, 6) for k, v in nan_te.items() if v > 0},
        kesinti_egitim_nan_ort=round(float(Xktr[KKOL].isna().mean().mean()), 6),
    )
    print(json.dumps(kunye, indent=1), flush=True)

    onem = {}
    cikti = {}
    for ad, Xa, Xb in (("taban", Xtr, Xte), ("kesinti", Xktr, Xkte)):
        ds = lgb.Dataset(Xa, ytr)
        parca = {}
        for nm, pk, tur in (("huber", HUB, TUR_HUB), ("l1", L1, TUR_L1)):
            acc = []
            for s in TOHUM:
                p = dict(V)
                p.update(pk)
                p.update(seed=s, bagging_seed=s + 1, feature_fraction_seed=s + 2)
                m = lgb.train(p, ds, tur)
                acc.append(m.predict(Xb))
                if ad == "kesinti" and nm == "huber" and s == TOHUM[0]:
                    g = pd.Series(m.feature_importance("gain"), index=Xa.columns)
                    onem["toplam_gain"] = float(g.sum())
                    onem["kesinti_gain_payi"] = float(g[KKOL].sum() / g.sum())
                    onem["en_iyi_10_kesinti"] = {
                        k: round(float(v / g.sum()), 5)
                        for k, v in g[KKOL].sort_values(ascending=False).head(10).items()
                    }
                print(f"  {ad}/{nm} tohum {s} ({time.time() - t0:.0f}s)", flush=True)
            parca[nm] = np.mean(acc, axis=0)
        cikti[ad] = (parca["huber"] + parca["l1"]) / 2
        np.save(os.path.join(BURA, f"k6_p_{ad}.npy"), cikti[ad])
        del ds

    d = cikti["kesinti"] - cikti["taban"]
    kunye["onem"] = onem
    kunye["delta"] = dict(
        Q=float((d**2).mean()),
        ort=float(d.mean()),
        std=float(d.std()),
        mutlak_maks=float(np.abs(d).max()),
    )
    print("DELTA:", json.dumps(kunye["delta"], indent=1))
    json.dump(
        kunye,
        open(os.path.join(BURA, "k6_kesinti_model.json"), "w", encoding="utf-8"),
        indent=1,
        ensure_ascii=False,
    )
    print(f"BITTI ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
