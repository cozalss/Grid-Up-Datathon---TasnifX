"""Z2b -- KOSULLU DAGILIM (kantil) yeniden kurulumu; nokta kestirimi degil.

Hata haritasinin en kararli bulgusu: EN DUSUK SEVIYE desilinde tahminleri
sistematik olarak +0,040 log SISIRIYORUZ; gorulebilen hatanin %22 kadari orada.
Bunun mekanik bir sebebi var: uretim hatti Huber(alfa=2) ile egitiliyor.
Huber, kosullu MEDYANA yakin bir kestirim verir. Dusuk tuketimli / soguk
trafolarda kosullu log1p dagilimi SOLA CARPIK (sifir kutlesi + uzun asagi
kuyruk), dolayisiyla medyan > ortalama -> sistematik FAZLA tahmin. Metrik ise
log uzayinda L2, yani dogru hedef kosullu ORTALAMA.

Bu aday kosullu ortalamayi dogrudan degil, KOSULLU DAGILIMI kurup integre
ederek bulur:
    7 kantil (0,05 ... 0,95) LightGBM pinball kaybiyla ayri ayri kestirilir
    Q(u) dugum noktalari arasinda PARCALI DOGRUSAL kabul edilir, kuyruklar sabit
    E[L] = integral_0^1 Q(u) du  = agirlikli toplam (agirliklar dugum genislikleri)
Egrilik nerede varsa duzeltme orada olur; simetrik kosullu dagilimda kestirim
Huber ile ayni kalir. Yani duzeltme HEDEFLIDIR, genel bir kaydirma degildir.

Kantil monotonlugu ihlal edilirse (agaclar bagimsiz egitildigi icin olabilir)
satir bazinda siralanarak onarilir.
"""

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
import z1_ortak as Z  # noqa: E402
import z2_tezgah as T  # noqa: E402
from m33_durust import VARSAYILAN  # noqa: E402

U = np.array([0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95])
TUR = 200
TOHUM = 7


def agirlik(u):
    """Parcali dogrusal kantil fonksiyonunun integral agirliklari (kuyruk sabit)."""
    w = np.zeros_like(u)
    w[0] = u[0] + (u[1] - u[0]) / 2
    w[-1] = (1 - u[-1]) + (u[-1] - u[-2]) / 2
    for i in range(1, len(u) - 1):
        w[i] = (u[i] - u[i - 1]) / 2 + (u[i + 1] - u[i]) / 2
    return w


def main():
    t0 = time.time()
    Xtr, ytr, Xte, tr, te = T.matrisler()
    tr["L"] = tr.ly
    msk = Z.maskeler(tr, te)
    A6 = Z.taban()
    Ltr = np.log1p(ytr)
    W = agirlik(U)
    assert abs(W.sum() - 1.0) < 1e-12
    print(f"kantiller {U.tolist()} agirliklar {np.round(W, 4).tolist()}", flush=True)

    ds = lgb.Dataset(Xtr, Ltr)
    Qm = np.zeros((len(Xte), len(U)))
    for i, u in enumerate(U):
        p = dict(VARSAYILAN)
        p.update(
            objective="quantile",
            alpha=float(u),
            metric="quantile",
            learning_rate=0.06,
            lambda_l2=20.0,
            seed=TOHUM,
            bagging_seed=TOHUM + 1,
            feature_fraction_seed=TOHUM + 2,
        )
        Qm[:, i] = lgb.train(p, ds, TUR).predict(Xte)
        print(
            f"  kantil {u:.2f} bitti, ort {Qm[:, i].mean():.4f} ({time.time() - t0:.0f}s)",
            flush=True,
        )

    ihlal = float((np.diff(Qm, axis=1) < 0).mean())
    Qm = np.sort(Qm, axis=1)  # monotonluk onarimi
    L = Qm @ W
    L = np.clip(L, 0.0, 14.0)
    print(f"monotonluk ihlali %{100 * ihlal:.2f}; E[L] ort {L.mean():.4f}", flush=True)

    rap = Z.bitir(L, te, msk, A6, "tuketim_z2_kantil.csv", kirp=2.0)
    rap["parametreler"] = dict(
        kantiller=U.tolist(), agirliklar=W.tolist(), tur=TUR, monoton_ihlal=ihlal
    )
    rap["kantil_ortalamalari"] = Qm.mean(axis=0).tolist()
    np.save(os.path.join(BURA, "z2_kantil_L.npy"), L)
    json.dump(rap, open(os.path.join(BURA, "z2_kantil.json"), "w"), indent=1)
    print(f"TAMAM ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
