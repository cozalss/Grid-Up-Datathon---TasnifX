"""K6-KONTROL -- GURULTU TABANI. Kesinti yonu gercek mi, tesaduf mu?

k6'daki DELTA = P(taban+kesinti) - P(taban). Bu fark iki kaynaktan gelebilir:
  (1) kesintinin gercek bilgisi
  (2) 35 fazladan kolonun feature_fraction=0.8 ornekmesini degistirmesi
      (ayni tohumda bile secilen kolon kumesi baskalasir) -- SAF GURULTU

Ayirmak icin AYNI 35 kolon, ILCE ETIKETLERI KARISTIRILMIS halde eklenir.
Marjinal dagilimlar, gun duzeyi toplamlar, NaN deseni AYNEN korunur; bozulan
tek sey (ilce,gun) eslesmesidir -- yani sadece bilginin kendisi.

  Q(gercek delta) ~= Q(sahte delta)  ->  yon GURULTU, alinmamali
  Q(gercek delta) >> Q(sahte delta)  ->  yon GERCEK bilgi tasiyor

Cikti: k6_p_sahte.npy, k6_kesinti_kontrol.json
"""

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)

from k5_kesinti_veri import anahtarla, filo, kesinti_panel  # noqa: E402
from k6_kesinti_model import AY, HUB, KESIM, L1, TOHUM, TUR_HUB, TUR_L1, V  # noqa: E402
from m30_ozellik import kur, yukle_ham  # noqa: E402
from m33_durust import hizala  # noqa: E402

TOHUM_KARISTIR = 101


def main():
    t0 = time.time()
    tr, te = yukle_ham()
    anahtarla(tr)
    anahtarla(te)
    KP = kesinti_panel(filo(tr, te))
    KKOL = list(KP.columns)

    # ilce etiketlerini KARISTIR: her ilcenin tum zaman serisi baska bir ilceye
    # takilir. Marjinaller ve gun ekseni ayni kalir, bilgi yok olur.
    ilceler = list(KP.index.levels[0])
    rng = np.random.default_rng(TOHUM_KARISTIR)
    perm = list(rng.permutation(ilceler))
    esleme = dict(zip(ilceler, perm))
    sabit = sum(1 for a, b in esleme.items() if a == b)
    KS = KP.copy()
    KS.index = pd.MultiIndex.from_arrays(
        [
            pd.Index([esleme[i] for i in KS.index.get_level_values(0)], name="ilce_key"),
            KS.index.get_level_values(1),
        ]
    )
    KS = KS.sort_index()
    print(f"karistirildi: {len(ilceler)} ilce, yerinde kalan {sabit} ({time.time() - t0:.0f}s)")

    def ekle(X, ilce, tarih):
        idx = pd.MultiIndex.from_arrays([np.asarray(ilce, dtype=object), np.asarray(tarih)])
        d = KS.reindex(idx)
        return pd.concat(
            [
                X.reset_index(drop=True),
                pd.DataFrame({c: d[c].to_numpy(dtype=np.float32) for c in KKOL}),
            ],
            axis=1,
        )

    Xs, ys = [], []
    tavan = pd.Timestamp(KESIM)
    for k in AY:
        kk = pd.Timestamp(k)
        son = min(kk + pd.DateOffset(months=4), tavan)
        gec = tr[tr.tarih <= kk]
        hed = tr[(tr.tarih > kk) & (tr.tarih <= son)]
        if len(hed) == 0:
            continue
        X = kur(gec, hed, kk, set(gec.tanim))
        Xs.append(ekle(X, hed.ilce_key.to_numpy(), hed.tarih.to_numpy()))
        ys.append(np.log1p(hed.tuketim.to_numpy()))
    Xtr = pd.concat(Xs, ignore_index=True)
    ytr = np.concatenate(ys)
    del Xs, ys
    Xte = ekle(kur(tr, te, tavan, set(tr.tanim)), te.ilce_key.to_numpy(), te.tarih.to_numpy())
    Xtr, Xte = hizala(Xtr, Xte)
    Xte = Xte[Xtr.columns]
    print(f"  matris hazir {Xtr.shape} ({time.time() - t0:.0f}s)", flush=True)

    ds = lgb.Dataset(Xtr, ytr)
    parca = {}
    gain = None
    for nm, pk, tur in (("huber", HUB, TUR_HUB), ("l1", L1, TUR_L1)):
        acc = []
        for s in TOHUM:
            p = dict(V)
            p.update(pk)
            p.update(seed=s, bagging_seed=s + 1, feature_fraction_seed=s + 2)
            m = lgb.train(p, ds, tur)
            acc.append(m.predict(Xte))
            if nm == "huber" and s == TOHUM[0]:
                g = pd.Series(m.feature_importance("gain"), index=Xtr.columns)
                gain = float(g[KKOL].sum() / g.sum())
            print(f"  sahte/{nm} tohum {s} ({time.time() - t0:.0f}s)", flush=True)
        parca[nm] = np.mean(acc, axis=0)
    p_sahte = (parca["huber"] + parca["l1"]) / 2
    np.save(os.path.join(BURA, "k6_p_sahte.npy"), p_sahte)

    p_tab = np.load(os.path.join(BURA, "k6_p_taban.npy"))
    p_kes = np.load(os.path.join(BURA, "k6_p_kesinti.npy"))
    dg = p_kes - p_tab
    dsh = p_sahte - p_tab
    Qg, Qs = float((dg**2).mean()), float((dsh**2).mean())
    kes_gain = json.load(open(os.path.join(BURA, "k6_kesinti_model.json"), encoding="utf-8"))[
        "onem"
    ]["kesinti_gain_payi"]
    r = dict(
        aciklama="ilce etiketleri karistirilmis kesinti kolonlari = GURULTU TABANI",
        karistirma_tohumu=TOHUM_KARISTIR,
        yerinde_kalan_ilce=int(sabit),
        Q_gercek_delta=Qg,
        Q_sahte_delta=Qs,
        oran_gercek_bolu_sahte=Qg / Qs if Qs > 0 else float("inf"),
        sinyal_ustu_gurultu_payi=float(min(1.0, Qs / Qg)) if Qg > 0 else float("nan"),
        kosinus_gercek_sahte=float((dg * dsh).mean() / np.sqrt(max(1e-30, Qg * Qs))),
        gain_payi_gercek=kes_gain,
        gain_payi_sahte=gain,
        HUKUM=("Q(gercek) ile Q(sahte) yakinsa fark BILGI DEGIL, kolon ornekleme gurultusudur."),
    )
    print(json.dumps(r, indent=1, ensure_ascii=False))
    json.dump(
        r,
        open(os.path.join(BURA, "k6_kesinti_kontrol.json"), "w", encoding="utf-8"),
        indent=1,
        ensure_ascii=False,
    )
    print(f"BITTI ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
